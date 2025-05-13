'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer
from torch.nn import LayerNorm

def build_text_encoder(args):
    clip_model_dict = {'large': './clip_model/clip-vit-large-patch14',
                       'huge': './clip_model/CLIP-ViT-H-14-laion2B-s32B-b79K',
                       'giga': './CLIP-Giga-config-fixed',
                       }

    clip_preprocess = CLIPImageProcessor(crop_size={'height': 224, 'width': 224},
                                         do_center_crop=True,
                                         do_convert_rgb=True,
                                         do_normalize=True,
                                         do_rescale=True,
                                         do_resize=True,
                                         image_mean=[0.48145466, 0.4578275, 0.40821073],
                                         image_std=[0.26862954, 0.26130258, 0.27577711],
                                         resample=3,
                                         size={'shortest_edge': 224},
                                         )

    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dict[args.clip_model_name], torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.float32, cache_dir=args.cache_dir)

    clip_text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_dict[args.clip_model_name], torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.float32, cache_dir=args.cache_dir)

    tokenizer = CLIPTokenizer.from_pretrained('./clip_model/stable-diffusion-xl-base-1.0', subfolder='tokenizer_2', cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':["[$]"]}) # NOTE: 49408

    return clip_vision_model, clip_preprocess, clip_text_model, tokenizer


import torch
import torch.nn as nn


class CLIPAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CLIPAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 多头自注意力模块
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播，支持 float16 类型。

        :param x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        :return: 输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 确保输入为 FP16

        # 多头自注意力
        attn_output, _ = self.multihead_attn(x, x, x)  # Q, K, V 相同
        x = x + self.dropout1(attn_output)  # 残差连接
        x = self.norm1(x)

        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # 残差连接
        x = self.norm2(x)

        return x


class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 线性注意力层
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播，支持 float16 类型。

        :param x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        :return: 输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 确保输入为 FP16（如果适用）
        seq_len, _ = x.size()

        # 查询、键、值的线性映射
        query = self.query_projection(x)  # (batch_size, seq_len, embed_dim)
        key = self.key_projection(x)  # (batch_size, seq_len, embed_dim)
        value = self.value_projection(x)  # (batch_size, seq_len, embed_dim)

        # 对键和值进行 Softmax 激活
        key = F.softmax(key, dim=-1)
        value = F.softmax(value, dim=-1)

        # 计算线性注意力（矩阵乘法代替了传统的自注意力计算）
        attention_output = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attention_output = torch.matmul(attention_output, value)  # (batch_size, seq_len, embed_dim)

        # 残差连接和归一化
        x = x + self.dropout1(attention_output)  # 残差连接
        x = self.norm1(x)

        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # 残差连接
        x = self.norm2(x)

        return x


class WindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift_size=0, dropout=0.1):
        super(WindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # 确保窗口大小是可以整除的
        assert window_size % num_heads == 0, "Window size should be divisible by number of heads."

        self.scale = embed_dim ** -0.5  # 缩放因子

        # 查询、键、值的投影层
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # LayerNorm
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def window_attention(self, query, key, value):
        """
        在窗口内计算注意力
        :param query: (batch_size, num_patches, embed_dim)
        :param key: (batch_size, num_patches, embed_dim)
        :param value: (batch_size, num_patches, embed_dim)
        """
        q = query.view(-1, self.window_size, self.window_size, self.embed_dim)  # (batch_size, window_size, window_size, embed_dim)
        k = key.view(-1, self.window_size, self.window_size, self.embed_dim)  # (batch_size, window_size, window_size, embed_dim)
        v = value.view(-1, self.window_size, self.window_size, self.embed_dim)  # (batch_size, window_size, window_size, embed_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, window_size, window_size, window_size)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)  # (batch_size, window_size, window_size, embed_dim)
        return output.view(-1, self.window_size * self.window_size, self.embed_dim)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        :return: 输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()

        # 查询、键、值的线性映射
        query = self.query_projection(x)  # (batch_size, seq_len, embed_dim)
        key = self.key_projection(x)  # (batch_size, seq_len, embed_dim)
        value = self.value_projection(x)  # (batch_size, seq_len, embed_dim)

        # 分割成窗口，并进行窗口内的注意力计算
        window_query = query.view(batch_size, seq_len // self.window_size, self.window_size, self.embed_dim)
        window_key = key.view(batch_size, seq_len // self.window_size, self.window_size, self.embed_dim)
        window_value = value.view(batch_size, seq_len // self.window_size, self.window_size, self.embed_dim)

        # 计算每个窗口内的注意力
        output = self.window_attention(window_query, window_key, window_value)

        # 残差连接和归一化
        x = x + self.dropout(output)  # 残差连接
        x = self.norm1(x)

        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # 残差连接
        x = self.norm2(x)

        return x


class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-work embedding.
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/phi.py
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        #x = F.normalize(x, dim=-1)
        return self.layers(x)


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters, decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.collected_params = None

        self.decay = decay
        self.optimization_step = 0

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        value = (1 + self.optimization_step) / (10 + self.optimization_step)
        one_minus_decay = 1 - min(self.decay, value)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict.
        This method is used by accelerate during checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Loads the ExponentialMovingAverage state.
        This method is used by accelerate during checkpointing to save the ema state dict.
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.optimization_step = state_dict["optimization_step"]
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.shadow_params = state_dict["shadow_params"]
        if not isinstance(self.shadow_params, list):
            raise ValueError("shadow_params must be a list")
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError("shadow_params must all be Tensors")

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            if not isinstance(self.collected_params, list):
                raise ValueError("collected_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.collected_params):
                raise ValueError("collected_params must all be Tensors")
            if len(self.collected_params) != len(self.shadow_params):
                raise ValueError("collected_params and shadow_params must have the same length")


class PIC2WORD(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)
