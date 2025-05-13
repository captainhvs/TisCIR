'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import torch
from clip.model import CLIP
from transformers import CLIPTextModelWithProjection




def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    Copy-paste from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L679-L693
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def encode_with_pseudo_tokens_HF(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1, return_last_states=False) -> torch.Tensor:
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    x = torch.where(text.unsqueeze(-1) == 259,pseudo_tokens.unsqueeze(1).type(clip_model.dtype),x) ###########
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(inputs_embeds=x,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False)
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
          text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
          ]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)

    if return_last_states:
        return x, x_last
    else:
        return x


def encode_with_pseudo_tokens_HF_with_3(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens1: torch.Tensor,
                                        pseudo_tokens2: torch.Tensor,pseudo_tokens3: torch.Tensor, return_last_states=False) -> torch.Tensor:
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)

    # [batch_size, n_ctx, d_model]




    # # 创建一个 mask，标记 text 中值为 259 的位置
    # mask = (text == 259).unsqueeze(-1)  # [32, 77, 1]
    # # 创建 pseudo_tokens 的映射矩阵，形状与 x 相同
    # pseudo_tokens_combined = torch.stack([pseudo_tokens1, pseudo_tokens2, pseudo_tokens3], dim=0)  # [3, 768]
    # # 创建一个索引矩阵，指示 text 的第 1、2、3 个出现的 259 分别使用 pseudo_token1、pseudo_token2 和 pseudo_token3
    # index_map = (torch.cumsum((text == 259).int(), dim=1) - 1).clamp(min=0, max=2) # [32, 77]
    #
    # index_map = index_map.unsqueeze(-1)  # [32, 77, 1]
    #
    # # 选择 pseudo_tokens_combined 中对应的 token
    # selected_pseudo_tokens = pseudo_tokens_combined[index_map.squeeze(-1)]  # [32, 77, 768]
    # # 替换 x 中对应位置的值
    # x = torch.where(mask, selected_pseudo_tokens.type(clip_model.dtype), x)




    # 假设 mask 是一个标志着 259 位置的张量，形状为 [batch_size, seq_len, 1]
    mask = (text == 259).unsqueeze(-1).to(x.device)  # [batch_size, seq_len, 1]

    # 创建一个索引矩阵，指示 text 的每个位置分别使用 pseudo_token1、pseudo_token2、pseudo_token3
    # index_map 由 0、1、2 组成，每个位置对应 0 -> pseudo_token1，1 -> pseudo_token2，2 -> pseudo_token3
    index_map = (torch.cumsum((text == 259).int(), dim=1) - 1).clamp(min=0, max=2)  # [batch_size, seq_len]
    #
    # torch.set_printoptions(threshold=100000)
    # print(index_map)

    # 选择 pseudo_token1, pseudo_token2, pseudo_token3
    pseudo_tokens_combined = torch.stack([pseudo_tokens1, pseudo_tokens2, pseudo_tokens3],
                                          dim=1)  # [batch_size, 3, feature_dim]

    # batch_size=index_map.shape[0]
    # seq_len=index_map.shape[1]
    # feature_dim= pseudo_tokens_combined.shape[2]
    # selected_pseudo_tokens = torch.zeros(batch_size, seq_len, feature_dim).to(x.device)
    #
    # for i in range(batch_size):  # 遍历每个 batch
    #     for j in range(seq_len):  # 遍历每个 seq_len
    #         token_index = index_map[i, j]  # 获取当前索引位置对应的伪 token 索引
    #         selected_pseudo_tokens[i, j] = pseudo_tokens_combined[i, token_index]


    # 使用 index_map 选择对应的 pseudo token
    selected_pseudo_tokens = pseudo_tokens_combined[
        torch.arange(index_map.shape[0]).unsqueeze(1), index_map]  # [batch_size, seq_len, feature_dim]

    # 替换 x 中对应位置的值
    x = torch.where(mask, selected_pseudo_tokens.type(clip_model.dtype), x)




    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(inputs_embeds=x,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False)
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
          text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
          ]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)

    if return_last_states:
        return x, x_last
    else:
        return x





def encode_with_pseudo_tokens_HF_without_phi(clip_model: CLIPTextModelWithProjection, text: torch.Tensor,
                              num_tokens=1, return_last_states=False) -> torch.Tensor:
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(inputs_embeds=x,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False)
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
          text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
          ]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)

    if return_last_states:
        return x, x_last
    else:
        return x