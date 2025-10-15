import json
import os
import gc
import random
import math
import torch.nn as nn
from memory_profiler import profile
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Set
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF_with_3
from models import build_text_encoder, Phi, EMAModel,CLIPAttention
from utils import extract_image_features, extract_embeddings_with_attention,extract_pseudo_tokens_without_msmp
from validate import cirr_compute_val_text
from loader import CIRRDataset
from PIL import Image
import transformers
from transformers import get_scheduler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from torchvision import transforms
import torch.multiprocessing as mp
from validate import cirr_compute_val_metrics_with_FGIA



# --cirr_dataset_path ./PatterCIR



logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--output_dir", default="./train_out/attention1", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--logging_dir", default="logs", type=str, help="tensorboard logs will saved here")
    parser.add_argument("--cache_dir", default="./hf_models", type=str,
                        help="Path to model cache folder")
    parser.add_argument("--report_to", default="tensorboard", type=str, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--clip_model_name", default="large", type=str,
                        help="CLIP model to use, e.g 'large', 'giga'")
    parser.add_argument("--cirr_dataset_path", default="./PatternCIR",type=str, help="Path to CIRR dataset", required=True)
    parser.add_argument("--keywords_path", type=str, help="Path to keywords json file")
    parser.add_argument("--resume", default="./train_out/checkpoints_FGIA/attention_best.pt", type=str, help="Path to pretrained attention ckpt")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_train_steps", type=int, default=50000, help="Total number of training steps to perform")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout probability for the attention")
    parser.add_argument("--l2_normalize",default=False, action="store_true", help="Whether or not to use l2 normalization")
    parser.add_argument("--batch_size", default=32, type=int, help="Attention training batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")
    parser.add_argument("--mixed_precision", default="fp16", type=str, choices=["no", "fp16", "bf16"], help="mixed precision")
    parser.add_argument("--validation_steps", default=1000, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--checkpointing_steps", default=1000, type=int, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--seed", type=int, default=12345, help="seed for reproducibility")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class CirrImageDataset(Dataset):
    def __init__(self, dataset_path, json_file, image_encoder,accelerator,transform=None):

        """
        Args:
            dataset_path (str): CIRR数据集路径
            json_file (str): 训练集JSON文件路径
            image_encoder: CLIP图像编码器
            transform: 对图像进行转换的预处理函数
        """
        self.dataset_path = dataset_path
        self.json_file = json_file
        self.image_encoder = image_encoder
        self.transform = transform
        # self.decive = accelerator.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 读取JSON文件，获取图像路径
        with open(json_file, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像的相对路径
        # mp.set_start_method("spawn", force=True)
        image_name, image_path = list(self.image_paths.items())[idx]
        image_path = os.path.join( self.dataset_path, image_path)

            # 加载图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with torch.no_grad():
            image_features = self.image_encoder(pixel_values=image.unsqueeze(0).to("cuda"))[0]
            image_features = torch.nn.functional.normalize(image_features, dim=-1)  # 标准化特征

        return image_features.squeeze(0).to('cpu'), image_name



def save_attention(name: str, cur_epoch: int, model_to_save: CLIPAttention, training_path: Path) -> None:
    """
    Save the weights of Phi during training
    """
    models_path = os.path.join(training_path, "checkpoints")
    os.makedirs(models_path, exist_ok=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, os.path.join(models_path, f'{name}.pt'))

# @profile
def train_attention(args):
    # We are going to use the pre-extracted clip image features. so we do not need image_encoder anymore.
    mp.set_start_method("spawn", force=True)
    ### init accelerator here
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    ### Define the text encoder from clip
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)



    ### GPU handling
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)



    ### Define the train datasets
    print('pytorch loader')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        # CLIP标准化
    ])

    # 创建CIRR数据集和DataLoader
    dataset_path = './cirr'


    attention_model = CLIPAttention(embed_dim=768, num_heads=4).to(accelerator.device, dtype=torch.float32)
    cirr_relative_val_dataset = CIRRDataset(args.cirr_dataset_path, "test", 'relative', clip_preprocess)
    cirr_classic_val_dataset = CIRRDataset(args.cirr_dataset_path, 'test', 'classic', clip_preprocess)
    cirr_val_index_features, cirr_val_index_names = extract_image_features(cirr_classic_val_dataset, image_encoder)

    if args.l2_normalize:
        cirr_val_index_features = F.normalize(cirr_val_index_features, dim=-1)

    if args.resume:
        attention_model.load_state_dict(
                torch.load(args.resume, map_location=accelerator.device)[
                attention_model.__class__.__name__])





    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(attention_model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes,
    )



    if accelerator.is_main_process:
        accelerator.init_trackers("zeroshot-cir", config=vars(args))


    attention_model, optimizer, lr_scheduler = accelerator.prepare(
        attention_model, optimizer, lr_scheduler
    )


    def check_gpu(check_point,idx):
        print(f"Allocated memory check_point{check_point} iteration{idx}: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Reserved memory check_point{check_point} iteration{idx}: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

    attention_model.train()
    global_step = 0
    epoch_loss = 0
    best_recall = -1






    if accelerator.is_main_process:


        cirr_val_image_embedding, cirr_val_ref_names_list = extract_embeddings_with_attention(image_encoder,
                                                                                         attention_model,
                                                                                         cirr_relative_val_dataset,
                                                                                         args)

        cirr_val_ref_names_list = extract_pseudo_tokens_without_msmp(image_encoder,cirr_relative_val_dataset,args)



        # Compute the CIRR validation metrics
        cirr_results_dict = cirr_compute_val_metrics_with_FGIA(cirr_relative_val_dataset, text_encoder,
                                                     cirr_val_index_features, cirr_val_index_names,
                                                     args,cirr_val_ref_names_list,cirr_val_image_embedding)
        check_list = ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_recall_at10', 'cirr_recall_at50']
        for check_key in check_list:
            accelerator.log({f"validate/{check_key}": cirr_results_dict[check_key]}, step=global_step)
        print(json.dumps(cirr_results_dict, indent=4))

    if args.checkpointing_steps:
        if cirr_results_dict['cirr_recall_at1'] > best_recall:
            best_recall = cirr_results_dict['cirr_recall_at1']
            logger.info(f"best model saving... step: {global_step}")


    gc.collect()
    torch.cuda.empty_cache()





if __name__ == '__main__':
    args = parse_args()

    train_attention(args)
