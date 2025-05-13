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


# --cirr_dataset_path ./PatternCIR




logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--output_dir", default="./train_out/checkpoints_FGIA", type=str,help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--logging_dir", default="logs", type=str, help="tensorboard logs will saved here")
    parser.add_argument("--cache_dir", default="./hf_models", type=str,help="Path to model cache folder")
    parser.add_argument("--report_to", default="tensorboard", type=str, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--clip_model_name", default="large", type=str,help="CLIP model to use, e.g 'large', 'giga'")
    parser.add_argument("--cirr_dataset_path", default="./PatternCIR",type=str, help="Path to CIRR dataset", required=True)
    parser.add_argument("--keywords_path", type=str, help="Path to keywords json file")
    parser.add_argument("--phi_resume1", default="./train_out/checkpoints_MSMP_1/phi_best.pt", type=str, help="Path to pretrained phi ckpt")
    parser.add_argument("--phi_resume2", default="./train_out/checkpoints_MSMP_2/phi_best.pt", type=str, help="Path to pretrained phi ckpt")
    parser.add_argument("--phi_resume3", default="./train_out/checkpoints_MSMP_3/phi_best.pt", type=str, help="Path to pretrained phi ckpt")
    parser.add_argument("--resume", default=None, type=str, help="Path to pretrained attention ckpt")
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

    ### Define the phi model
    phi1 = Phi(input_dim=text_encoder.config.projection_dim,
                    hidden_dim=text_encoder.config.projection_dim * 4,
                    output_dim=text_encoder.config.hidden_size, dropout=args.dropout)

    phi2 = Phi(input_dim=text_encoder.config.projection_dim,
               hidden_dim=text_encoder.config.projection_dim * 4,
               output_dim=text_encoder.config.hidden_size, dropout=args.dropout)

    phi3 = Phi(input_dim=text_encoder.config.projection_dim,
               hidden_dim=text_encoder.config.projection_dim * 4,
               output_dim=text_encoder.config.hidden_size, dropout=args.dropout)

    ### Load the phi model
    phi1.load_state_dict(
        torch.load(args.phi_resume1, map_location=accelerator.device)[
            phi1.__class__.__name__])

    phi2.load_state_dict(
        torch.load(args.phi_resume2, map_location=accelerator.device)[
            phi2.__class__.__name__])

    phi3.load_state_dict(
        torch.load(args.phi_resume3, map_location=accelerator.device)[
            phi3.__class__.__name__])




    ### GPU handling
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16



    phi1.to(accelerator.device, dtype=weight_dtype)
    phi2.to(accelerator.device, dtype=weight_dtype)
    phi3.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)



    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)

    phi1.requires_grad_(False)
    phi2.requires_grad_(False)
    phi3.requires_grad_(False)







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
    # json_file = os.path.join(args.cirr_dataset_path,dataset_path, './image_splits/split.rc2.train.json')

    json_file = os.path.join(args.cirr_dataset_path, dataset_path, './image_splits/split.rc2.val.json')
    batch_size = args.batch_size
    cirr_train_dataset = CirrImageDataset(args.cirr_dataset_path, json_file, image_encoder, accelerator,transform)
    train_loader = DataLoader(cirr_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False,drop_last=True,persistent_workers=True)
    # train_dataset = accelerator.prepare(train_loader)
    attention_model = CLIPAttention(embed_dim=768, num_heads=4).to(accelerator.device, dtype=torch.float32)

    cirr_relative_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'relative', clip_preprocess)
    cirr_classic_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'classic', clip_preprocess)
    cirr_val_index_features, cirr_val_index_names = extract_image_features(cirr_classic_val_dataset, image_encoder)
    if args.l2_normalize:
        cirr_val_index_features = F.normalize(cirr_val_index_features, dim=-1)

    if args.resume:
        attention_model.load_state_dict(
                torch.load(args.resume, map_location=accelerator.device)[
                attention_model.__class__.__name__])




    #Define optimizer and lr
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


    attention_model, optimizer, lr_scheduler, train_dataset = accelerator.prepare(
        attention_model, optimizer, lr_scheduler, train_loader
    )


    def check_gpu(check_point,idx):
        print(f"Allocated memory check_point{check_point} iteration{idx}: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Reserved memory check_point{check_point} iteration{idx}: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

    attention_model.train()
    global_step = 0
    epoch_loss = 0
    best_recall = -1
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    while True:
        for idx, (batch, name) in enumerate(train_dataset):
            # epoch_loss = 0  3412


            image_embeddings_1=batch.to(accelerator.device)
            input_features = image_embeddings_1.clone()
            # input_features += 1.0 * torch.rand(input_features.shape[0], device=input_features.device).unsqueeze(
            #     -1) * torch.randn(input_features.shape, device=input_features.device)

            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)
                image_embeddings_1 = F.normalize(image_embeddings_1, dim=-1)




            with torch.no_grad():
                estimated_token_embeddings1 = phi1(input_features)
                estimated_token_embeddings2 = phi2(input_features)
                estimated_token_embeddings3 = phi3(input_features)

            replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'

            # for i in range(0,batch.shape[0],1):
            # replaced_caption.append(f'A satellite image of [$] with [$] in the background of [$]')



            replaced_token_dict = tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length',
                                                 truncation=True)
            replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], \
            replaced_token_dict['attention_mask'][0]
            replaced_tokens = torch.where(replaced_tokens == 49408,torch.ones_like(replaced_tokens) * 259,replaced_tokens)

            replaced_tokens = replaced_tokens.unsqueeze(0).repeat(batch_size, 1)
            replaced_tokens=replaced_tokens.to(accelerator.device)
            replaced_text_embeddings_3, replaced_last_hidden_states = encode_with_pseudo_tokens_HF_with_3(text_encoder,
                                                                                                 replaced_tokens,
                                                                                                 estimated_token_embeddings1,
                                                                                                 estimated_token_embeddings2,
                                                                                                 estimated_token_embeddings3,
                                                                                                 return_last_states=True)
            if args.l2_normalize:
                replaced_text_embeddings_3 = F.normalize(replaced_text_embeddings_3, dim=-1)

            image_embeddings_1=image_embeddings_1.to(torch.float32)
            replaced_text_embeddings_3 = replaced_text_embeddings_3.to(torch.float32)


            # with accelerator.accumulate(attention_model):
            image_embeddings_2 = attention_model(image_embeddings_1)
            difference = image_embeddings_1 - image_embeddings_2 - replaced_text_embeddings_3
            epoch_loss = torch.norm(difference, p=2)
            accelerator.backward(epoch_loss)


            if accelerator.sync_gradients and args.max_grad_norm is not None:
                accelerator.clip_grad_norm_(attention_model.parameters(), arg.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            del replaced_text_embeddings_3,image_embeddings_2,image_embeddings_1,difference,input_features,replaced_tokens
            torch.cuda.empty_cache()



            global_step += 1
            progress_bar.update(1)
            accelerator.log({"train/train_loss": epoch_loss.item()}, step=global_step)




            if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    logger.info(f"model saving... step: {global_step}")
                    save_attention(f"attention_{global_step:09}", global_step, accelerator.unwrap_model(attention_model), args.output_dir)
                    save_attention(f"attention_latest", global_step, accelerator.unwrap_model(attention_model), args.output_dir)

                    cirr_val_image_embedding, cirr_val_ref_names_list = extract_embeddings_with_attention(image_encoder,
                                                                                                     attention_model,
                                                                                                     cirr_relative_val_dataset,
                                                                                                     args)

                    cirr_val_ref_names_list = extract_pseudo_tokens_without_msmp(image_encoder,
                                                                                 cirr_relative_val_dataset, args)

                    # Compute the CIRR validation metrics
                    cirr_results_dict = cirr_compute_val_metrics_with_FGIA(cirr_relative_val_dataset, text_encoder,
                                                                           cirr_val_index_features,
                                                                           cirr_val_index_names,
                                                                           args, cirr_val_ref_names_list,
                                                                           cirr_val_image_embedding)
                    check_list = ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_recall_at10', 'cirr_recall_at50']
                    for check_key in check_list:
                        accelerator.log({f"validate/{check_key}": cirr_results_dict[check_key]}, step=global_step)
                    print(json.dumps(cirr_results_dict, indent=4))

                if args.checkpointing_steps:
                    if cirr_results_dict['cirr_recall_at1'] > best_recall:
                        best_recall = cirr_results_dict['cirr_recall_at1']
                        logger.info(f"best model saving... step: {global_step}")
                        save_attention("attention_best", global_step, accelerator.unwrap_model(attention_model), args.output_dir)
            gc.collect()
            torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                exit()
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    args = parse_args()

    train_attention(args)