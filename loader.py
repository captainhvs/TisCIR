'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import os
import functools
import glob
import random
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
import webdataset as wds
import spacy
import numpy as np
import datasets


def extract_keywords(spacy_nlp, caption):
    candidates = []
    nlp_caption = caption

    doc = spacy_nlp(nlp_caption)

    tmp = ''
    for word in doc:
        if word.pos_ == 'ADJ':
            if tmp == '':
                tmp += word.text
            else:
                tmp += ' ' + word.text
        elif word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
            if tmp == '':
                tmp += word.text
            else:
                tmp += ' ' + word.text
        else:
            if tmp != '':
                candidates.append(tmp)
            tmp = ''
    if tmp != '':
        candidates.append(tmp)

    candidates = list(set(candidates))

    return candidates


def extract_keywords_spacy(spacy_nlp, caption):
    sequences = []
    current_sequence = []
    doc = spacy_nlp(caption)
    for token in doc:
        # Check if the token is a noun, proper noun, or adjective
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'DET']:
            current_sequence.append(token.text)
        else:
            # If we encounter a token that's not one of the desired POS and current_sequence is not empty
            if current_sequence:
                sequences.append(" ".join(current_sequence))
                current_sequence = []

    # Adding any remaining sequence after the loop
    if current_sequence:
        sequences.append(" ".join(current_sequence))

    return sequences


def clean_caption(caption, tokenizer):
    if caption is None:
        caption = ''
    if '<PERSON>' in caption: # to handle with GCC12M
        caption = caption.replace('<PERSON>', 'person')
    caption = caption.lower().replace('$', '').strip()
    tokens = tokenizer.encode(caption, padding='longest', return_tensors='pt')
    if tokens.shape[1] > 77:
        caption = tokenizer.batch_decode(tokens[:,1:76])[0]
    return caption


# def preprocess_precomputed_base(sample, spacy_nlp, keywords_list, tokenizer):
#     '''
#     'image_feature.npy','json'
#     '''
#     image_feature, image_feature_giga, meta = sample
#
#     caption = clean_caption(meta['source_caption'], tokenizer)
#
#     keywords = ['']
#     try:
#         keywords = extract_keywords_spacy(spacy_nlp, caption)
#     except Exception as e:
#         #print(e)
#         pass
#
#     # for keywords
#     # indicator = 1
#     # replaced_caption = caption
#     # for keyword in keywords:
#     #     if keyword != '' and keyword in caption:
#     #         replaced_caption = replaced_caption.replace(keyword, '[$]')
#     #     else:
#     #         tmp_keywords = caption.split(' ')
#     #         if len(tmp_keywords) > 0:
#     #             selected_keywords = random.sample(tmp_keywords, k=min(int(len(tmp_keywords) * 1.0), 1))
#     #             for selected_keyword in selected_keywords:
#     #                 replaced_caption = replaced_caption.replace(selected_keyword, '[$]')
#     #         else:
#     #             replaced_caption = f'a photo of [$] that {caption}'
#     #             indicator = 0
#     #         break
#
#     replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'
#
#     token_dict = tokenizer(text=caption, return_tensors='pt', padding='max_length', truncation=True)
#     tokens, attention_mask = token_dict['input_ids'][0], token_dict['attention_mask'][0]
#
#     replaced_token_dict = tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
#     replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]
#
#     replaced_tokens = torch.where(replaced_tokens == 49408,
#                                   torch.ones_like(replaced_tokens) * 259,
#                                   replaced_tokens)
#
#     if 259 not in replaced_tokens:
#         replaced_caption = 'a photo of [$]'
#         replaced_token_dict = tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
#         replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]
#
#         replaced_tokens = torch.where(replaced_tokens == 49408,
#                                       torch.ones_like(replaced_tokens) * 259,
#                                       replaced_tokens)
#         indicator = 0
#
#     new_sample = [tokens, replaced_tokens, indicator]
#
#     return tuple(new_sample)





class CaptionDataset_with_id(Dataset):
    def __init__(self, captions, tokenizer, position_id):
        self.captions = captions
        self.tokenizer = tokenizer
        self.id = position_id

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]

        caption = clean_caption(caption, self.tokenizer)



        # for keywords
        indicator = 1
        start_1 = "a satellite image of"
        start_2 = "with"
        start_3 = "in the background of"

        # 初始化列表
        key_part = []


        part_1 = caption[caption.find(start_1) + len(start_1): caption.find(start_2)].strip()
        key_part.append(part_1)

        part_2 = caption[caption.find(start_2) + len(start_2): caption.find(start_3)].strip()
        key_part.append(part_2)


        part_3 = caption[caption.find(start_3) + len(start_3):].strip()
        key_part.append(part_3)

        replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'

        result_caption = ""

        count = 0  # 记录[$]的出现次数
        i = 0  # 遍历字符串的索引

        while i < len(replaced_caption):
            if replaced_caption[i:i + 3] == "[$]":  # 检测到[$]
                count += 1
                if count == self.id:
                    # 保留第self.id个[$]
                    result_caption += "[$]"
                else:
                    # 替换为key_part中的对应部分
                    result_caption += key_part[count - 1]
                i += 3  # 跳过"[$]"
            else:
                # 非[$]部分直接添加到结果字符串
                result_caption += replaced_caption[i]
                i += 1
        replaced_caption=result_caption

        token_dict = self.tokenizer(text=caption, return_tensors='pt', padding='max_length', truncation=True)
        tokens, attention_mask = token_dict['input_ids'][0], token_dict['attention_mask'][0]

        replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
        replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]
        replaced_tokens = torch.where(replaced_tokens == 49408,
                                      torch.ones_like(replaced_tokens) * 259,
                                      replaced_tokens)

        # if 259 not in replaced_tokens:
        #     replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'
        #     replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
        #     replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]
        #
        #     replaced_tokens = torch.where(replaced_tokens == 49408,
        #                                   torch.ones_like(replaced_tokens) * 259,
        #                                   replaced_tokens)
        #     indicator = 0

        return tokens, replaced_tokens, indicator

class CaptionDataset(Dataset):
    def __init__(self, captions, tokenizer, spacy_nlp):
        self.captions = captions
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]

        caption = clean_caption(caption, self.tokenizer)

        keywords = [""]
        try:
            keywords = extract_keywords_spacy(self.spacy_nlp, caption)
        except Exception as e:
            #print(e)
            pass

        # for keywords
        indicator = 1
        replaced_caption = caption

        if len(keywords) == 0:
            keywords = [""]

        # for keyword in keywords:
        #     if keyword != '' and keyword in caption:
        #         replaced_caption = replaced_caption.replace(keyword, '[$]')
        #     else:
        #         tmp_keywords = caption.split(' ')
        #         if len(tmp_keywords) > 0:
        #             selected_keywords = random.sample(tmp_keywords, k=min(int(len(tmp_keywords) * 1.0), 1))
        #             for selected_keyword in selected_keywords:
        #                 replaced_caption = replaced_caption.replace(selected_keyword, '[$]')
        #         else:
        #             replaced_caption = f'a photo of [$] that {caption}'
        #             indicator = 0
        #         break

        replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'
        token_dict = self.tokenizer(text=caption, return_tensors='pt', padding='max_length', truncation=True)
        tokens, attention_mask = token_dict['input_ids'][0], token_dict['attention_mask'][0]

        replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
        replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]
        replaced_tokens = torch.where(replaced_tokens == 49408,
                                      torch.ones_like(replaced_tokens) * 259,
                                      replaced_tokens)

        if 259 not in replaced_tokens:
            replaced_caption = f'A satellite image of [$] with [$] in the background of [$]'
            replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True)
            replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]

            replaced_tokens = torch.where(replaced_tokens == 49408,
                                          torch.ones_like(replaced_tokens) * 259,
                                          replaced_tokens)
            indicator = 0

        return tokens, replaced_tokens, indicator





def build_loader_with_id(args, tokenizer, accelerator,position_id):
    data_names = {'dataset1': './patterncir_captions',
                  'dataset2': './patterncir_captions',
                  # 'dataset3': '/mnt/sdd1/pyprojects/lincir/midjourney-prompts-only',  '/mnt/sdd1/pyprojects/lincir/gcc_caption_only',
                  }

    for k, v in data_names.items():
        if not os.path.exists(os.path.join('./datasets', k)):
            if accelerator.is_main_process:
                print('Downloading captions is required')
                db = datasets.load_dataset(v, cache_dir=os.path.join('./datasets', k))

    captions = []
    for k, v in data_names.items():
        db = datasets.load_dataset(v, cache_dir=os.path.join('./datasets', k))
        captions += db['train']['text']

    dataset = CaptionDataset_with_id(captions, tokenizer,position_id)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

    return data_loader


class FashionIQDataset(Dataset):
    """
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Remove duplicats from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]
                    target_name = self.triplets[index]['target']
                    target_image_path = self.dataset_path / 'images' / f"{target_name}.jpg"
                    target_image = self.preprocess(PIL.Image.open(target_image_path), return_tensors='pt')['pixel_values'][0]

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_name,
                        'relative_captions': relative_captions
                    }

                elif self.split == 'test':
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.dataset_path / 'images' / f"{image_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path), return_tensors='pt')['pixel_values'][0]

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
   Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
   CIRR dataset class for PyTorch dataloader.
   The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates

        if split == "test":
            split = "test1"
            self.split = "test1"

        # Validate inputs
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # Remove duplicates from triplets
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['reference'] not in seen:
                    seen.add(triplet['reference'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get a mapping from image name to relative path
        with open(dataset_path / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                relative_caption = self.triplets[index]['caption']

                if self.split in ['train', 'val','test1']:
                    reference_image_path = self.dataset_path / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.dataset_path / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path), return_tensors='pt')['pixel_values'][0]

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_hard_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members
                    }

                # elif self.split == 'test1':
                #     pair_id = self.triplets[index]['pairid']
                #     reference_image_path = self.dataset_path / self.name_to_relpath[reference_name]
                #     reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]
                #     return {
                #         'reference_image': reference_image,
                #         'reference_name': reference_name,
                #         'relative_caption': relative_caption,
                #         'group_members': group_members,
                #         'pair_id': pair_id
                #     }

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.dataset_path / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im, return_tensors='pt')['pixel_values'][0]

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path), return_tensors='pt')['pixel_values'][0]

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path), return_tensors='pt')['pixel_values'][0]

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path), return_tensors='pt')['pixel_values'][0]
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
