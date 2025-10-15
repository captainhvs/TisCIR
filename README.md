## PatternCIR Benchmark and TisCIR: Advancing Zero-Shot Composed Image Retrieval in Remote Sensing（IJCAI 2025）


[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://www.ijcai.org/proceedings/2025/0171.pdf)




Welcome to the official implementation of PatternCIR Benchmark and TisCIR!



Explore **PatternCIR**, the first fine-grained benchmark dataset for Composed Image Retrieval in remote sensing.   
Alongside it, our **TisCIR**, a pioneering framework driving the next generation of Zero-Shot Composed Image Retrieval (ZS-CIR) in the remote sensing domain. Experience how our benchmark and zero-shot model redefine compositional understanding and retrieval capabilities through the lens of pattern intelligence.


**Authors**: 

**[Zhechun Liang](https://github.com/captainhvs)\*<sup>1</sup>, Tao Huang\*<sup>2</sup>, Fangfang Wu\*<sup>3</sup>,Shiwen Xue\*<sup>4</sup>,Zhenyu Wang\*<sup>5</sup>, Weisheng Dong\*<sup>6</sup>,Xin Li\*<sup>7</sup>,Guangming Shi\*<sup>8</sup>**

<sup>1</sup> Xidian University <sup>2</sup> State University of New York at Albany <sup>3</sup> Key Laboratory of Intelligent Perception and Image Understanding of Ministry of Education


## ⭐ Overview



### Text-image Sequential Training of Composed Image Retrieval (TisCIR)

<img width="400" height="200" alt="stage1" src="https://github.com/user-attachments/assets/94c00e10-74fa-4917-8d7d-159e7af6b4e7" /><br><br>

<img width="400" height="200" alt="stage2" src="https://github.com/user-attachments/assets/293371a3-4f9c-4850-8ea1-01ccb9d66179" /><br><br>

<img width="400" height="200" alt="stage3" src="https://github.com/user-attachments/assets/01df7152-665c-4e9d-90fb-2f77137f234b" /><br><br>

TisCIR enhances zero-shot composed image retrieval by sequentially training on text and image features. It first extracts structured semantics from text via a Multiple Self-Masking Projection (MSMP) module, then refines image embeddings with a Fine-Grained Image Attention (FGIA) module to remove conflicting information. During retrieval, both refined embeddings are combined for more accurate and semantically consistent matching.



## 🛠️ Installation
Get started with TisCIR by installing the necessary dependencies:

```bash
$ cd TisCIR
$ conda create -n tiscir python==3.9
$ conda activate tiscir
$ pip install -r requirements.txt
```


## 📂 Dataset Preparation

Our PatternCIR dataset, including the training, validation, and test sets, is fully open-source.

Please refer to [here](https://1drv.ms/u/c/3181ea346496e56a/EWKxCKAC8kxBjnRvcMfU4bMBWIkx4KtHRy7_q1R9suYE3A?e=O7inh2) to download zip file.


## 💯 How to Evaluate TisCIR


When evaluating TisCIR, first download the PatternCIR dataset and the FGIA checkpoint `.pt` files.  

- After extracting the PatternCIR dataset, place it in the **project root directory**.  
- The FGIA checkpoints are by default located in `./train_out/checkpoints_FGIA/`.


Evaluate TisCIR on the PatternCIR test set with the following command:


```bash
$ python test.py \
--cirr_dataset_path /path/to/PatternCIR \
--resume  /path/to/FGIA_checkpoints
```


## 📚 How to Train TisCIR


When training TisCIR, the process involves two sequential training stages. The command for the first stage is as follows:

```bash
$ python train_MSMP.py \
--cirr_dataset_path /path/to/PatternCIR \
--position_id 1/2/3
```
In the first stage, three projections need to be trained separately. The `position_id` should be set to **1**, **2**, and **3**, respectively.

The trained weights will be saved in `./train_out/checkpoints_MSMP_{position_id}/`, where `{position_id}` should be replaced with the corresponding projection ID (1, 2, or 3).

The command for the second stage is as follows:
```bash
$ python train_FGIA.py \
--cirr_dataset_path /path/to/PatternCIR 
```
The trained weights will be saved in `./train_out/checkpoints_FGIA/`.


## Citation

```
@inproceedings{ijcai2025p171,
  title     = {PatternCIR Benchmark and TisCIR: Advancing Zero-Shot Composed Image Retrieval in Remote Sensing},
  author    = {Liang, Zhechun and Huang, Tao and Wu, Fangfang and Xue, Shiwen and Wang, Zhenyu and Dong, Weisheng and Li, Xin and Shi, Guangming},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {1530--1538},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/171},
  url       = {https://doi.org/10.24963/ijcai.2025/171},
}
```


## Acknowledgement
We would like to express our special gratitude to the authors of [LinCIR](https://github.com/navervision/lincir) for their invaluable contributions, as our code draws significant inspiration from this open-source project.


