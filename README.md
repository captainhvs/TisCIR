## PatternCIR Benchmark and TisCIR: Advancing Zero-Shot Composed Image Retrieval in Remote Sensing（IJCAI 2025）


[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4090.pdf)




Welcome to the official implementation of PatternCIR Benchmark and TisCIR!



Explore **PatternCIR**, the first fine-grained benchmark dataset for Composed Image Retrieval in remote sensing.   
Alongside it, our **TisCIR**, a pioneering framework driving the next generation of Zero-Shot Composed Image Retrieval (ZS-CIR) in the remote sensing domain. Experience how our benchmark and zero-shot model redefine compositional understanding and retrieval capabilities through the lens of pattern intelligence.


**Authors**: 

**[Zhechun Liang](https://github.com/captainhvs)\*<sup>1</sup>, Tao Huang\*<sup>2</sup>, Fangfang Wu\*<sup>3</sup>,Shiwen Xue\*<sup>4</sup>,Zhenyu Wang\*<sup>5</sup>, Weisheng Dong\*<sup>6</sup>,Xin Li\*<sup>7</sup>,Guangming Shi\*<sup>8</sup>**

<sup>1</sup> Xidian University <sup>2</sup> State University of New York at Albany <sup>3</sup> Key Laboratory of Intelligent Perception and Image Understanding of Ministry of Education


## ⭐ Overview


**Zero-Shot Query Text Generator (ZS-QTG)**



<img width="600" height="300" alt="zsqtg" src="https://github.com/user-attachments/assets/8050f58e-9b29-44a1-9bd1-68275351e19d" />



ZS-QTG is designed to automatically generate text queries that closely match a given target image in a zero-shot manner. It leverages CBART as a language backbone to propose fluent candidate sentences and introduces visual guidance from RemoteCLIP to evaluate image–text similarity. By combining the linguistic probability from CBART with the visual–semantic score from RemoteCLIP, ZS-QTG adaptively selects the most visually relevant words during generation, ensuring that the final query accurately reflects the image content.


**Text-image Sequential Training of Composed Image Retrieval (TisCIR)**

<img width="600" height="300" alt="stage1" src="https://github.com/user-attachments/assets/94c00e10-74fa-4917-8d7d-159e7af6b4e7" />

<img width="600" height="300" alt="stage2" src="https://github.com/user-attachments/assets/293371a3-4f9c-4850-8ea1-01ccb9d66179" />

<img width="600" height="300" alt="stage3" src="https://github.com/user-attachments/assets/01df7152-665c-4e9d-90fb-2f77137f234b" />


TisCIR enhances zero-shot composed image retrieval by sequentially training on text and image features. It first extracts structured semantics from text via a Multiple Self-Masking Projection (MSMP) module, then refines image embeddings with a Fine-Grained Image Attention (FGIA) module to remove conflicting information. During retrieval, both refined embeddings are combined for more accurate and semantically consistent matching.


