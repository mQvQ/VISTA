# VISTA: A Diffusion-based Generative Framework for Virtual In Situ Transcriptome Analysis

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-pink.svg)](https://arxiv.org/abs/2510.12345)
[![Project Page](https://img.shields.io/badge/Project-Page-orange)](https://your-project-page-url.github.io/)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for **VISTA**.

[Here you can place an image, GIF, or architecture diagram that best represents the core idea or key results of your project]
![VISTA Overview](assets/overview.png) <!-- It's recommended to create an assets folder in your repository to store images -->

## 1. What is VISTA?

 VISTA is a diffusion-based generative framework for virtual spatial transcriptomics modeling. 
 Its core idea is to conditionally generate biologically coherent spatial gene expression profiles. 
 By leveraging a flexible conditional architecture to integrate diverse priors, including histological morphology, cancer type, spatial relationships, and cell type proportion, VISTA can achieve excellent performance on diverse downstream applications, including gene prediction, cancer grading, and survival analysis.

Key Innovation


## 2. Installation

We recommend installing the dependencies in a virtual environment (e.g., `conda` or `venv`) to avoid package version conflicts.

```bash
# 1. Clone this repository
git clone [https://github.com/your-username/VISTA.git](https://github.com/your-username/VISTA.git)
cd VISTA

# 2. (Optional but recommended) Create and activate a conda environment
conda create -n vista python=3.9 -y
conda activate vista

# 3. Install all dependencies
# Install basic requirements
pip install -r requirements.txt

# Install PyTorch (choose the right command for your CUDA version from https://pytorch.org/get-started/locally/)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install pytorch-lightning
pip install pytorch-lightning==1.4.2

# Install CONCH
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
pip install -e .

```
## 3. Download and Organizing data
The dataset is curated on [HEST](https://github.com/mahmoodlab/HEST), and the preprocessed data can be downloaded from [huggingface](https://huggingface.co/datasets/u3011706/VISTA-Dataset).
dataset 包含:
train_set_log1p.h5 
test_set_log1p.h5
test_set_standard_norm_celltype_newest.json
train_set_standard_norm_celltype_newest.json

## 4. ckpts & latent embeddings
you can download the pretrained models and latent embeddings from:
[huggingface](https://huggingface.co/datasets/u3011706/VISTA-Weights)
the repo includes
1. GAE ckpt 
2. latent embeddings extracted from the graph encoder
3. latent diffusion ckpt

## 5. Train
```bash
python st_main.py -t --base configs/st_unet_mixcontrolv6.yaml --gpus 0,1,2,3, --logdir /path/to/VISTA/logs
```

## 6. Sample
```bash
python -m infer.sample_wo_celltype \
--split test --batch_size 96 --scale 1 --st_shape 1 256 \
--exp_name '10-10T17-05_st_unet_mixcontrolv6' \
--ckpt_path '/path/to/last.ckpt' \
--config_path '/path/to/configs/10-10T17-05-project.yaml' \
--results_save_dir '/path/to/results' \
--emb_path 'multi_organ_gae_test_set_log1p_d256_embs_3+1.pkl' --device 'cuda:0'
```





