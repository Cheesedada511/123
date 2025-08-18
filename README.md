# PrismNet

Code for "Spectral decomposition of chemical semantics for activity cliffs-aware molecular property prediction".

[![python](https://badge.ttsalpha.com/api?icon=python&label=python&status=3.7.13)](https://example.com)  [![pytorch](https://badge.ttsalpha.com/api?icon=pytorch&label=pytorch&status=1.10.0&color=yellow)](https://example.com)  [![cuda](https://badge.ttsalpha.com/api?icon=nvidia&label=cuda&status=11.3&color=green)](https://example.com)

## Abstract

Accurately predicting physicochemical and biological properties for molecules is of great significance for modern drug discovery, yet existing deep learning models often fail to emulate the holistic, multi-level reasoning of human chemists. These models, typically reliant on single molecular graphs, struggle to capture the synergistic interplay between global scaffolds, functional groups, and pharmacophoric patterns, and are often insensitive to the subtle local perturbations that cause “activity cliffs”. Here, we introduce **PrismNet**, a chemistry-inspired spectral graph network that emulates chemical intuition through a computational prism analogy. PrismNet decomposes complex molecular information through a dual-decomposition strategy. It first refracts molecules into fundamental chemical perspectives—scaffolds, functional groups, and pharmacophores—and then resolves these views into distinct spectral frequencies. Augmented by a dynamic learning strategy to handle data heterogeneity, PrismNet achieves state-of-the-art performance across a comprehensive suite of 64 benchmark datasets on property prediction, with 30 challenging activity cliff datasets included. More importantly, its decisions are chemically interpretable, as it autonomously identifies critical substructures that align with established structure-activity relationships. This work establishes a framework for learning chemically trustworthy representations by unifying multi-scale semantics with spectral decomposition, paving the way for more reliable in silico screens in drug discovery.

![framework](./images/PrismNet.png)

## Installation

```bash
conda env create -f environment.yml
conda activate PrismNet
```
## Download datasets

To download the datasets: https://drive.google.com/file/d/1JF_ePa4LQTlDBrql3oDUrDCTrSTMYnNa.

Then unzip the file and put it into the `data` directory.

## Pretraining

### Preparing dataset

Preprocess dataset:

```bash
cd data_process
python prepare_pretrain_dataset.py
```
Then you can get the processed pretrain dataset.

### Pretraining
Commands for pretrain:
```bash
cd scripts
python pretrain.py --dataset zinc15_250K  --seed 19  --epochs 150  --gpu 0
```

Usage:
```bash
options:
  -h, --help            show this help message
  --dataset             pretraining dataset
  --seed SEED           random seeds
  --epochs EPOCHS       number of total epochs to run
  --gpu GPU             gpu id
```
All hyperparameters can be tuned in the `scripts/utils.py`.

You can also download the pretrained model at: https://drive.google.com/file/d/1gk8FDLceQGb4xWlGyq9iy_K2ncf32ipz.

Then put it into the `ckpts` directory.

## Finetune

### Preparing dataset
```bash
cd data_process
python preprocess.py
```
Then you can get the processed dataset.

### Finetune
Fine-tune pre-trained model on a specific downstream task:

```bash
cd scripts
python finetune.py  --gpu 0  --dataset bace  --seed 19  --epochs 150 --ckpt_path  ../ckpts/pretrain.pth
```

Usage:
```bash
options:
  -h, --help            show this help message
  --dataset             fine-tuning dataset
  --seed SEED           random seeds
  --epochs EPOCHS       number of total epochs to run
  --gpu GPU             gpu id
  --ckpt_path           pretrained model path
```
All hyperparameters can be tuned in the `scripts/utils.py`.

## Evaluation
We provide the fine-tuned model for 11 datasets, to guarantee the reproducibility of the test results reported in our paper.

### Download finetuned models

To download the fine-tuned models: https://drive.google.com/file/d/15HlEKCjd-Jhx3o4AH5yyu5IUcbAQDNKj

Then unzip it and put the files in the `ckpts` directory.

### Reproduce the results

Then the results can be reproduced by:
```bash
cd scripts
python finetune.py  --type evaluate --gpu 0  --dataset bace  --seed 19 --ckpt_path  ../ckpts/bace.pt
```