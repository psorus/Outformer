# Outformer
ICML 2026 Paper implementation

## Installation

Create the `outformer` conda environment from [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate outformer
```

The environment uses conda-forge only. If conda stops with a Terms of Service error for
`repo.anaconda.com`, use this instead:

```bash
conda create --override-channels -c conda-forge --file environment.yml
```

PyTorch is installed from PyPI, and the Linux wheels include the CUDA runtime, so you don't
need a separate CUDA toolkit. You still need a GPU with an NVIDIA driver for training.

## Benchmarks

Outformer is evaluated on three tabular outlier-detection benchmarks. OddBench and OvRBench
were introduced with the paper,
[From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](https://arxiv.org/abs/2602.03018).
Download the datasets into `benchmarks/`, which git ignores.

| Benchmark | Datasets | What it contains |
|---|---|---|
| [ADBench](https://github.com/Minqi824/ADBench) | 57 | The established outlier-detection benchmark: classical tabular datasets, plus image and text datasets turned into feature vectors with pretrained encoders |
| [OddBench](https://huggingface.co/datasets/MacrOData-CMU/OddBench) | 690 | Real-world tables from Tablib whose metadata marks semantic anomalies such as fraud, failure or defect |
| [OvRBench](https://huggingface.co/datasets/MacrOData-CMU/OvRBench) | 755 | Classification benchmarks (TabArena, TabRepo, TabZilla and others) recast as outlier detection: one class is kept as inliers and the other classes are subsampled as outliers |

### OddBench and OvRBench

Both are hosted on Hugging Face under [MacrOData-CMU](https://huggingface.co/MacrOData-CMU).
The `hf` command comes with `huggingface_hub`, which is part of the `outformer` environment.

```bash
hf download MacrOData-CMU/OddBench --repo-type dataset --local-dir benchmarks/oddbench
hf download MacrOData-CMU/OvRBench --repo-type dataset --local-dir benchmarks/ovrbench
```

Each repo has two folders. `public/` holds the full benchmark (about 0.5 GB for OddBench and
2.2 GB for OvRBench, including both folders), and `representative/` holds a 50-dataset subset.
To download only the subset, add `--include "representative/*"`.

Each `.npz` file is already split into `train` and `test`, with labels in `train_labels` and
`test_labels`. It also holds metadata such as `anomaly_fraction` and `feature_count`.

### ADBench

ADBench keeps its datasets in its GitHub repository (about 2 GB). This clones only the dataset
folder and copies it into place:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/Minqi824/ADBench.git /tmp/ADBench
git -C /tmp/ADBench sparse-checkout set adbench/datasets
mkdir -p benchmarks/adbench
cp -r /tmp/ADBench/adbench/datasets/{Classical,CV_by_ResNet18,CV_by_ViT,NLP_by_BERT,NLP_by_RoBERTa} benchmarks/adbench/
rm -rf /tmp/ADBench
```

Keep the subfolders. The two CV folders use the same file names (for example
`CIFAR10_0.npz`), and so do the two NLP folders. Each `.npz` file has a feature matrix `X` and
labels `y` (1 = anomaly).

## Pretraining

```bash
python pretrain.py
```

Settings are read from [config/pretrain_config.yaml](config/pretrain_config.yaml) with
[Hydra](https://hydra.cc), so you can override any of them on the command line.
