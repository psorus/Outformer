# Outformer
Official implementation of Outformer, a foundation model for zero-shot outlier detection reaching state-of-the-art performance.

## Setup Instructions
Please follow the instructions in [FoMo-0D](https://anonymous.4open.science/r/PFN40D/README.md) (https://anonymous.4open.science/r/PFN40D/README.md)

## Pretraining Outformer

To **pretrain** our model, use `CUDA_VISIBLE_DEVICES=0 python3 pretrain_parallel_torch.py`

**Hyperparameters** are given in configuration/

# Checkpoints

As anonymized repositories allow for only files up to 100mb, we are only able to publish our checkpoints after the review process.

# Benchmark datasets

Our proposed benchmark datasets are given in benchmarks/oddbench and benchmarks/ovrbench. Please note that due to github size limitations, we had to remove 14 datasets from ovrbench until after review. The removed datasets will be released unchanged after review.

# Individual results

Per-dataset evaluation results will be released in a cleaned, structured, and searchable format after the review process.


