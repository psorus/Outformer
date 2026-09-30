# Outformer
Official implementation of Outformer, a foundation model for zero-shot outlier detection reaching state-of-the-art performance.

## Setup Instructions
Please follow the instructions in [FoMo-0D](https://anonymous.4open.science/r/PFN40D/README.md) (https://anonymous.4open.science/r/PFN40D/README.md)

## Pretraining Outformer

To **pretrain** our model, use `CUDA_VISIBLE_DEVICES=0 python3 pretrain_parallel_torch.py`

**Hyperparameters** are given in configuration/

# Checkpoints

We publish our checkpoint at [https://huggingface.co/MacrOData-CMU/OutFormer](https://huggingface.co/MacrOData-CMU/OutFormer/blob/main/outformer.ckpt)

# Benchmark datasets

All our datasets are available at https://huggingface.co/MacrOData-CMU/datasets

# Individual results

Further results, individual performance and a leaderboard is available at https://huggingface.co/spaces/MacrOData-CMU/MacrOData


