import os.path
import os
os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

from pytorch_lightning.strategies import DDPStrategy
import torch
import time
import random
from torch import nn
from pfns import encoders
import hydra
from omegaconf import DictConfig
from data_priors.data_generator import PriorTrainDataGenerator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import wandb
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datetime import datetime
from trainer.cl_trainer import ZeroShotOD

@rank_zero_only
def ensure_save_dir(path: str):
    """Ensure directory exists before saving checkpoints."""
    os.makedirs(path, exist_ok=True)


class SingleEvalPosGenerator:
    """Generate single evaluation position for test samples during training."""
    def __init__(self, mode, num_test_x, seq_len, num_R):
        self.mode = mode
        self.num_test_x = num_test_x
        self.seq_len = seq_len
        self.single_eval_pos = None
        self.num_R = 0 if num_R is None else num_R

        if self.mode == 'constant':
            assert self.num_test_x > self.num_R, (
                f"Constant mode: ensure seq_len={self.seq_len} > "
                f"num_test_x={self.num_test_x}+num_R={self.num_R}"
            )

    def generate(self, seed=None):
        """Generate evaluation position based on mode (constant or random)."""
        if self.mode == 'constant':
            self.single_eval_pos = self.seq_len - self.num_test_x + self.num_R
        else:
            if seed is None:
                self.single_eval_pos = random.choices(range(0, self.seq_len - self.num_R))[0] + self.num_R
            else:
                rng = random.Random(seed)
                self.single_eval_pos = rng.choices(range(0, self.seq_len - self.num_R))[0] + self.num_R
        return self.single_eval_pos


def make_pl_model(cfg, get_batch_function, seq_len, num_features, hps,
                  generator_mode='constant', num_class=2, num_R=None,
                  model_para_dict=None, train_extra_dict=None, resume_from_ckpt=False):
    """Create a PyTorch Lightning model for zero-shot OD training."""
    criterion = nn.CrossEntropyLoss(
        weight=torch.ones(size=(num_class,)) / num_class,
        reduction='none',
        ignore_index=hps['ignore_index']
    )

    single_eval_pos_gen = SingleEvalPosGenerator(
        mode=generator_mode,
        num_test_x=hps['num_test_x'],
        seq_len=seq_len,
        num_R=num_R
    )
    print(f"T0 value: {cfg.train.T0}")
    
    pl_model = ZeroShotOD(
        cfg=cfg,
        priordataloader_class_or_get_batch=get_batch_function,
        criterion=criterion,
        encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        y_encoder_generator=encoders.Linear,
        extra_prior_kwargs_dict={
            'num_features': num_features,
            'hyperparameters': hps,
            'pt_dataloader': {'num_workers': 0, 'pin_memory': True},
            'num_R': num_R
        },
        single_eval_pos_gen=single_eval_pos_gen,
        progress_bar=True,
        train_extra_dict=train_extra_dict,
        resume_from_ckpt=resume_from_ckpt,
        T0=cfg.train.T0,
        num_bins=cfg.train.num_bins,
        **(model_para_dict if model_para_dict else {})
    )
    return pl_model


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy, random, torch, and cuda."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@hydra.main(version_base='1.3', config_path='config', config_name='pretrain_config')
def main(cfg: DictConfig):
    """Main training entry point for model pretraining with distributed DDP training."""
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)
    train_cfg = cfg.train
    
    # Train hyperparameters
    seq_len = train_cfg.seq_len
    hyperparameters = train_cfg.hyperparameters
    batch_size = train_cfg.batch_size
    epochs = train_cfg.epochs
    steps_per_epoch = train_cfg.steps_per_epoch
    lr = train_cfg.lr
    emsize = train_cfg.emsize
    nhead = train_cfg.nhead
    nhid = train_cfg.nhid
    nlayer = train_cfg.nlayer
    num_R = train_cfg.num_R
    gen_one_train_one = train_cfg.gen_one_train_one
    resume_from_ckpt = train_cfg.resume_from_ckpt
    apply_linear_transform = train_cfg.apply_linear_transform
    seed = train_cfg.seed
    num_device = train_cfg.num_device
    T0 = train_cfg.T0
    
    # Prior hyperparameters
    max_feature_dim = cfg.prior.mixture.max_feature_dim
    num_bins = train_cfg.num_bins
    set_seed(seed=seed)

    generator_mode = hyperparameters['mode']
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    config_details = (
        f"fomo_bin{num_bins}_temp{train_cfg.temperature}_"
        f"scheduler{train_cfg.filterscheduler}_dim{max_feature_dim}_"
        f"context{seq_len}.feat{max_feature_dim}.R{num_R}."
        f"LT{apply_linear_transform}.gen1tr1{gen_one_train_one}."
        f"reuse{train_cfg.reuse_data_every_n}.E{epochs}.step{steps_per_epoch}."
        f"bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}."
        f"nlayer{nlayer}.ndevice{num_device}.T0{T0}_{current_time}"
    )
    
    if train_cfg.last_layer_no_R:
        config_details = f"last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}"

    if train_cfg.extra_heading != '':
        config_details = f"{train_cfg.extra_heading}.{config_details}"

    save_path = f"{train_cfg.model_dir}/{config_details}/seed{seed}"
    ensure_save_dir(save_path)

    start_time = time.time()
    zero_shot_od_pl_model = make_pl_model(
        cfg=cfg,
        get_batch_function=prior_train_data_gen.get_batch_all_models,
        seq_len=seq_len,
        num_features=max_feature_dim,
        num_class=2,
        generator_mode=generator_mode,
        hps=hyperparameters,
        num_R=num_R,
        model_para_dict={'num_R': num_R, 'last_layer_no_R': train_cfg.last_layer_no_R},
        train_extra_dict=(
            {'prior_train_data_gen': prior_train_data_gen} if gen_one_train_one else None
        ),
        resume_from_ckpt=resume_from_ckpt
    )

    if train_cfg.logging:
        logger = WandbLogger(project='ZeroShotOD_parallel-PSC', name=config_details)
    else:
        logger = None

    # Setup checkpointing callbacks
    ckpt_callbacks = []
    
    train_ckpt_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        save_top_k=3,
        dirpath=save_path,
        filename='min-trainloss-{epoch:02d}-{train_loss:.2f}',
        verbose=True,
        save_last=True
    )
    ckpt_callbacks.append(train_ckpt_callback)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="{epoch:02d}-{train_loss:.2f}",
        save_top_k=-1,
        every_n_epochs=100
    )
    ckpt_callbacks.append(checkpoint_callback)

    print(f"Setting max epochs to {epochs}")
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=ckpt_callbacks,
        logger=logger,
        max_epochs=epochs,
        enable_progress_bar=True,
        limit_val_batches=None if train_cfg.use_validation else 0,
        check_val_every_n_epoch=1,
        devices=num_device,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        num_sanity_val_steps=1,
        use_distributed_sampler=False
    )
    
    trainer.fit(
        zero_shot_od_pl_model,
        ckpt_path=f"{save_path}/last.ckpt" if resume_from_ckpt else None
    )

    # Log training time
    train_time = time.time() - start_time
    print(f"Total training time: {train_time / 60:.2f} minutes")


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    main()