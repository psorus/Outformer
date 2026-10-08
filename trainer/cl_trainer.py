from __future__ import annotations
import torch.distributed as dist
import time
import torch
from torch import nn
import random
import pickle
from trainer import utils
from data_priors.batch import *
from pfns.transformer import TransformerModel
from trainer.utils import get_cosine_schedule_with_warmup, get_openai_lr,get_cosine_schedule_with_warmup_min_lr
from pfns import positional_encodings
import pytorch_lightning as pl
from data_priors.dataset import EpochDataset
from torch.utils.data import DataLoader
import gc
import torch.nn.functional as F
from itertools import combinations, count
import torch
from torch.nn.utils import clip_grad_norm_
import os
import json
from collections import Counter
import numpy as np
from trainer.curriculum_scheduler import CurriculumScheduler
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm



def make_model_od(criterion, encoder_generator,
                  emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0, seq_len=10,
                  input_normalization=False,
                  y_encoder_generator=None, pos_encoder_generator=None, decoder_dict={}, extra_prior_kwargs_dict={},
                  initializer=None,
                  efficient_eval_masking=True, num_global_att_tokens=0, **model_extra_args):
    style_encoder = None
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
    
    # Criterion is always CrossEntropyLoss in retraining path
    n_out = criterion.weight.shape[0] if hasattr(criterion, 'weight') else 2

    decoder_dict = decoder_dict if decoder_dict else {'standard': (None, n_out)}

    decoder_once_dict = {}

    encoder = encoder_generator(extra_prior_kwargs_dict['num_features'], emsize)
    model = TransformerModel(encoder=encoder
                             , nhead=nhead
                             , ninp=emsize
                             , nhid=nhid
                             , nlayers=nlayers
                             , dropout=dropout
                             , style_encoder=style_encoder
                             , y_encoder=y_encoder_generator(1, emsize)
                             , input_normalization=input_normalization
                             , pos_encoder=pos_encoder
                             , decoder_dict=decoder_dict
                             , init_method=initializer
                             , efficient_eval_masking=efficient_eval_masking
                             , decoder_once_dict=decoder_once_dict
                             , num_global_att_tokens=num_global_att_tokens
                             , **model_extra_args
                             )
    model.criterion = criterion
    import pdb;pdb.set_trace()
    print(model.summary())
    return model


class MetricRecorder:
    def __init__(self,
                 seq_len, 
                 steps_per_epoch,
                 categories,
                 verbose):
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose

        self.total_loss = 0.0
        self.total_positional_losses = torch.zeros(self.seq_len)
        self.total_positional_losses_recorded = torch.zeros(self.seq_len)
        self.nan_steps = 0.0
        self.ignore_steps = 0.0
        self.epoch_start_time = 0.0
        self.total_step_time = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.dependence_copula_loss = 0.0
        self.dependence_copula_step_count = 0
        self.probablistic_copula_loss = 0.0
        self.probablistic_copula_step_count = 0
        self.structure_scm_loss = 0.0
        self.structure_scm_step_count = 0
        self.measure_scm_loss = 0.0
        self.measure_scm_step_count = 0
        self.model_names_list = ['gmm', 'dependence_copula', 'probablistic_copula', 'structure_scm', 'measure_scm']
        self.categories = categories
        categories_loss = [0.0 for i in self.categories] 
        self.category_loss = dict(zip(self.categories, categories_loss))
        categories_counts = [0 for i in self.categories] 
        self.category_counts = dict(zip(self.categories, categories_counts))
        

    def reset(self):
        self.total_loss = 0.0
        self.nan_steps = 0.0
        self.ignore_steps = 0.0
        self.epoch_start_time = 0.0
        self.total_step_time = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.dependence_copula_loss = 0.0
        self.dependence_copula_step_count = 0
        self.probablistic_copula_loss = 0.0
        self.probablistic_copula_step_count = 0
        self.structure_scm_loss = 0.0
        self.structure_scm_step_count = 0
        self.measure_scm_loss = 0.0
        self.measure_scm_step_count = 0
        categories_loss = [0.0 for i in self.categories] 
        self.category_loss = dict(zip(self.categories, categories_loss))
        categories_counts = [0 for i in self.categories] 
        self.category_counts = dict(zip(self.categories, categories_counts))
        
        
        
    def update(self,
               loss, 
               losses, 
               single_eval_pos,
               targets,
               nan_share,
               step_time, 
               categories):
        if  (not loss is None) and not torch.isnan(loss):
            self.total_loss += loss.cpu().detach().item()
            if categories is not None and losses is not None:
                for i,category_name in enumerate(categories):
                    if losses[i] is None:
                        continue
                    l = losses[i]
                    name = category_name[1]
                    if name == 'gmm':
                        self.gmm_loss += l
                        self.gmm_step_count += 1
                    elif name == 'dependence_copula':
                        self.dependence_copula_loss += l
                        self.dependence_copula_step_count += 1
                    elif name == 'probablistic_copula':
                        self.probablistic_copula_loss += l
                        self.probablistic_copula_step_count += 1
                    elif name == 'structure_scm':
                        self.structure_scm_loss += l
                        self.structure_scm_step_count += 1
                    elif name == 'measure_scm':
                        self.measure_scm_loss += l
                        self.measure_scm_step_count += 1
                    self.category_loss[category_name] += l
                    self.category_counts[category_name] += 1         
                    
            self.nan_steps += nan_share.cpu().item()
            self.ignore_steps += (targets == -100).float().mean().cpu().item()

        self.total_step_time += step_time

   


    def fetch_and_print(self, epoch=None, lr=None):
        avg_loss = self.total_loss / self.steps_per_epoch
        avg_gmm_loss = self.gmm_loss / self.gmm_step_count if self.gmm_step_count != 0 else 0
        avg_dependence_copula_loss = self.dependence_copula_loss / self.dependence_copula_step_count if self.dependence_copula_step_count != 0 else 0
        avg_probablistic_copula_loss = self.probablistic_copula_loss / self.probablistic_copula_step_count if self.probablistic_copula_step_count != 0 else 0
        avg_structure_scm_loss = self.structure_scm_loss / self.structure_scm_step_count if self.structure_scm_step_count != 0 else 0
        avg_measure_scm_loss = self.measure_scm_loss / self.measure_scm_step_count if self.measure_scm_step_count != 0 else 0

        #here avg losses per category can also be printed if needed
        avg_category_losses = {}
        for category_name in self.categories:
            count = self.category_counts[category_name]
            if count > 0:
                avg_cat_loss = self.category_loss[category_name] / count
            else:
                avg_cat_loss = 0.0
            if self.verbose:
                print(f" Avg loss for category {category_name}: {avg_cat_loss:.4f} over {count} steps")
            avg_category_losses[category_name] = avg_cat_loss
            
        nan_share = self.nan_steps / self.steps_per_epoch
        ignore_share = self.ignore_steps / self.steps_per_epoch
        total_time = time.time() - self.epoch_start_time
        
             
        if self.verbose:
            print('-' * 89)
            print(
                f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                f' | end of epoch {epoch:3d} | time: {total_time:5.2f}s | (approx) step time: {self.total_step_time:5.2f}s | '
                f'(approx) data time: {total_time - self.total_step_time:5.2f}s | mean loss {avg_loss:5.2f} | lr {lr}'
            )
            print(f" Avg losses: GMM={avg_gmm_loss:.4f}, DependenceCopula={avg_dependence_copula_loss:.4f}, ProbablisticCopula={avg_probablistic_copula_loss:.4f}, "
                f"StructureSCM={avg_structure_scm_loss:.4f}, MeasureSCM={avg_measure_scm_loss:.4f}")
            print('-' * 89)
            

        # avg_cosine_similarities computed and printed, but not returned
        return {
            'avg_loss': avg_loss,
            'nan_share': nan_share,
            'ignore_share': ignore_share,
            'total_time': total_time,
            'avg_gmm_loss': avg_gmm_loss,
            'avg_dependence_copula_loss': avg_dependence_copula_loss,
            'avg_probablistic_copula_loss': avg_probablistic_copula_loss,
            'avg_structure_scm_loss': avg_structure_scm_loss,
            'avg_measure_scm_loss': avg_measure_scm_loss,
            'avg_category_losses': avg_category_losses
        }


class ZeroShotOD(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 priordataloader_class_or_get_batch: PriorDataLoader | callable,
                 criterion,
                 encoder_generator, 
                 dropout=0.0,
                 weight_decay=0.0,
                 input_normalization=False,
                 y_encoder_generator=None,
                 pos_encoder_generator=None, 
                 decoder_dict=None,
                 extra_prior_kwargs_dict=None,
                 train_extra_dict=None, 
                 resume_from_ckpt=False,
                 scheduler=get_cosine_schedule_with_warmup_min_lr,
                 single_eval_pos_gen=None,
                 verbose=False, 
                 initializer=None, 
                 efficient_eval_masking=True,
                 num_global_att_tokens=0,
                 num_bins=5,
                 progress_bar=False,
                 **model_extra_args):
        super(ZeroShotOD, self).__init__()

        train_cfg = cfg.train
        prior_gmm_cfg = cfg.prior.mixture.gmm
        
        # Train hyperparameters
        seq_len = train_cfg.seq_len
        self.batch_size = train_cfg.batch_size
        epochs = train_cfg.epochs
        self.steps_per_epoch = train_cfg.steps_per_epoch
        emsize = train_cfg.emsize
        nhead = train_cfg.nhead
        nhid = train_cfg.nhid
        nlayers = train_cfg.nlayer
        self.reuse_data_every_n = train_cfg.reuse_data_every_n
        num_device = train_cfg.num_device
        self.num_device = num_device
        lr = train_cfg.lr
        
        # Prior hyperparameters
        self.max_feature_dim = prior_gmm_cfg.max_feature_dim
        self.min_feature_dim = 2
        self.max_model_dim = prior_gmm_cfg.max_model_dim
        self.max_num_cluster = prior_gmm_cfg.max_num_cluster
        self.inflate_full = prior_gmm_cfg.inflate_full
        self.model_names_list = ['gmm', 'dependence_copula', 'probablistic_copula', 'structure_scm', 'measure_scm'] 

        # Generate-one-train-one mode
        self.gen_one_train_one = False if train_extra_dict is None else True
        self.prior_train_data_gen = None if train_extra_dict is None else train_extra_dict['prior_train_data_gen']
        
        # Criterion (fixed to CrossEntropyLoss for this retraining path)
        self.criterion = criterion
        self.apply_linear_transform = train_cfg.apply_linear_transform
        self.dataloader_para = extra_prior_kwargs_dict.get('pt_dataloader', {'num_workers': 0, 'pin_memory': True})

       
        # Determine training data path
        # train data path is None here since we generate on the fly
        train_data_path = None

        self.train_dataset = EpochDataset(batch_size=self.batch_size, 
                                          seq_len=seq_len,
                                          steps_per_epoch=self.steps_per_epoch,
                                          hyperparameters=extra_prior_kwargs_dict['hyperparameters'],
                                          reuse_data_every_n=self.reuse_data_every_n, max_model_dim=self.max_model_dim,
                                          max_num_cluster=self.max_num_cluster,
                                          get_batch_method=priordataloader_class_or_get_batch,
                                          rank=0, num_device=num_device,  # rank is not yet set in __init__
                                          training=True, single_eval_pos_gen=single_eval_pos_gen,
                                          data_path=train_data_path,
                                          is_source_numpy=False if self.gen_one_train_one else True)
        # stored data currently is always numpy
        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.train_dataset.prior_batch_collate_fn,
            **self.dataloader_para
        )
        
        # Encoder setup
        style_encoder = None
        pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
        
        # Criterion is fixed to CrossEntropyLoss; infer output classes from criterion weight
        self.n_out = self.criterion.weight.shape[0] if hasattr(self.criterion, 'weight') else 2

        # Initialize model
        decoder_dict = decoder_dict or {'standard': (None, self.n_out)}
        encoder = encoder_generator(extra_prior_kwargs_dict['num_features'], emsize)
        
        self.model = TransformerModel(
            encoder=encoder,
            nhead=nhead,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            style_encoder=style_encoder,
            y_encoder=y_encoder_generator(1, emsize),
            input_normalization=input_normalization,
            pos_encoder=pos_encoder,
            decoder_dict=decoder_dict,
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            num_global_att_tokens=num_global_att_tokens,
            model_para_dict=model_extra_args
        )
        self.model.criterion = self.criterion

        print(
            f"Using a Transformer with {sum(p.numel() for p in self.model.parameters()) / 1000 / 1000:.{2}f} M parameters")

        # Note: Model initialization from another model is not currently supported in retraining

        # Optimizer and scheduler parameters
        self.lr = lr
        self.scheduler_fn = scheduler
        self.warmup_epochs = epochs // 10
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        # Curriculum learning parameters
        self.gamma = 0.1  # smoothing factor for moving average
        self.temperature = train_cfg.temperature
        self.filterscheduler = train_cfg.filterscheduler
        
        self.curriculum_scheduler = CurriculumScheduler(
            total_steps=self.epochs,
            a=0.8,
            b=0.2,
            scheduler_name=self.filterscheduler,
            max_value=0.95
        )
        
        # Divide dimensions into power-law bins for curriculum learning
        self.num_bins = num_bins
        alpha = 2.0  # >1 => more bins near min_feature_dim; <1 => near max_feature_dim
        lin = torch.linspace(0, 1, steps=self.num_bins + 1)
        edges = (
            self.min_feature_dim +
            (self.max_feature_dim - self.min_feature_dim) * (lin ** alpha)
        ).round().long()
        edges[-1] = self.max_feature_dim + 1  # ensure coverage
        self.bin_ranges = [(edges[i].item(), edges[i + 1].item()) for i in range(self.num_bins)]

        # Create B x D grid of (bin_range, model_name) pairs
        self.categories = [
            (bin_range, model_name)
            for bin_range in self.bin_ranges
            for model_name in self.model_names_list
        ]
        
        self.categories_weights = [0.0 for i in self.categories] 
        self.data_weights_map = dict(zip(self.categories, self.categories_weights))

        self.total_data_size = self.steps_per_epoch * self.batch_size
        self.data_samples = self.sample(k=self.total_data_size, temperature=self.temperature)
        self.batch_mask = None  # default: no masking (all samples used)

        # Validate API compatibility
        utils.check_compatibility(self.train_dl)

        # Training dynamics tracking
        self.train_recorder = MetricRecorder(
            seq_len=seq_len,
            steps_per_epoch=self.steps_per_epoch,
            verbose=verbose,
            categories=self.categories
        )
        self.train_losses = []
        
    # ==================== Softmax Sampling ====================
    @staticmethod
    def _softmax(logits, temp):
        if temp <= 0:
            raise ValueError("temperature must be > 0")
        x = logits / temp
        x = x - np.max(x)            # numerical stability
        ex = np.exp(x)
        return ex / ex.sum()
    
    
    def _weights_array(self):
        # Keep weights in the same order as self.categories
        return np.array([self.data_weights_map[c] for c in self.categories], dtype=float)


    def sample(self, k, temperature=1.0, rng=None):
        """
        k: number of samples to draw
        temperature: > 0; lower -> peakier, higher -> flatter
        replace: True = with replacement (i.i.d.), False = without replacement
        rng: optional numpy Generator (np.random.default_rng(seed))
        """
        if rng is None:
            rng = np.random.default_rng()
        logits = self._weights_array()
        # i.i.d. draws from softmax
        p = self._softmax(logits, temperature)
        idx = rng.choice(len(self.categories), size=k, replace=True, p=p)
        data_samples = [self.categories[i] for i in np.atleast_1d(idx)]
        return data_samples
    # ==================== End Softmax Sampling ====================
    
    
    
    def _hb(self, tag):
        """Heartbeat logging for distributed training debugging."""
        try:
            ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            ws = -1
        print(
            f"[HB] epoch={self.current_epoch} rank={self.global_rank} world={ws} tag={tag} time={time.time():.3f}",
            flush=True
        )
        
  
    def configure_optimizers(self):
        # learning rate
        if self.lr is None:
            self.lr = get_openai_lr(self.model)
            print(f"Using OpenAI max lr of {self.lr}.")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.scheduler_fn(optimizer, self.warmup_epochs,
                                      self.epochs if self.epochs is not None else 100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor a validation metric
                'interval': 'epoch',  # How often to step (options: 'epoch', 'step')
                'frequency': 1,  # How many epochs/steps between each step
            }
        }

    def on_fit_start(self) -> None:
        print('on_fit_start---setting ranks...')
        self.train_dataset.set_rank(rank=self.global_rank)
        # if self.val_dataset is not None:
        #     self.val_dataset.set_rank(rank=self.global_rank)
        if self.trainer.ckpt_path:
            print(f"Resuming training from checkpoint: {self.trainer.ckpt_path}")
        else:
            print("Training from scratch.")


    def train_dataloader(self):
        # Load data samples for this epoch using curriculum learning
        print(f'Preparing data samples for epoch {self.current_epoch}...')
        self.data_samples = self.sample(k=self.total_data_size, temperature=self.temperature)
        
        if self.global_rank == 0:
            category_counts = Counter(self.data_samples)
            print(f'Data sample counts per category for epoch {self.current_epoch}:')
            logits = self._weights_array()
            p_all = self._softmax(logits, self.temperature)
            for category, count in category_counts.items():
                p = p_all[self.categories.index(category)]
                print(
                    f"  Category {category}: {count} samples, "
                    f"weight: {self.data_weights_map[category]:.4f}, p: {p:.4f}"
                )
            
            self._pending_dl_metrics = {
                f'category_normalized_weights_bin{cat[0]}_{cat[1]}': float(p_all[self.categories.index(cat)])
                for cat in self.categories
            }
        
        # Shuffle and distribute data samples across ranks
        if torch.distributed.is_initialized():
            if self.global_rank == 0:
                random.shuffle(self.data_samples)
            
            # Convert to indices for broadcasting
            if self.global_rank == 0:
                model_to_idx = {name: i for i, name in enumerate(self.categories)}
                data_indices = [model_to_idx[name] for name in self.data_samples]
            else:
                data_indices = [0] * len(self.data_samples)
            
            data_tensor = torch.tensor(data_indices, device=self.device)
            torch.distributed.broadcast(data_tensor, src=0)
            
            # Convert indices back to model names
            idx_to_model = {i: name for i, name in enumerate(self.categories)}
            shuffled_samples = [idx_to_model[idx.item()] for idx in data_tensor]
            
            # Each device takes its portion
            samples_per_device = len(shuffled_samples) // self.num_device
            start_idx = self.global_rank * samples_per_device
            end_idx = start_idx + samples_per_device
            self.individual_data_samples = shuffled_samples[start_idx:end_idx]
        else:
            # Single-device case
            random.shuffle(self.data_samples)
            self.individual_data_samples = self.data_samples
        
        # Generate data on the fly (generate-one-train-one paradigm)
        if self.gen_one_train_one:
            print('Generating new data on the fly...')
            self.train_dataset.free_data()
            data_dict = self.generate_new_data_for_train()
        else:
            raise NotImplementedError("Loading pre-generated data is not supported.")

        self.train_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)

        if data_dict is None:
            return self.train_dl
        else:
            del data_dict
            gc.collect()
            torch.cuda.empty_cache()
            
            self.train_dl = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.train_dataset.prior_batch_collate_fn,
                **self.dataloader_para
            )
            print(f'Batches per epoch: {len(self.train_dl)}')
            return self.train_dl


    def generate_new_data_for_train(self):
        self.prior_train_data_gen.gen1tr1_epoch_id = self.current_epoch
        self.prior_train_data_gen.device = f"cuda:{self.global_rank}"
        
        if self.apply_linear_transform:
            raise NotImplementedError("Linear transform + gen1tr1 not implemented yet.")
        
        inliners_list, LA_list, _, model_names = (
            self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=None,
                save_data=False,
                categories=self.individual_data_samples
            )
        )
        
        min_required = int(self.steps_per_epoch * self.batch_size / self.num_device)
        assert len(LA_list) >= min_required, (
            f"Generated {len(LA_list)} samples but need {min_required}"
        )
        return {'in': inliners_list, 'la': LA_list, 'model_names': model_names}


    def forward(self, full_data, use_mask=False, batch_idx=None):
        data = (
            full_data.style.to(self.device) if full_data.style is not None else None,
            full_data.x.to(self.device),
            full_data.y.to(self.device) if full_data.y is not None else None
        )
        
        model_names = full_data.model_names
        targets = full_data.target_y.to(self.device)
        single_eval_pos = full_data.single_eval_pos
        try:
            out = self.model(
                tuple(e.to(self.device) if torch.is_tensor(e) else e for e in data),
                single_eval_pos=single_eval_pos,
                only_return_standard_out=False
            )
            out, _ = out if isinstance(out, tuple) else (out, None)
            output = out['standard'] if isinstance(out, dict) else out

            if single_eval_pos is not None:
                targets = targets[single_eval_pos:]

            if len(targets.shape) == len(output.shape):
                targets = targets.squeeze(-1)
            assert targets.shape == output.shape[:-1], (
                f"Target shape {targets.shape} does not match output shape {output.shape}"
            )
            assert not torch.isinf(output).any(), "Inf in outputs"
            assert output.shape[-1] == 2, "Each output must have 2 logits (classes)"
            assert targets.min() >= 0 and targets.max() < 2, "Target out of range"
            
            if use_mask:
                assert batch_idx is not None, "batch_idx must be provided when use_mask is True"

                S = output.shape[1]  # number of positions
                side_keys = [int(batch_idx * S + s) for s in range(S)]
                side_masks = [self.batch_mask.get(k, None) for k in side_keys]
                
                # Skip batch if no side is in the mask
                if all(m is None or len(m) == 0 for m in side_masks):
                    print(f"Global rank {self.global_rank} Batch {batch_idx} not in mask, skipping (zero-loss).")
                    zero_loss = output.sum() * 0.0
                    nan_share = torch.tensor(0.0, device=self.device)
                    mean_losses = [None] * S
                    return zero_loss, [None] * S, single_eval_pos, torch.empty(0, device=self.device), nan_share, model_names, mean_losses

                # Slice outputs/targets for each side with a non-empty mask
                out_parts = []
                tgt_parts = []
                sizes = []

                for s in range(S):
                    m = side_masks[s]
                    if m is not None and len(m) > 0:
                        out_s = output[m, s, :]
                        tgt_s = targets[m, s]
                        out_parts.append(out_s)
                        tgt_parts.append(tgt_s.long())
                        sizes.append(out_s.shape[0])
                    else:
                        sizes.append(0)
                        
                # Concatenate and compute loss across all sides
                combined_output = torch.cat(out_parts, dim=0) if out_parts else output[:0, 0, :]
                combined_targets = torch.cat(tgt_parts, dim=0) if tgt_parts else targets[:0, 0]
                losses_flat = self.criterion(combined_output, combined_targets)

                # Split losses back to per-side
                losses_split = []
                offset = 0
                for n in sizes:
                    if n > 0:
                        losses_split.append(losses_flat[offset:offset + n])
                        offset += n
                    else:
                        losses_split.append(None)

                mean_losses = [
                    l.mean(0).cpu().detach().item() if l is not None else None
                    for l in losses_split
                ]
                loss, nan_share = utils.torch_nanmean(losses_flat, axis=0, return_nanshare=True)
                targets = combined_targets
                return loss, losses_split, single_eval_pos, targets, nan_share, model_names, mean_losses

            # No-mask path: compute loss across all samples
            losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
            S = output.shape[1]
            losses = losses.view(-1, S)
            loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
            losses = [losses[:, s] for s in range(S)]
            mean_losses = [l.mean(0).cpu().detach().item() if l is not None else None for l in losses]
                
        except Exception as e:
            print("Invalid step encountered, skipping...")
            print(e)
            raise e
        return loss, losses, single_eval_pos, targets, nan_share, model_names, mean_losses


    def training_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, single_eval_pos, targets, nan_share, model_names, mean_losses = self.forward(
            full_data=batch,
            use_mask=True,
            batch_idx=batch_idx
        )
        skip_batch = isinstance(losses, (list, tuple)) and all(l is None for l in losses)
        step_time = time.time() - step_start
        
        if not skip_batch:
            self.train_recorder.update(
                loss=loss,
                losses=mean_losses,
                single_eval_pos=single_eval_pos,
                targets=targets,
                nan_share=nan_share,
                step_time=step_time,
                categories=model_names
            )
        else:
            print(f"Global rank {self.global_rank} Batch {batch_idx} skipped (zero-loss, no metrics update).")
        
        if batch_idx == len(self.train_dl) - 1:
            self._hb("last_batch_reached")
        return loss 
    
    
    
    # ==================== Filtering & Curriculum Learning ====================
    def _collect_all_sample_losses(self):
        """Collect losses for all samples across all ranks."""
        all_losses_with_rank_info = []
        for batch_idx, batch in tqdm(enumerate(self.train_dl)):
            with torch.no_grad():
                _, losses, _, _, _, model_names, _ = self.forward(
                    full_data=batch,
                    use_mask=False,
                    batch_idx=batch_idx
                )
                for loss_index, loss in enumerate(losses):
                    loss_list = loss.detach().cpu().tolist()
                    batch_idx_list = [batch_idx * len(losses) + loss_index] * len(loss_list)
                    rank_idx_list = [self.global_rank] * len(loss_list)
                    sample_idx_list = list(range(len(loss_list)))
                    model_names_list = [model_names[loss_index]] * len(loss_list)
                    all_losses_with_rank_info.extend(
                        list(zip(rank_idx_list, batch_idx_list, sample_idx_list, model_names_list, loss_list))
                    )
        return all_losses_with_rank_info
    
    
    def _filter_top_distributed(self, filter_ratio):
        losses = self._collect_all_sample_losses()
        print(f"Collected {len(losses)} losses on global rank {self.global_rank}. Beginning filtering...")
        if torch.distributed.is_initialized():
            all_losses_gathered = [None] * self.num_device
            torch.distributed.all_gather_object(all_losses_gathered, losses)
            
            if self.global_rank == 0:
                flattened_data = sum(all_losses_gathered, [])
                flattened_data.sort(key=lambda x: x[-1])
                k = int(len(flattened_data) * filter_ratio)
                top_k_data = flattened_data[:k]
                print(f"Selected top {k} samples from {len(flattened_data)} total samples")
                print(flattened_data[0:5])  # Print first 5 entries for debugging
                category_sorted_data = sorted(flattened_data, key=itemgetter(3)) #top_k_data
                category_loss = {}
                for g, rest in groupby(category_sorted_data, key=itemgetter(3)):
                    category_loss[g] = [item[-1] for item in rest]
                
                for category in self.categories:
                    loss_entropy = np.array(category_loss[category]) - np.mean(category_loss[category])**2
                    loss_entropy = np.mean(loss_entropy)
                    self.data_weights_map[category] = (
                                        self.gamma * self.data_weights_map[category] + (1 - self.gamma) * float(loss_entropy))
                    
                    print(f"Updated weight for category {category}: {self.data_weights_map[category]:.4f}, entropy={loss_entropy:.4f}")
                    self.log(f'category_data_weights_bin{category[0]}_{category[1]}', self.data_weights_map[category], sync_dist=False)
                    self.log(f'current_loss_entropy_bin{category[0]}_{category[1]}', loss_entropy, sync_dist=False)
                
                sorted_data = sorted(top_k_data, key=itemgetter(0, 1))
                mask = {}
                for (rk, bidx), grp in groupby(sorted_data, key=itemgetter(0, 1)):
                    mask.setdefault(rk, {})[bidx] = [g[2] for g in grp]
                broadcast_data = [mask]
            else:
                broadcast_data = [{}]
                        
            torch.distributed.broadcast_object_list(broadcast_data, src=0)
            weights_container = [self.data_weights_map] if self.global_rank == 0 else [None]
            torch.distributed.broadcast_object_list(weights_container, src=0)
            self.data_weights_map = weights_container[0]
            
            # Each rank extracts its own mask
            full_mask_dict = broadcast_data[0]
            self.batch_mask = full_mask_dict.get(self.global_rank, {})
            print('End of filtering step for global rank', self.global_rank)
            torch.distributed.barrier()
        else:
            # Single device case
            losses.sort(key=lambda x: x[-1])
            k = int(len(losses) * filter_ratio)
            top_k_data = losses[:k]
            print(f"Selected top {k} samples from {len(losses)} total samples")
            
            sorted_data = sorted(top_k_data, key=itemgetter(1))
            self.batch_mask = {}
            for bidx, grp in groupby(sorted_data, key=itemgetter(1)):
                self.batch_mask[bidx] = [g[2] for g in grp]
    # ==================== End Filtering & Curriculum Learning ====================
    
    def on_train_epoch_start(self) -> None:
        filter_ratio = self.curriculum_scheduler.get_current_value(self.current_epoch)
        self.filter_ratio = filter_ratio
        print(f"Current epoch: {self.current_epoch}, filter ratio: {filter_ratio}")
        self._filter_top_distributed(filter_ratio)
        self._hb(f"mask_ready: batches_in_mask={len(self.batch_mask)}")

       
    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        print(f"Current epoch: {current_epoch}, global_rank: {self.global_rank}")
        lr = self.lr_schedulers().get_last_lr()[0]

        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)

        # Log main metrics
        self.log('train_loss', train_metric['avg_loss'], sync_dist=True)
        self.log('train_time', train_metric['total_time'], sync_dist=True)
        self.log('lr', lr, sync_dist=True)
        self.log('filter_ratio', self.filter_ratio, sync_dist=True)

        # Log model-specific losses
        self.log('train_gmm_loss', train_metric['avg_gmm_loss'], sync_dist=True)
        self.log('train_dependence_copula_loss', train_metric['avg_dependence_copula_loss'], sync_dist=True)
        self.log('train_probablistic_copula_loss', train_metric['avg_probablistic_copula_loss'], sync_dist=True)
        self.log('train_structure_scm_loss', train_metric['avg_structure_scm_loss'], sync_dist=True)
        self.log('train_measure_scm_loss', train_metric['avg_measure_scm_loss'], sync_dist=True)
        
        # Log category losses
        for category in self.categories:
            cat_loss = train_metric['avg_category_losses'][category]
            self.log(f'category_loss_bin{category[0]}_{category[1]}', cat_loss, sync_dist=True)
        
        # Record average loss
        self.train_losses.append(train_metric['avg_loss'])
        
        # Log category sample counts
        counts = Counter(self.data_samples)
        for category in self.categories:
            count = counts.get(category, 0)
            if self.global_rank == 0:
                self.log(f'category_count_bin{category[0]}_{category[1]}', count, sync_dist=False)
        
        if self.global_rank == 0:
            if getattr(self, "_pending_dl_metrics", None) is not None:
                for k, v in self._pending_dl_metrics.items():
                    self.log(k, v, sync_dist=False)
            self._pending_dl_metrics = None

        # Cleanup
        self.train_recorder.reset()
        gc.collect()
        torch.cuda.empty_cache()     

    
    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint['train_losses'] = self.train_losses
        checkpoint['data_weights_map'] = self.data_weights_map
        

    def on_load_checkpoint(self, checkpoint):
        # Load the lists of train and val losses
        self.train_losses = checkpoint.get('train_losses', [])
        self.data_weights_map = checkpoint.get('data_weights_map', self.data_weights_map)
        print('-' * 20)
        print(f'getting the train losses of length {len(self.train_losses)}  from the latest ckpt')
        train_losses_len = len(self.train_losses)
        print('-' * 20)
