from __future__ import annotations
import torch.distributed as dist
import time
import torch
from torch import nn
import random
import pickle
from . import utils
from .priors import prior
from .fomo_transformer import TransformerModel
from .bar_distribution import BarDistribution
from .utils import get_cosine_schedule_with_warmup, get_openai_lr,get_cosine_schedule_with_warmup_min_lr
from . import positional_encodings
import pytorch_lightning as pl
from .parallel_dataset_torch import EpochDataset, ValidationDataset
from torch.utils.data import DataLoader
from data_prior.GMM_torch import make_NdMclusterGMM, generate_linear_transform, transform_samples
import gc
import torch.nn.functional as F
from itertools import combinations, count
import torch
from torch.nn.utils import clip_grad_norm_
import os
import json
from collections import Counter
import numpy as np
from .curriculum_scheduler import CurriculumScheduler
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
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion,
                    BarDistribution) or "BarDistribution" in criterion.__class__.__name__:
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    # border_decoder = None if border_decoder is None else border_decoder(emsize, criterion.num_bars + 1).to(device)

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
        self.density_tree_loss = 0.0
        self.scm_prob_loss =0.0
        self.scm_contextual_loss = 0.0
        self.copula_loss = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.copula_step_count = 0
        self.scm_prob_step_count = 0
        self.scm_contextual_step_count = 0
        self.density_tree_step_count = 0
        self.model_names_list = ['gmm', 'corpula', 'prob', 'contextual'] #, 'density']
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
        self.density_tree_loss = 0.0
        self.scm_prob_loss =0.0
        self.scm_contextual_loss = 0.0
        self.copula_loss = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.copula_step_count = 0
        self.scm_prob_step_count = 0
        self.scm_contextual_step_count = 0
        self.density_tree_step_count = 0
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
                    elif name == 'corpula':
                        self.copula_loss += l
                        self.copula_step_count += 1
                    elif name == 'contextual':
                        self.scm_contextual_loss += l
                        self.scm_contextual_step_count += 1
                    elif name == 'prob':
                        self.scm_prob_loss += l
                        self.scm_prob_step_count += 1
                    elif name == 'density':
                        self.density_tree_loss += l
                        self.density_tree_step_count += 1
                    self.category_loss[category_name] += l
                    self.category_counts[category_name] += 1         
                    
            self.nan_steps += nan_share.cpu().item()
            self.ignore_steps += (targets == -100).float().mean().cpu().item()

        self.total_step_time += step_time

   


    def fetch_and_print(self, epoch=None, lr=None):
        avg_loss = self.total_loss / self.steps_per_epoch
        avg_gmm_loss = self.gmm_loss / self.gmm_step_count if self.gmm_step_count != 0 else 0
        avg_copula_loss = self.copula_loss / self.copula_step_count if self.copula_step_count != 0 else 0
        avg_contextual_loss = self.scm_contextual_loss / self.scm_contextual_step_count if self.scm_contextual_step_count != 0 else 0
        avg_prob_loss = self.scm_prob_loss / self.scm_prob_step_count if self.scm_prob_step_count != 0 else 0
        avg_density_loss = self.density_tree_loss / self.density_tree_step_count if self.density_tree_step_count != 0 else 0

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
            print(f" Avg losses: GMM={avg_gmm_loss:.4f}, Copula={avg_copula_loss:.4f}, "
                f"Contextual={avg_contextual_loss:.4f}, Prob={avg_prob_loss:.4f}, Density={avg_density_loss:.4f}")
            print('-' * 89)
            

        # avg_cosine_similarities computed and printed, but not returned
        return {
            'avg_loss': avg_loss,
            'nan_share': nan_share,
            'ignore_share': ignore_share,
            'total_time': total_time,
            'avg_gmm_loss': avg_gmm_loss,
            'avg_copula_loss': avg_copula_loss,
            'avg_contextual_loss': avg_contextual_loss,
            'avg_prob_loss': avg_prob_loss,
            'avg_density_loss': avg_density_loss,
            'avg_category_losses': avg_category_losses
        }


class ZeroShotOD(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 priordataloader_class_or_get_batch: prior.PriorDataLoader | callable, criterion,
                 encoder_generator, 
                 dropout=0.0,
                 weight_decay=0.0,
                 input_normalization=False,
                 y_encoder_generator=None,
                 pos_encoder_generator=None, 
                 decoder_dict={},
                 extra_prior_kwargs_dict={},
                 train_extra_dict=None, 
                 resume_from_ckpt=False,  # added here
                 scheduler=get_cosine_schedule_with_warmup_min_lr, #get_cosine_schedule_with_warmup,
                 load_weights_from_this_state_dict=None, 
                 validation_period=10, 
                 single_eval_pos_gen=None,
                 gpu_device='cuda:0',
                 aggregate_k_gradients=1, 
                 verbose=False, 
                 style_encoder_generator=None, 
                 epoch_callback=None,
                 step_callback=None,
                 continue_model=None,
                 initializer=None, 
                 initialize_with_model=None, 
                 train_mixed_precision=False, 
                 efficient_eval_masking=True,
                 border_decoder=None,
                 num_global_att_tokens=0,
                 T0 = 0, 
                 num_bins = 5,
                 progress_bar=False,
                 **model_extra_args):
        super(ZeroShotOD, self).__init__()

        train_cfg = cfg.train
        prior_gmm_cfg = cfg.prior.mixture.gmm
        # train hyperparameters
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
        self.steps_per_epoch = train_cfg.steps_per_epoch
        lr = train_cfg.lr  #/ num_device
        
        #max_dimension
        self.max_feature_dim = cfg.prior.mixture.max_feature_dim
        self.min_feature_dim = 2

        # prior hyperparameters
        self.max_feature_dim = prior_gmm_cfg.max_feature_dim
        self.max_model_dim = prior_gmm_cfg.max_model_dim
        self.max_num_cluster = prior_gmm_cfg.max_num_cluster
        self.inflate_full = prior_gmm_cfg.inflate_full
        self.model_names_list = ['gmm', 'corpula', 'prob', 'contextual'] 

        # specifics for generate-one-train-one
        self.gen_one_train_one = False if train_extra_dict is None else True
        self.prior_train_data_gen = None if train_extra_dict is None else train_extra_dict['prior_train_data_gen']

        #assert self.batch_size % num_device == 0
        # define train/val data loader
        self.criterion = criterion

        self.apply_linear_transform = train_cfg.apply_linear_transform
        self.dataloader_para = extra_prior_kwargs_dict['pt_dataloader']

        self.base_data_path = f'{prior_gmm_cfg.data_dir}/num_feat_{self.max_feature_dim}'
        print('train data loader')
        if not self.apply_linear_transform and not self.gen_one_train_one:
            # meaning using stored data and use no LT (then at least 1 epoch is already generated)
            train_data_path = f'{self.base_data_path}/train'  # if provided, by default `epoch0` will be loaded
        else:
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
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,  #change shuffle = False, so we can retrieve the shuffled data
                                   collate_fn=self.train_dataset.prior_batch_collate_fn,
                                   **self.dataloader_para)
        
        if train_cfg.use_validation:
            print('no validation')
            self.val_dataset = None
            self.val_dl = None
            
        #print(f'Style definition of first 3 examples: {None}')
        style_encoder = None
        pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
        if isinstance(self.criterion, nn.GaussianNLLLoss):
            self.n_out = 2
        elif isinstance(self.criterion,
                        BarDistribution) or "BarDistribution" in self.criterion.__class__.__name__:
            self.n_out = self.criterion.num_bars
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            self.n_out = self.criterion.weight.shape[0]
        else:
            self.n_out = 1

        # initialize model
        if continue_model:
            raise NotImplementedError
        else:
            decoder_dict = decoder_dict if decoder_dict else {'standard': (None, self.n_out)}

            decoder_once_dict = {}

            encoder = encoder_generator(extra_prior_kwargs_dict['num_features'], emsize)
            self.model = TransformerModel(encoder=encoder
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
            print(self.model)
        self.model.criterion = self.criterion

        print(
            f"Using a Transformer with {sum(p.numel() for p in self.model.parameters()) / 1000 / 1000:.{2}f} M parameters")

        try:
            for (k, v), (k2, v2) in zip(self.model.state_dict().items(), initialize_with_model.state_dict().items()):
                print(k, ((v - v2) / v).abs().mean(), v.shape)
        except Exception:
            pass

        # define parameters for optimizer & scheduler
        self.lr = lr
        self.scheduler_fn = scheduler
        self.warmup_epochs = epochs // 10  #epochs // 5  #epochs // 10 # warmup epochs, usually 1/10 of total epochs
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        #define parameters for curriculum learning
        self.gamma =0.1 #smoothing factor for moving average
        self.temperature= 0.3
        
        #define a list to store the losses
        self.all_losses_with_rank_info = []
        
        self.curriculum_scheduler = CurriculumScheduler(total_steps = self.epochs, 
                                                        a=0.8, 
                                                        b=0.2, 
                                                        scheduler_name='root',
                                                        max_value=0.95)
        
        #divide the dimensions into equal size bins
        self.num_bins = num_bins
        self.bin_size = (self.max_feature_dim - self.min_feature_dim) // self.num_bins
        self.bin_ranges = [(self.min_feature_dim + i * self.bin_size, 
                    self.min_feature_dim + (i + 1) * self.bin_size) for i in range(self.num_bins)]
        self.bin_ranges[-1] = (self.bin_ranges[-1][0], self.max_feature_dim + 1)  # Create a new tuple to replace the last bin

        #set the categories
        # Create B x D grid of (bin_range, model_name) pairs
        self.categories = [(bin_range, model_name)
            for bin_range in self.bin_ranges
            for model_name in self.model_names_list
        ]
        
        self.categories_weights = [0.0 for i in self.categories] 
        self.data_weights_map = dict(zip(self.categories, self.categories_weights))

        self.total_data_size = self.steps_per_epoch * self.batch_size
        self.data_samples = self.sample(k=self.total_data_size, temperature=self.temperature)
        
        self.all_category_losses = None # to be updated after each epoch
        self.batch_mask = None  # default do not use any mask

        # check that everything uses up-to-date APIs
        utils.check_compatibility(self.train_dl)
        #utils.check_compatibility(self.val_dl)

        # training & validation dynamics
        self.train_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=verbose,categories=self.categories)
        #self.val_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=False)
        #self.test_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=False)
        self.train_losses = []
        #self.val_losses = []
        
    #========================= For Softmax Sampling =========================#    
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
    #=========================END For Softmax Sampling =========================#
    
    
    # Debug
    # Add near the top of the class (or as a free function)
    def _hb(self, tag):
        try:
            ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            ws = -1
        print(f"[HB] epoch={self.current_epoch} rank={self.global_rank} world={ws} tag={tag} time={time.time():.3f}", flush=True)
        
  
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
        # first load the data samples for this epoch baseed on curriculum learning
        print(f'Preparing data samples for epoch {self.current_epoch}...')
        self.data_samples = self.sample(k=self.total_data_size, temperature=self.temperature)
        #count the data samples for each category and print
        if self.global_rank == 0:
            category_counts = Counter(self.data_samples)
            print(f'Data sample counts per category for epoch {self.current_epoch}:')
            for category, count in category_counts.items():
                logits = self._weights_array()
                p_all = self._softmax(logits, self.temperature)
                p = p_all[self.categories.index(category)]
                print(f'  Category {category}: {count} samples, weight: {self.data_weights_map[category]:.4f}, p: {p:.4f}')
                self._pending_dl_metrics = {
                        f'category_normalized_weights_bin{cat[0]}_{cat[1]}': float(p_all[self.categories.index(cat)])
                        for cat in self.categories
                    }
        
        # Shuffle data_samples on rank 0 and broadcast to all devices
        if torch.distributed.is_initialized():
            if self.global_rank == 0:
                random.shuffle(self.data_samples)
            # Broadcast the entire shuffled list to all devices
            # Convert to indices for easier broadcasting
            if self.global_rank == 0:
                # Create a mapping from model names to indices
                model_to_idx = {name: i for i, name in enumerate(self.categories)}
                data_indices = [model_to_idx[name] for name in self.data_samples]
            else:
                data_indices = [0] * len(self.data_samples)
            
            # Broadcast as tensor
            data_tensor = torch.tensor(data_indices, device=self.device)
            torch.distributed.broadcast(data_tensor, src=0)
            
            # Convert back to model names and take device portion
            idx_to_model = {i: name for i, name in enumerate(self.categories)}
            shuffled_samples = [idx_to_model[idx.item()] for idx in data_tensor]
            
            # Each device takes its portion
            samples_per_device = len(shuffled_samples) // self.num_device
            start_idx = self.global_rank * samples_per_device
            end_idx = start_idx + samples_per_device
            self.individual_data_samples = shuffled_samples[start_idx:end_idx]
        else:
            # Single device case
            random.shuffle(self.data_samples) 
            self.individual_data_samples = self.data_samples
        
        # generate data on the fly (if with "generate-one-train-one" paradigm)
        if self.gen_one_train_one:
            print('generate new data on the fly now....')
            self.train_dataset.free_data()  # free data to allow space for data generation
            data_dict = self.generate_new_data_for_train()
        else:  # just loading, which will be handled internally by the train_dataset
            raise NotImplementedError("Loading pre-generated data is not supported.")

        self.train_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)

        if data_dict is None:  # using the existing data (reuse 1 epoch of data)
            return self.train_dl
        else:  # load new data
            del data_dict
            gc.collect()
            torch.cuda.empty_cache()
            # set the epoch such that it will load different data
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                       collate_fn=self.train_dataset.prior_batch_collate_fn,
                                       **self.dataloader_para)#,num_workers=4)
            print( f'num batches per epoch={len(self.train_dl)}')
            return self.train_dl


    def generate_new_data_for_train(self):
        self.prior_train_data_gen.gen1tr1_epoch_id = self.current_epoch
        # set the generation epoch to avoid repetitive generation
        if self.apply_linear_transform:
            raise NotImplementedError("LT + gen1tr1 not implemented yet.")
        else:  # generate `steps_per_epoch * batch_size` all together on the fly
            inliners_list, LA_list, _ , model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=None,  # generate `steps_per_epoch * batch_size`
                save_data=False,
                categories=self.individual_data_samples)
            
        assert len(LA_list) >= int(self.steps_per_epoch * self.batch_size /self.num_device ), \
            print(
                f'number of training instances={len(LA_list)} should be >= {self.steps_per_epoch * self.batch_size}')
        return {'in': inliners_list, 'la': LA_list, 'model_names':model_names}


    def val_dataloader(self):
        # DataLoader for validation
        raise NotImplementedError("Validation dataloader is not implemented.")
        #return self.val_dl
    
    
    def test_dataloader(self):
        # DataLoader for testing
        raise NotImplementedError("Test dataloader is not implemented.")
        #return self.test_dl
    

    def forward(self, full_data, use_mask=False,batch_idx = None):
        data = (full_data.style.to(self.device) if full_data.style is not None else None, full_data.x.to(self.device),
                full_data.y.to(self.device) if full_data.y is not None else None)
        
        model_names = full_data.model_names if full_data.model_names is not None else None
        
        # added here (do not include y in the feature)
        targets = full_data.target_y.to(self.device)
        single_eval_pos = full_data.single_eval_pos
        
        # If style is set to None, it should not be transferred to device
        out = self.model(tuple(e.to(self.device) if torch.is_tensor(e) else e for e in data),
                            single_eval_pos=single_eval_pos, only_return_standard_out=False)

        # this handling is for training old models only, this can be deleted soon(ish)
        # to only support models that return a tuple of dicts
        out, output_once = out if isinstance(out, tuple) else (out, None)
        output = out['standard'] if isinstance(out, dict) else out

        if single_eval_pos is not None:
            targets = targets[single_eval_pos:]

        if len(targets.shape) == len(output.shape):
            # this implies the prior uses a trailing 1 dimesnion
            # below we assume this not to be the case
            targets = targets.squeeze(-1)
        assert targets.shape == output.shape[:-1], f"Target shape {targets.shape} " \
                                                    f"does not match output shape {output.shape}"      
        assert not torch.isinf(output).any(), "Inf in outputs"
        assert output.shape[-1] == 2, "Each output must have 2 logits (classes)"
        assert targets.min() >= 0 and targets.max() < 2, "Target out of range"
            
        if use_mask is True:
            assert batch_idx is not None, "batch_idx must be provided when use_mask is True"

            # ---- Generalize the no-mask path too (for S sides) ----
            losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
            
            #first filter out the batch based on the g(ab)t
            per_elem_losses = losses.view(-1)

            # Exclude non-finite values from selection
            valid_mask = torch.isfinite(per_elem_losses)
            valid_losses = per_elem_losses[valid_mask]

            k = int(self.filter_ratio * valid_losses.numel())
            k = max(1, min(valid_losses.numel(), k))
            topk_vals, topk_idx_in_valid = torch.topk(valid_losses, k, largest=False)

            # Map selected indices back to the original flattened space
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            selected_indices = valid_indices[topk_idx_in_valid]

            selected_mask = torch.zeros_like(per_elem_losses, dtype=torch.bool)
            selected_mask[selected_indices] = True
            loss = topk_vals.mean()

            B = output.shape[0]
            T = per_elem_losses.numel() // B
            per_elem_losses_2d = per_elem_losses.view(B, T)
            selected_mask_2d = selected_mask.view(B, T)

            mean_losses = []
            for i in range(B):
                sel = selected_mask_2d[i]
                if sel.any():
                    mean_losses.append(per_elem_losses_2d[i][sel].mean().detach().item())
                else:
                    mean_losses.append(None)
            # reshape unfiltered losses to a list [step] -> tensor(B,)
            losses = [per_elem_losses_2d[:, s] for s in range(T)]
            filtered_losses = []
            for s in range(T):
                step_losses = per_elem_losses_2d[:, s]
                step_mask = selected_mask_2d[:, s]
                filtered_losses.append(step_losses[step_mask] if step_mask.any() else step_losses.new_empty(0))
            return loss, losses, filtered_losses, single_eval_pos, targets, None, model_names, mean_losses

        else:
            losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
            S = output.shape[1]
            losses = losses.view(-1, S)  # shape: (N, S)
            loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
            losses = [losses[:, s] for s in range(S)]
            mean_losses = [l.mean(0).cpu().detach().item() if l is not None else None for l in losses]
            return loss, losses, losses, single_eval_pos, targets, nan_share, model_names, mean_losses



    def training_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, filtered_losses, single_eval_pos, targets, nan_share, model_names, mean_losses = self.forward(full_data=batch, use_mask=True,batch_idx =batch_idx)
        skip_batch = isinstance(losses, (list, tuple)) and all(l is None for l in losses)
        step_time = time.time() - step_start
        if not skip_batch:
            # Collect loss information for each sample
            if self.use_filtered_loss is True:
                losses = filtered_losses
                for loss_index, loss in enumerate(losses):
                        loss_list = loss.detach().cpu().tolist()
                        batch_idx_list = [batch_idx * len(losses) + loss_index] * len(loss_list)
                        rank_idx_list = [self.global_rank] * len(loss_list)
                        sample_idx_list = list(range(len(loss_list)))
                        model_names_list = [model_names[loss_index]] * len(loss_list)
                        self.all_losses_with_rank_info.extend(
                        list(zip(rank_idx_list, batch_idx_list, sample_idx_list, model_names_list, loss_list))
                        )
            self.train_recorder.update(loss=loss,
                                       losses=mean_losses,
                                       single_eval_pos=single_eval_pos, 
                                       targets=targets,
                                       nan_share=nan_share, 
                                       step_time=step_time, 
                                       categories=model_names)
        else:
            print(f"Global rank {self.global_rank} Batch {batch_idx} skipped (zero-loss, no metrics update).")
        
        if batch_idx == len(self.train_dl) - 1:
            self._hb("last_batch_reached")
        return loss 
      
    
    def _calculate_prior_entropy(self):
        if torch.distributed.is_initialized():
            all_losses_gathered = [None] * self.num_device
            torch.distributed.all_gather_object(all_losses_gathered, self.all_losses_with_rank_info)
            
            if self.global_rank == 0:
                flattened_data = sum(all_losses_gathered, [])
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
                        
            weights_container = [self.data_weights_map] if self.global_rank == 0 else [None]
            torch.distributed.broadcast_object_list(weights_container, src=0)
            self.data_weights_map = weights_container[0]
            
            
                
                
    #=========================END Filtering Batches ==============================#
    def on_train_epoch_start(self) -> None:
        filter_ratio = self.curriculum_scheduler.get_current_value(self.current_epoch)
        self.filter_ratio = filter_ratio
        print(f"Current epoch: {self.current_epoch}, filter ratio: {filter_ratio}")
       
       
    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        print(f"Current epoch: {current_epoch}, global_rank: {self.global_rank}")
        lr = self.lr_schedulers().get_last_lr()[0]

        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)

        # Log main metrics
        self.log('train_loss', train_metric['avg_loss'], sync_dist=True)
        self.log('train_time', train_metric['total_time'], sync_dist=True)
        
        # Log learning rate
        self.log('lr',lr, sync_dist=True)
        self.log('filter_ratio', self.filter_ratio, sync_dist=True)

        # Log additional losses
        self.log('train_gmm_loss', train_metric['avg_gmm_loss'], sync_dist=True)
        self.log('train_copula_loss', train_metric['avg_copula_loss'], sync_dist=True)
        self.log('train_contextual_loss', train_metric['avg_contextual_loss'], sync_dist=True)
        self.log('train_prob_loss', train_metric['avg_prob_loss'], sync_dist=True)
        self.log('train_density_loss', train_metric['avg_density_loss'], sync_dist=True)
        
        # Log dataset weights
        for i, category in enumerate(self.categories):
            cat_loss = train_metric['avg_category_losses'][category]
            self.log(f'category_loss_bin{category[0]}_{category[1]}', cat_loss, sync_dist=True)

        # Record average loss
        self.train_losses.append(train_metric['avg_loss'])
        
        counts = Counter(self.data_samples)
        sample_counts_by_category = {cat: counts.get(cat, 0) for cat in self.categories}
        for i, category in enumerate(self.categories):
            count = sample_counts_by_category[category]
            if self.global_rank == 0:
                self.log(f'category_count_bin{category[0]}_{category[1]}', count, sync_dist=False)
                print(f"Global rank {self.global_rank} Category {category}: sampled {count} times.") 
        
        if self.global_rank == 0:
            if getattr(self, "_pending_dl_metrics", None) is not None:
                for k, v in self._pending_dl_metrics.items():
                    self.log(k, v, sync_dist=False)
            self._pending_dl_metrics = None
        
        self._calculate_prior_entropy()
        self.all_losses_with_rank_info = [] # reset for next epoch

        # Reset metrics and clean up
        self.train_recorder.reset()
        gc.collect()
        torch.cuda.empty_cache()     

    
    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint['train_losses'] = self.train_losses
        

    def on_load_checkpoint(self, checkpoint):
        # Load the lists of train and val losses
        self.train_losses = checkpoint.get('train_losses', [])
        print('-' * 20)
        print(f'getting the train losses of length {len(self.train_losses)}  from the latest ckpt')
        train_losses_len = len(self.train_losses)
        print('-' * 20)
