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
from .utils import get_cosine_schedule_with_warmup, get_openai_lr
from . import positional_encodings
import pytorch_lightning as pl
from .parallel_dataset_torch import EpochDataset, ValidationDataset
from torch.utils.data import DataLoader
from data_prior.GMM_torch import make_NdMclusterGMM, generate_linear_transform, transform_samples
import gc
import torch.nn.functional as F
from itertools import combinations
import torch
from torch.nn.utils import clip_grad_norm_
import os
import json
from collections import Counter
import numpy as np

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


# =====================
# Alpha schedule
# =====================



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
    def __init__(self, seq_len, steps_per_epoch, verbose):
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
        
        
        

    def update(self, loss, losses, single_eval_pos, targets, nan_share, step_time, model_names,grad_vectors=None, cos_sim=None):
        if not torch.isnan(loss):
            self.total_loss += loss.cpu().detach().item()
            if model_names is not None and losses is not None:
                mean_loss = losses.mean(0)
                for i,name in enumerate(model_names):
                    l = mean_loss[i].cpu().detach().item()
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
            'avg_density_loss': avg_density_loss
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
                 scheduler=get_cosine_schedule_with_warmup,
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
                 border_decoder=None
                 , num_global_att_tokens=0,
                 T0 = 0, 
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

        # LT is False & gen1tr1 is False: pre-generate n epochs of data, directly load & do nothing
        # LT is True & gen1tr1 is False: pre-generate n epochs of data, directly load & add LT

        # LT is False & gen1tr1 is True: generate 1 epoch of data on the fly for each epoch
        # LT is True & gen1tr1 is True: generate (<1 epoch of data + augment with LT) on the fly for each epoch
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
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                   collate_fn=self.train_dataset.prior_batch_collate_fn,
                                   **self.dataloader_para)
        
        if train_cfg.use_validation:
            print('validation data loader')
            if os.path.exists('/home/xding2/FoMo-0D-explore/pfns/val_dataset_dict.json'):
                with open('/home/xding2/FoMo-0D-explore/pfns/val_dataset_dict.json', 'r') as f:
                    dataset_dict = json.load(f)
            if os.path.exists('/home/xding2/FoMo-0D-explore/pfns/val_name_to_tgt.json'):
                with open('/home/xding2/FoMo-0D-explore/pfns/val_name_to_tgt.json', 'r') as f:
                    name_to_tgt = json.load(f)
            if os.path.exists('/home/xding2/FoMo-0D-explore/pfns/val_dataset_list.txt'):
                with open('/home/xding2/FoMo-0D-explore/pfns/val_dataset_list.txt', 'r') as f:
                    dataset_list = f.read().splitlines()
            self.val_dataset = ValidationDataset(max_feature_dim=self.max_feature_dim,
                                                  anomaly_ratio=0.2,
                                                  dataset_dict=dataset_dict,
                                                  name_to_tgt=name_to_tgt,
                                                  dataset_list=dataset_list,
                                                  val_data_path='/home/xding2/FoMo-0D-explore/data/val_data')
            self.val_dl = DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                     collate_fn=self.val_dataset.prior_batch_collate_fn,
                                     **self.dataloader_para)
        else:
            print('no validation')
            self.val_dataset = None
            self.val_dl = None

        print(f'Style definition of first 3 examples: {None}')
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
        self.warmup_epochs =  0 #epochs // 10 # warmup epochs, usually 1/10 of total epochs
        self.weight_decay = weight_decay
        self.epochs = epochs
        

        self.intial_assigned_weights = {}
        for i in self.model_names_list:
            self.intial_assigned_weights[i] = 1.0 if i == 'gmm' else 0.0
        self.data_weights = [1.0 if i == 'gmm' else 0.0 for i in self.model_names_list]
        self.data_weights_map = dict(zip(self.model_names_list, self.data_weights))
        
        self.total_data_size = self.steps_per_epoch * self.batch_size
        self.data_samples = [[name] * int(weight * self.total_data_size) for name, weight in self.data_weights_map.items()]
        self.data_samples = sum(self.data_samples, [])

        # check that everything uses up-to-date APIs
        utils.check_compatibility(self.train_dl)
        utils.check_compatibility(self.val_dl)

        # training & validation dynamics
        self.train_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=verbose)
        self.val_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=False)
        self.test_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=False)
        self.train_losses = []
        self.val_losses = []
        
        # algorithm for updates exp3 active learning
        self.T0 = T0
        self.eta = 0.1 #learning rate for weights updates
        self.weight_variable = dict(zip(self.model_names_list, [1.0 / len(self.model_names_list)] * len(self.model_names_list)))
        self.eps = 0.02
        


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
        if self.val_dataset is not None:
            self.val_dataset.set_rank(rank=self.global_rank)
        if self.trainer.ckpt_path:
            print(f"Resuming training from checkpoint: {self.trainer.ckpt_path}")
        else:
            print("Training from scratch.")


    def train_dataloader(self):
        # Shuffle data_samples on rank 0 and broadcast to all devices
        if torch.distributed.is_initialized():
            if self.global_rank == 0:
                random.shuffle(self.data_samples)
            # Broadcast the entire shuffled list to all devices
            # Convert to indices for easier broadcasting
            if self.global_rank == 0:
                # Create a mapping from model names to indices
                model_to_idx = {name: i for i, name in enumerate(self.model_names_list)}
                data_indices = [model_to_idx[name] for name in self.data_samples]
            else:
                data_indices = [0] * len(self.data_samples)
            
            # Broadcast as tensor
            data_tensor = torch.tensor(data_indices, device=self.device)
            torch.distributed.broadcast(data_tensor, src=0)
            
            # Convert back to model names and take device portion
            idx_to_model = {i: name for i, name in enumerate(self.model_names_list)}
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
            self.train_dataset.free_data()  # free data to allow space for data generation
            data_dict = self.generate_new_data_for_train()
        else:  # just loading, which will be handled internally by the train_dataset
            data_dict = self.load_generated_data_for_train()

        self.train_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)

        if data_dict is None:  # using the existing data (reuse 1 epoch of data)
            return self.train_dl
        else:  # load new data
            del data_dict
            gc.collect()
            torch.cuda.empty_cache()
            # set the epoch such that it will load different data
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.train_dataset.prior_batch_collate_fn,
                                       **self.dataloader_para)#,num_workers=4)
            print( f'num batches per epoch={len(self.train_dl)}')
            return self.train_dl


    def generate_new_data_for_train(self):
        self.prior_train_data_gen.gen1tr1_epoch_id = self.current_epoch
        # set the generation epoch to avoid repetitive generation
        if self.apply_linear_transform:
            # generate less on the fly, and use LT to fill up to `steps_per_epoch * batch_size`
            inliners_list, LA_list, sub_dims_list, model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=1,  # generate `max_feature_dim * max_num_cluster`
                save_data=False,
                model_weights= self.individual_data_samples)
            inliners_list, LA_list = self.increase_datasets_via_LT(inliners_list=inliners_list, LA_list=LA_list,
                                                                   sub_dims_list=sub_dims_list, transform_all=False,
                                                                   is_source_numpy=False)
        else:  # generate `steps_per_epoch * batch_size` all together on the fly
            inliners_list, LA_list, _ , model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=None,  # generate `steps_per_epoch * batch_size`
                save_data=False,
                model_weights=self.individual_data_samples)
            
        assert len(LA_list) >= int(self.steps_per_epoch * self.batch_size /self.num_device ), \
            print(
                f'number of training instances={len(LA_list)} should be >= {self.steps_per_epoch * self.batch_size}')

        return {'in': inliners_list, 'la': LA_list, 'model_names':model_names}


    def load_generated_data_for_train(self):
        print('using pre-generated data')
        if self.apply_linear_transform:  # some datasets are pre-generated, apply LT to them dynamically
            pre_gen_epoch = self.current_epoch % self.reuse_data_every_n
            print(f'loading from pre-generated data for epoch={pre_gen_epoch} and apply LT for epochs >={self.reuse_data_every_n}')
            inliners_list = load_pickle(
                file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/in.pickle')
            LA_list = load_pickle(
                file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/la.pickle')

            if self.current_epoch >= self.reuse_data_every_n:
                # apply LT only for epochs >= reuse_data_every_n epochs
                print(f'applying LT...')
                if self.inflate_full:  # no sub dims
                    sub_dims_list = None
                else:
                    sub_dims_list = load_pickle(
                        file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/sub_dims.pickle')

                inliners_list, LA_list = self.increase_datasets_via_LT(inliners_list=inliners_list, LA_list=LA_list,
                                                                       sub_dims_list=sub_dims_list,
                                                                       transform_all=True,
                                                                       is_source_numpy=True,)
            data_dict = {'in': inliners_list, 'la': LA_list}
        else:
            if self.current_epoch % self.reuse_data_every_n != 0:  # reusing more than 1 epoch
                pre_gen_epoch = self.current_epoch % self.reuse_data_every_n
                print(f'loading from pre-generated data for epoch={pre_gen_epoch} and apply no LT')
                inliners_list = load_pickle(
                    file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/in.pickle')
                LA_list = load_pickle(
                    file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/la.pickle')
                data_dict = {'in': inliners_list, 'la': LA_list}
            else:
                data_dict = None  # reuse_every_n=1, then no need to always load the data

        return data_dict


    def increase_datasets_via_LT(self, inliners_list, LA_list, sub_dims_list, transform_all, is_source_numpy=False):
        LT_in_list, LT_LA_list = [], []
        # needs to be at least (`steps_per_epoch * batch_size` - len(inliners_list))
        data_size = len(inliners_list)
        indices = list(range(data_size))  # Create a list of numbers from 0 to n-1
        random.shuffle(indices)  # Shuffle the list to get a random order
        if transform_all:
            transform_times = 1
        else:
            transform_times = (self.steps_per_epoch * self.batch_size + data_size) // data_size - 1

        for i in indices:
            for _ in range(transform_times):
                if sub_dims_list is None:
                    inliners, LA, sub_dims = inliners_list[i], LA_list[i], None
                    A, b = generate_linear_transform(dim=inliners.shape[-1],
                                                     device=None if is_source_numpy else inliners.device,
                                                     A_scale=1, b_scale=1,)
                else:
                    inliners, LA, sub_dims = inliners_list[i], LA_list[i], sub_dims_list[i]
                    A, b = generate_linear_transform(dim=len(sub_dims), device=None if is_source_numpy else inliners.device,
                                                     A_scale=1, b_scale=1,)

                inliners = transform_samples(samples=inliners, sub_dims=sub_dims, A=A, b=b,
                                             is_source_numpy=is_source_numpy)
                LA = transform_samples(samples=LA, sub_dims=sub_dims, A=A, b=b,
                                       is_source_numpy=is_source_numpy)

                LT_in_list.append(inliners)
                LT_LA_list.append(LA)

        if transform_all:
            return LT_in_list, LT_LA_list
        else:
            inliners_list = inliners_list + LT_in_list
            LA_list = LA_list + LT_LA_list
            return inliners_list, LA_list


    def val_dataloader(self):
        # DataLoader for validation
        return self.val_dl
    
    
    def test_dataloader(self):
        return self.test_dl
    
    
    def val_forward(self, full_data):
        data = (full_data.style.to(self.device, dtype=torch.float32) if full_data.style is not None else None,
        full_data.x.to(self.device, dtype=torch.float32),
        full_data.y.to(self.device, dtype=torch.float32) if full_data.y is not None else None)
        model_names = full_data.model_names if full_data.model_names is not None else None
        # added here (do not include y in the feature)
        targets = full_data.target_y.to(self.device)
        single_eval_pos = full_data.single_eval_pos
        with torch.no_grad():
            try:
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
                
                if isinstance(self.criterion, nn.GaussianNLLLoss):
                    assert output.shape[-1] == 2, \
                        'need to write a little bit of code to handle multiple regression targets at once'

                    mean_pred = output[..., 0]
                    var_pred = output[..., 1].abs()
                    losses = self.criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
                elif isinstance(self.criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                    targets[torch.isnan(targets)] = -100
                    losses = self.criterion(output.flatten(), targets.flatten())
                elif isinstance(self.criterion, nn.CrossEntropyLoss):
                    targets[torch.isnan(targets)] = -100
                    # print(f"{targets.min()=}, {targets.max()=}")
                    losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
                else:
                    losses = self.criterion(output, targets)

                losses = losses.view(-1, output.shape[1])  # sometimes the seq length can be one off
                #print(full_data.x.shape, targets.shape,losses.shape, output.shape)
                # that is because bar dist appends the mean
                loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
            except Exception as e:
                print("Invalid step encountered, skipping...")
                print(e)
                raise e
            return loss, None, single_eval_pos, targets, nan_share, model_names, None, None


    def forward(self, full_data):
        data = (full_data.style.to(self.device) if full_data.style is not None else None, full_data.x.to(self.device),
                full_data.y.to(self.device) if full_data.y is not None else None)
        
        model_names = full_data.model_names if full_data.model_names is not None else None
        # added here (do not include y in the feature)
        targets = full_data.target_y.to(self.device)
        single_eval_pos = full_data.single_eval_pos
        try:
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
            
            if isinstance(self.criterion, nn.GaussianNLLLoss):
                assert output.shape[-1] == 2, \
                    'need to write a little bit of code to handle multiple regression targets at once'

                mean_pred = output[..., 0]
                var_pred = output[..., 1].abs()
                losses = self.criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
            elif isinstance(self.criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                targets[torch.isnan(targets)] = -100
                losses = self.criterion(output.flatten(), targets.flatten())
            elif isinstance(self.criterion, nn.CrossEntropyLoss):
                targets[torch.isnan(targets)] = -100
                # print(f"{targets.min()=}, {targets.max()=}")
                losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
            else:
                losses = self.criterion(output, targets)
            #print('losses shape before view', losses.shape)
            losses = losses.view(-1, output.shape[1]) 
            loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

        except Exception as e:
            print("Invalid step encountered, skipping...")
            print(e)
            raise e

        return loss, losses, single_eval_pos, targets, nan_share, model_names

    def training_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, single_eval_pos, targets, nan_share, model_names = self.forward(full_data=batch)
        step_time = time.time() - step_start
        self.train_recorder.update(loss=loss,
                                   losses=losses,
                                   single_eval_pos=single_eval_pos, 
                                   targets=targets,
                                   nan_share=nan_share, 
                                   step_time=step_time, 
                                   model_names = model_names, 
                                   grad_vectors = None,
                                   cos_sim=None)
        return loss

    def on_train_epoch_start(self) -> None:
        # Record the start time for the DataLoader to prepare the next batch
        self.train_recorder.epoch_start_time = time.time()
        
        if self.current_epoch < self.warmup_epochs:
            print(f"Warmup epoch {self.current_epoch+1}/{self.warmup_epochs}")
            return 
            
        # print current epoch
        print(f"\n--- Starting epoch {self.current_epoch} ---")
        
                    
        def alpha_power(t: int, T: int, gamma: float) -> float:
            return float((t / max(1, T)) ** gamma)

        def alpha_cosine(t: int, T: int) -> float:
            from math import cos, pi
            return 0.5 * (1 - cos(pi * t / max(1, T)))

        def get_alpha(t: int, T: int, scheduler='cosine') -> float:
            if scheduler == "cosine":
                return alpha_cosine(t, T)
            return alpha_power(t, T, 0.5)
        
        if self.current_epoch % self.T0 == 0:
            # Run on rank 0 only
            print(f"Current epoch: {self.current_epoch}, T0: {self.T0}")
            if self.global_rank == 0:
                rewards = self._calculate_gradient_norms()
                v0 = np.array([self.data_weights_map[key] for key in sorted(self.data_weights_map.keys())])
                w_np = np.array([self.weight_variable[key] for key in sorted(self.weight_variable.keys())])
                rewards_np = np.array([rewards[key] for key in sorted(rewards.keys())])
                alpha = get_alpha(self.current_epoch - self.warmup_epochs, self.epochs - self.warmup_epochs, "cosine")
                w_new = w_np * np.exp(self.eta * rewards_np)
                w_new = w_new / w_new.sum()
                w_new = np.maximum(w_new, self.eps)
                w_new /= w_new.sum()
                u = np.array([0.25,0.25,0.25,0.25])
                # get vt
                beta = np.exp(-(self.current_epoch - self.warmup_epochs) / ((self.epochs - self.warmup_epochs) / 50))
                v = (1 - alpha) * ((1 - beta) * w_new + beta * v0) + alpha * u
                v = np.maximum(v, self.eps); v /= v.sum()
                
                self.data_weights_map = dict(zip(sorted(self.data_weights_map.keys()), v.tolist()))
                print(f"Updated data weights: {self.data_weights_map}")
                self.weight_variable = dict(zip(sorted(self.weight_variable.keys()), w_new.tolist()))
                self.data_samples = [[name] * int(weight * self.total_data_size) for name, weight in self.data_weights_map.items()]
                self.data_samples = sum(self.data_samples, [])

                # If data_samples are less than total_data_size, append with random selections
                if len(self.data_samples) < self.total_data_size:
                    additional_samples = random.choices(self.model_names_list, k=self.total_data_size - len(self.data_samples))
                    self.data_samples.extend(additional_samples)
                    
        if self.global_rank == 0:
            for i in self.data_weights_map:
                self.log(f"data_weights/{i}", torch.tensor(self.data_weights_map[i], device=self.device), sync_dist=False)
                # Print the count of each data sample in self.data_samples
                sample_counts = Counter(self.data_samples)
                self.log(f"data_weights/assigned_counts_{i}", torch.tensor(sample_counts.get(i,0), device=self.device), sync_dist=False)

    # def _calculate_gradient_norms(self):
    #     """Calculate gradient norms for different model types without updating weights"""
        
    #     # Create a smaller dataset for gradient norm calculation
    #     grad_calc_dataset = EpochDataset(
    #         batch_size=8,
    #         seq_len=self.train_dataset.seq_len,
    #         steps_per_epoch=25,  # Use fewer steps for efficiency
    #         hyperparameters=self.train_dataset.hyperparameters,
    #         reuse_data_every_n=self.train_dataset.reuse_data_every_n,
    #         max_model_dim=self.max_model_dim,
    #         max_num_cluster=self.max_num_cluster,
    #         get_batch_method=self.train_dataset.get_batch_method,
    #         rank=self.global_rank,  # Use current rank
    #         num_device=1,
    #         training=True,
    #         single_eval_pos_gen=self.train_dataset.single_eval_pos_gen,
    #         data_path=None,  # Generate fresh data
    #         is_source_numpy=False
    #     )
    #     data_weights_map ={'gmm': 0.25, 'corpula': 0.25, 'prob': 0.25, 'contextual': 0.25}
    #     data_samples = [[name] * int(weight * 200) for name, weight in data_weights_map.items()]
    #     data_samples = sum(data_samples, [])
        
    #     # Set epoch and generate data
    #     inliners_list, LA_list, _ , model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
    #             every_n_dim=None,  # generate `steps_per_epoch * batch_size`
    #             save_data=False,
    #             model_weights=data_samples,
    #             total_tasks=200)
    
    #     data_dict = {'in': inliners_list, 'la': LA_list, 'model_names': model_names}
        
    #     grad_calc_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)
        
    #     # Create dataloader - no need for DDP sampler since we're only using rank 0
    #     grad_calc_dl = DataLoader(
    #         grad_calc_dataset, 
    #         batch_size=8, 
    #         shuffle=False,
    #         collate_fn=grad_calc_dataset.prior_batch_collate_fn,
    #         **self.dataloader_para
    #     )
        
    #     # Calculate gradients without updating weights
    #     gradient_norms = {}
    #     params = [p for p in self.model.parameters() if p.requires_grad]
        
    #     # Process each model type separately to avoid gradient accumulation issues
    #     for target_model_type in ['gmm', 'corpula', 'prob', 'contextual']:
    #         model_type_norms = []
            
    #         for batch_idx, batch in enumerate(grad_calc_dl):
    #             self.model.zero_grad()
    #             # Forward pass
    #             loss, losses, single_eval_pos, targets, nan_share, model_names = self.forward(full_data=batch)
    #             if not torch.isnan(loss) and model_names is not None:
    #                 # Find indices for this specific model type
    #                 model_indices = [i for i, name in enumerate(model_names) if name == target_model_type]
    #                 if model_indices:  # Only process if this model type exists in the batch
    #                     # Extract losses for this model type and compute mean
    #                     model_losses = losses[:, model_indices].mean()
    #                     grads = torch.autograd.grad(model_losses,
    #                                                 params,
    #                                                 retain_graph=False,   # reuse graph for other model types
    #                                                 create_graph=False,
    #                                                 allow_unused=True)
    #                     total_sq = 0.0
    #                     for g in grads:
    #                         if g is not None:
    #                             total_sq += g.norm(2).item() ** 2
    #                     model_type_norms.append(total_sq ** 0.5)
                
    #             self.model.zero_grad()
    #             for p in params:
    #                 p.grad = None
            
    #         # Store average gradient norm for this model type
    #         if model_type_norms:
    #             gradient_norms[target_model_type] = sum(model_type_norms) / len(model_type_norms)
    #             print(f"[Rank {self.global_rank}] Average gradient norm for {target_model_type}: {gradient_norms[target_model_type]:.6f}")
    #     # Clean up
    #     del grad_calc_dataset, grad_calc_dl
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
    #     return gradient_norms
    
    def _calculate_gradient_norms(self):
        """
        Lightweight per-model-type gradient norm estimation.
        Does NOT touch real optimizer state or parameter .grad.
        """
        if self.prior_train_data_gen is None:
            return {k: 0.0 for k in self.model_names_list}

        # Small probe dataset
        probe_steps = 12
        probe_total_tasks = 120
        probe_dataset = EpochDataset(
            batch_size=8,
            seq_len=self.train_dataset.seq_len,
            steps_per_epoch=probe_steps,
            hyperparameters=self.train_dataset.hyperparameters,
            reuse_data_every_n=self.train_dataset.reuse_data_every_n,
            max_model_dim=self.max_model_dim,
            max_num_cluster=self.max_num_cluster,
            get_batch_method=self.train_dataset.get_batch_method,
            rank=self.global_rank,
            num_device=1,
            training=True,
            single_eval_pos_gen=self.train_dataset.single_eval_pos_gen,
            data_path=None,
            is_source_numpy=False
        )

        base_mix = {k: 0.25 for k in self.model_names_list}
        sampling_list = sum([[k]*int(v*probe_total_tasks) for k,v in base_mix.items()], [])

        inliners_list, LA_list, _, model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
            every_n_dim=None,
            save_data=False,
            model_weights=sampling_list,
            total_tasks=probe_total_tasks
        )
        data_dict = {'in': inliners_list, 'la': LA_list, 'model_names': model_names}
        probe_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)

        dl = DataLoader(
            probe_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=probe_dataset.prior_batch_collate_fn,
            **self.dataloader_para
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        accum = {k: [] for k in self.model_names_list}


        for batch in dl:
            # Single forward to know which types are present
            _, losses, _, _, _, names = self.forward(full_data=batch)
            if losses is None or names is None:
                continue
            names = list(names)
            type_to_indices = {}
            for idx, nm in enumerate(names):
                type_to_indices.setdefault(nm, []).append(idx)

            # Re-run forward per type (simplifies graph handling)
            for mtype, idxs in type_to_indices.items():
                if mtype not in accum:
                    continue
                # fresh forward
                _, losses_t, _, _, _, _ = self.forward(full_data=batch)
                type_loss = losses_t[:, idxs].mean()
                grads = torch.autograd.grad(type_loss, params, allow_unused=True, create_graph=False)
                sq = 0.0
                for g in grads:
                    if g is not None:
                        sq += g.norm(2).item() ** 2
                accum[mtype].append(sq ** 0.5)

        # Average
        result = {}
        for k, vals in accum.items():
            if vals:
                result[k] = sum(vals)/len(vals)
                print(f"[Rank {self.global_rank}] GradNorm {k}: {result[k]:.6f}")
        # Cleanup
        del dl, probe_dataset
        gc.collect()
        torch.cuda.empty_cache()
        return result
        
     

      
    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        print(f"Current epoch: {current_epoch}")
        lr = self.lr_schedulers().get_last_lr()[0]
 
        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)
        
        self.log(f"data_weights/gmm_step_counts", self.train_recorder.gmm_step_count, sync_dist=True)
        self.log(f"data_weights/corpula_step_counts", self.train_recorder.copula_step_count, sync_dist=True)
        self.log(f"data_weights/prob_step_counts", self.train_recorder.scm_prob_step_count, sync_dist=True)
        self.log(f"data_weights/contextual_step_counts", self.train_recorder.scm_contextual_step_count, sync_dist=True)
    
        # Log main metrics
        self.log('train_loss', train_metric['avg_loss'], sync_dist=True)
        self.log('train_time', train_metric['total_time'], sync_dist=True)
        
        # Log learning rate
        self.log('lr',lr, sync_dist=True)

        # Log additional losses
        self.log('train_gmm_loss', train_metric['avg_gmm_loss'], sync_dist=True)
        self.log('train_copula_loss', train_metric['avg_copula_loss'], sync_dist=True)
        self.log('train_contextual_loss', train_metric['avg_contextual_loss'], sync_dist=True)
        self.log('train_prob_loss', train_metric['avg_prob_loss'], sync_dist=True)
        self.log('train_density_loss', train_metric['avg_density_loss'], sync_dist=True)
        
        # Record average loss
        self.train_losses.append(train_metric['avg_loss'])

        # Reset metrics and clean up
        self.train_recorder.reset()
        gc.collect()
        torch.cuda.empty_cache()            


    def on_validation_epoch_start(self) -> None:
        # Record the start time for the DataLoader to prepare the next batch
        self.val_recorder.epoch_start_time = time.time()

    def validation_step(self, batch, batch_idx):
        step_start = time.time()
        loss, _, single_eval_pos, targets, nan_share, model_names, _,_ = self.val_forward(full_data=batch)
        step_time = time.time() - step_start
        self.val_recorder.update(loss=loss,
                                   losses=None,
                                   single_eval_pos=single_eval_pos, 
                                   targets=targets,
                                   nan_share=nan_share, 
                                   step_time=step_time, 
                                   model_names = model_names)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        step_start = time.time()
        loss, _, single_eval_pos, targets, nan_share, model_names, _,_ = self.val_forward(full_data=batch)
        step_time = time.time() - step_start
        self.test_recorder.update(loss=loss,
                                   losses=None,
                                   single_eval_pos=single_eval_pos, 
                                   targets=targets,
                                   nan_share=nan_share, 
                                   step_time=step_time, 
                                   model_names = model_names)
        return loss
    

    def on_validation_epoch_end(self) -> None:
        val_metric = self.val_recorder.fetch_and_print(epoch=self.current_epoch, lr=None)
        self.log('val_loss', val_metric['avg_loss'], sync_dist=True)
        # Record average loss
        self.val_losses.append(val_metric['avg_loss'])
        # Reset metrics and clean up
        self.val_recorder.reset()
        gc.collect()
        torch.cuda.empty_cache()
        
    def on_test_epoch_end(self) -> None:
        test_metric = self.test_recorder.fetch_and_print(epoch=self.current_epoch, lr=None)
        
        # Print all available test metrics
        print(f"=== Test Results for Epoch {self.current_epoch} ===")
        print(f"Test Loss: {test_metric['avg_loss']}")
        
    
    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint['train_losses'] = self.train_losses
        checkpoint['val_losses'] = self.val_losses

    def on_load_checkpoint(self, checkpoint):
        # Load the lists of train and val losses
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print('-' * 20)
        print(f'getting the train losses of length {len(self.train_losses)} & the val losses of length '
              f'{len(self.val_losses)} from the latest ckpt')
        train_losses_len = len(self.train_losses)
        val_losses_len = len(self.val_losses)
        if train_losses_len > val_losses_len:  # training collapsed after train epoch & before val epoch
            self.train_losses = self.train_losses[:val_losses_len]
        elif val_losses_len > train_losses_len:
            raise Exception  # then sth. is wrong
        print('-' * 20)
