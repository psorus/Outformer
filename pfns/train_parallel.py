from __future__ import annotations

import time
import torch
from torch import nn
import random
import pickle
from . import utils
from .priors import prior
#from .transformer import TransformerModel
from .fomo_transformer import TransformerModel
from .bar_distribution import BarDistribution
from .utils import get_cosine_schedule_with_warmup, get_openai_lr
from . import positional_encodings
import pytorch_lightning as pl
from .parallel_dataset import EpochDataset
from torch.utils.data import DataLoader
from data_prior.GMM import make_NdMclusterGMM, generate_linear_transform, transform_samples


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


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

    def reset(self):
        self.total_loss = 0.0
        self.total_positional_losses = torch.zeros(self.seq_len)
        self.total_positional_losses_recorded = torch.zeros(self.seq_len)
        self.nan_steps = 0.0
        self.ignore_steps = 0.0
        self.epoch_start_time = 0.0
        self.total_step_time = 0.0

    def update(self, loss, losses, single_eval_pos, targets, nan_share, step_time):
        if not torch.isnan(loss):
            self.total_loss += loss.cpu().detach().item()
            self.total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                nn.functional.one_hot(torch.tensor(single_eval_pos), self.seq_len) * \
                utils.torch_nanmean(losses[:self.seq_len - single_eval_pos].mean(0)).cpu().detach()

            self.total_positional_losses_recorded += torch.ones(self.seq_len) if single_eval_pos is None else \
                nn.functional.one_hot(torch.tensor(single_eval_pos), self.seq_len)

            self.nan_steps += nan_share.cpu().item()
            self.ignore_steps += (targets == -100).float().mean().cpu().item()

        self.total_step_time += step_time

    def fetch_and_print(self, epoch=None, lr=None):
        avg_loss = self.total_loss / self.steps_per_epoch
        total_positional_losses = (
                self.total_positional_losses / self.total_positional_losses_recorded).tolist()
        nan_share = self.nan_steps / self.steps_per_epoch
        ignore_share = self.ignore_steps / self.steps_per_epoch
        total_time = time.time() - self.epoch_start_time

        if self.verbose:  # only used in training
            print('-' * 89)
            print(
                # f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, "
                f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                f'| end of epoch {epoch:3d} | time: {total_time :5.2f}s | (approx) step time: {self.total_step_time :5.2f}s | '
                f'(approx) data time: {total_time - self.total_step_time :5.2f}s | mean loss {avg_loss:5.2f} | lr {lr}'
            )
            print('-' * 89)
        return {'avr_loss': avg_loss, 'pos_loss': total_positional_losses, 'nan_share': nan_share,
                'ignore_share': ignore_share, 'total_time': total_time}


class ZeroShotOD(pl.LightningModule):
    def __init__(self, cfg, priordataloader_class_or_get_batch: prior.PriorDataLoader | callable, criterion,
                 encoder_generator, dropout=0.0, weight_decay=0.0,
                 input_normalization=False,
                 y_encoder_generator=None, pos_encoder_generator=None, decoder_dict={}, extra_prior_kwargs_dict={},
                 train_extra_dict=None, resume_from_ckpt=False,  # added here
                 scheduler=get_cosine_schedule_with_warmup,
                 load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None,
                 gpu_device='cuda:0',
                 aggregate_k_gradients=1, verbose=False, style_encoder_generator=None, epoch_callback=None,
                 step_callback=None,
                 continue_model=None,
                 initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True,
                 border_decoder=None
                 , num_global_att_tokens=0, progress_bar=False, **model_extra_args):
        super(ZeroShotOD, self).__init__()

        train_cfg = cfg.train
        prior_gmm_cfg = cfg.prior.gmm
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
        lr = train_cfg.lr / num_device
        print(f'original lr={train_cfg.lr}, dividing it by num_device={num_device}, so we get the new lr={lr}')
        # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
        # the lr interacts heavily with `batch_size` (smaller `batch_size` -> smaller best `lr`)

        # prior hyperparameters
        self.max_feature_dim = prior_gmm_cfg.max_feature_dim
        self.max_model_dim = prior_gmm_cfg.max_model_dim
        self.max_num_cluster = prior_gmm_cfg.max_num_cluster

        # specifics for generate-one-train-one
        self.gen_one_train_one = False if train_extra_dict is None else True
        self.prior_train_data_gen = None if train_extra_dict is None else train_extra_dict['prior_train_data_gen']

        if self.prior_train_data_gen is not None:
            num_workers = self.prior_train_data_gen.num_workers
            import multiprocessing as mp
            num_cpus = mp.cpu_count() if num_workers is None else num_workers
            print(f'using {num_cpus} cpus to generate data')

        assert self.batch_size % num_device == 0
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

        self.train_dataset = EpochDataset(batch_size=self.batch_size, seq_len=seq_len,
                                          steps_per_epoch=self.steps_per_epoch,
                                          hyperparameters=extra_prior_kwargs_dict['hyperparameters'],
                                          reuse_data_every_n=self.reuse_data_every_n, max_model_dim=self.max_model_dim,
                                          max_num_cluster=self.max_num_cluster,
                                          get_batch_method=priordataloader_class_or_get_batch,
                                          rank=0, num_device=num_device,  # rank is not yet set in __init__
                                          training=True, single_eval_pos_gen=single_eval_pos_gen,
                                          data_path=train_data_path,
                                          )
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                   collate_fn=self.train_dataset.prior_batch_collate_fn,
                                   **self.dataloader_para)
        if train_cfg.use_validation:
            print('validation data loader')
            self.val_dataset = EpochDataset(batch_size=self.batch_size, seq_len=seq_len,
                                            steps_per_epoch=self.steps_per_epoch,
                                            hyperparameters=extra_prior_kwargs_dict['hyperparameters'],
                                            reuse_data_every_n=self.reuse_data_every_n, max_model_dim=self.max_model_dim,
                                            max_num_cluster=self.max_num_cluster,
                                            get_batch_method=priordataloader_class_or_get_batch,
                                            rank=0, num_device=num_device,  # rank is not yet set in __init__
                                            training=False, single_eval_pos_gen=single_eval_pos_gen,
                                            data_path=f'{self.base_data_path}/val',
                                            )
            self.val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size * 4, shuffle=False,
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
        self.warmup_epochs = epochs // 4
        self.weight_decay = weight_decay
        self.epochs = epochs

        # check that everything uses up-to-date APIs
        utils.check_compatibility(self.train_dl)
        utils.check_compatibility(self.val_dl)

        # training & validation dynamics
        self.train_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=verbose)
        self.val_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=False)
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        # learning rate
        if self.lr is None:
            self.lr = get_openai_lr(self.model)
            print(f"Using OpenAI max lr of {self.lr}.")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.scheduler_fn(optimizer, self.warmup_epochs,
                                      self.epochs if self.epochs is not None else 100)
        # scheduler func returns LambdaLR(optimizer, lr_lambda, last_epoch)

        # Return both optimizer and scheduler
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
        # generate data on the fly (if with "generate-one-train-one" paradigm)
        if self.gen_one_train_one:
            self.prior_train_data_gen.gen1tr1_epoch_id = self.current_epoch
            # set the generation epoch to avoid repetitive generation
            if self.apply_linear_transform:
                # generate less on the fly, and use LT to fill up to `steps_per_epoch * batch_size`
                inliners_list, LA_list, sub_dims_list = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                    every_n_dim=1,  # generate `max_feature_dim * max_num_cluster`
                    save_data=False)
                inliners_list, LA_list = self.increase_datasets_via_LT(inliners_list=inliners_list, LA_list=LA_list,
                                                                       sub_dims_list=sub_dims_list)

            else:  # generate `steps_per_epoch * batch_size` all together on the fly
                inliners_list, LA_list, sub_dims_list = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                    every_n_dim=None,
                    # generate `steps_per_epoch * batch_size`
                    save_data=False)

            assert len(LA_list) >= self.steps_per_epoch * self.batch_size, \
                print(
                    f'number of training instances={len(LA_list)} should be >= {self.steps_per_epoch * self.batch_size}')

            data_dict = {'in': inliners_list, 'la': LA_list}
        else:  # just loading, which will be handled internally by the train_dataset
            print('using pre-generated data')
            if self.apply_linear_transform:  # some datasets are pre-generated, apply LT to them dynamically
                print(f'using pre-generated epochs and then apply LT for epochs >={self.reuse_data_every_n}')
                pre_gen_epoch = self.current_epoch % self.reuse_data_every_n
                print(f'loading from pre-generated data for epoch={pre_gen_epoch}')
                inliners_list = load_pickle(
                    file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/in.pickle')
                LA_list = load_pickle(
                    file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/la.pickle')

                if self.current_epoch >= self.reuse_data_every_n:
                    # apply LT only for epochs >= reuse_data_every_n epochs
                    sub_dims_list = load_pickle(
                        file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/sub_dims.pickle')
                    inliners_list, LA_list = self.increase_datasets_via_LT(inliners_list=inliners_list, LA_list=LA_list,
                                                                           sub_dims_list=sub_dims_list)
                data_dict = {'in': inliners_list, 'la': LA_list}
            else:
                if self.current_epoch % self.reuse_data_every_n != 0:  # reusing more than 1 epoch
                    pre_gen_epoch = self.current_epoch % self.reuse_data_every_n
                    print(f'loading from pre-generated data for epoch={pre_gen_epoch}')
                    inliners_list = load_pickle(
                        file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/in.pickle')
                    LA_list = load_pickle(
                        file_path=f'{self.base_data_path}/train/epoch{pre_gen_epoch}/la.pickle')
                    data_dict = {'in': inliners_list, 'la': LA_list}
                else:
                    data_dict = None  # reuse_every_n=1, then no need to always load the data
        self.train_dataset.set_epoch_and_data(epoch=self.current_epoch,
                                              data_dict=data_dict)
        # set the epoch such that it will load different data

        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                   collate_fn=self.train_dataset.prior_batch_collate_fn,
                                   **self.dataloader_para)
        return self.train_dl

    def val_dataloader(self):
        # DataLoader for validation
        return self.val_dl

    def forward(self, full_data):
        data = (full_data.style.to(self.device) if full_data.style is not None else None, full_data.x.to(self.device),
                full_data.y.to(self.device) if full_data.y is not None else None)
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
            # that is because bar dist appends the mean
            loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

        except Exception as e:
            print("Invalid step encountered, skipping...")
            print(e)
            raise e

        return loss, losses, single_eval_pos, targets, nan_share

    def increase_datasets_via_LT(self, inliners_list, LA_list, sub_dims_list):
        LT_in_list, LT_LA_list = [], []
        # needs to be at least (`steps_per_epoch * batch_size` - len(inliners_list))
        data_size = len(inliners_list)
        indices = list(range(data_size))  # Create a list of numbers from 0 to n-1
        random.shuffle(indices)  # Shuffle the list to get a random order
        transform_times = (self.steps_per_epoch * self.batch_size + data_size) // data_size - 1

        for i in indices:
            for _ in range(transform_times):
                inliners, LA, sub_dims = inliners_list[i], LA_list[i], sub_dims_list[i]
                A, b = generate_linear_transform(dim=len(sub_dims), A_scale=1, b_scale=1)
                inliners = transform_samples(samples=inliners, sub_dims=sub_dims, A=A, b=b)
                LA = transform_samples(samples=LA, sub_dims=sub_dims, A=A, b=b)

                LT_in_list.append(inliners)
                LT_LA_list.append(LA)

        inliners_list = inliners_list + LT_in_list
        LA_list = LA_list + LT_LA_list
        return inliners_list, LA_list

    def training_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, single_eval_pos, targets, nan_share = self.forward(full_data=batch)
        step_time = time.time() - step_start
        self.train_recorder.update(loss=loss, losses=losses, single_eval_pos=single_eval_pos, targets=targets,
                                   nan_share=nan_share, step_time=step_time)
        return loss

    def on_train_epoch_start(self) -> None:
        # Record the start time for the DataLoader to prepare the next batch
        self.train_recorder.epoch_start_time = time.time()

        for i, layer in enumerate(self.model.transformer_encoder.layers):
            print(f'==============layer{i}==================')
            print('self-att')
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                print(attn.out_proj.weight[0][:10])
                print(attn.out_proj.bias[:10])

            print('att-compressor')
            attns = layer.router_att.att_compressor if isinstance(layer.router_att.att_compressor, nn.ModuleList) else [
                layer.router_att.att_compressor]
            for attn in attns:
                print(attn.out_proj.weight[0][:10])
                print(attn.out_proj.bias[:10])

            print('att-recover')
            attns = layer.router_att.att_recover if isinstance(layer.router_att.att_recover, nn.ModuleList) else [
                layer.router_att.att_recover]
            for attn in attns:
                print(attn.out_proj.weight[0][:10])
                print(attn.out_proj.bias[:10])

    def on_train_epoch_end(self) -> None:
        lr = self.lr_schedulers().get_last_lr()[0]
        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)
        self.log('train_loss', train_metric['avr_loss'], sync_dist=True)
        self.log('train_time', train_metric['total_time'], sync_dist=True)
        self.train_losses.append(train_metric['avr_loss'])
        self.train_recorder.reset()  # reset the recorded dynamics for the next epoch

    def on_validation_epoch_start(self) -> None:
        # Record the start time for the DataLoader to prepare the next batch
        self.val_recorder.epoch_start_time = time.time()

    def validation_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, single_eval_pos, targets, nan_share = self.forward(full_data=batch)
        step_time = time.time() - step_start
        self.val_recorder.update(loss=loss, losses=losses, single_eval_pos=single_eval_pos, targets=targets,
                                 nan_share=nan_share, step_time=step_time)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_metric = self.val_recorder.fetch_and_print(epoch=None, lr=None)
        if not self.trainer.sanity_checking:
            self.log('val_loss', val_metric['avr_loss'], sync_dist=True)
            self.log('val_time', val_metric['total_time'], sync_dist=True)
            self.val_losses.append(val_metric['avr_loss'])
        self.val_recorder.reset()  # reset the recorded dynamics for the next epoch

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