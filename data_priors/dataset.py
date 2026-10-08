import hashlib
import os
import gc
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


class EpochDataset(Dataset):
    def __init__(self, batch_size, seq_len, steps_per_epoch, hyperparameters, reuse_data_every_n, max_model_dim,
                 max_num_cluster, get_batch_method, rank, num_device, training, single_eval_pos_gen, data_path,
                 is_source_numpy):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.hyperparameters = hyperparameters
        self.reuse_data_every_n = reuse_data_every_n
        self.max_model_dim = max_model_dim
        self.max_num_cluster = max_num_cluster
        self.current_epoch = 0

        self.get_batch_method = get_batch_method
        self.rank = rank
        self.num_device = num_device
        self.training = training
        self.data_path = data_path
        self.single_eval_pos_gen = single_eval_pos_gen
        self.is_source_numpy = is_source_numpy

        self.in_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/in.pickle')
        self.la_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/la.pickle')
        self.model_name = None
        self.cached_single_eval_pos = None
        self.cached_single_eval_pos_epoch = None

    def set_epoch_and_data(self, epoch, data_dict=None):
        """
        Update the dataset to use data from the specified epoch.
        """
        self.current_epoch = epoch
        print('setting current epoch to:', epoch)

        if data_dict is not None:  # reuse saved data
            print('new data loaded...')
            self.in_data = data_dict['in']
            print('in data shape:', len(self.in_data))
            self.la_data = data_dict['la']
            print('la data shape:', len(self.la_data))
            self.model_names = data_dict['model_names']
            print('model names length:', len(self.model_names))

    def free_data(self):
        self.in_data = None
        self.la_data = None
        self.model_names = None
        gc.collect()
        torch.cuda.empty_cache()

    def set_training_mode(self, training):
        self.training = training

    def set_rank(self, rank):
        self.rank = rank
        print(f'rank is successfully set to {rank} out of {self.num_device} devices')

    def __len__(self):
        return int(self.steps_per_epoch * self.batch_size / self.num_device)

    def __getitem__(self, idx):
        if self.is_source_numpy:  # train/validation (stored in numpy)
            return {'in': torch.from_numpy(self.in_data[idx]).to(torch.float),
                    'la': torch.from_numpy(self.la_data[idx]).to(torch.float),
                    'model_name': None}
        else:  # train (generate on the fly)
            return {'in': self.in_data[idx], 'la': self.la_data[idx], 'model_name': self.model_names[idx], 'idx':idx}

    def prior_batch_collate_fn(self, batch_list):
        random_seed = triple_seed(base_seed=42,
                                  epoch=self.current_epoch,
                                  idx=batch_list[0]['idx'], 
                                  rank=self.rank)
        single_eval_pos = self.single_eval_pos_gen.generate(random_seed) 
        batch = self.get_batch_method(list_of_data=batch_list, seq_len=self.seq_len,
                                      hyperparameters=self.hyperparameters,
                                      training=self.training,
                                      single_eval_pos=single_eval_pos, )
        return batch



def load_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def triple_seed(base_seed: int, epoch: int, idx: int, rank: int, bits: int = 64) -> int:
    """
    Deterministically map (base_seed, epoch, idx, rank) -> integer seed.
    Uses SHA-256 so it's stable across processes and Python versions.
    
    Args:
        base_seed: Base seed value
        epoch: Training epoch
        idx: Data sample index
        rank: Process rank in distributed training
        bits: Number of bits to extract from hash (default: 64)
    
    Returns:
        Integer seed derived from the hash
    """
    msg = f"{base_seed}:{epoch}:{idx}:{rank}".encode()
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:bits // 8], "big")
