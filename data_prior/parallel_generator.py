import os.path
import random
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data_prior.GMM import make_NdMclusterGMM
import multiprocessing as mp
import pickle
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from data_prior.feature_transform import FeatureTransform
import time

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

from pfns.priors import Batch


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


def call_process_function(args):
    # Unpack the arguments
    process_function, params = args
    return process_function(*params)


class PriorTrainDataGenerator:  # generate synthetic data for training
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.prior_gmm_cfg = cfg.prior.gmm
        self.test_cfg = cfg.test
        # train hyperparameters
        self.seq_len = self.train_cfg.seq_len
        self.hyperparameters = self.train_cfg.hyperparameters
        self.device = self.train_cfg.device
        self.batch_size = self.train_cfg.batch_size
        self.epochs = self.train_cfg.epochs
        self.steps_per_epoch = self.train_cfg.steps_per_epoch
        self.reuse_data_every_n = self.train_cfg.reuse_data_every_n
        self.gen_one_train_one = self.train_cfg.gen_one_train_one
        self.apply_linear_transform = self.train_cfg.apply_linear_transform

        # prior hyperparameters
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim
        self.max_model_dim = self.prior_gmm_cfg.max_model_dim
        self.max_num_cluster = self.prior_gmm_cfg.max_num_cluster
        self.max_mean = self.prior_gmm_cfg.max_mean
        self.max_var = self.prior_gmm_cfg.max_var
        self.inflate_full = self.prior_gmm_cfg.inflate_full
        self.diversity = self.prior_gmm_cfg.diversity
        self.percentile = self.prior_gmm_cfg.percentile
        # data generation:
        self.base_dir = f'./{self.prior_gmm_cfg.data_dir}/num_feat_{self.max_feature_dim}'
        self.gen1tr1_epoch_id = 0
        # feature transform:
        self.FT = FeatureTransform(cfg=cfg)
        self.num_workers = None

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def generate_from_GMM(self, model):  # TODO: currently only in & la are supported
        num_inliners = self.seq_len
        num_LA = self.seq_len  # int(self.seq_len / 2) + 1

        inliners, LA = model.draw_batched_data(num_inliners=num_inliners, num_local_anomalies=num_LA)

        return inliners, LA

    @staticmethod  # we refer to one entry in the batch as one "dataset" (seq-len, #feat)
    def process_one_dataset(epoch_id, step, max_num_cluster, max_model_dim, every_n_dim, diversity, max_mean, max_var,
                            inflate_full, percentile, generate_fn):
        # Use a combination of epoch, step, and process ID to set the seed
        np.random.seed(epoch_id + step + os.getpid())

        if every_n_dim is None:
            # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
            dim = np.random.randint(low=2, high=max_model_dim + 1)  # draw from [2, max_model_dim]
            num_cluster = np.random.randint(low=2, high=max_num_cluster + 1)  # draw from [2, max_num_cluster]
        else:
            dim = (step // max_num_cluster + 1) * every_n_dim
            if dim == 1:  # for the case every_n_dim=1, dim ranges from [1, max_model_dim],
                # we reset this case to dim=max_model_dim
                dim = max_model_dim
            num_cluster = step % max_num_cluster + 1  # takes value in [1, max_num_cluster]

        max_mean = np.random.randint(low=2, high=max_mean + 1)  # draw from [2, max_mean]
        max_var = np.random.randint(low=2, high=max_var + 1)  # draw from [2, max_var]

        model = make_NdMclusterGMM(dim=dim, num_cluster=num_cluster, weights=[1 / num_cluster] * num_cluster,
                                   max_mean=max_mean, max_var=max_var, inflate_full=inflate_full, sub_dims=None,
                                   percentile=percentile, delta=0.05)
        inliners, LA = generate_fn(model=model)
        sub_dims = model.sub_dims
        return inliners, LA, sub_dims

    def generate_batches(self, epoch, process_function=None, every_n_dim=10):
        # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
        if process_function is None:
            process_function = self.process_one_dataset

        num_cores = mp.cpu_count() if self.num_workers is None else self.num_workers
        if every_n_dim is None:
            total_tasks = self.steps_per_epoch * self.batch_size
            print(f'generating {self.steps_per_epoch * self.batch_size} datasets')
        else:
            total_tasks = (self.max_model_dim // every_n_dim) * self.max_num_cluster
            print(f'generating models with dim from 2 to {self.max_model_dim} with an interval of {every_n_dim},'
                  f' each dim has {self.max_num_cluster} model(s) with num-of-clusters '
                  f'from 1 to {self.max_num_cluster}')

        print(f'using {num_cores} of cpus to generate in parallel')

        tasks = [
            (process_function, (
                epoch * total_tasks, step, self.max_num_cluster, self.max_model_dim, every_n_dim, self.diversity,
                self.max_mean, self.max_var, self.inflate_full, self.percentile, self.generate_from_GMM))
            for step in range(total_tasks)
        ]

        with mp.Pool(processes=num_cores) as pool:
            # Use tqdm with imap_unordered for progress tracking
            results = list(tqdm(pool.imap_unordered(call_process_function, tasks), total=total_tasks))

        # Unpack results
        inliners, LA, sub_dims = zip(*results)
        return list(inliners), list(LA), list(sub_dims)

    def generate_one_epoch(self, epoch, every_n_dim, epoch_dir, save_data):
        # Generate all batches for the epoch at once using parallelization
        inliners, LA, sub_dims = self.generate_batches(epoch=epoch,
                                                       process_function=self.process_one_dataset,
                                                       every_n_dim=every_n_dim)
        if save_data:
            with open(os.path.join(epoch_dir, f'in.pickle'), 'wb') as handle:
                pickle.dump(inliners, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(epoch_dir, f'la.pickle'), 'wb') as handle:
                pickle.dump(LA, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(epoch_dir, f'sub_dims.pickle'), 'wb') as handle:
                pickle.dump(sub_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return inliners, LA, sub_dims

    def make_train_data(self, every_n_dim):
        base_dir = f'{self.base_dir}/train'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        print('generating training data...')
        if self.epochs > self.reuse_data_every_n:
            print(
                f'we are generating with #epochs={self.reuse_data_every_n} while the #-of-train-epochs={self.epochs}, '
                f'please make sure this is the desired behavior')

        for epoch in tqdm(range(self.reuse_data_every_n)):
            epoch_dir = os.path.join(base_dir, f'epoch{epoch}_every_n_{every_n_dim}')
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            self.generate_one_epoch(epoch=epoch, every_n_dim=every_n_dim, epoch_dir=epoch_dir, save_data=True)

    def generate_one_epoch_then_train_one(self, every_n_dim, save_data):
        # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
        base_dir = f'{self.base_dir}/train'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        s = time.time()
        epoch = 'gen1tr1'
        epoch_dir = os.path.join(base_dir, f'epoch{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        print('current gen1tr1 epoch_id: {}'.format(self.gen1tr1_epoch_id))
        inliners, LA, sub_dims = self.generate_one_epoch(epoch=self.gen1tr1_epoch_id, every_n_dim=every_n_dim,
                                                         epoch_dir=epoch_dir, save_data=save_data)

        self.gen1tr1_epoch_id += 1
        print('generation time: {} min'.format((time.time() - s) / 60))
        return inliners, LA, sub_dims

    def make_val_data(self, every_n_dim):
        base_dir = f'{self.base_dir}/val'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        print('generating validation data...')
        epoch = 0
        epoch_dir = os.path.join(base_dir, f'epoch{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        # make sure the epoch for val is different from that for train
        self.generate_one_epoch(epoch=epoch + self.epochs + random.randint(a=1, b=100),
                                every_n_dim=every_n_dim, epoch_dir=epoch_dir, save_data=True)

    def get_batch_for_NdMclusterGaussian(self, list_of_data, seq_len=100, hyperparameters=None, **kwargs):
        # this will be part of the collate_fn to help prepare batched data

        xs = []
        ys = []
        is_train = kwargs['training']
        single_eval_pos = kwargs['single_eval_pos'] if is_train else seq_len - 1

        num_inliners = single_eval_pos
        num_test_x = seq_len - single_eval_pos
        ignore_index = hyperparameters['ignore_index']

        def make_x_y_with_stored_data(train_test_in, test_la):
            train_test_in = np.random.permutation(train_test_in)
            test_la = np.random.permutation(test_la)

            inliners = train_test_in[:num_inliners]

            test_inliner = train_test_in[num_inliners:]
            test_la = test_la[:num_test_x]

            test_x = np.concatenate([test_inliner, test_la], axis=0)
            test_y = np.array([0] * num_test_x + [1] * num_test_x)

            sample_indices = np.random.choice(2 * num_test_x, num_test_x, replace=False)

            test_x = test_x[sample_indices]
            test_y = test_y[sample_indices]

            x = np.concatenate([inliners, test_x], axis=0)  # (num_inliners+num_test_x, dim)
            y = np.concatenate([np.array([ignore_index] * num_inliners), test_y], axis=0)

            feature_dim = x.shape[-1]
            if feature_dim < self.max_feature_dim:
                x = self.FT.feature_padding(x=x, num_feature=feature_dim)
            return torch.from_numpy(x), torch.from_numpy(y)

        for data in list_of_data:
            # a list containing 'batch_size' number of {'in':..., 'la':..., 'sub_dims':...}
            inliners = data['in'][:self.seq_len, :]
            la = data['la'][:self.seq_len, :]
            x, y = make_x_y_with_stored_data(train_test_in=inliners, test_la=la)
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs, dim=0).to(torch.float)  # (bs, seq_len, dim)
        ys = torch.stack(ys, dim=0).to(torch.float)  # (bs, seq_len)

        return Batch(x=xs.transpose(0, 1), y=None, target_y=ys.transpose(0, 1), single_eval_pos=single_eval_pos)


@hydra.main(version_base='1.3', config_path='../configuration', config_name='config')
def main(cfg: DictConfig):
    num_workers = 32
    # use maximum number of cpu
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)
    prior_train_data_gen.set_num_workers(num_workers=num_workers)

    # exemplary usage:
    # prior_train_data_gen.generate_one_epoch_then_train_one(every_n_dim=1, save_data=False)
    # prior_train_data_gen.make_train_data(every_n_dim=1)
    # prior_train_data_gen.make_val_data(every_n_dim=1)
    # prior_train_data_gen.make_train_data(every_n_dim=None)
    prior_train_data_gen.make_val_data(every_n_dim=None)


if __name__ == "__main__":
    main()
