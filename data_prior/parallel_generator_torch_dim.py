import os.path
import random
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data_prior.GMM_torch import make_NdMclusterGMM
import multiprocessing as mp
import pickle
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from data_prior.feature_transform import FeatureTransform
import time
from data_prior.SCM_torch_speed import make_contextualSCM, make_probSCM
from data_prior.density_tree_torch_speed2 import make_density
from data_prior.corpula_torch import make_corpula
from copy import deepcopy
import math 

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

from pfns.priors import Batch


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


class PriorTrainDataGenerator:  # generate synthetic data for training
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_cfg = cfg.train
        
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
        
        self.prior_probscm_cfg = cfg.prior.mixture.scm_prob
        self.prior_contextual_cfg = cfg.prior.mixture.scm_contextual
        self.prior_density = cfg.prior.mixture.density
        self.prior_gmm_cfg = cfg.prior.mixture.gmm
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim
        
        self.use_dim = self.train_cfg.use_dim
        self.bin_dim = self.train_cfg.bin_dim
        
        print('use_dim:', self.use_dim)
        print('bin_dim:', self.bin_dim)
        
        # data generation:
        self.base_dir = f'./{self.prior_gmm_cfg.data_dir}/num_feat_{self.max_feature_dim}'
        self.gen1tr1_epoch_id = 0
        # feature transform:
        self.FT = FeatureTransform(cfg=cfg)
        self.num_workers = None
        self.update_model_parameters()

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers


    def generate_from_GMM(self, model):  # TODO: currently only in & la are supported
        num_inliers = self.seq_len
        num_LA = self.seq_len  # int(self.seq_len / 2) + 1 # self.seq_len  # int(self.seq_len / 2) + 1
        #print(num_inliers, num_LA)
        inliners, LA = model.draw_batched_data(num_inliers, num_LA)
        return inliners, LA
    
    
    def generate_from_mixture(self, model): #generate from any distributions
        num_inliers = self.seq_len
        num_LA = self.seq_len 
        #print(num_inliers, num_LA)
        inliners, LA = model.draw_batched_data(num_inliers, num_LA)
        return inliners, LA
    
    
    def update_model_parameters(self):
        model_choices = []
        # corpula model
        corpula_params = dict(generate_fn=self.generate_from_mixture)
        model_choices.append(("corpula", make_corpula, corpula_params))
        # GMM Model
        gmm_params = dict(
                max_num_cluster=self.prior_gmm_cfg.max_num_cluster,
                max_model_dim=self.prior_gmm_cfg.max_model_dim, 
                diversity=self.prior_gmm_cfg.diversity,
                max_mean=self.prior_gmm_cfg.max_mean,
                max_var=self.prior_gmm_cfg.max_var, 
                inflate_full=self.prior_gmm_cfg.inflate_full,
                percentile=self.prior_gmm_cfg.percentile,
                generate_fn=self.generate_from_mixture)
        model_choices.append(("gmm", make_NdMclusterGMM, gmm_params))
        # Contextual SCM Model
        contextual_params = dict(
                max_feature_dim=self.prior_contextual_cfg.max_feature_dim,
                min_num_layer=self.prior_contextual_cfg.min_num_layer,
                max_num_layer=self.prior_contextual_cfg.max_num_layer,
                min_hidden_size=self.prior_contextual_cfg.min_hidden_size,
                max_hidden_size=self.prior_contextual_cfg.max_hidden_size,
                alpha=self.prior_contextual_cfg.alpha,
                beta=self.prior_contextual_cfg.beta,
                generate_fn=self.generate_from_mixture
        )
        model_choices.append(("contextual", make_contextualSCM, contextual_params))
        #Prob SCM Model
        prob_params = dict(
                max_feature_dim=self.prior_probscm_cfg.max_feature_dim,
                min_num_layer=self.prior_probscm_cfg.min_num_layer,
                max_num_layer=self.prior_probscm_cfg.max_num_layer,
                min_hidden_size=self.prior_probscm_cfg.min_hidden_size,
                max_hidden_size=self.prior_probscm_cfg.max_hidden_size,
                alpha=self.prior_probscm_cfg.alpha,
                beta=self.prior_probscm_cfg.beta,
                generate_fn=self.generate_from_mixture
        )
        model_choices.append(("prob", make_probSCM, prob_params))
        # #Density Model
        # density_params = dict(
        #     max_feature_dim=self.prior_density.max_feature_dim,
        #     min_cat=self.prior_density.min_cat,
        #     max_cat=self.prior_density.max_cat,
        #     tree_depth=self.prior_density.tree_depth,
        #     prune_prob=self.prior_density.prune_prob,
        #     generate_fn=self.generate_from_mixture
        # )
        # model_choices.append(("density", make_density, density_params))
        self.model_choices = model_choices

    @staticmethod
    def process_one_dataset(epoch_id,
                            step, 
                            device,
                            every_n_dim,
                            model_choices,
                            origin_epoch_id=0,
                            use_dim = False,
                            bin_dim = False,
                            model_weights = None,
                            total_epoch = 1000,): 
        if model_weights is None:
            model_index, model_entry = random.choice(list(enumerate(model_choices)))
        else:
            model_entry = next((name, constructor, params) for name, constructor, params in model_choices if name == model_weights)
            
        model_name, model_constructor, params = model_entry
        # Set random seed for reproducibility
        seed = epoch_id + step + os.getpid()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        #print('Processing dataset - Epoch ID:', origin_epoch_id, 'Step:', step, 'Process ID:', os.getpid())
        #print(model_name)

        # Deepcopy params to avoid mutation
        params = deepcopy(params)
        params['device'] = device

        # You can have model-specific logic below
        if model_name == "gmm":
            #print('generating with gmm')
            #print('every n dim',every_n_dim)
            max_model_dim = params['max_model_dim']
            max_num_cluster = params['max_num_cluster']
            # if every_n_dim is None:
            #     # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
            #     dim = np.random.randint(low=2, high=max_model_dim + 1)  # draw from [2, max_model_dim]
            #     num_cluster = np.random.randint(low=2, high=max_num_cluster + 1)  # draw from [2, max_num_cluster]
            # else: 
            #     dim = (step // max_num_cluster + 1) * every_n_dim
            #     if dim == 1:  # for the case every_n_dim=1, dim ranges from [1, max_model_dim],
            #         # we reset this case to dim=max_model_dim
            #         dim = max_model_dim
            #     #print(f'generating with dim={dim}')
            
            if use_dim:
                dim = int((origin_epoch_id / total_epoch) * max_model_dim)
                dim = max(dim,2) #minimum 2
                if step == 0:
                    print(f'generating with dim={dim}')
            elif bin_dim:
                bins = [(2,10), (2,20), (2,30), (2,40), (2,50), (2,60), (2,70), (2,80), (2,90), (2,100)]
                progress = origin_epoch_id / total_epoch
                biased_progress = progress ** 0.5 
                bin_index = int(biased_progress * len(bins))
                bin_index = min(bin_index, len(bins) - 1) 
                min_dim, max_dim = bins[bin_index]
                dim = np.random.randint(min_dim, max_dim + 1)
                if step == 0:
                    print(f'generating with bin={bins[bin_index]}, dim={dim}')
            else:
                dim = np.random.randint(low=2, high=max_model_dim + 1)
                
    
            num_cluster = np.random.randint(low=2, high=params["max_num_cluster"] + 1)
            max_mean = np.random.randint(low=2, high=params["max_mean"] + 1)
            max_var = np.random.randint(low=2, high=params["max_var"] + 1)
            model = model_constructor(
                dim=dim,
                num_cluster=num_cluster,
                weights=torch.tensor([1 / num_cluster] * num_cluster, device=device),
                max_mean=max_mean,
                max_var=max_var,
                inflate_full=params["inflate_full"],
                sub_dims=None,
                percentile=params["percentile"],
                delta=0.05,
                device=device
            )
            inliers, LA = params["generate_fn"](model=model)
            sub_dims = model.sub_dims
            del model
            return inliers, LA, sub_dims, model_name

        elif model_name == "contextual" or model_name == 'prob':
            # Sample random contextual model hyperparameters
            #print('generating with scm')
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            params['min_num_layer']=max(int(np.sqrt(feature_dim))-3,2)
            params['min_hidden_size'] = max(int(math.floor(feature_dim / params['min_num_layer'])) + 2 ,2)
            params['max_hidden_size'] = min(params['min_hidden_size']+ 7, params['max_hidden_size'])
            model = model_constructor(feature_dim,
                                      params['min_num_layer'],
                                      params['max_num_layer'],
                                      params['min_hidden_size'],
                                      params['max_hidden_size'],
                                      params['alpha'],
                                      params['beta'], 
                                      device= device)
            inliers, LA = params["generate_fn"](model=model)
            del model
            return inliers, LA, None, model_name

        elif model_name == "density":
            n_numeric = np.random.randint(2,100)
            n_categorical = 0 #np.random.randint(0,1) #10)
            tree_depth = np.random.randint(4,8)
            #n_samples = 5000
            model = model_constructor(n_numeric, n_categorical, tree_depth,device= device)
            inliers, LA = params["generate_fn"](model=model)
            del model
            return inliers, LA, None, model_name
        
        elif model_name == "corpula":
            model = model_constructor(device= device)
            inliers, LA = params["generate_fn"](model=model)
            del model
            return inliers, LA, None, model_name
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
          
    
    def generate_batches(self, 
                         epoch, 
                         process_function=None, 
                         every_n_dim=10,
                         model_weights = None,
                         total_tasks = None,):
        # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
        if process_function is None:
            process_function = self.process_one_dataset  
        if total_tasks is None:
            if every_n_dim is None:
                total_tasks = int(self.steps_per_epoch * (self.batch_size) / self.cfg.train.num_device)
                #print(f'generating {self.steps_per_epoch * self.batch_size} datasets')
            else:
                total_tasks = (self.max_model_dim // every_n_dim) * self.max_num_cluster
                print(self.max_num_cluster)
                print(self.max_model_dim, every_n_dim) 
                print(f'generating models with dim from 2 to {self.max_model_dim} with an interval of {every_n_dim},'
                    f' each dim has {self.max_num_cluster} model(s) with num-of-clusters '
                    f'from 1 to {self.max_num_cluster}')

        print(f'using GPU to generate fast')
        inliners_list, LA_list, sub_dims_list = [], [], []
        model_name_list = []
        for step in tqdm(range(total_tasks)):
            model_name = model_weights[step]
            inliners, LA, sub_dims, model_name = process_function(
                epoch_id=epoch*total_tasks,
                step=step,
                device = self.device,
                every_n_dim =every_n_dim,
                model_choices = self.model_choices,
                origin_epoch_id = epoch,
                model_weights = model_name,
                total_epoch = self.epochs,
                use_dim = self.use_dim,
                bin_dim = self.bin_dim,
                )
            inliners_list.append(inliners)
            LA_list.append(LA)
            sub_dims_list.append(sub_dims)
            model_name_list.append(model_name)

        return inliners_list, LA_list, sub_dims_list,model_name_list 

    def generate_one_epoch(self, 
                           epoch,
                           every_n_dim, 
                           epoch_dir, 
                           save_data, 
                           model_weights=None,
                           total_tasks = None):
        # Generate all batches for the epoch at once using parallelization
        inliners, LA, sub_dims, model_names = self.generate_batches(epoch=epoch,
                                                       process_function=self.process_one_dataset,
                                                       every_n_dim=every_n_dim,
                                                       model_weights = model_weights,
                                                       total_tasks = total_tasks)
        if save_data:
            with open(os.path.join(epoch_dir, f'in.pickle'), 'wb') as handle:
                pickle.dump(inliners, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(epoch_dir, f'la.pickle'), 'wb') as handle:
                pickle.dump(LA, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(epoch_dir, f'sub_dims.pickle'), 'wb') as handle:
                pickle.dump(sub_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return inliners, LA, sub_dims,model_names

    def make_train_data(self, every_n_dim,model_weights=None):
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

            self.generate_one_epoch(epoch=epoch, every_n_dim=every_n_dim, epoch_dir=epoch_dir, save_data=True,model_weights=model_weights)

    def generate_one_epoch_then_train_one(self,
                                          every_n_dim, 
                                          save_data,
                                          model_weights = None,
                                          total_tasks = None,):
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
        inliners, LA, sub_dims, model_names = self.generate_one_epoch(epoch=self.gen1tr1_epoch_id, 
                                                                      every_n_dim=every_n_dim,
                                                         epoch_dir=epoch_dir, 
                                                         save_data=save_data,
                                                         model_weights=model_weights,
                                                         total_tasks=total_tasks)

        self.gen1tr1_epoch_id += 1
        print('generation time: {} min'.format((time.time() - s) / 60))
        return inliners, LA, sub_dims, model_names

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
                                every_n_dim=every_n_dim, epoch_dir=epoch_dir, save_data=True,model_weights=None)

    def get_batch_all_models(self, list_of_data, seq_len=100, hyperparameters=None, **kwargs):
        #print(len(list_of_data))
        # this will be part of the collate_fn to help prepare batched data
        xs = []
        ys = []
        model_names = []
        is_train = kwargs['training']
        single_eval_pos = kwargs['single_eval_pos'] if is_train else seq_len - 1
        # print('seq_len:', seq_len)
        # print('single_eval_pos:', single_eval_pos)
        num_inliners = single_eval_pos
        num_test_x = seq_len - single_eval_pos
        # print(num_inliners, num_test_x)
        ignore_index = hyperparameters['ignore_index']
        # print('ignore_index:', ignore_index)

        def make_x_y_with_stored_data(train_test_in, test_la):
            train_test_in = train_test_in[torch.randperm(train_test_in.shape[0])]
            test_la = test_la[torch.randperm(test_la.shape[0])]

            inliners = train_test_in[:num_inliners]

            test_inliner = train_test_in[num_inliners:]
            test_la = test_la[:num_test_x]

            test_x = torch.cat([test_inliner, test_la], dim=0)
            test_y = torch.tensor([0] * num_test_x + [1] * num_test_x)

            sample_indices = torch.randperm(2 * num_test_x)[:num_test_x]

            test_x = test_x[sample_indices]
            test_y = test_y[sample_indices]

            x = torch.cat([inliners, test_x], dim=0)  # (num_inliners+num_test_x, dim)
            y = torch.cat([torch.tensor([ignore_index] * num_inliners), test_y], dim=0)

            feature_dim = x.shape[-1]
            if feature_dim < self.max_feature_dim:
                x = self.FT.feature_padding_torch(x=x, num_feature=feature_dim)
            return x, y

        for data in list_of_data:
            # print(data['in'].shape,'ninliers')
            # print(data['la'].shape,'noutliers')
            #print(data['model_name'])
            # a list containing 'batch_size' number of {'in':..., 'la':..., 'sub_dims':...}
            inliners = data['in'][:self.seq_len, :]
            la = data['la'][:self.seq_len, :]
            model_name = data['model_name']
            x, y = make_x_y_with_stored_data(train_test_in=inliners, test_la=la)
            xs.append(x)
            ys.append(y)
            model_names.append(model_name)

        xs = torch.stack(xs, dim=0)  #.to(torch.float)  # (bs, seq_len, dim)
        ys = torch.stack(ys, dim=0)  #.to(torch.float)  # (bs, seq_len)
        # print(model_names)
        # print(seq_len, single_eval_pos, xs.shape, ys.shape)
        # exit(0)
        return Batch(x=xs.transpose(0, 1), y=None, target_y=ys.transpose(0, 1),model_names = model_names, single_eval_pos=single_eval_pos)


@hydra.main(version_base='1.3', config_path='../configuration', config_name='config')
def main(cfg: DictConfig):
    num_workers = 32
    # use maximum number of cpu
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)
    prior_train_data_gen.set_num_workers(num_workers=num_workers)

    # exemplary usage:
    prior_train_data_gen.generate_one_epoch_then_train_one(every_n_dim=1, save_data=False)
    prior_train_data_gen.generate_one_epoch_then_train_one(every_n_dim=None, save_data=False)


if __name__ == "__main__":
    main()
