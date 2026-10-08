import os
import sys
import pickle
import math
import time
from copy import deepcopy

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from data_priors.feature_transform import FeatureTransform
from data_priors.scm import make_structureSCM, make_measureSCM
from data_priors.copula import make_dependence_copula, make_probablistic_copula
from data_priors.gmm import make_NdMclusterGMM
from data_priors.batch import Batch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_pickle(file_path):
    """Load and return a pickle file."""
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


class PriorTrainDataGenerator:
    """Generate synthetic data for training using various prior distributions."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.seq_len = self.train_cfg.seq_len
        self.hyperparameters = self.train_cfg.hyperparameters
        self.device = self.train_cfg.device
        self.batch_size = self.train_cfg.batch_size
        self.steps_per_epoch = self.train_cfg.steps_per_epoch
        self.apply_linear_transform = self.train_cfg.apply_linear_transform
        
        # Prior configuration
        self.prior_gmm_cfg = cfg.prior.mixture.gmm
        self.prior_probscm_cfg = cfg.prior.mixture.scm_prob
        self.prior_contextual_cfg = cfg.prior.mixture.scm_contextual
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim
        
        # Data paths and initialization
        self.gen1tr1_epoch_id = 0
        self.FT = FeatureTransform(cfg=cfg)
        self.update_model_parameters()

    def set_num_workers(self, num_workers):
        """Set number of workers for data loading."""
        self.num_workers = num_workers

    def generate_from_mixture(self, model):
        """Generate inliers and anomalies from a model."""
        num_inliers = self.seq_len
        num_anomalies = self.seq_len
        inliers, anomalies = model.draw_batched_data(num_inliers, num_anomalies)
        return inliers, anomalies
    
    
    def update_model_parameters(self):
        """Initialize model choices with their configurations."""
        model_choices = []
        
        # Copula models
        copula_params = dict(generate_fn=self.generate_from_mixture)
        model_choices.append(("dependence_copula", make_dependence_copula, copula_params))
        model_choices.append(("probablistic_copula", make_probablistic_copula, copula_params))
        
        # GMM model
        gmm_params = dict(
            max_num_cluster=self.prior_gmm_cfg.max_num_cluster,
            max_model_dim=self.prior_gmm_cfg.max_model_dim, 
            diversity=self.prior_gmm_cfg.diversity,
            max_mean=self.prior_gmm_cfg.max_mean,
            max_var=self.prior_gmm_cfg.max_var, 
            inflate_full=self.prior_gmm_cfg.inflate_full,
            percentile=self.prior_gmm_cfg.percentile,
            generate_fn=self.generate_from_mixture
        )
        model_choices.append(("gmm", make_NdMclusterGMM, gmm_params))
        
        # Structure SCM model
        structure_params = dict(
            max_feature_dim=self.prior_contextual_cfg.max_feature_dim,
            min_num_layer=self.prior_contextual_cfg.min_num_layer,
            max_num_layer=self.prior_contextual_cfg.max_num_layer,
            min_hidden_size=self.prior_contextual_cfg.min_hidden_size,
            max_hidden_size=self.prior_contextual_cfg.max_hidden_size,
            alpha=self.prior_contextual_cfg.alpha,
            beta=self.prior_contextual_cfg.beta,
            generate_fn=self.generate_from_mixture
        )
        model_choices.append(("structure_scm", make_structureSCM, structure_params))
        
        # Measure SCM model
        measure_params = dict(
            max_feature_dim=self.prior_probscm_cfg.max_feature_dim,
            min_num_layer=self.prior_probscm_cfg.min_num_layer,
            max_num_layer=self.prior_probscm_cfg.max_num_layer,
            min_hidden_size=self.prior_probscm_cfg.min_hidden_size,
            max_hidden_size=self.prior_probscm_cfg.max_hidden_size,
            alpha=self.prior_probscm_cfg.alpha,
            beta=self.prior_probscm_cfg.beta,
            generate_fn=self.generate_from_mixture
        )
        model_choices.append(("measure_scm", make_measureSCM, measure_params))
        
        self.model_choices = model_choices

    @staticmethod
    def process_one_dataset(epoch_id, step, device, every_n_dim, model_choices, category, 
                            origin_epoch_id=0, total_epoch=1000): 
        """Process one dataset sample by instantiating and generating data from a model.
        
        Args:
            epoch_id: Epoch identifier for seeding
            step: Step within epoch
            device: GPU device to use
            every_n_dim: Dimension subsampling parameter (unused in current path)
            model_choices: List of (name, constructor, params) tuples
            category: (min_dim, max_dim), model_name tuple defining data category
            origin_epoch_id: Original epoch ID (unused)
            total_epoch: Total epochs (unused)
        
        Returns:
            Tuple of (inliers, anomalies, sub_dims, category)
        """
        (min_dim, max_dim), prior_name = category
        model_entry = next((name, constructor, params) for name, constructor, params in model_choices 
                          if name == prior_name)
        model_name, model_constructor, params = model_entry
        dim = np.random.randint(min_dim, max_dim)
        
        # Set random seed for reproducibility across processes
        seed = epoch_id + step + os.getpid()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Deepcopy params to avoid mutation
        params = deepcopy(params)
        params['device'] = device

        if model_name == "gmm":
            # GMM model generation with random hyperparameters
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
            inliers, anomalies = params["generate_fn"](model=model)
            sub_dims = model.sub_dims
            del model
            return inliers, anomalies, sub_dims, category

        elif model_name == "structure_scm" or model_name == 'measure_scm':
            # SCM models: adjust layer sizes based on feature dimension
            feature_dim = dim
            params['min_num_layer'] = max(int(np.sqrt(feature_dim)) - 3, 2)
            params['min_hidden_size'] = max(int(math.floor(feature_dim / params['min_num_layer'])) + 2, 2)
            params['max_hidden_size'] = min(params['min_hidden_size'] + 7, params['max_hidden_size'])
            
            model = model_constructor(
                feature_dim,
                params['min_num_layer'],
                params['max_num_layer'],
                params['min_hidden_size'],
                params['max_hidden_size'],
                params['alpha'],
                params['beta'], 
                device=device
            )
            inliers, anomalies = params["generate_fn"](model=model)
            del model
            return inliers, anomalies, None, category
        
        elif model_name == "dependence_copula" or model_name == "probablistic_copula":
            # Copula models
            model = model_constructor(device=device, dim=dim)
            inliers, anomalies = params["generate_fn"](model=model)
            del model
            return inliers, anomalies, None, category
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
          
    
    def generate_batches(self, epoch, categories, process_function=None, every_n_dim=10, total_tasks=None):
        """Generate all batches for an epoch.
        
        Args:
            epoch: Epoch number
            categories: List of (min_dim, max_dim), model_name categories
            process_function: Function to process each dataset
            every_n_dim: Dimension sampling interval (for validation)
            total_tasks: Total number of tasks to generate
        
        Returns:
            Tuple of (inliners_list, anomalies_list, sub_dims_list, model_name_list)
        """
        if process_function is None:
            process_function = self.process_one_dataset  
        
        if total_tasks is None:
            if every_n_dim is None:
                total_tasks = int(self.steps_per_epoch * self.batch_size / self.cfg.train.num_device)
            else:
                total_tasks = (self.max_feature_dim // every_n_dim) * self.prior_gmm_cfg.max_num_cluster

        print(f'Generating {total_tasks} batches using GPU')
        
        inliners_list, anomalies_list, sub_dims_list = [], [], []
        model_name_list = []
        
        for step in tqdm(range(total_tasks)):
            category = categories[step]
            inliners, anomalies, sub_dims, model_name = process_function(
                epoch_id=epoch * total_tasks,
                step=step,
                device=self.device,
                every_n_dim=every_n_dim,
                model_choices=self.model_choices,
                origin_epoch_id=epoch,
                category=category,
                total_epoch=self.cfg.train.epochs
            )
            
            inliners_list.append(inliners)
            anomalies_list.append(anomalies)
            sub_dims_list.append(sub_dims)
            model_name_list.append(model_name)

        return inliners_list, anomalies_list, sub_dims_list, model_name_list 

    def generate_one_epoch(self, epoch, every_n_dim, save_data, categories, total_tasks=None):
        """Generate all data for one epoch.
        
        Args:
            epoch: Epoch number
            every_n_dim: Dimension sampling interval
            save_data: Whether to save data to disk
            categories: List of data categories
            total_tasks: Total number of tasks
        
        Returns:
            Tuple of (inliners_list, anomalies_list, sub_dims_list, model_names)
        """
        inliners, anomalies, sub_dims, model_names = self.generate_batches(
            epoch=epoch,
            process_function=self.process_one_dataset,
            every_n_dim=every_n_dim,
            categories=categories,
            total_tasks=total_tasks
        )
        return inliners, anomalies, sub_dims, model_names

    def generate_one_epoch_then_train_one(self, every_n_dim, save_data, categories, total_tasks=None):
        """Generate one epoch of data using the generate-one-train-one paradigm.
        
        Args:
            every_n_dim: Dimension sampling interval
            save_data: Whether to save data to disk
            categories: List of data categories
            total_tasks: Total number of tasks
        
        Returns:
            Tuple of (inliners_list, anomalies_list, sub_dims_list, model_names)
        """
        
        print(f'Current gen1tr1 epoch_id: {self.gen1tr1_epoch_id}')
        start_time = time.time()
        inliners, anomalies, sub_dims, model_names = self.generate_one_epoch(
            epoch=self.gen1tr1_epoch_id, 
            every_n_dim=every_n_dim,
            save_data=save_data,
            categories=categories,
            total_tasks=total_tasks
        )

        self.gen1tr1_epoch_id += 1
        print(f'Generation time: {(time.time() - start_time) / 60:.2f} min')
        return inliners, anomalies, sub_dims, model_names


    def get_batch_all_models(self, list_of_data, seq_len=100, hyperparameters=None, **kwargs):
        """Prepare batched data for training by mixing inliers and anomalies.
        
        Args:
            list_of_data: List of data dictionaries with 'in', 'la', 'model_name'
            seq_len: Sequence length
            hyperparameters: Dictionary with training hyperparameters (e.g., ignore_index)
            **kwargs: Additional arguments including 'training' and 'single_eval_pos'
        
        Returns:
            Batch object with stacked tensors and model names
        """
        xs = []
        ys = []
        model_names = []
        
        is_train = kwargs['training']
        single_eval_pos = kwargs['single_eval_pos'] if is_train else seq_len - 1
        num_inliners = single_eval_pos
        num_test = seq_len - single_eval_pos
        ignore_index = hyperparameters['ignore_index']

        def prepare_sample(train_test_in, test_anomaly):
            """Prepare a single training sample by mixing train inliers with test anomalies."""
            train_test_in = train_test_in[torch.randperm(train_test_in.shape[0])]
            test_anomaly = test_anomaly[torch.randperm(test_anomaly.shape[0])]

            inliners = train_test_in[:num_inliners]
            test_inlier = train_test_in[num_inliners:]
            test_anomaly = test_anomaly[:num_test]

            test_x = torch.cat([test_inlier, test_anomaly], dim=0)
            test_y = torch.tensor([0] * num_test + [1] * num_test)

            # Randomly shuffle and select test samples
            sample_indices = torch.randperm(2 * num_test)[:num_test]
            test_x = test_x[sample_indices]
            test_y = test_y[sample_indices]

            x = torch.cat([inliners, test_x], dim=0)
            y = torch.cat([torch.tensor([ignore_index] * num_inliners), test_y], dim=0)

            # Pad features to max dimension if needed
            feature_dim = x.shape[-1]
            if feature_dim < self.max_feature_dim:
                x = self.FT.feature_padding_torch(x=x, num_feature=feature_dim)
            
            return x, y

        for data in list_of_data:
            inliners = data['in'][:self.seq_len, :]
            anomalies = data['la'][:self.seq_len, :]
            model_name = data['model_name']
            
            x, y = prepare_sample(train_test_in=inliners, test_anomaly=anomalies)
            xs.append(x)
            ys.append(y)
            model_names.append(model_name)

        xs = torch.stack(xs, dim=0)  # (batch_size, seq_len, dim)
        ys = torch.stack(ys, dim=0)  # (batch_size, seq_len)
        
        return Batch(x=xs.transpose(0, 1), y=None, target_y=ys.transpose(0, 1),
                     model_names=model_names, single_eval_pos=single_eval_pos)


@hydra.main(version_base='1.3', config_path='../configuration', config_name='config')
def main(cfg: DictConfig):
    """Main entry point for data generation.
    
    Args:
        cfg: Hydra configuration dictionary
    """
    num_workers = 32
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)
    prior_train_data_gen.set_num_workers(num_workers=num_workers)

    # Example usage: generate one epoch using generate-one-train-one paradigm
    prior_train_data_gen.generate_one_epoch_then_train_one(every_n_dim=1, save_data=False)
    prior_train_data_gen.generate_one_epoch_then_train_one(every_n_dim=None, save_data=False)


if __name__ == "__main__":
    main()
