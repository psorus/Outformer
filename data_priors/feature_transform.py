import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

from pfns.utils import normalize_data, to_ranking_low_mem, remove_outliers
from pfns.utils import NOP, normalize_by_used_features_f


class FeatureTransform:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prior_gmm_cfg = cfg.prior.mixture.gmm
        # prior hyperparameters
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim  # max_allowed_features; #input_features

    def feature_scale(self, x, num_feature, rescale_with_sqrt=False):
        if rescale_with_sqrt:
            return x / (num_feature / self.max_feature_dim) ** (1 / 2)
        return x / (num_feature / self.max_feature_dim)

    def feature_padding(self, x, num_feature):
        x = self.feature_scale(x=x, num_feature=num_feature)
        x = np.concatenate([x, np.zeros(shape=(x.shape[0], self.max_feature_dim - num_feature))], axis=-1)
        # from line 487 of pfn/scripts/tabpfn_inferface.py
        # eval_xs_ = torch.cat([eval_xs_, torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)
        return x

    def feature_padding_torch(self, x, num_feature):
        x = self.feature_scale(x=x, num_feature=num_feature)
        x = torch.cat([x, torch.zeros(x.shape[0], self.max_feature_dim - num_feature, device=x.device)], dim=-1)
        # from line 487 of pfn/scripts/tabpfn_inferface.py
        # eval_xs_ = torch.cat([eval_xs_, torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)
        return x

    def feature_subsampling(self, x, num_feature):
        # inliners and anomalies will share the same sub-samping order, so we expect x to be the concatenation of both
        # during inference on real dataset
        x = x[:, sorted(np.random.choice(num_feature, self.max_feature_dim, replace=False))]
        # from line 366 of tabpfn_interface.py
        # eval_xs = eval_xs[:, :, sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))]
        return x

    def feature_sparse_projection(self, x, num_feature):
        # inliners and anomalies will share the same sub-samping order, so we expect x to be the concatenation of both
        # during inference on real dataset
        scale = np.sqrt(3 / self.max_feature_dim)  # sqrt(3/K)
        projection = [self.generate_1d_projection(num_feature=num_feature, scale=scale)
                      for _ in range(self.max_feature_dim)]
        projection = np.array(projection)  # (max_allowed_features, num_feature)
        x = x @ projection.T  # (seq-len, num_feature) ---> (seq-len, max_allowed_features)
        return x

    @staticmethod
    def generate_1d_projection(num_feature, scale=1.0):
        # Calculate the number of 0s, 1s, and -1s
        num_zeros = int(num_feature * 2 / 3)
        num_ones = int(num_feature / 6)
        num_neg_ones = num_feature - num_zeros - num_ones

        # Create the array with the specified numbers of 0s, 1s, and -1s
        array = np.array([0] * num_zeros + [1] * num_ones + [-1] * num_neg_ones) * scale

        # Shuffle the array to randomize the distribution of elements
        np.random.shuffle(array)

        return array

    # in the official implementation:
    # 1. (potential) feature sub-sampling ---> 2. normalize data ---> 3. transform data ---> 4. rescale feature
    # later outside of this program: 5. feature padding
    def pfn_inference_transform(self, eval_xs: np.ndarray, preprocess_transform: str, eval_position: int,
                                normalize_with_test: bool = False, rescale_with_sqrt: bool = False):
        """
        :param eval_xs: the inputs
        :param preprocess_transform: str, 'none', 'power', 'quantile', 'robust'
        :param eval_position: train-x <--eval_position---> test_x
        :param normalize_with_test: when perform (x-mean)/std, whether to include test_x
        :param rescale_with_sqrt: when rescale the features, whether to use "* sqrt(num-max-feat/num-used-feat)"
        """
        import warnings
        if len(eval_xs.shape) != 2:
            raise Exception(
                "Transforms only allow input of shape: (#sampled, #feat), but we have: {}".format(eval_xs.shape))

        num_feature = eval_xs.shape[-1]
        if num_feature > self.max_feature_dim:
            eval_xs = self.feature_subsampling(x=eval_xs, num_feature=num_feature)

        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)

        eval_xs = torch.from_numpy(eval_xs)  # convert to torch
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)
        # perform (x-mean)/std normalization
        eval_xs = eval_xs.cpu().numpy()  # convert back to numpy

        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            print('feature preprocessing transform with {}'.format(preprocess_transform))
            # prev:
            # feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
            #     range(eval_xs.shape[1])) - set(
            #     categorical_feats)
            feats = set(range(eval_xs.shape[1]))
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col:col + 1] = trans
                except:
                    pass
        warnings.simplefilter('default')

        # Rescale X
        # prev:
        # eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
        #                                        normalize_with_sqrt=normalize_with_sqrt)
        eval_xs = self.feature_scale(x=eval_xs, num_feature=eval_xs.shape[-1], rescale_with_sqrt=rescale_with_sqrt)
        if eval_xs.shape[-1] < self.max_feature_dim:
            eval_xs = np.concatenate([eval_xs, np.zeros(shape=(eval_xs.shape[0], self.max_feature_dim - num_feature))],
                                     axis=-1)
        return eval_xs
