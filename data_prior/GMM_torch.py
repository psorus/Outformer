import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import chi2
import time


class GaussianMixtureModel:
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor,
                 percentile=0.80, delta=0.05, inflate_scale=5.0, inflate_full=False, sub_dims=None, device='cpu'):
        self.device = device
        self.means = means
        self.covariances = covariances
        self.weights = weights

        # assert torch.isclose(self.weights.sum(), torch.tensor(1.0, device=self.device)), 'weights need to sum up to 1'
        # assert len(means) == len(covariances)
        # assert len(covariances) == self.weights.shape[0]

        self.num_cluster = len(means)

        d = self.means[0].shape[0]
        self.d = d
        n = d if inflate_full else np.random.randint(1, d + 1)  # Generate random integer between 1 and d, inclusive
        if inflate_full and sub_dims is not None:
            print('we are inflating all the dimensions, however, sub_dims is provided')
            raise Exception
        self.sub_dims = torch.sort(torch.randperm(d)[:n]).values.to(self.device) if sub_dims is None else sub_dims.to(
            self.device)

        self.threshold_plus_delta = chi2.ppf(percentile + delta, df=len(self.sub_dims))
        self.threshold_minus_delta = chi2.ppf(percentile - delta, df=len(self.sub_dims))

        self.inflated_covariances = []
        self.inv_sub_covariances = []

        for cov in self.covariances:
            cov_copy = cov.clone()
            sub_cov = cov_copy[self.sub_dims, :][:, self.sub_dims]  # the sub_cov defines a new (smaller) Gaussian
            # n x n ---> sub_dims x n ---> sub_dims x sub_dims
            self.inv_sub_covariances.append(torch.linalg.inv(sub_cov))
            cov_copy[self.sub_dims[:, None], self.sub_dims] *= inflate_scale
            self.inflated_covariances.append(cov_copy)

        self.inflated_covariances = torch.stack(self.inflated_covariances)
        self.inv_sub_covariances = torch.stack(self.inv_sub_covariances)

        self.GMM4sample = [MultivariateNormal(self.means[cluster_id], self.covariances[cluster_id])
                           for cluster_id in range(len(self.weights))]

        self.GMM4inf = [MultivariateNormal(self.means[cluster_id], self.inflated_covariances[cluster_id])
                        for cluster_id in range(len(self.weights))]

    def draw_samples(self, num_samples):
        """
        Draws samples from the d-dimensional Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        torch.Tensor: samples drawn from the GMM.
        """
        samples = torch.zeros(num_samples, self.d, device=self.device)
        component_choices = torch.multinomial(self.weights, num_samples, replacement=True)
        for cluster_id in range(self.num_cluster):
            mask = (component_choices == cluster_id)
            num_cluster_samples = mask.sum().item()
            if num_cluster_samples > 0:
                sample = self.GMM4sample[cluster_id].sample((num_cluster_samples,))
                samples[mask] = sample
        return samples

    # def draw_samples(self, num_samples):
    #     """
    #     Draws samples from the d-dimensional Gaussian Mixture Model.
    #
    #     Parameters:
    #     num_samples (int): Number of samples to draw.
    #
    #     Returns:
    #     torch.Tensor: samples drawn from the GMM.
    #     """
    #     # Sample cluster assignments
    #     component_choices = torch.multinomial(self.weights, num_samples, replacement=True)  # (num_samples,)
    #
    #     # Stack means and standard deviations
    #     cluster_variances = self.covariances.diagonal(dim1=1, dim2=2)  # Shape: (num_clusters, dim)
    #     cluster_std_devs = cluster_variances.sqrt()  # Shape: (num_clusters, dim)
    #
    #     # Get per-sample means and std_devs based on cluster assignments
    #     per_sample_means = self.means[component_choices]  # Shape: (num_samples, dim)
    #     per_sample_std_devs = cluster_std_devs[component_choices]  # Shape: (num_samples, dim)
    #
    #     # Sample standard normals
    #     standard_normals = torch.randn(num_samples, self.d, device=self.device)
    #
    #     # Generate samples
    #     samples = per_sample_means + standard_normals * per_sample_std_devs  # Shape: (num_samples, dim)
    #
    #     return samples

    def draw_inflated_samples(self, num_samples):
        """
        Draws samples from the d-dimensional inflated Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        torch.Tensor: samples drawn from the inflated GMM.
        """
        samples = torch.zeros(num_samples, self.d, device=self.device)
        component_choices = torch.multinomial(self.weights, num_samples, replacement=True)

        for cluster_id in range(self.num_cluster):
            mask = (component_choices == cluster_id)
            num_cluster_samples = mask.sum().item()
            if num_cluster_samples > 0:
                sample = self.GMM4inf[cluster_id].sample((num_cluster_samples,))  # .type_as(self.weights)
                samples[mask] = sample
        return samples

    # def draw_inflated_samples(self, num_samples):
    #     # Sample cluster assignments
    #     component_choices = torch.multinomial(self.weights, num_samples, replacement=True)  # (num_samples,)
    #
    #     # Stack means and standard deviations for inflated covariances
    #     cluster_variances = self.inflated_covariances.diagonal(dim1=1, dim2=2)  # Shape: (num_clusters, dim)
    #     cluster_std_devs = cluster_variances.sqrt()  # Shape: (num_clusters, dim)
    #
    #     # Get per-sample means and std_devs
    #     per_sample_means = self.means[component_choices]  # Shape: (num_samples, dim)
    #     per_sample_std_devs = cluster_std_devs[component_choices]  # Shape: (num_samples, dim)
    #
    #     # Sample standard normals
    #     standard_normals = torch.randn(num_samples, self.d, device=self.device)
    #
    #     # Generate samples
    #     samples = per_sample_means + standard_normals * per_sample_std_devs
    #
    #     return samples

    def mahalanobis_distance(self, sample, mean, inv_covariance):
        """
        Computes the Mahalanobis distance of a sample from a given mean and inverse covariance matrix.

        Parameters:
        sample (torch.Tensor): Sample point. (d, )
        mean (torch.Tensor): Mean vector.  (d, )
        inv_covariance (torch.Tensor): Inverse covariance matrix.  (sub-dims, )

        Returns:
        float: Mahalanobis distance of the sample from the mean.
        """
        delta = sample[self.sub_dims] - mean[self.sub_dims]
        return torch.sqrt((delta @ inv_covariance @ delta).sum())

    def batched_squared_mahalanobis_distance(self, X, mean, inv_cov):
        delta = X[:, self.sub_dims] - mean[self.sub_dims]
        return torch.diag(delta @ inv_cov @ delta.T)

    # def batched_squared_mahalanobis_distance(self, X, mean, inv_cov):
    #     delta = X[:, self.sub_dims] - mean[self.sub_dims]
    #     left = torch.matmul(delta, inv_cov)
    #     squared_distances = (left * delta).sum(dim=1)
    #     # print(squared_distances)
    #     # print(torch.diag(delta @ inv_cov @ delta.T))
    #     return squared_distances

    # def draw_inliers(self, num_samples):
    #     """
    #     Draws samples from the d-dimensional Gaussian Mixture Model where points are within n-th percentile.
    #
    #     Parameters:
    #     num_samples (int): Number of samples to draw.
    #
    #     Returns:
    #     torch.Tensor: Array of samples drawn from the GMM that are within n-th percentile.
    #     """
    #     samples = []
    #     while len(samples) < num_samples:
    #         sample = self.draw_samples(1)[0]  # (1, d) ---> (d, )
    #         distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
    #                      zip(self.means, self.inv_sub_covariances)]
    #         if min(distances) ** 2 < self.threshold_minus_delta:
    #             samples.append(sample)
    #     return torch.stack(samples, dim=0)

    def draw_inliers(self, num_samples):
        batch_size = max(num_samples * 2, 1000)
        samples = []
        total_samples_needed = num_samples
        while total_samples_needed > 0:
            raw_samples = self.draw_samples(batch_size)
            batch_distances = self.get_squared_batched_dist(raw_samples)
            min_squared_distances = torch.min(batch_distances, dim=1).values
            inliner_mask = min_squared_distances < self.threshold_minus_delta
            selected_samples = raw_samples[inliner_mask]
            num_selected = selected_samples.shape[0]
            if num_selected > 0:
                if num_selected >= total_samples_needed:
                    samples.append(selected_samples[:total_samples_needed])
                    total_samples_needed = 0
                else:
                    samples.append(selected_samples)
                    total_samples_needed -= num_selected
        samples = torch.cat(samples)
        return samples

    # def draw_local_anomalies(self, num_samples):
    #     samples = []
    #     while len(samples) < num_samples:
    #         sample = self.draw_inflated_samples(1)[0]  # (1, d) ---> (d, )
    #         distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
    #                      zip(self.means, self.inv_sub_covariances)]
    #         if min(distances) ** 2 > self.threshold_plus_delta:
    #             samples.append(sample)
    #     return torch.stack(samples, dim=0)

    def draw_local_anomalies(self, num_samples):
        batch_size = max(num_samples * 2, 1000)
        samples = []
        total_samples_needed = num_samples
        while total_samples_needed > 0:
            raw_samples = self.draw_inflated_samples(batch_size)
            batch_distances = self.get_squared_batched_dist(raw_samples)
            min_squared_distances = torch.min(batch_distances, dim=1).values
            anomaly_mask = min_squared_distances > self.threshold_plus_delta
            selected_samples = raw_samples[anomaly_mask]
            num_selected = selected_samples.shape[0]
            if num_selected > 0:
                if num_selected >= total_samples_needed:
                    samples.append(selected_samples[:total_samples_needed])
                    total_samples_needed = 0
                else:
                    samples.append(selected_samples)
                    total_samples_needed -= num_selected
        samples = torch.cat(samples)
        return samples

    def assert_inliers(self, samples):
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 < self.threshold_minus_delta

    def assert_local_anomalies(self, samples):
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 > self.threshold_plus_delta

    # def get_squared_batched_dist(self, raw_samples):
    #     num_samples = raw_samples.size(0)
    #     delta = raw_samples[:, None, self.sub_dims] - torch.stack([mean[self.sub_dims] for mean in self.means])[None, :, :]
    #     delta = delta.view(-1, len(self.sub_dims))
    #     inv_covs = self.inv_sub_covariances.repeat(num_samples, 1, 1)
    #     left = torch.bmm(delta.unsqueeze(1), inv_covs)
    #     squared_distances = (left.squeeze(1) * delta).sum(dim=1)
    #     squared_distances = squared_distances.view(num_samples, self.num_cluster)
    #     return squared_distances

    def get_squared_batched_dist(self, raw_samples):
        batch_dist = []
        for mean, inv_cov in zip(self.means, self.inv_sub_covariances):
            distances = self.batched_squared_mahalanobis_distance(X=raw_samples, mean=mean, inv_cov=inv_cov)
            batch_dist.append(distances)
        return torch.stack(batch_dist, dim=1)  # (#samples, num_cluster)

    def draw_batched_data(self, num_inliers, num_local_anomalies):
        raw_inliers = self.draw_samples(num_samples=int(num_inliers * 2))
        raw_local_anomalies = self.draw_inflated_samples(num_samples=int(num_local_anomalies * 2))

        inliers_squared_dist = self.get_squared_batched_dist(raw_samples=raw_inliers)
        local_anomalies_squared_dist = self.get_squared_batched_dist(raw_samples=raw_local_anomalies)

        min_inliers_squared_dist = torch.min(inliers_squared_dist, dim=1).values
        min_local_anomalies_squared_dist = torch.min(local_anomalies_squared_dist, dim=1).values

        inliers_mask = min_inliers_squared_dist < self.threshold_minus_delta  # (#raw-inliers, )
        local_anomalies_mask = min_local_anomalies_squared_dist > self.threshold_plus_delta  # (#raw-la, )

        inliers = raw_inliers[inliers_mask][:num_inliers]
        local_anomalies = raw_local_anomalies[local_anomalies_mask][:num_local_anomalies]

        def add_extra(existing_samples, target_num_samples, draw_func):
            if existing_samples.shape[0] < target_num_samples:
                extra_samples = draw_func(num_samples=target_num_samples - existing_samples.shape[0])
                existing_samples = torch.concat([existing_samples, extra_samples], dim=0)
            return existing_samples

        inliers = add_extra(existing_samples=inliers, target_num_samples=num_inliers, draw_func=self.draw_inliers)
        local_anomalies = add_extra(existing_samples=local_anomalies, target_num_samples=num_local_anomalies,
                                    draw_func=self.draw_local_anomalies)
        return inliers, local_anomalies


# def make_NdMclusterGMM(dim: int, num_cluster: int, weights: torch.Tensor, max_mean: int, max_var: int,
#                        inflate_full: bool, device, sub_dims=None, percentile=0.80, delta=0.05, ):
#     means = [torch.rand(dim, ) * torch.randint(low=-max_mean, high=max_mean+1, size=(dim, ))
#              for _ in range(num_cluster)]
#     covariances = []
#     for _ in range(num_cluster):
#         diag = torch.rand(dim, ) * torch.randint(low=1, high=max_var+1, size=(dim, ))
#         diag[diag == 0] = max_var / 2
#         cov = torch.diag(diag)
#         covariances.append(cov)
#     N_d_M_cluster_gaussian = GaussianMixtureModel(means=means, covariances=covariances, weights=weights,
#                                                   inflate_full=inflate_full, sub_dims=sub_dims,
#                                                   percentile=percentile, delta=delta, device=device)
#     return N_d_M_cluster_gaussian

def make_NdMclusterGMM(dim: int, num_cluster: int, weights: torch.Tensor, max_mean: int, max_var: int,
                       inflate_full: bool, device, sub_dims=None, percentile=0.80, delta=0.05):
    # Generate means between -max_mean and max_mean
    means = torch.rand(num_cluster, dim, device=device) * \
            torch.randint(low=-max_mean, high=max_mean+1, size=(num_cluster, dim, ), device=device)

    # Generate diagonal covariance matrices with positive entries between 1 and max_var
    diag_values = torch.rand(num_cluster, dim, device=device) * \
                  torch.randint(low=1, high=max_var+1, size=(num_cluster, dim, ), device=device)
    diag_values[diag_values == 0] = max_var / 2

    # Create batch of diagonal covariance matrices
    covariances = torch.diag_embed(diag_values)  # Shape: (num_cluster, dim, dim)

    N_d_M_cluster_gaussian = GaussianMixtureModel(
        means=means,
        covariances=covariances,
        weights=weights,
        inflate_full=inflate_full,
        sub_dims=sub_dims,
        percentile=percentile,
        delta=delta,
        device=device
    )
    return N_d_M_cluster_gaussian


def generate_constrained_eigenvals(d):
    # Generate uniformly distributed values in the range (-0.8, -0.2)
    low_range = np.random.uniform(-1.0, -0.1, size=d)

    # Generate uniformly distributed values in the range (0.2, 0.8)
    high_range = np.random.uniform(0.1, 1.0, size=d)

    # Randomly choose between the two ranges for each element
    choice = np.random.choice([0, 1], size=d)
    vector = np.where(choice == 0, low_range, high_range)

    return vector


def generate_full_rank_matrix(dim, device, scale=1):
    # Generate a random orthogonal matrix using QR decomposition
    A = np.random.rand(dim, dim)
    Q, _ = np.linalg.qr(A)

    eigenvals = generate_constrained_eigenvals(d=dim)
    eigenvals = np.diag(eigenvals)

    full_rank_matrix = Q @ eigenvals @ Q.T
    assert np.linalg.matrix_rank(full_rank_matrix) == dim
    if device is None:  # source is numpy
        return full_rank_matrix
    else:
        return torch.from_numpy(full_rank_matrix).to(dtype=torch.float, device=device)


def generate_linear_transform(dim, device, A_scale=1, b_scale=1):
    A = generate_full_rank_matrix(dim=dim, device=device, scale=A_scale)
    b = np.random.rand(dim) * np.random.randint(low=-b_scale, high=b_scale + 1, size=dim)  # [low, high)

    if device is not None:  # source is torch, transfer from numpy to torch
        b = torch.from_numpy(b).to(dtype=torch.float, device=device)
    return A, b


def transform_means(means, sub_dims, A, b):
    trans = []
    for mean in means:
        new_mean = mean.clone()
        new_mean[sub_dims] = A @ new_mean[sub_dims] + b
        trans.append(new_mean)
    return torch.stack(trans)


def transform_covs(covs, sub_dims, A):
    trans = []
    for cov in covs:
        new_cov = cov.clone()
        new_cov[sub_dims[:, None], sub_dims] = A @ new_cov[sub_dims[:, None], sub_dims] @ A.T
        trans.append(new_cov)
    return torch.stack(trans)


def transform_samples(samples, sub_dims, A, b, is_source_numpy=False):
    if is_source_numpy:
        new_samples = samples.copy()
    else:
        new_samples = samples.clone()

    if sub_dims is None:
        new_samples = new_samples @ A.T + b
    else:
        new_samples[:, sub_dims] = new_samples[:, sub_dims] @ A.T + b

    return new_samples


if __name__ == "__main__":
    # TODO: need to check why the generated LA is usually smaller than the target number,
    #  which is very strange (Yuchen/2024/10/31) & the LT is also problematic
    s = time.time()
    device = 'cuda:0'
    dim = np.random.randint(low=2, high=41)  # draw from [2, 20]
    num_cluster = np.random.randint(low=2, high=6)  # draw from [2, 5]
    max_mean = np.random.randint(low=2, high=6)  # draw from [2, 5]
    max_var = np.random.randint(low=2, high=6)  # draw from [2, 5]
    print('num cluster', num_cluster)
    print('dim', dim)
    model = make_NdMclusterGMM(dim=dim, num_cluster=num_cluster, weights=torch.tensor([1 / num_cluster] * num_cluster, device=device),
                               max_mean=max_mean, max_var=max_var, inflate_full=False, sub_dims=None,
                               percentile=0.9, delta=0.05, device=device)

    num_samples = 5000

    print(f'drawing {num_samples} inliers and outliers')

    # test the batch data follows the inliner/local_anomaly criterion
    inliers, local_anomalies = model.draw_batched_data(num_samples, num_samples)
    print(time.time()-s)
    model.assert_inliers(inliers)
    model.assert_local_anomalies(local_anomalies)

    # test whether linear transform preserves percentiles
    for _ in range(num_samples):
        full_rank_matrix = generate_full_rank_matrix(dim=dim, device=device)
        assert torch.linalg.matrix_rank(full_rank_matrix) == dim
    print('full rank matrix generation asserted')

    sub_dims = model.sub_dims
    A, b = generate_linear_transform(dim=len(sub_dims), device=device)

    model_T = GaussianMixtureModel(means=transform_means(model.means, sub_dims, A, b),
                                   covariances=transform_covs(model.covariances, sub_dims, A),
                                   weights=torch.tensor([1 / num_cluster] * num_cluster),
                                   sub_dims=sub_dims, device=device, percentile=0.9, delta=0.05)

    in_T = transform_samples(inliers, sub_dims, A, b)
    la_T = transform_samples(local_anomalies, sub_dims, A, b)

    model_T.assert_inliers(in_T)  # linear transformed inliers remain within the n-th percentiles under new model
    model_T.assert_local_anomalies(la_T)  # similar to above
    print('linear transform successfully asserted')
