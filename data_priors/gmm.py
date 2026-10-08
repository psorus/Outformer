import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import chi2
import time


class GaussianMixtureModel:
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor,
                 percentile=0.80, delta=0.05, inflate_scale=5.0, inflate_full=False, sub_dims=None, device='cpu'):
        """Initialize a Gaussian mixture model used for inlier/anomaly sampling.

        The model can optionally inflate selected covariance dimensions to generate
        local anomalies and computes chi-square thresholds for distance-based
        filtering in a subspace.

        Args:
            means: Component means with shape (num_cluster, d).
            covariances: Component covariance matrices with shape (num_cluster, d, d).
            weights: Mixture weights with shape (num_cluster,).
            percentile: Base chi-square percentile for distance thresholds.
            delta: Margin added/subtracted from percentile for anomaly/inlier bands.
            inflate_scale: Multiplicative scale applied to selected covariance entries.
            inflate_full: If True, use all dimensions for inflation.
            sub_dims: Optional predefined subspace indices; sampled if None.
            device: Target device for tensors.

        Raises:
            Exception: If inflate_full is True while sub_dims is explicitly provided.
        """
        self.device = device
        self.means = means
        self.covariances = covariances
        self.weights = weights

        self.num_cluster = len(means)

        d = self.means[0].shape[0]
        self.d = d
        # Choose the inflation subspace size. If not full inflation, sample in [1, d].
        n = d if inflate_full else np.random.randint(1, d + 1)
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
            # Extract the covariance block in the selected subspace.
            sub_cov = cov_copy[self.sub_dims, :][:, self.sub_dims]
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
        """Draw samples from the original Gaussian mixture.

        Args:
            num_samples: Number of points to sample.

        Returns:
            Tensor of shape (num_samples, d).
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


    def draw_inflated_samples(self, num_samples):
        """Draw samples from the covariance-inflated Gaussian mixture.

        Args:
            num_samples: Number of points to sample.

        Returns:
            Tensor of shape (num_samples, d).
        """
        samples = torch.zeros(num_samples, self.d, device=self.device)
        component_choices = torch.multinomial(self.weights, num_samples, replacement=True)

        for cluster_id in range(self.num_cluster):
            mask = (component_choices == cluster_id)
            num_cluster_samples = mask.sum().item()
            if num_cluster_samples > 0:
                sample = self.GMM4inf[cluster_id].sample((num_cluster_samples,))
                samples[mask] = sample
        return samples


    def mahalanobis_distance(self, sample, mean, inv_covariance):
        """Compute the Mahalanobis distance in the selected subspace.

        Args:
            sample: Input sample with shape (d,).
            mean: Component mean with shape (d,).
            inv_covariance: Inverse covariance in subspace with shape (k, k).

        Returns:
            Scalar distance value.
        """
        delta = sample[self.sub_dims] - mean[self.sub_dims]
        return torch.sqrt((delta @ inv_covariance @ delta).sum())

    def batched_squared_mahalanobis_distance(self, X, mean, inv_cov):
        """Compute squared Mahalanobis distances for a batch in the subspace.

        Args:
            X: Samples with shape (n, d).
            mean: Component mean with shape (d,).
            inv_cov: Inverse covariance in subspace with shape (k, k).

        Returns:
            Tensor of shape (n,) with squared distances.
        """
        delta = X[:, self.sub_dims] - mean[self.sub_dims]
        return torch.diag(delta @ inv_cov @ delta.T)

    def draw_inliers(self, num_samples):
        """Sample inliers whose minimum squared distance is below the lower threshold.

        Args:
            num_samples: Number of inlier samples to return.

        Returns:
            Tensor of shape (num_samples, d).
        """
        batch_size = max(num_samples * 2, 1000)
        samples = []
        total_samples_needed = num_samples
        while total_samples_needed > 0:
            raw_samples = self.draw_samples(batch_size)
            batch_distances = self.get_squared_batched_dist(raw_samples)
            min_squared_distances = torch.min(batch_distances, dim=1).values
            inlier_mask = min_squared_distances < self.threshold_minus_delta
            selected_samples = raw_samples[inlier_mask]
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

    def draw_local_anomalies(self, num_samples):
        """Sample local anomalies above the upper squared-distance threshold.

        Args:
            num_samples: Number of local anomaly samples to return.

        Returns:
            Tensor of shape (num_samples, d).
        """
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
        """Assert that each sample is an inlier under at least one component."""
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 < self.threshold_minus_delta

    def assert_local_anomalies(self, samples):
        """Assert that each sample is a local anomaly under all components."""
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 > self.threshold_plus_delta

    def get_squared_batched_dist(self, raw_samples):
        """Compute per-component squared distances for a batch of samples.

        Args:
            raw_samples: Samples with shape (n, d).

        Returns:
            Tensor with shape (n, num_cluster).
        """
        batch_dist = []
        for mean, inv_cov in zip(self.means, self.inv_sub_covariances):
            distances = self.batched_squared_mahalanobis_distance(X=raw_samples, mean=mean, inv_cov=inv_cov)
            batch_dist.append(distances)
        return torch.stack(batch_dist, dim=1)

    def draw_batched_data(self, num_inliers, num_local_anomalies):
        """Generate inliers and local anomalies in one call.

        This method first draws oversized raw batches and filters by threshold.
        If either class is underfilled, it backfills using iterative samplers.

        Args:
            num_inliers: Target number of inliers.
            num_local_anomalies: Target number of local anomalies.

        Returns:
            Tuple (inliers, local_anomalies) with target counts.
        """
        raw_inliers = self.draw_samples(num_samples=int(num_inliers * 2))
        raw_local_anomalies = self.draw_inflated_samples(num_samples=int(num_local_anomalies * 2))

        inliers_squared_dist = self.get_squared_batched_dist(raw_samples=raw_inliers)
        local_anomalies_squared_dist = self.get_squared_batched_dist(raw_samples=raw_local_anomalies)

        min_inliers_squared_dist = torch.min(inliers_squared_dist, dim=1).values
        min_local_anomalies_squared_dist = torch.min(local_anomalies_squared_dist, dim=1).values

        inliers_mask = min_inliers_squared_dist < self.threshold_minus_delta
        local_anomalies_mask = min_local_anomalies_squared_dist > self.threshold_plus_delta

        inliers = raw_inliers[inliers_mask][:num_inliers]
        local_anomalies = raw_local_anomalies[local_anomalies_mask][:num_local_anomalies]

        def add_extra(existing_samples, target_num_samples, draw_func):
            """Backfill samples when threshold filtering returns too few points."""
            if existing_samples.shape[0] < target_num_samples:
                extra_samples = draw_func(num_samples=target_num_samples - existing_samples.shape[0])
                existing_samples = torch.concat([existing_samples, extra_samples], dim=0)
            return existing_samples

        inliers = add_extra(existing_samples=inliers, target_num_samples=num_inliers, draw_func=self.draw_inliers)
        local_anomalies = add_extra(existing_samples=local_anomalies, target_num_samples=num_local_anomalies,
                                    draw_func=self.draw_local_anomalies)
        return inliers, local_anomalies

def make_NdMclusterGMM(dim: int, num_cluster: int, weights: torch.Tensor, max_mean: int, max_var: int,
                       inflate_full: bool, device, sub_dims=None, percentile=0.80, delta=0.05):
    """Construct a diagonal-covariance GMM with random means/variances.

    Args:
        dim: Feature dimension.
        num_cluster: Number of mixture components.
        weights: Mixture weights with shape (num_cluster,).
        max_mean: Absolute bound used to scale random means.
        max_var: Upper bound used to scale random diagonal variances.
        inflate_full: Whether inflation should use all dimensions.
        device: Target torch device.
        sub_dims: Optional predefined inflation subspace.
        percentile: Base chi-square percentile.
        delta: Margin around percentile.

    Returns:
        Configured GaussianMixtureModel instance.
    """
    # Generate means with random sign and magnitude scaling.
    means = torch.rand(num_cluster, dim, device=device) * \
            torch.randint(low=-max_mean, high=max_mean+1, size=(num_cluster, dim, ), device=device)

    # Generate diagonal covariance values in (0, max_var].
    diag_values = torch.rand(num_cluster, dim, device=device) * \
                  torch.randint(low=1, high=max_var+1, size=(num_cluster, dim, ), device=device)
    diag_values[diag_values == 0] = max_var / 2

    # Assemble one diagonal covariance matrix per component.
    covariances = torch.diag_embed(diag_values)

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
    """Generate non-near-zero eigenvalues with mixed signs.

    Args:
        d: Number of eigenvalues.

    Returns:
        NumPy array of shape (d,) with values sampled away from zero.
    """
    # Sample negative values away from zero.
    low_range = np.random.uniform(-1.0, -0.1, size=d)

    # Sample positive values away from zero.
    high_range = np.random.uniform(0.1, 1.0, size=d)

    # Mix signs element-wise to avoid degenerate directional structure.
    choice = np.random.choice([0, 1], size=d)
    vector = np.where(choice == 0, low_range, high_range)

    return vector


def generate_full_rank_matrix(dim, device, scale=1):
    """Generate a full-rank square matrix via orthogonal eigen decomposition.

    Args:
        dim: Matrix dimension.
        device: Target torch device, or None to return NumPy.
        scale: Reserved argument for compatibility.

    Returns:
        Full-rank matrix as NumPy array (if device is None) or torch.Tensor.
    """
    # Build an orthogonal basis with QR decomposition.
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
    """Generate an affine transform x -> Ax + b on the selected dimensions.

    Args:
        dim: Transform dimension.
        device: Target torch device, or None for NumPy outputs.
        A_scale: Reserved argument passed to matrix generation.
        b_scale: Integer bound controlling translation magnitude.

    Returns:
        Tuple (A, b) where A is full rank and b is a random shift vector.
    """
    A = generate_full_rank_matrix(dim=dim, device=device, scale=A_scale)
    b = np.random.rand(dim) * np.random.randint(low=-b_scale, high=b_scale + 1, size=dim)

    # Convert the translation vector to torch when needed.
    if device is not None:
        b = torch.from_numpy(b).to(dtype=torch.float, device=device)
    return A, b


def transform_means(means, sub_dims, A, b):
    """Apply an affine transform to mean vectors on selected dimensions.

    Args:
        means: Mean matrix with shape (num_cluster, d).
        sub_dims: Indices of transformed dimensions.
        A: Linear transform matrix with shape (k, k).
        b: Translation vector with shape (k,).

    Returns:
        Transformed means with the same shape as input.
    """
    trans = []
    for mean in means:
        new_mean = mean.clone()
        new_mean[sub_dims] = A @ new_mean[sub_dims] + b
        trans.append(new_mean)
    return torch.stack(trans)


def transform_covs(covs, sub_dims, A):
    """Apply covariance pushforward Sigma -> A Sigma A^T on selected dimensions.

    Args:
        covs: Covariance batch with shape (num_cluster, d, d).
        sub_dims: Indices of transformed dimensions.
        A: Linear transform matrix with shape (k, k).

    Returns:
        Transformed covariance batch with the same shape as input.
    """
    trans = []
    for cov in covs:
        new_cov = cov.clone()
        new_cov[sub_dims[:, None], sub_dims] = A @ new_cov[sub_dims[:, None], sub_dims] @ A.T
        trans.append(new_cov)
    return torch.stack(trans)


def transform_samples(samples, sub_dims, A, b, is_source_numpy=False):
    """Apply an affine transform to samples globally or on a subspace.

    Args:
        samples: Sample matrix with shape (n, d).
        sub_dims: Transformed indices, or None to transform all dimensions.
        A: Linear transform matrix.
        b: Translation vector.
        is_source_numpy: Whether input samples are NumPy arrays.

    Returns:
        Samples with transformed coordinates and preserved input type.
    """
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
    # Quick self-check for sampling logic and transform consistency.
    s = time.time()
    device = 'cuda:0'
    dim = np.random.randint(low=2, high=41)  # Draw from [2, 40].
    num_cluster = np.random.randint(low=2, high=6)  # Draw from [2, 5].
    max_mean = np.random.randint(low=2, high=6)  # Draw from [2, 5].
    max_var = np.random.randint(low=2, high=6)  # Draw from [2, 5].
    print('num cluster', num_cluster)
    print('dim', dim)
    model = make_NdMclusterGMM(dim=dim, num_cluster=num_cluster, weights=torch.tensor([1 / num_cluster] * num_cluster, device=device),
                               max_mean=max_mean, max_var=max_var, inflate_full=False, sub_dims=None,
                               percentile=0.9, delta=0.05, device=device)

    num_samples = 5000

    print(f'drawing {num_samples} inliers and outliers')

    # Validate that generated samples satisfy inlier/anomaly constraints.
    inliers, local_anomalies = model.draw_batched_data(num_samples, num_samples)
    print(time.time()-s)
    model.assert_inliers(inliers)
    model.assert_local_anomalies(local_anomalies)

    # Validate that generated transform matrices are full rank.
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

    # Validate that affine transformation preserves sample class under transformed model.
    model_T.assert_inliers(in_T)
    model_T.assert_local_anomalies(la_T)
    print('linear transform successfully asserted')
