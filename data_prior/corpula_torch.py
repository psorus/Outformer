import random, time, math, numpy as np
from contextlib import contextmanager

import torch
from torch.distributions import Normal, Beta, Exponential, StudentT
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics  import roc_auc_score
import scipy
from scipy.stats import beta as scipy_beta
import torch, math, multiprocessing as mp, numpy as np
import scipy.stats as sps
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import betainc, betaincinv


import time
import torch

# Make sure to sync for accurate GPU timings
def tic():
    torch.cuda.synchronize()
    return time.time()

def toc(start, label=""):
    torch.cuda.synchronize()
    end = time.time()
    print(f"{label}: {end - start:.6f} seconds")

# ───────────────────────── helper: icdf() calculation on torch ──────────────────────────
def _neg_log1p(u):
    # stable for u≈0
    return (-torch.log1p(-u)).clamp_min(0.0)


def exp_icdf(u, rate):
    """
    u     : (...,)  in (0,1)
    rate  : broadcastable with u   (λ > 0)
    """
    return _neg_log1p(u) / rate


def beta_icdf(p, a, b, n_grid: int = 1000):
    p = torch.as_tensor(p, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.size() == torch.Size([]):
      a = a.expand(1)
    if b.size() == torch.Size([]):
      b = b.expand(1)
    x = torch.linspace(0.0, 1.0, n_grid + 1, dtype=torch.float32, device=p.device)
    mid = (x[:-1] + x[1:]) / 2.0
    mid = mid.view(-1, 1)  # shape [n_grid, 1]
    dx = 1.0 / n_grid

    log_norm = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    pdf_mid = torch.exp((a - 1) * torch.log(mid) +
                        (b - 1) * torch.log1p(-mid) -
                        log_norm)  # [n_grid, D]
    cdf = torch.cumsum(pdf_mid, dim=0) * dx  # [n_grid, D]
    cdf = torch.cat([torch.zeros(1, cdf.shape[1], device=p.device), cdf], dim=0)  # [n_grid+1, D]
    # Expand p to match shape [N, D]
    if p.ndim == 1:
        p = p.unsqueeze(1)
    if a.ndim == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    if p.shape[1] != a.shape[1]:
        p = p.expand(-1, a.shape[1])
    # Searchsorted per-dimension
    N, D = p.shape
    idx = torch.empty((N, D), dtype=torch.long, device=p.device)
    for d in range(D):
        idx[:, d] = torch.searchsorted(cdf[:, d], p[:, d], right=False).clamp(1, n_grid)
    # Gather x and y for interpolation
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = torch.empty_like(p)
    y1 = torch.empty_like(p)
    for d in range(D):
        y0[:, d] = cdf[idx[:, d] - 1, d]
        y1[:, d] = cdf[idx[:, d], d]

    t = (p - y0) / (y1 - y0 + 1e-12)
    return torch.clamp(x0 + t * (x1 - x0), 0., 1.)


def student_t_icdf(u, df):
    u = torch.as_tensor(u, dtype=torch.float32)
    df = torch.as_tensor(df, dtype=torch.float32)

    p = 2.0 * torch.minimum(u, 1 - u)
    p = p.to(df.device)
    x = beta_icdf(p, 0.5 * df, torch.tensor(0.5,device = p.device))
    t = torch.sqrt(df * (1.0 / x - 1.0))
    return torch.where(u > 0.5, t, -t)


def normal_cdf(x):  
    return 0.5*(1+torch.special.erf(x/math.sqrt(2)))


def normal_ppf(u):  
    return Normal(0,1).icdf(u)


# ─────────────── low-level helpers ───────────────
def rand_corr_batch(batch, d, identity=False,device='cuda'):
    """(B,d,d) stack of correlation matrices on device/torch.float32"""
    if identity:
        eye = torch.eye(d, device=device, dtype=torch.float32)
        return eye.expand(batch, -1, -1).clone()

    A   = torch.rand(batch, d, d, device=device, dtype=torch.float32)
    psd = A @ A.transpose(-1, -2)
    diag= torch.diagonal(psd, dim1=-2, dim2=-1)
    norm= torch.sqrt(torch.clamp(diag, 1e-12))
    C   = psd / (norm.unsqueeze(-1)*norm.unsqueeze(-2))
    C.diagonal(dim1=-2, dim2=-1).fill_(1.)
    C += torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0)*1e-6
    return C


def mvn_sample(chol,device):       # chol: (B,d,d) or (d,d)
    if chol.dim() == 2: chol = chol.unsqueeze(0)
    B,d = chol.shape[:2]
    z   = torch.randn(B, d, 1, device=device, dtype=torch.float32)
    return (chol @ z).squeeze(-1)           # (B,d)


# ─────────────── mixture helpers ───────────────
class GaussianMix:
    def __init__(self, comps, device='cpu'):
        w, mu, sigma = zip(*comps)
        self.device = device
        self.w = torch.tensor(w, device=device, dtype=torch.float32)
        self.mu = torch.tensor(mu, device=device, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.w /= self.w.sum()

    def cdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        z = (x.unsqueeze(-1) - self.mu) / (self.sigma * math.sqrt(2))
        return (self.w * (0.5 * (1 + torch.erf(z)))).sum(-1)

    def ppf_bounds(self):
        lo = (self.mu - 5 * self.sigma).min().item()
        hi = (self.mu + 5 * self.sigma).max().item()
        return lo, hi

    def ppf(self, u, tol=1e-5, max_iter=100):
        """
        Numerical inverse CDF using bisection.
        u: tensor of shape (...)
        Returns: tensor of same shape
        """
        u = torch.as_tensor(u, device=self.device, dtype=torch.float32)
        low, high = self.ppf_bounds()
        low = torch.full_like(u, low)
        high = torch.full_like(u, high)

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            cdf_mid = self.cdf(mid)
            low = torch.where(cdf_mid < u, mid, low)
            high = torch.where(cdf_mid >= u, mid, high)
            if torch.max(high - low) < tol:
                break
        return 0.5 * (low + high)


class BetaMix:
    def __init__(self, comps,device='cuda'):
        self.comps = comps
        self.w = torch.tensor([c[0] for c in comps], device=device, dtype=torch.float32)
        self.a = torch.tensor([c[1] for c in comps], device=device, dtype=torch.float32)
        self.b = torch.tensor([c[2] for c in comps], device=device, dtype=torch.float32)
        self.loc = torch.tensor([c[3] for c in comps], device=device, dtype=torch.float32)
        self.sc = torch.tensor([c[4] for c in comps], device=device, dtype=torch.float32)
        self.w /= self.w.sum()
        self.device = device

    def cdf(self, x: torch.Tensor):
        # Fallback to CPU-based computation
        x_cpu = x.detach().cpu().numpy()
        cdf_vals = np.zeros_like(x_cpu)
        for w, a, b, loc, scale in zip(self.w, self.a, self.b, self.loc, self.sc):
            dist = scipy_beta(a=a.item(), b=b.item(), loc=loc.item(), scale=scale.item())
            cdf_vals += w.item() * dist.cdf(x_cpu)
        return torch.tensor(cdf_vals, device=self.device, dtype=torch.float32)

    def ppf_bounds(self):
        lo = (self.loc).min().item()
        hi = (self.loc + self.sc).max().item()
        return lo, hi

# ─────────────── marginal catalogue ───────────────
def rand_def(device='cuda',
             PPF_GRID = 1_000):
    cat = random.choice(["normal","beta","beta_mix","expo","gauss_mix","beta_mix","student"])
    if cat=="normal":
        return dict(kind="torch", dist=Normal(torch.tensor(random.uniform(-1,1)),
                                              torch.tensor(random.uniform(1.5,2))))
    if cat=="beta":
        return dict(kind="torch", dist=Beta(torch.tensor(random.uniform(1,5)),
                                            torch.tensor(random.uniform(1,5))))
    if cat=="expo":
        return dict(kind="torch", dist=Exponential(torch.tensor(random.uniform(1,2))))
    if cat=="student":
        return dict(kind="torch", dist=StudentT(torch.tensor(random.randint(3,10))))
    if cat=="gauss_mix":
        comps = [(random.uniform(0.3,0.7),
                  random.uniform(-3,3),
                  random.uniform(0.5,1.5)) for _ in range(random.randint(1,3))]
        mix   = GaussianMix(comps)
    else:
        comps = [(random.uniform(0.3,0.7),
                  random.uniform(1,5),
                  random.uniform(1,5),
                  -5.0, 10.0) for _ in range(random.randint(1,3))]
        mix   = BetaMix(comps,device=device)
    # build interpolation grids
    u_grid = torch.linspace(0.001,0.999,PPF_GRID, device=device, dtype=torch.float32)
    lo,hi  = mix.ppf_bounds()
    x_grid = torch.linspace(lo,hi,PPF_GRID, device=device, dtype=torch.float32)
    #cdf_vals = mix.cdf(x_grid)
    return dict(kind="interp", u=u_grid, x=x_grid, lo=lo, hi=hi)


class CorpulaGenerator:
    def __init__(self, num_dims, device="cuda", ppf_grid=2_000):
        self.num_dims = num_dims
        self.device = device
        self.dtype = torch.float32
        self.ppf_grid = ppf_grid
        self.chol_base = torch.linalg.cholesky(
            rand_corr_batch(1, num_dims, device=device)[0]
        )
        self.specs = [rand_def(device=device, PPF_GRID=ppf_grid) for _ in range(num_dims)]


    def sample_inliers(self, num_inliers):
        """Sample clean multivariate normal data & transform to marginals"""
        z = torch.randn(num_inliers, self.num_dims, device=self.device, dtype=self.dtype)
        samples = (z @ self.chol_base.T)
        return samples

    def sample_outliers(self,
                        num_outliers,
                        method="perturb_u_values",
                        strength=0.2):
        """Generate outliers using perturb-u or disturbed covariance"""
        if method == "perturb_u_values":
            # 1. Sample from base MVN
            z = torch.randn(num_outliers, self.num_dims, device=self.device, dtype=self.dtype)
            samples = (z @ self.chol_base.T)

            # 2. Transform to uniform
            U = normal_cdf(samples)
            min_k = max(1, math.ceil(0.02 * self.num_dims))
            max_k = max(min_k, math.floor(0.2 * self.num_dims))

            # Ensure min_k < max_k + 1 to avoid invalid range
            if min_k >= max_k + 1:
                max_k = min_k # fallback: both min_k and max_k equal, will result in k_row = min_k

            k_row = torch.randint(min_k, max_k + 1, (num_outliers,), device=self.device)
            #k_row = torch.randint(5, 6, (num_outliers,), device=self.device) 
            perm = torch.rand(num_outliers, self.num_dims, device=self.device).argsort(dim=1)
            sel_mask = torch.arange(self.num_dims, device=self.device).expand(num_outliers, -1) < k_row.unsqueeze(1)
            mask = torch.zeros_like(U, dtype=torch.bool).scatter(1, perm, sel_mask)

            push0 = torch.rand_like(U) < 0.5
            z_mask, o_mask = mask & push0, mask & ~push0
            noise = torch.empty_like(U)
            noise[z_mask] = strength * torch.rand(z_mask.sum(), device=self.device) * 0.5
            noise[o_mask] = 1.0 - strength * torch.rand(o_mask.sum(), device=self.device) * 0.5
            U[mask] = noise[mask]

            samples = Normal(0, 1).icdf(U)
            return samples

        elif method == "disturb_covariance":
            d, device = self.num_dims, self.device
            base_L    = self.chol_base                       
            # 1) Random block lengths k  and start indices i0  –– now 1 … d inclusive 
            lowerbound = int(1+ d//3)
            upperbound = min(int(1+ 2 * d//3),d+1)
            if lowerbound == upperbound:
              upperbound += 1
            k      = torch.randint(lowerbound, upperbound, (num_outliers,), device=device)     # (B,) #int(1+ d//2)
            k_max  = int(k.max())

            # Vectorised start positions: i0[b] ∈ {0 … d-k[b]}
            # torch.randint can’t take a per-element “high”, so we synthesise i0 using rand():
            max_start = d - k                          # (B,)
            i0 = (torch.rand(num_outliers, device=device) * (max_start + 1)
                ).floor().long()                      # (B,)

            i1 = i0 + k                                # (B,)  exclusive end index
            L_rand_full = torch.linalg.cholesky(
                rand_corr_batch(num_outliers, k_max, identity=True, device=device)          # (B,k_max,k_max)
            )
            rows  = torch.arange(k_max, device=device).view(1, k_max, 1)     # (1,k_max,1)
            cols  = torch.arange(k_max, device=device).view(1, 1, k_max)     # (1,1,k_max)
            keep  = (rows < k.view(-1, 1, 1)) & (cols < k.view(-1, 1, 1))    # (B,k_max,k_max)

            L_rand_pad = torch.zeros(num_outliers, d, d, device=device)      # (B,d,d)
            L_rand_pad[:, :k_max, :k_max][keep] = L_rand_full[keep]
            L_mix = base_L.expand(num_outliers, -1, -1).clone()

            row = torch.arange(d, device=device).view(1, d)        # (1,d)
            mask_rows = (row >= i0.view(-1, 1)) & (row < i1.view(-1, 1))  # (B,d)
            col = torch.arange(d, device=device).view(1, 1, d)     # (1,1,d)
            mask = mask_rows.unsqueeze(-1) & (col < row.unsqueeze(-1) + 1)

            L_mix = torch.where(mask,
                                (1- strength) * L_mix + strength *L_rand_pad,
                                L_mix)
            L_mix.diagonal(dim1=-2, dim2=-1).clamp_(min=1e-6)
            return mvn_sample(L_mix, device=device)  
        else:
            raise ValueError(f"Unsupported outlier injection method: {method}")

    def _transform(self, samples):
        """
        Fully GPU-based transform.
        • Normal, Beta, Exp, StudentT → vectorized GPU code
        • Mixture (interp) → fast interpolation
        """
        U = normal_cdf(samples)
        eps = 1e-6
        U = torch.clamp(U, eps, 1-eps) #make sure U does not approch infty
        N, D = U.shape
        X = torch.empty_like(U)
        # Group columns by distribution type
        interp_cols, normal_cols, beta_cols, exp_cols, student_cols = [], [], [], [], []
        for d, spec in enumerate(self.specs):
            if spec["kind"] == "interp":
                interp_cols.append(d)
            elif spec["kind"] == "torch":
                dist = spec["dist"]
                if isinstance(dist, Normal):
                    normal_cols.append(d)
                elif isinstance(dist, Beta):
                    beta_cols.append(d)
                elif isinstance(dist, Exponential):
                    exp_cols.append(d)
                elif isinstance(dist, StudentT):
                    student_cols.append(d)
        # ---------- Interp mixture sampling ----------
        for d in interp_cols:
            u = U[:, d]
            spec = self.specs[d]
            grid_min = spec["u"][0]
            grid_max = spec["u"][-1]
            n_bins = len(spec["u"]) - 1
            bin_width = (grid_max - grid_min) / n_bins
            # Clamp u to grid range
            u_clamped = torch.clamp(u, grid_min, grid_max - 1e-6)
            # Fast approximate bin index (assumes linear CDF grid)
            idx = ((u_clamped - grid_min) / bin_width).long().clamp(0, n_bins - 1)
            idx = torch.clamp(idx, 1, len(spec["u"]) - 1)
            u_lo, u_hi = spec["u"][idx - 1], spec["u"][idx]
            x_lo, x_hi = spec["x"][idx - 1], spec["x"][idx]
            X[:, d] = x_lo + (u - u_lo) * (x_hi - x_lo) / (u_hi - u_lo)
        # ---------- Normal ----------
        if normal_cols:
            loc = torch.tensor([self.specs[d]["dist"].loc for d in normal_cols],
                            device=self.device).view(1, -1)
            scale = torch.tensor([self.specs[d]["dist"].scale for d in normal_cols],
                                device=self.device).view(1, -1)
            dist = Normal(loc, scale)
            X[:, normal_cols] = dist.icdf(U[:, normal_cols])
        # ---------- Beta ----------
        if beta_cols:
            a = torch.tensor([self.specs[d]["dist"].concentration1 for d in beta_cols],
                            device=self.device).view(1, -1)
            b = torch.tensor([self.specs[d]["dist"].concentration0 for d in beta_cols],
                            device=self.device).view(1, -1)
            X[:, beta_cols] = beta_icdf(U[:, beta_cols], a, b)
        # ---------- Exponential ----------
        if exp_cols:
            rate = torch.tensor([self.specs[d]["dist"].rate for d in exp_cols],
                                device=self.device).view(1, -1)
            X[:, exp_cols] = exp_icdf(U[:, exp_cols], rate)
        # ---------- StudentT ----------
        if student_cols:
            df = torch.tensor([self.specs[d]["dist"].df for d in student_cols],
                            device=self.device).view(1, -1)
            X[:, student_cols] = student_t_icdf(U[:, student_cols], df)
        return X



    @torch.no_grad()
    def draw_batched_data(self,
                          num_inliers,
                          num_local_anomalies):
        METHOD = random.choice(["disturb_covariance","perturb_u_values"])  # or add "perturb_u_values"
        STRENGTH = random.uniform(0.2,0.4) if METHOD == 'perturb_u_values' else random.uniform(0.97,0.99)
        inliers = self.sample_inliers(num_inliers)

        outliers =  self.sample_outliers(num_local_anomalies, method=METHOD, strength=STRENGTH)
        combined = torch.cat([inliers, outliers], dim=0)

        X_combined = self._transform(combined)
        X_inliers = X_combined[:num_inliers]
        X_outliers = X_combined[num_inliers:]

        return X_inliers, X_outliers


def make_corpula(device, 
                 max_feature_dim=100,
                 min_feature_dim = 2,
                 dim = None):
    if dim is not None:
        num_features = dim
    else:
        num_features = np.random.randint(min_feature_dim, max_feature_dim)
    return CorpulaGenerator(num_dims=num_features, device=device)



# ───────────────────────── main / smoke-test ──────────────────────────
if __name__ == "__main__":
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics  import roc_auc_score
    from contextlib import contextmanager
    import time, torch, random
    import pandas as pd

    @contextmanager
    def timer(name):
        t0 = time.perf_counter()
        yield
        print(f"{name}: {time.perf_counter()-t0:6.3f} s")

    #random.seed(0);  np.random.seed(0);  torch.manual_seed(0)
    # ------- choose GPU if available ----------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning on {DEVICE.upper()}")
    NUM_DIMS      = np.random.randint(2, 100)         # random feature count
    INLIERS       = 1_000
    OUTLIERS      = 100
    gen = CorpulaGenerator(num_dims=NUM_DIMS, device=DEVICE)
    with timer("Synthetic data draw   "):
        X_in, X_out = gen.draw_batched_data(INLIERS, OUTLIERS)
    X   = torch.cat([X_in, X_out]).cpu().numpy()
    y   = np.concatenate([np.zeros(INLIERS), np.ones(OUTLIERS)])  # 1 ⇒ outlier
    print(X)
    contamination = OUTLIERS / (INLIERS + OUTLIERS)
    with timer("LOF fit+score         "):
        lof = LocalOutlierFactor(n_neighbors=20,
                                 contamination=contamination)
        _   = lof.fit_predict(X)                      # triggers fit
        scores = -lof.negative_outlier_factor_       # higher ⇒ more anomalous
    auc = roc_auc_score(y, scores)
    print(f"ROC-AUC (LOF vs truth): {auc:.4f}")

     # Get number of dimensions excluding is_outlier column
    NUM_SAMPLES = INLIERS + OUTLIERS

    # Create column names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["is_outlier"] = y

    print(f"\nGenerated {NUM_SAMPLES} synthetic data points with {NUM_DIMS} dimensions.")
    print("\nFirst 5 rows of the synthetic dataset:")
    print(df.head())

    print("\nDescriptive statistics of the synthetic dataset:")
    print(df.describe())

    # Verify if features are within [-5, 5] range
    feature_cols = [col for col in df.columns if col != 'is_outlier']
    min_vals = df[feature_cols].min()
    max_vals = df[feature_cols].max()
    print("\nMinimum values per dimension:", min_vals.tolist())
    print("Maximum values per dimension:", max_vals.tolist())

    """Display comprehensive analysis of the synthetic dataset."""



    if (min_vals >= -5).all() and (max_vals <= 5).all():
        print("All generated features are within the [-5, 5] range.")
    else:
        print("WARNING: Some generated features might be outside the [-5, 5] range.")

    # Add scatter plots for first 5 features
    first_5_features = feature_cols[:5]
    if len(first_5_features) >= 2:
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))

        for i in range(5):
            for j in range(5):
                feature1 = first_5_features[i]
                feature2 = first_5_features[j]

                # Clean data for plotting
                clean_data = df[[feature1, feature2, 'is_outlier']].replace([np.inf, -np.inf], np.nan).dropna()

                # If on diagonal, plot histogram
                if i == j:
                    try:
                        # Fix: Convert series to proper format for histplot
                        data_to_plot = clean_data[feature1].values
                        sns.histplot(data=data_to_plot, kde=True, ax=axes[i,j])
                        axes[i,j].set_title(f'{feature1}')
                    except Exception as e:
                        print(f"Warning: Could not create histogram for {feature1}: {e}")
                        axes[i,j].text(0.5, 0.5, 'Plot Error', ha='center', va='center')
                else:

                      # Plot non-outliers
                      normal_mask = clean_data['is_outlier'] == 0.0
                      if normal_mask.any():
                          axes[i,j].scatter(clean_data[normal_mask][feature1],
                                          clean_data[normal_mask][feature2],
                                          c='blue', alpha=0.5, s=10)

                      # Plot outliers
                      outlier_mask = clean_data['is_outlier'] ==1.0
                      if outlier_mask.any():
                          axes[i,j].scatter(clean_data[outlier_mask][feature1],
                                          clean_data[outlier_mask][feature2],
                                          c='red', alpha=0.7, s=20)

                # Add labels
                if i == 4:  # Only bottom row gets x labels
                    axes[i,j].set_xlabel(feature1)
                else:
                    axes[i,j].set_xlabel('')

                if j == 0:  # Only leftmost column gets y labels
                    axes[i,j].set_ylabel(feature2)
                else:
                    axes[i,j].set_ylabel('')

                # Remove title except for diagonal
                if i != j:
                    axes[i,j].set_title('')

                # Set reasonable axis limits
                if not i == j:
                    try:
                        xlim = np.percentile(clean_data[feature1].replace([np.inf, -np.inf], np.nan).dropna(), [1, 99])
                        ylim = np.percentile(clean_data[feature2].replace([np.inf, -np.inf], np.nan).dropna(), [1, 99])
                        axes[i,j].set_xlim(xlim)
                        axes[i,j].set_ylim(ylim)
                    except Exception:
                        pass

        # Add single legend for the entire figure
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='blue', markersize=10,
                                    alpha=0.5, label='Normal'),
                          plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='red', markersize=10,
                                    alpha=0.7, label='Outlier')]
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))

        plt.tight_layout()
        plt.suptitle('Pairwise Scatter Plots of First 5 Features\n(Outliers in Red)', y=1.02)
        plt.show()
