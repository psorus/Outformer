import numpy as np
import random
import torch
from itertools import count

class IsolationTreeNode:
    def __init__(self, depth, max_depth, numeric_bounds, cat_options):
        self.depth = depth
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.numeric_bounds = {k: v for k, v in numeric_bounds.items()}
        self.cat_options = {k: list(v) for k, v in cat_options.items()}
        self.is_leaf = False
        self.leaf_id = None
        self.is_outlier = False

class DensityTreeSampler:
    def __init__(self, n_numeric=3, n_categorical=2, n_categories_range=(2, 4), tree_depth=5, random_seed=42, min_leaves=5, max_leaves=20 ,device = 'cuda'):
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.n_categories_range = n_categories_range
        self.tree_depth = tree_depth
        self.random_seed = random_seed
        self.min_leaves = min_leaves
        self.max_leaves = max_leaves
        self.device = device

        self.numeric_features, self.categorical_features, self.numeric_bounds, self.cat_options = self.random_feature_definitions()
        self.leaf_id_counter = count(0)
        self.tree = None
        self.leaves = None
        self.cat_value_lists = None
        self.cat_value_to_idx = None

        self._build_tree()

    def random_feature_definitions(self):
        random.seed(self.random_seed)
        numeric_features = [f"num_{i}" for i in range(self.n_numeric)]
        categorical_features = [f"cat_{i}" for i in range(self.n_categorical)]
        numeric_bounds = {f: (random.uniform(-20, 0), random.uniform(1, 20)) for f in numeric_features}
        numeric_bounds = {f: (min(a, b), max(a, b)) for f, (a, b) in numeric_bounds.items()}
        cat_options = {f: [f"{f}_val_{i}" for i in range(random.randint(*self.n_categories_range))] for f in categorical_features}
        return numeric_features, categorical_features, numeric_bounds, cat_options

    def build_isolation_tree(self, numeric_bounds, cat_options, depth, max_depth, leaf_id_counter):
        node = IsolationTreeNode(depth, max_depth, numeric_bounds, cat_options)
        if depth == max_depth:
            node.is_leaf = True
            node.leaf_id = next(leaf_id_counter)
            return node

        possible_features = []
        if numeric_bounds:
            possible_features.append('numeric')
        if cat_options:
            possible_features.append('categorical')
        if not possible_features:
            node.is_leaf = True
            node.leaf_id = next(leaf_id_counter)
            return node

        feature_type = random.choice(possible_features)
        if feature_type == 'numeric':
            feature = random.choice(list(numeric_bounds.keys()))
            low, high = numeric_bounds[feature]
            if high - low < 1e-6:
                node.is_leaf = True
                node.leaf_id = next(leaf_id_counter)
                return node
            split_val = random.uniform(low, high)
            node.split_feature = feature
            node.split_value = split_val
            left_bounds = numeric_bounds.copy()
            right_bounds = numeric_bounds.copy()
            left_bounds[feature] = (low, split_val)
            right_bounds[feature] = (split_val, high)
            node.left = self.build_isolation_tree(left_bounds, cat_options, depth + 1, max_depth, leaf_id_counter)
            node.right = self.build_isolation_tree(right_bounds, cat_options, depth + 1, max_depth, leaf_id_counter)
        elif feature_type == 'categorical':
            feature = random.choice(list(cat_options.keys()))
            categories = cat_options[feature]
            if len(categories) <= 1:
                node.is_leaf = True
                node.leaf_id = next(leaf_id_counter)
                return node
            split_cat = random.choice(categories)
            node.split_feature = feature
            node.split_value = split_cat
            left_options = cat_options.copy()
            right_options = cat_options.copy()
            left_options[feature] = [c for c in categories if c != split_cat]
            right_options[feature] = [split_cat]
            node.left = self.build_isolation_tree(numeric_bounds, left_options, depth + 1, max_depth, leaf_id_counter)
            node.right = self.build_isolation_tree(numeric_bounds, right_options, depth + 1, max_depth, leaf_id_counter)
        else:
            node.is_leaf = True
            node.leaf_id = next(leaf_id_counter)
        return node

    def get_non_leaf_nodes(self, tree):
        nodes = []
        def traverse(node):
            if not node.is_leaf:
                nodes.append(node)
                traverse(node.left)
                traverse(node.right)
        traverse(tree)
        return nodes

    def get_leaves(self, tree):
        leaves = []
        def traverse(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                traverse(node.left)
                traverse(node.right)
        traverse(tree)
        return leaves

    def prune_random_nodes(self, tree, n_prune=1, leaf_id_counter=None, random_seed=None, min_leaves=4):
        if random_seed is not None:
            random.seed(random_seed)
        nodes = self.get_non_leaf_nodes(tree)
        leaves = self.get_leaves(tree)
        n_leaves_now = len(leaves)
        if len(nodes) <= 1 or n_prune < 1 or n_leaves_now <= min_leaves:
            return
        max_prune = n_leaves_now - min_leaves
        if max_prune < 1:
            return
        n_prune_actual = min(n_prune, max_prune, len(nodes) - 1)  # don't prune root
        if n_prune_actual < 1:
            return
        chosen_nodes = random.sample(nodes[1:], n_prune_actual)
        for chosen in chosen_nodes:
            chosen.left = None
            chosen.right = None
            chosen.is_leaf = True
            if leaf_id_counter is not None:
                chosen.leaf_id = next(leaf_id_counter)
            else:
                chosen.leaf_id = 10000 + random.randint(0, 9999)

    def calculate_leaf_area(self, leaf):
        numeric_volume = 1.0
        for feat in self.numeric_features:
            low, high = leaf.numeric_bounds[feat]
            numeric_volume *= (high - low)
        categorical_volume = 1
        for feat in self.categorical_features:
            categorical_volume *= len(leaf.cat_options[feat])
        return numeric_volume * categorical_volume

    def mark_largest_area_leaves_as_outliers(self, tree, n_inlier_leaves=2):
        leaves = self.get_leaves(tree)
        leaf_areas = [
            (leaf, self.calculate_leaf_area(leaf))
            for leaf in leaves
        ]
        leaf_areas_sorted = sorted(leaf_areas, key=lambda x: x[1])
        for leaf, _ in leaf_areas_sorted[:n_inlier_leaves]:
            leaf.is_outlier = False
        for leaf, _ in leaf_areas_sorted[n_inlier_leaves:]:
            leaf.is_outlier = True
        return leaves

    def _build_tree(self):
        while True:
            self.leaf_id_counter = iter(range(5000000))
            tree = self.build_isolation_tree(
                self.numeric_bounds, self.cat_options, depth=0,
                max_depth=self.tree_depth, leaf_id_counter=self.leaf_id_counter
            )
            leaves = self.get_leaves(tree)
            n_leaf_nodes = len(leaves)
            if n_leaf_nodes < self.min_leaves or n_leaf_nodes > self.max_leaves:
                # Prune random nodes to control number of leaves
                n_nodes = np.max([1, np.random.randint(0, max(1, int(n_leaf_nodes/2)))]) if n_leaf_nodes > 1 else 1
                self.prune_random_nodes(tree, n_nodes, leaf_id_counter=self.leaf_id_counter) #, random_seed=self.random_seed)
                leaves = self.get_leaves(tree)
                n_leaf_nodes = len(leaves)
            if self.min_leaves <= n_leaf_nodes <= self.max_leaves:
                self.tree = tree
                self.leaves = leaves
                break
        self.mark_largest_area_leaves_as_outliers(self.tree)

    def draw_batched_data(self, num_inliers, num_local_anomalies):
        n_inlier_samples = num_inliers
        n_outlier_samples = num_local_anomalies
        leaves = self.leaves
        inlier_leaves = [leaf for leaf in leaves if not getattr(leaf, 'is_outlier', False)]
        outlier_leaves = [leaf for leaf in leaves if getattr(leaf, 'is_outlier', False)]
        n_inlier_leaves = len(inlier_leaves)
        n_outlier_leaves = len(outlier_leaves)
        per_inlier_leaf = (n_inlier_samples // n_inlier_leaves + 1) if n_inlier_leaves > 0 else 0
        per_outlier_leaf = (n_outlier_samples // n_outlier_leaves + 1) if n_outlier_leaves > 0 else 0

        # Gather all possible categorical values for encoding
        if self.categorical_features:
            cat_value_sets = {f: set() for f in self.categorical_features}
            for leaf in leaves:
                for f in self.categorical_features:
                    cat_value_sets[f].update(leaf.cat_options[f])
            cat_value_lists = {f: sorted(list(s)) for f, s in cat_value_sets.items()}
            cat_value_to_idx = {f: {v: i for i, v in enumerate(vals)} for f, vals in cat_value_lists.items()}
        else:
            cat_value_lists = {}
            cat_value_to_idx = {}

        def fast_sample_categorical(leaf_list, per_leaf):
            if not self.categorical_features or not leaf_list:
                return np.empty((0, 0), dtype=int)
            n_leaves = len(leaf_list)
            n_cat = len(self.categorical_features)
            max_cat_options = [max(len(leaf.cat_options[f]) for leaf in leaf_list) for f in self.categorical_features]
            options_tensors = []
            for fi, f in enumerate(self.categorical_features):
                options = []
                for leaf in leaf_list:
                    opts = leaf.cat_options[f]
                    idxs = [cat_value_to_idx[f][val] for val in opts]
                    idxs += [-1] * (max_cat_options[fi] - len(idxs))
                    options.append(idxs)
                options_tensors.append(torch.tensor(options, device=self.device))
            X_cat = torch.empty((n_leaves * per_leaf, n_cat), dtype=torch.long, device=self.device)
            for fi, f in enumerate(self.categorical_features):
                opts_tensor = options_tensors[fi]
                num_opts = torch.tensor([len(leaf.cat_options[f]) for leaf in leaf_list], device=self.device)
                idxs = torch.cat([
                    torch.randint(0, num_opts[li], (per_leaf,), device=self.device) for li in range(n_leaves)
                ])
                gather_base = (torch.arange(n_leaves, device=self.device).repeat_interleave(per_leaf) * max_cat_options[fi]).long()
                idxs_flat = gather_base + idxs
                opts_tensor_flat = opts_tensor.flatten(0, 1)
                chosen = opts_tensor_flat[idxs_flat]
                X_cat[:, fi] = chosen
            return X_cat.cpu().numpy()

        def sample_numeric(leaf_list, per_leaf):
            if not leaf_list:
                return np.empty((0, len(self.numeric_features)))
            n_leaves = len(leaf_list)
            n_num = len(self.numeric_features)
            bounds = np.array([[leaf.numeric_bounds[f] for f in self.numeric_features] for leaf in leaf_list])
            lows = torch.tensor(bounds[:, :, 0], device=self.device)
            highs = torch.tensor(bounds[:, :, 1], device=self.device)
            unif = torch.rand((n_leaves, n_num, per_leaf), device=self.device)
            nums = lows.unsqueeze(2) + unif * (highs - lows).unsqueeze(2)
            nums = nums.permute(0, 2, 1).reshape(-1, n_num).cpu().numpy()
            return nums

        # Inliers
        X_num_in = sample_numeric(inlier_leaves, per_inlier_leaf)[:n_inlier_samples, :]
        X_cat_in = fast_sample_categorical(inlier_leaves, per_inlier_leaf)[:n_inlier_samples, :]

        # Outliers
        X_num_out = sample_numeric(outlier_leaves, per_outlier_leaf)[:n_outlier_samples, :]
        X_cat_out = fast_sample_categorical(outlier_leaves, per_outlier_leaf)[:n_outlier_samples, :]

        # One-hot encoding if necessary
        def onehot_cat(X_cat):
            if self.categorical_features and X_cat.size:
                onehot_columns = []
                for i, f in enumerate(self.categorical_features):
                    n_vals = len(cat_value_lists[f])
                    mask = (X_cat[:, i] >= 0)
                    vals = X_cat[:, i].copy()
                    vals[~mask] = 0
                    onehot = np.eye(n_vals)[vals]
                    onehot[~mask] = 0
                    onehot_columns.append(onehot)
                return np.concatenate(onehot_columns, axis=1)
            else:
                return np.empty((X_cat.shape[0], 0))
        
        if not self.categorical_features:
            x_inliers = torch.from_numpy(X_num_in).float().to(self.device)
            x_outliers = torch.from_numpy(X_num_out).float().to(self.device)
            return x_inliers, x_outliers
        
        X_cat_in_onehot = onehot_cat(X_cat_in)
        X_cat_out_onehot = onehot_cat(X_cat_out)

        # Final arrays
        x_inliers = np.concatenate([X_num_in, X_cat_in_onehot], axis=1)
        x_outliers = np.concatenate([X_num_out, X_cat_out_onehot], axis=1)

        # Convert to tensors if you want, or keep as numpy
        x_inliers = torch.from_numpy(x_inliers).float().to(self.device)
        x_outliers = torch.from_numpy(x_outliers).float().to(self.device)
        return x_inliers, x_outliers



def make_density(n_numeric: int,
                 n_categorical: int,
                 tree_depth: int,
                 random_seed: int = 42,
                 n_categories_range =(2, 4),
                 min_leaves=5,
                 max_leaves=20,
                 device= 'cuda'):
    sampler = DensityTreeSampler(
                                n_numeric,
                                n_categorical,   
                                n_categories_range,
                                tree_depth=tree_depth,
                                random_seed=random_seed,
                                min_leaves=min_leaves,
                                max_leaves=max_leaves,
                                device=device,
    )
    return sampler