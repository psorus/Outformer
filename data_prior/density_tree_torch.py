import torch
import random

def random_split_integer(total, n_parts, min_val=2, max_val=10):
    if n_parts == 1:
        return [total]
    min_val = max(min_val, 1)
    min_val = min(min_val, total)
    max_val = min(max_val, total)
    if n_parts * min_val > total:
        # Not possible, so set all to min_val except last
        vals = [min_val] * (n_parts - 1)
        vals.append(total - min_val * (n_parts - 1))
        return vals
    if n_parts * max_val < total:
        # Not possible, so set all to max_val except last
        vals = [max_val] * (n_parts - 1)
        vals.append(total - max_val * (n_parts - 1))
        return vals
    assert n_parts * min_val <= total <= n_parts * max_val, "Impossible split after adjustment!"
    cuts = sorted(random.sample(range(1, total), n_parts - 1))
    vals = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, n_parts - 1)] + [total - cuts[-1]]
    if all(min_val <= v <= max_val for v in vals):
        return vals
    # Fallback: assign min_val to most, adjust last
    vals = [min_val] * (n_parts - 1)
    vals.append(total - min_val * (n_parts - 1))
    return vals


def random_feature_definitions_with_sum(n_numeric=20, target_dim=100, n_categorical=10, min_cat=2, max_cat=10, device='cpu'):
    numeric_features = [f"num_{i}" for i in range(n_numeric)]
    cat_total = target_dim - n_numeric
    numeric_bounds = {f: (random.uniform(-20, 0), random.uniform(1, 20)) for f in numeric_features}
    numeric_bounds = {f: (min(a, b), max(a, b)) for f, (a, b) in numeric_bounds.items()}

    if cat_total <= 0:
        return numeric_features, [], numeric_bounds, {}

    n_categorical = max(1, min(n_categorical, cat_total))
    min_cat = min(max(min_cat, 1), cat_total)
    max_cat = min(max_cat, cat_total)

    # Adjust n_categorical if constraints are not possible
    if n_categorical * min_cat > cat_total:
        n_categorical = cat_total // min_cat
        n_categorical = max(1, n_categorical)
    if n_categorical * max_cat < cat_total:
        n_categorical = cat_total // max_cat
        n_categorical = max(1, n_categorical)
    if n_categorical < 1:
        n_categorical = 1
    if min_cat > max_cat:
        max_cat = min_cat

    # Edge case: fallback to a single categorical feature
    if n_categorical * min_cat > cat_total or n_categorical * max_cat < cat_total:
        n_categorical = 1
        min_cat = max_cat = cat_total

    cat_sizes = random_split_integer(cat_total, n_categorical, min_cat, max_cat)
    categorical_features = [f"cat_{i}" for i in range(n_categorical)]
    cat_options = {f: [f"{f}_val_{i}" for i in range(size)] for f, size in zip(categorical_features, cat_sizes)}
    return numeric_features, categorical_features, numeric_bounds, cat_options


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
        self.probability = None
        self.leaf_id = None
        self.is_outlier = False

class SyntheticInlierOutlierGenerator:
    def __init__(self, 
                 n_numeric=20,
                 target_dim=100,
                 n_categorical=10, 
                 min_cat=2, 
                 max_cat=10,
                 tree_depth=3,
                 prune_prob=0.65,
                 device='cpu',
                 outlier_leaf_prob=0.005,
                 inlier_range=(3, 10)):
        self.device = device
        self.n_numeric = n_numeric
        self.target_dim = target_dim
        self.n_categorical = n_categorical
        self.min_cat = min_cat
        self.max_cat = max_cat
        self.tree_depth = tree_depth
        self.outlier_leaf_prob = outlier_leaf_prob
        self.inlier_range = inlier_range

        self.numeric_features, self.categorical_features, self.numeric_bounds, self.cat_options = random_feature_definitions_with_sum(
            n_numeric=self.n_numeric,
            target_dim=self.target_dim,
            n_categorical=self.n_categorical,
            min_cat=self.min_cat,
            max_cat=self.max_cat,
            device=self.device
        )
        leaf_id_counter = iter(range(10000))
        self.tree = self.build_isolation_tree(
            self.numeric_bounds, self.cat_options, depth=0, max_depth=self.tree_depth, leaf_id_counter=leaf_id_counter)
        self.leaves = self.assign_leaf_probabilities_fixed_outlier(
            self.tree, outlier_prob=self.outlier_leaf_prob, inlier_range=self.inlier_range)
        while len(self.leaves) < 4:
            self.tree = self.build_isolation_tree(self.numeric_bounds, self.cat_options, depth=0, max_depth=self.tree_depth, leaf_id_counter=leaf_id_counter)
            self.leaves = self.assign_leaf_probabilities_fixed_outlier(
                self.tree, outlier_prob=self.outlier_leaf_prob, inlier_range=self.inlier_range)
        #self.prune_tree(prune_fraction=prune_prob)
        
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
            node.left = self.build_isolation_tree(left_bounds, cat_options, depth+1, max_depth, leaf_id_counter)
            node.right = self.build_isolation_tree(right_bounds, cat_options, depth+1, max_depth, leaf_id_counter)
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
            node.left = self.build_isolation_tree(numeric_bounds, left_options, depth+1, max_depth, leaf_id_counter)
            node.right = self.build_isolation_tree(numeric_bounds, right_options, depth+1, max_depth, leaf_id_counter)
        else:
            node.is_leaf = True
            node.leaf_id = next(leaf_id_counter)
        return node

    def prune_tree(self, prune_fraction=0.4):
        # Only prune inlier nodes!
        inlier_internal_nodes = []
        def traverse(node, parent=None, side=None):
            if node is None:
                return
            if parent is not None:
                # Only add internal nodes whose subtrees are entirely inlier
                leaves = []
                def gather_leaves(n):
                    if n is None:
                        return
                    if n.is_leaf:
                        leaves.append(n)
                    else:
                        if n.left: gather_leaves(n.left)
                        if n.right: gather_leaves(n.right)
                gather_leaves(node)
                if all(not leaf.is_outlier for leaf in leaves):
                    inlier_internal_nodes.append((node, parent, side))
            if node.left: traverse(node.left, node, 'left')
            if node.right: traverse(node.right, node, 'right')
        traverse(self.tree)

        n_prune = int(prune_fraction * len(inlier_internal_nodes))
        if n_prune == 0:
            return
        to_prune = random.sample(inlier_internal_nodes, n_prune)
        pruned_ids = set()
        for node, parent, side in to_prune:
            if id(parent) in pruned_ids:
                continue
            if side == 'left':
                parent.left = None
            elif side == 'right':
                parent.right = None
            pruned_ids.add(id(node))

        # After pruning, reassign leaf probabilities with fixed outlier leaf prob!
        self.leaves = self.assign_leaf_probabilities_fixed_outlier(
            self.tree, outlier_prob=self.outlier_leaf_prob, inlier_range=self.inlier_range)

    def assign_leaf_probabilities_fixed_outlier(self, tree, outlier_prob=0.005, inlier_range=(3, 10)):
        leaves = []
        def traverse(node):
            if node is None:
                return
            if node.is_leaf:
                leaves.append(node)
            else:
                traverse(node.left)
                traverse(node.right)
        traverse(tree)
        n = len(leaves)
        min_inlier, max_inlier = inlier_range
        min_inlier = max(1, min_inlier)
        max_inlier = min(max_inlier, n)

        # Randomly choose n_inliers in [min_inlier, max_inlier]
        n_inliers = random.randint(min_inlier, max_inlier)

        if n_inliers == n:
            inlier_indices = set(range(n))
            n_outliers = 0
        else:
            inlier_indices = set(random.sample(range(n), n_inliers))
            n_outliers = n - n_inliers

        inlier_mass = 1.0 - n_outliers * outlier_prob

        # Assign outlier/inlier labels
        for i, leaf in enumerate(leaves):
            leaf.is_outlier = (i not in inlier_indices)
        for i, leaf in enumerate(leaves):
            if leaf.is_outlier:
                leaf.probability = outlier_prob
            else:
                leaf.probability = inlier_mass / n_inliers if n_inliers > 0 else 0.0

        # **Ensure at least one outlier exists**
        if not any(leaf.is_outlier for leaf in leaves):
            # Pick the leaf with the smallest probability and make it the outlier
            min_prob_leaf = min(leaves, key=lambda leaf: leaf.probability)
            min_prob_leaf.is_outlier = True
            min_prob_leaf.probability = outlier_prob
            # Renormalize inlier probabilities
            inlier_leaves = [leaf for leaf in leaves if not leaf.is_outlier]
            total_inlier_mass = 1.0 - sum(leaf.probability for leaf in leaves if leaf.is_outlier)
            n_inlier = len(inlier_leaves)
            for leaf in inlier_leaves:
                leaf.probability = total_inlier_mass / n_inlier if n_inlier > 0 else 0.0

        return leaves

    def sample_data(self, n_samples):
        numeric_features = self.numeric_features
        categorical_features = self.categorical_features
        cat_options = self.cat_options
        leaves = self.leaves
        device = self.device

        probs = torch.tensor([leaf.probability for leaf in leaves], device=device)
        probs = probs / probs.sum()
        idxs = torch.multinomial(probs, n_samples, replacement=True)
        idxs_np = idxs.cpu().numpy()  # For fast indexing below

        # Prepare arrays for all numeric/categorical bounds/options per leaf
        # Numeric bounds
        low = torch.tensor(
            [[leaves[leaf_idx].numeric_bounds[feat][0] for feat in numeric_features] for leaf_idx in idxs_np], 
            device=device
        )
        high = torch.tensor(
            [[leaves[leaf_idx].numeric_bounds[feat][1] for feat in numeric_features] for leaf_idx in idxs_np], 
            device=device
        )

        # Categorical option sizes per feature
        cat_sizes = [len(cat_options[feat]) for feat in categorical_features]
        total_cat_len = sum(cat_sizes)
        total_feat_len = len(numeric_features) + total_cat_len
        assert total_feat_len == self.target_dim, f"Total feature length is {total_feat_len}, expected {self.target_dim}"

        # Numeric: Sample all at once!
        X_numeric = low + (high - low) * torch.rand((n_samples, len(numeric_features)), device=device)

        # Categorical: One-hot
        X_categorical = torch.zeros((n_samples, total_cat_len), device=device)
        offset = 0
        for k, feat in enumerate(categorical_features):
            options = cat_options[feat]
            n_cat = len(options)
            # For each sample, get leaf and options (always the same, but just in case)
            cat_options_this_leaf = [leaves[leaf_idx].cat_options[feat] for leaf_idx in idxs_np]
            # All options should be the same length per call (if the tree doesn't do weird splits)
            # To be robust, get the size for each sample (but assume always == n_cat)
            # Sample indices for each row
            cat_idx = torch.randint(0, n_cat, (n_samples,), device=device)
            X_categorical[torch.arange(n_samples, device=device), offset + cat_idx] = 1.0
            offset += n_cat

        # Merge
        X = torch.cat([X_numeric, X_categorical], dim=1)

        # y label: outlier or not
        y = torch.tensor([1 if leaves[leaf_idx].is_outlier else 0 for leaf_idx in idxs_np], device=device, dtype=torch.long)

        return X, y

    def generate_synthetic_inlier_outlier_data(self, n_samples):
        return self.sample_data(n_samples)
    
    def print_leaf_probabilities(self):
        print("Leaf nodes and their probabilities:")
        for i, leaf in enumerate(self.leaves):
            print(f"Leaf {i:3d} (ID={leaf.leaf_id:4d}): "
                f"prob={leaf.probability:.6f}, "
                f"is_outlier={int(leaf.is_outlier)}")
    
    def draw_batched_data(self, num_inliers, num_anomalies):
        #self.print_leaf_probabilities()
        X_inlier, X_outlier = [], []
        batch_size = max(num_inliers, num_anomalies, int(num_inliers * 2.5))
        inlier_count = 0
        outlier_count = 0
        #count = 0
        while inlier_count < num_inliers or outlier_count < num_anomalies:
            #count += 1
            #print(inlier_count, outlier_count)
            X, y = self.generate_synthetic_inlier_outlier_data(n_samples=batch_size)
            is_inlier = (y == 0)
            is_outlier = (y == 1)
            if is_inlier.any():
                X_inlier.append(X[is_inlier])
                inlier_count += X[is_inlier].shape[0]
            if is_outlier.any():
                X_outlier.append(X[is_outlier])
                outlier_count += X[is_outlier].shape[0]
        X_inlier = torch.cat(X_inlier, dim=0)[:num_inliers]
        X_outlier = torch.cat(X_outlier, dim=0)[:num_anomalies]
        return X_inlier, X_outlier

    
    
    
def make_density(max_feature_dim: int,
                 n_numeric: int,
                 n_categorial: int,
                 min_cat,
                 max_cat,
                 tree_depth,
                 prune_prob,
                 device,
                 ):
    gen = SyntheticInlierOutlierGenerator(
        n_numeric=n_numeric,
        target_dim=max_feature_dim,
        n_categorical=n_categorial,
        min_cat=min_cat,
        max_cat=max_cat,
        tree_depth=random.randint(tree_depth, tree_depth+1),
        prune_prob = prune_prob,
        device=device,
        outlier_leaf_prob=0.01,
        inlier_range=(1,3)
    )
    #gen.print_leaf_probabilities()
    return gen
    