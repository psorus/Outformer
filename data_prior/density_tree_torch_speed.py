import numpy as np
import random
import json
import pprint
import torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm

# --- Feature definition ---
def random_feature_definitions(n_numeric=3, n_categorical=2, n_categories_range=(2,4)):
    numeric_features = [f"num_{i}" for i in range(n_numeric)]
    categorical_features = [f"cat_{i}" for i in range(n_categorical)]
    numeric_bounds = {f: (random.uniform(-20, 0), random.uniform(1, 20)) for f in numeric_features}
    numeric_bounds = {f: (min(a, b), max(a, b)) for f, (a, b) in numeric_bounds.items()}
    cat_options = {f: [f"{f}_val_{i}" for i in range(random.randint(*n_categories_range))] for f in categorical_features}
    return numeric_features, categorical_features, numeric_bounds, cat_options

# --- Tree node class ---
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
        self.is_outlier = False  # Add this attribute

# --- Tree builder ---
def build_isolation_tree(numeric_bounds, cat_options, depth, max_depth, leaf_id_counter):
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
        node.left = build_isolation_tree(left_bounds, cat_options, depth+1, max_depth, leaf_id_counter)
        node.right = build_isolation_tree(right_bounds, cat_options, depth+1, max_depth, leaf_id_counter)
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
        node.left = build_isolation_tree(numeric_bounds, left_options, depth+1, max_depth, leaf_id_counter)
        node.right = build_isolation_tree(numeric_bounds, right_options, depth+1, max_depth, leaf_id_counter)
    else:
        node.is_leaf = True
        node.leaf_id = next(leaf_id_counter)
    return node

# --- Pruning ---
def get_non_leaf_nodes(tree):
    nodes = []
    def traverse(node):
        if not node.is_leaf:
            nodes.append(node)
            traverse(node.left)
            traverse(node.right)
    traverse(tree)
    return nodes

def get_leaves(tree):
    leaves = []
    def traverse(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            traverse(node.left)
            traverse(node.right)
    traverse(tree)
    return leaves

def prune_random_nodes(tree, n_prune=1, leaf_id_counter=None, random_seed=None, min_leaves=4):
    if random_seed is not None:
        random.seed(random_seed)
    nodes = get_non_leaf_nodes(tree)
    leaves = get_leaves(tree)
    n_leaves_now = len(leaves)
    if len(nodes) <= 1 or n_prune < 1 or n_leaves_now <= min_leaves:
        return
    max_prune = n_leaves_now - min_leaves
    if max_prune < 1:
        return
    n_prune_actual = min(n_prune, max_prune, len(nodes) - 1)  # don't prune root
    if n_prune_actual < 1:
        return
    chosen_nodes = random.sample(nodes[1:], n_prune_actual)  # skip root
    for chosen in chosen_nodes:
        chosen.left = None
        chosen.right = None
        chosen.is_leaf = True
        if leaf_id_counter is not None:
            chosen.leaf_id = next(leaf_id_counter)
        else:
            chosen.leaf_id = 10000 + random.randint(0, 9999)

# --- Area calculation ---
def calculate_leaf_area(leaf, numeric_features, categorical_features):
    numeric_volume = 1.0
    for feat in numeric_features:
        low, high = leaf.numeric_bounds[feat]
        numeric_volume *= (high - low)
    categorical_volume = 1
    for feat in categorical_features:
        categorical_volume *= len(leaf.cat_options[feat])
    return numeric_volume * categorical_volume

# --- Tree structure export with normalized area ---
def get_leaf_areas(node, numeric_features, categorical_features):
    areas = []
    def traverse(n):
        if n.is_leaf:
            area = calculate_leaf_area(n, numeric_features, categorical_features)
            areas.append((n, area))
        else:
            traverse(n.left)
            traverse(n.right)
    traverse(node)
    return areas

def get_tree_structure_with_areas(node, numeric_features, categorical_features, area_normalizer=1.0):
    if node.is_leaf:
        area = calculate_leaf_area(node, numeric_features, categorical_features)
        return {
            'is_leaf': True,
            'leaf_id': node.leaf_id,
            'area': area / area_normalizer,
            'numeric_bounds': node.numeric_bounds,
            'cat_options': node.cat_options,
            'is_outlier': node.is_outlier
        }
    else:
        return {
            'is_leaf': False,
            'split_feature': node.split_feature,
            'split_value': node.split_value,
            'left': get_tree_structure_with_areas(node.left, numeric_features, categorical_features, area_normalizer),
            'right': get_tree_structure_with_areas(node.right, numeric_features, categorical_features, area_normalizer)
        }

# --- Outlier assignment ---
def mark_largest_area_leaves_as_outliers(tree, numeric_features, categorical_features, n_inlier_leaves=2):
    leaves = get_leaves(tree)
    leaf_areas = [
        (leaf, calculate_leaf_area(leaf, numeric_features, categorical_features))
        for leaf in leaves
    ]
    leaf_areas_sorted = sorted(leaf_areas, key=lambda x: x[1])
    for leaf, _ in leaf_areas_sorted[:n_inlier_leaves]:
        leaf.is_outlier = False
    for leaf, _ in leaf_areas_sorted[n_inlier_leaves:]:
        leaf.is_outlier = True
    return leaves

# --- Sampling from tree ---
def sample_data(
    leaves,
    n_inlier_samples,
    n_outlier_samples,
    numeric_features,
    categorical_features,
    device='cuda'
):
    inlier_leaves = [leaf for leaf in leaves if not getattr(leaf, 'is_outlier', False)]
    outlier_leaves = [leaf for leaf in leaves if getattr(leaf, 'is_outlier', False)]
    n_inlier_leaves = len(inlier_leaves)
    n_outlier_leaves = len(outlier_leaves)
    per_inlier_leaf = (n_inlier_samples // n_inlier_leaves + 1) if n_inlier_leaves > 0 else 0
    per_outlier_leaf = (n_outlier_samples // n_outlier_leaves + 1) if n_outlier_leaves > 0 else 0

    # Gather all possible categorical values for encoding
    if categorical_features:
        cat_value_sets = {f: set() for f in categorical_features}
        for leaf in leaves:
            for f in categorical_features:
                cat_value_sets[f].update(leaf.cat_options[f])
        cat_value_lists = {f: sorted(list(s)) for f, s in cat_value_sets.items()}
        cat_value_to_idx = {f: {v: i for i, v in enumerate(vals)} for f, vals in cat_value_lists.items()}
    else:
        cat_value_lists = {}
        cat_value_to_idx = {}

    def fast_sample_categorical(leaf_list, per_leaf):
        if not categorical_features or not leaf_list:
            return np.empty((0, 0), dtype=int)
        n_leaves = len(leaf_list)
        n_cat = len(categorical_features)
        max_cat_options = [max(len(leaf.cat_options[f]) for leaf in leaf_list) for f in categorical_features]

        options_tensors = []
        for fi, f in enumerate(categorical_features):
            options = []
            for leaf in leaf_list:
                opts = leaf.cat_options[f]
                idxs = [cat_value_to_idx[f][val] for val in opts]
                idxs += [-1] * (max_cat_options[fi] - len(idxs))
                options.append(idxs)
            options_tensors.append(torch.tensor(options, device=device))
        X_cat = torch.empty((n_leaves * per_leaf, n_cat), dtype=torch.long, device=device)
        for fi, f in enumerate(categorical_features):
            opts_tensor = options_tensors[fi]
            num_opts = torch.tensor([len(leaf.cat_options[f]) for leaf in leaf_list], device=device)
            idxs = torch.cat([
                torch.randint(0, num_opts[li], (per_leaf,), device=device) for li in range(n_leaves)
            ])
            gather_base = (torch.arange(n_leaves, device=device).repeat_interleave(per_leaf) * max_cat_options[fi]).long()
            idxs_flat = gather_base + idxs
            opts_tensor_flat = opts_tensor.flatten(0, 1)
            chosen = opts_tensor_flat[idxs_flat]
            X_cat[:, fi] = chosen
        return X_cat.cpu().numpy()

    def sample_numeric(leaf_list, per_leaf):
        if not leaf_list:
            return np.empty((0, len(numeric_features)))
        n_leaves = len(leaf_list)
        n_num = len(numeric_features)
        bounds = np.array([[leaf.numeric_bounds[f] for f in numeric_features] for leaf in leaf_list])
        lows = torch.tensor(bounds[:,:,0], device=device)
        highs = torch.tensor(bounds[:,:,1], device=device)
        unif = torch.rand((n_leaves, n_num, per_leaf), device=device)
        nums = lows.unsqueeze(2) + unif * (highs - lows).unsqueeze(2)
        nums = nums.permute(0,2,1).reshape(-1, n_num).cpu().numpy()
        return nums

    def sample_batch(leaf_list, per_leaf, label):
        nums = sample_numeric(leaf_list, per_leaf)
        cats = fast_sample_categorical(leaf_list, per_leaf) if categorical_features else np.empty((len(nums), 0), dtype=int)
        y = np.full((len(nums),), label, dtype=int)
        return nums, cats, y

    X_num_in, X_cat_in, y_in = sample_batch(inlier_leaves, per_inlier_leaf, 0)
    X_num_out, X_cat_out, y_out = sample_batch(outlier_leaves, per_outlier_leaf, 1)
    X_numeric_all = np.vstack([X_num_in, X_num_out]) if X_num_in.size and X_num_out.size else (X_num_in if X_num_out.size == 0 else X_num_out)
    X_categorical_all = np.vstack([X_cat_in, X_cat_out]) if X_cat_in.size and X_cat_out.size else (X_cat_in if X_cat_out.size == 0 else X_cat_out)
    y_all = np.concatenate([y_in, y_out])

    # One-hot encoding
    if categorical_features and X_categorical_all.size:
        onehot_columns = []
        for i, f in enumerate(categorical_features):
            n_vals = len(cat_value_lists[f])
            mask = (X_categorical_all[:, i] >= 0)
            vals = X_categorical_all[:, i].copy()
            vals[~mask] = 0
            onehot = np.eye(n_vals)[vals]
            onehot[~mask] = 0
            onehot_columns.append(onehot)
        X_categorical_onehot = np.concatenate(onehot_columns, axis=1)
        X_all = np.concatenate([X_numeric_all, X_categorical_onehot], axis=1)
    else:
        X_all = X_numeric_all

    X_all_tensor = torch.from_numpy(X_all).float().to(device)
    y_all_tensor = torch.from_numpy(y_all).long().to(device)
    return X_all_tensor, y_all_tensor, cat_value_lists, cat_value_to_idx


def dict_to_tree(node_dict):
    if node_dict['is_leaf']:
        node = IsolationTreeNode(
            depth=0,
            max_depth=0,
            numeric_bounds=node_dict['numeric_bounds'],
            cat_options=node_dict['cat_options']
        )
        node.is_leaf = True
        node.leaf_id = node_dict['leaf_id']
        node.is_outlier = node_dict.get('is_outlier', False)
        node.area = node_dict.get('area', None)
        return node
    else:
        node = IsolationTreeNode(
            depth=0,
            max_depth=0,
            numeric_bounds={},
            cat_options={}
        )
        node.is_leaf = False
        node.split_feature = node_dict['split_feature']
        node.split_value = node_dict['split_value']
        node.left = dict_to_tree(node_dict['left'])
        node.right = dict_to_tree(node_dict['right'])
        return node
    
    
    @torch.no_grad()
    def draw_batched_data(self, 
                          num_inliers, 
                          num_local_anomalies):
        #start = time.time()
        raw_inliers = self.sample_inliers(num_inliers)
        #print('sample inliers', time.time()-start)
        if self.outlier_type == 'prob':
            raw_local_anomalies = self.sample_prob_outliers(num_samples=num_local_anomalies)
        elif self.outlier_type == 'contextual':
            raw_local_anomalies = self.sample_contextual_outliers(num_samples=num_local_anomalies)
        return raw_inliers, raw_local_anomalies
    
    
    
def make_contextualSCM(max_feature_dim: int,
                 min_num_layer: int,
                 max_num_layer: int,
                 min_hidden_size: int,
                 max_hidden_size: int,
                 alpha: float,
                 beta: float,
                 device):
    return StructuralCausalModel(num_features = max_feature_dim,
                                 min_num_layer=min_num_layer,
                                 max_num_layer = max_num_layer,
                                 min_hidden_size = min_hidden_size,
                                 max_hidden_size = max_hidden_size,
                                 device = device,
                                 outlier_type = 'contextual',
                                 drop_weight_prob = 0.6)
    
    

if __name__ == "__main__":
    for repeat in tqdm(range(200000)):
        n_numeric = np.random.randint(2,100)
        n_categorical = 0 #np.random.randint(0,1) #10)
        tree_depth = np.random.randint(4,8)
        n_samples = 5000
        #random_seed = 42

        numeric_features, categorical_features, numeric_bounds, cat_options = random_feature_definitions(
            n_numeric=n_numeric, n_categorical=n_categorical)
        leaf_id_counter = iter(range(500000))

        n_num = len(numeric_features)
        if categorical_features:
            n_cat_onehot = sum(len(cat_options[f]) for f in categorical_features)
        else:
            n_cat_onehot = 0
        n_total = n_num + n_cat_onehot

        while n_total > 100:
            n_numeric = np.random.randint(2,100)
            n_categorical = np.random.randint(0,2)
            tree_depth = np.random.randint(5,10)
            numeric_features, categorical_features, numeric_bounds, cat_options = random_feature_definitions(
            n_numeric=n_numeric, n_categorical=n_categorical)
            n_num = len(numeric_features)
            if categorical_features:
                n_cat_onehot = sum(len(cat_options[f]) for f in categorical_features)
            else:
                n_cat_onehot = 0
            n_total = n_num + n_cat_onehot

        n_leaf_nodes = 0
        while(n_leaf_nodes < 5 or n_leaf_nodes > 20):
            tree = build_isolation_tree(
                numeric_bounds, cat_options, depth=0, max_depth=tree_depth, leaf_id_counter=leaf_id_counter)
            leaves = get_leaves(tree)
            n_leaf_nodes = len(leaves)
            n_nodes = np.max([1, np.random.randint(0, int(n_leaf_nodes/2))]) if n_leaf_nodes > 1 else 1
            prune_random_nodes(tree, n_nodes) #, random_seed=random_seed)
            n_leaf_nodes = len(get_leaves(tree))

        mark_largest_area_leaves_as_outliers(tree, numeric_features, categorical_features)
        leaf_areas = get_leaf_areas(tree, numeric_features, categorical_features)
        total_area = sum(area for _, area in leaf_areas)

        tree_info = get_tree_structure_with_areas(tree, numeric_features, categorical_features, area_normalizer=total_area)
        data_to_save = {
            'tree_info': tree_info,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        output = 200000+repeat
        with open(f"/home/xding2/FoMo-0D-explore/density_trees/tree_info_{output}.json", "w") as f:
            json.dump(data_to_save, f, indent=2)

        # # === 2. Load tree and sample ===
        # with open("tree_info.json", "r") as f:
        #     loaded = json.load(f)
        # loaded_tree_info = loaded['tree_info']
        # numeric_features = loaded['numeric_features']
        # categorical_features = loaded['categorical_features']

        # tree_loaded = dict_to_tree(loaded_tree_info)
        # leaves = get_leaves(tree_loaded)

        # #print("\nLeaf nodes with outlier status (from loaded tree):")
        # #for leaf in leaves:
        # #    print(f"Leaf ID: {leaf.leaf_id}, area: {getattr(leaf, 'area', -1):.3f}, is_outlier: {leaf.is_outlier}")

        # X_all, y_all, cat_value_lists, cat_value_to_idx = sample_data(
        #     leaves, n_samples, n_samples,
        #     numeric_features=numeric_features,
        #     categorical_features=categorical_features,
        #     device='cuda'
        # )
        # #print(f"\nX_all shape: {X_all.shape}")
        # #print(f"y_all shape: {y_all.shape}")

        # lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
        # _ = lof.fit_predict(X_all.detach().cpu().numpy())
        # lof_scores = -lof.negative_outlier_factor_
        # auc = roc_auc_score(y_all.detach().cpu().numpy(), lof_scores)
        # print(f"AUC-ROC for LOF on SCM contextual anomalies: {auc:.4f}")