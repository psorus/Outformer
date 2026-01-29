import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pandas as pd
import torch
import seaborn as sns


def lognormal_discrete(mu,
                       sigma,
                       minval:int,
                       maxval:int):
    # sample from lognormal distribtuion, making it discrete
    # input: mu, sigma, minval (int), maxval(int)
    # return: a integer value
    val = int(np.round(np.random.lognormal(mu, sigma)))
    return int(np.clip(val, minval, maxval))


def sample_layers_and_nodes(min_num_layer =2,
                            max_num_layer =5,
                            min_hidden_size = 3,
                            max_hidden_size = 8):
    #return: a randomly sampled hidden layer and number of layers
    l = lognormal_discrete(mu=0.7, sigma=0.4, minval=min_num_layer, maxval=max_num_layer)  # num layers
    h = lognormal_discrete(mu=1.2, sigma=0.5, minval=min_hidden_size, maxval=max_hidden_size)  # hidden size
    return l, h


def build_mlp_dag(l, h):
    # input: l = number of layers, h = number of hidden nodes
    # return: G: a fully connected graph
    # layers: the name of the layers within the graph
    G = nx.DiGraph()
    layers = []
    for i in range(l):
        layer = []
        for j in range(h):
            node_name = f"L{i}_N{j}"
            G.add_node(node_name)
            layer.append(node_name)
        layers.append(layer)
    for i in range(l-1):
        for src in layers[i]:
            for dst in layers[i+1]:
                G.add_edge(src, dst)
    return G, layers


def sample_weights(G, min_abs=0.35, device='cpu'):
    # Sample edge weights from N(0,1), but force |w| >= min_abs by clamping
    weights = {}
    for edge in G.edges():
        w = torch.normal(mean=0., std=1., size=(1,), device=device)
        # Clamp the magnitude to be at least min_abs, keep the sign
        abs_w = torch.clamp(torch.abs(w), min=min_abs)
        w_clipped = torch.sign(w) * abs_w
        weights[edge] = w_clipped.item()
    return weights


def drop_random_edges(G, layers, k, alpha=1.5, beta=4.0):
    drop_prob = np.random.beta(alpha, beta)
    G_new = G.copy()
    for layer_idx in range(1, len(layers)):
        for node in layers[layer_idx]:
            parents = list(G_new.predecessors(node))
            if len(parents) <= 1:
                continue
            keep_parent = np.random.choice(parents)
            for p in parents:
                if p == keep_parent:
                    continue
                if np.random.rand() < drop_prob:
                    G_new.remove_edge(p, node)
    # Remove isolated nodes
    isolated = list(nx.isolates(G_new))
    G_new.remove_nodes_from(isolated)

    # If not enough nodes, randomly add back missing nodes (without edges)
    if G_new.number_of_nodes() < k:
        all_nodes = set(G.nodes())
        current_nodes = set(G_new.nodes())
        missing_nodes = list(all_nodes - current_nodes)
        if len(missing_nodes) > 0:
            nodes_to_add = np.random.choice(missing_nodes, size=k - G_new.number_of_nodes(), replace=False)
            G_new.add_nodes_from(nodes_to_add)
    # Optionally, you could restore the original edges for these added nodes if you want.
    # Check for DAG property (should be true as adding isolated nodes can't create cycles)
    assert nx.is_directed_acyclic_graph(G_new)
    return G_new


def sample_activation(device='cpu'):
    # sample activation functions (PyTorch version)
    activations = [
        ("tanh", torch.tanh),
        ("leaky_relu", lambda x: torch.where(x > 0, x, 0.01 * x)),
        ("elu", lambda x: torch.where(x > 0, x, torch.exp(x) - 1)),
        ("identity", lambda x: x),
    ]
    idx = torch.randint(0, len(activations), (1,), device=device).item()
    return activations[idx]


def sample_noise_distribution(device='cpu'):
    # Sample noise distributions (log-normal) using PyTorch
    mu = (torch.rand(1, device=device) - 0.5).item()  # Uniform in [-0.5, 0.5)
    sigma = (torch.rand(1, device=device) * (0.5 - 0.05) + 0.05).item()  # Uniform in [0.05, 0.5)
    return lambda n: torch.exp(
        mu + sigma * torch.randn(n, device=device)
    )
    
    
def sample_feature_nodes(G, k):
    # Sample k feature nodes from all nodes (no label)
    nodes = list(G.nodes)
    Nx = np.random.choice(nodes, k, replace=False)
    return list(Nx)




class StructuralCausalModel:
    def __init__(self,
                num_features: int = 3,
                min_num_layer: int = 3,
                max_num_layer: int = 5,
                min_hidden_size: int = 8,
                max_hidden_size: int = 8,
                alpha: float = 1.5,
                beta: float = 4.0,
                device = 'gpu',
                outlier_type = 'contextual',
                ):
        self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        while self.l * self.h < num_features:
            self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        self.G, self.layers = build_mlp_dag(self.l, self.h)
        self.weights = sample_weights(self.G,device=device)
        self.G_dag = drop_random_edges(self.G, self.layers, num_features, alpha=alpha, beta=beta)
        self.activation_name, self.activation = sample_activation(device=device)
        self.node_noises = sample_noise_distribution(device=device)
        self.k = num_features
        self.Nx = sample_feature_nodes(self.G_dag, self.k)
        self.topo_order = list(nx.topological_sort(self.G_dag))
        self.outlier_type = outlier_type
        self.device = device


    def sample(self): #single node generation, not used !!!
        values = {}
        for node in self.topo_order:
            parents = list(self.G_dag.predecessors(node)) #find the parents of the node
            noise = self.node_noises(1).to(self.device)  # sample a noise for each feature
            if parents:
                parent_sum = sum(
                    self.weights.get((p, node), 0) * values[p] for p in parents
                )
                # parent_sum and noise are both tensors, need to sum properly
                z = self.activation(parent_sum + noise)
            else:
                z = self.activation(noise)
            values[node] = z
        return torch.stack([values[n] for n in self.Nx]).squeeze()
    
    
    def draw_inliers(self, n_samples=100): #parallel node generation
        # Pre-allocate: all node values, shape (n_samples, num_nodes)
        node_list = self.topo_order
        num_nodes = len(node_list)
        node_idx = {node: i for i, node in enumerate(node_list)}
        vals = torch.zeros((n_samples, num_nodes), device=self.device)
        
        # Map from node name to index in vals
        for i, node in enumerate(node_list):
            parents = list(self.G_dag.predecessors(node))
            noises = self.node_noises(n_samples).to(self.device)
            if parents:
                parent_indices = [node_idx[p] for p in parents]
                parent_vals = vals[:, parent_indices]  # shape: (n_samples, num_parents)
                parent_weights = torch.tensor([self.weights.get((p, node), 0) for p in parents],
                                            device=self.device).float()
                parent_sum = (parent_vals * parent_weights).sum(dim=1)  # (n_samples,)
                z = self.activation(parent_sum + noises)
            else:
                z = self.activation(noises)
            vals[:, i] = z
        
        # Output only feature nodes
        Nx_indices = [node_idx[n] for n in self.Nx]
        return vals[:, Nx_indices]
    
        
    def sample_prob_outlier_batch(
        self,
        n_samples=100,
        n_perturbed_node=1,
        scale=2.0,
        outlier_node=None
    ):
        """
        Vectorized: Batch of probabilistic outlier samples.
        All samples share the same outlier feature nodes.
        Returns: (n_samples, n_features), outlier_node_names
        """
        device = self.device
        node_list = self.topo_order
        num_nodes = len(node_list)
        k = self.k
        node_idx = {node: i for i, node in enumerate(node_list)}
        Nx_indices = [node_idx[n] for n in self.Nx]

        # Choose perturbed feature node indices (same for all samples)
        
        if outlier_node is None:
            n_perturbed_node = max(1, min(n_perturbed_node, len(Nx_indices)))
            outlier_idxs = np.random.choice(Nx_indices, n_perturbed_node, replace=False)
            outlier_node_names = [node_list[i] for i in outlier_idxs]
        else:
            if isinstance(outlier_node, str):
                outlier_node_names = [outlier_node]
            else:
                outlier_node_names = outlier_node
            outlier_idxs = [node_idx[n] for n in outlier_node_names]

        # Prepare noises and scale mask
        noises = self.node_noises((n_samples, num_nodes)).to(device)
        scale_mask = torch.ones((n_samples, num_nodes), device=device)
        scale_mask[:, outlier_idxs] = scale

        # Apply scaling
        noises = noises * scale_mask

        # Batched forward propagation
        vals = torch.zeros((n_samples, num_nodes), device=device)
        for i, node in enumerate(node_list):
            parents = list(self.G_dag.predecessors(node))
            if parents:
                parent_indices = [node_idx[p] for p in parents]
                parent_vals = vals[:, parent_indices]  # shape: (n_samples, n_parents)
                parent_weights = torch.tensor(
                    [self.weights.get((p, node), 0) for p in parents], device=device
                ).float()
                parent_sum = (parent_vals * parent_weights).sum(dim=1)  # (n_samples,)
                z = self.activation(parent_sum + noises[:, i])
            else:
                z = self.activation(noises[:, i])
            vals[:, i] = z

        # Output only feature nodes
        return vals[:, Nx_indices], outlier_node_names



    def sample_contextual_outlier_batch(
        self,
        n_samples=1000,
        affected_edges=None,
        weights_modified=None,
        weight_fraction=0.05,
    ):
        """
        Batched contextual outlier generation: 
        All samples use the same perturbed weights and affected edges.
        Returns: (n_samples, n_features), affected_nodes, affected_edges, weights_modified
        """
        device = self.device
        node_list = self.topo_order
        num_nodes = len(node_list)
        k = self.k
        node_idx = {node: i for i, node in enumerate(node_list)}
        Nx_indices = [node_idx[n] for n in self.Nx]

        # Prepare perturbed weights/edges
        if weights_modified is None and affected_edges is None:
            weights_modified = copy.deepcopy(self.weights)
            all_edges = list(self.G_dag.edges())
            num_to_modify = max(1, int(torch.ceil(torch.tensor(weight_fraction * len(all_edges)))))
            edge_indices = torch.randperm(len(all_edges), device=device)[:num_to_modify].cpu().tolist()
            affected_edges = {all_edges[i] for i in edge_indices}
            # Perturb selected edges (flip sign)
            for (u, v) in affected_edges:
                val = -torch.randint(0, 2, (1,), device=device).item()
                weights_modified[(u, v)] = val * weights_modified[(u, v)]

        # Affected nodes
        affected_targets = {v for (u, v) in affected_edges}
        affected_nodes = set(affected_targets)
        for node in affected_targets:
            affected_nodes.update(nx.descendants(self.G_dag, node))

        # Ensure at least one feature node is affected
        if not any(n in affected_nodes for n in self.Nx):
            nx_feat = np.random.choice(self.Nx)
            parents = list(self.G_dag.predecessors(nx_feat))
            if parents:
                forced_parent = np.random.choice(parents)
                edge = (forced_parent, nx_feat)
                affected_edges.add(edge)
                val = -torch.randint(0, 2, (1,), device=device).item()
                weights_modified[edge] = val * weights_modified.get(edge, 1.0)
                affected_nodes.add(nx_feat)
                affected_nodes.update(nx.descendants(self.G_dag, nx_feat))

        # Batched noise
        noises = self.node_noises((n_samples, num_nodes)).to(device)

        # Batched forward propagation
        vals = torch.zeros((n_samples, num_nodes), device=device)
        for i, node in enumerate(node_list):
            parents = list(self.G_dag.predecessors(node))
            if parents:
                parent_indices = [node_idx[p] for p in parents]
                parent_vals = vals[:, parent_indices]  # (n_samples, n_parents)
                parent_weights = torch.tensor(
                    [weights_modified.get((p, node), 0) for p in parents], device=device
                ).float()
                parent_sum = (parent_vals * parent_weights).sum(dim=1)  # (n_samples,)
                z = self.activation(parent_sum + noises[:, i])
            else:
                z = self.activation(noises[:, i])
            vals[:, i] = z

        # Output only feature nodes
        return vals[:, Nx_indices], affected_nodes, affected_edges, weights_modified



    
    
    def describe(self):
        print(f"Layers: {self.l}, Hidden size: {self.h}")
        print(f"Dropout DAG: {self.G_dag.number_of_edges()} edges (original: {self.G.number_of_edges()})")
        print(f"Activation: {self.activation_name}")
        print(f"Feature nodes: {self.Nx}")
        
    
    def draw_inflated_samples(self,
                            n_samples=100,
                            batch_size=500,
                            outlier_node=None,
                            outlier_edge=None,
                            outlier_weight=None):
        """
        Generate n_samples inflated outlier samples, using batching for efficiency.
        Optionally split into batches (set batch_size for large n_samples).
        Returns: (n_samples, k)
        """
        all_outliers = []
        n_iters = int(np.ceil(n_samples / batch_size))
        for i in range(n_iters):
            current_batch = min(batch_size, n_samples - i * batch_size)
            if self.outlier_type == 'prob':
                scale = torch.rand(1, device=self.device).item() * (3.0 - 2.0) + 2.0  # Uniform [2.0, 3.0)
                n_perturbed_node = torch.randint(5, 11, (1,), device=self.device).item()
                batch_outliers, _ = self.sample_prob_outlier_batch(
                    n_samples=current_batch,
                    n_perturbed_node=n_perturbed_node,
                    scale=scale,
                    outlier_node=outlier_node
                )
                all_outliers.append(batch_outliers)
            elif self.outlier_type == 'contextual':
                weight_fraction = torch.rand(1, device=self.device).item() * (0.03 - 0.01) + 0.01
                batch_outliers, _, _, _ = self.sample_contextual_outlier_batch(
                    n_samples=current_batch,
                    affected_edges=outlier_edge,
                    weights_modified=outlier_weight,
                    weight_fraction=weight_fraction
                )
                all_outliers.append(batch_outliers)
            else:
                raise ValueError("Invalid outlier type. Must be 'contextual' or 'prob'.")

        return torch.cat(all_outliers, dim=0)[:n_samples]  # Trim if over




    def draw_batched_data(self, 
                          num_inliers, 
                          num_local_anomalies):
        raw_inliers = self.draw_inliers(num_inliers)
        raw_local_anomalies = self.draw_inflated_samples(n_samples=num_local_anomalies)
        return raw_inliers, raw_local_anomalies
        




def make_probSCM(max_feature_dim: int,
                 min_num_layer: int,
                 max_num_layer: int,
                 min_hidden_size: int,
                 max_hidden_size: int,
                 alpha: float,
                 beta: float,
                 device):
    return StructuralCausalModel(max_feature_dim,
                                 min_num_layer,
                                 max_num_layer,
                                 min_hidden_size,
                                 max_hidden_size,
                                 alpha,
                                 beta,
                                 device,
                                 outlier_type = 'prob')
    
        
        
        
def make_contextualSCM(max_feature_dim: int,
                 min_num_layer: int,
                 max_num_layer: int,
                 min_hidden_size: int,
                 max_hidden_size: int,
                 alpha: float,
                 beta: float,
                 device):
    return StructuralCausalModel(max_feature_dim,
                                 min_num_layer,
                                 max_num_layer,
                                 min_hidden_size,
                                 max_hidden_size,
                                 alpha,
                                 beta,
                                 device,
                                 outlier_type = 'contextual')
    
    
        
        
    
