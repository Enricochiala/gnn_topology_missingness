"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np

from locale import normalize
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_adj

from graph_utils import get_symmetrically_normalized_adjacency, get_row_normalized_adjacency
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def random_filling(X):
    return torch.randn_like(X)

def zero_filling(X):
    return torch.zeros_like(X)

def mean_filling(X, feature_mask):
    n_nodes = X.shape[0]
    return compute_mean(X, feature_mask).repeat(n_nodes, 1)


def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X.clone()
    X_zero_filled[~feature_mask] = 0.0


    src, dst = edge_index[0], edge_index[1]


    summed_features = torch.zeros_like(X)
    valid_counts = torch.zeros(n_nodes, dtype=torch.float, device=X.device)

    for i in range(edge_index.shape[1]):
        s = src[i]
        d = dst[i]
        summed_features[d] += X_zero_filled[s]
        valid_counts[d] += feature_mask[s].float().sum() / feature_mask.size(1)


    mean_features = summed_features / (valid_counts.unsqueeze(-1) + 1e-10)
    mean_features[mean_features.isnan()] = 0.0

    return mean_features

def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)

def filling(filling_method, edge_index, X, feature_mask, num_iterations=None):
    if filling_method == "random":
        X_reconstructed = random_filling(X)
    elif filling_method == "zero":
        X_reconstructed = zero_filling(X)
    elif filling_method == "mean":
        X_reconstructed = mean_filling(X, feature_mask)
    elif filling_method == "neighborhood_mean":
        X_reconstructed = neighborhood_mean_filling(edge_index, X, feature_mask)
    elif filling_method == "fp":
        X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations)
    else:
        raise ValueError(f"{filling_method} method not implemented")
    return X_reconstructed

def compute_mean(X, feature_mask):
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    num_of_non_zero = torch.count_nonzero(feature_mask, dim=0)
    mean_features = torch.sum(X_zero_filled, axis=0) / num_of_non_zero

    mean_features[mean_features.isnan()] = 0

    return mean_features

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations


    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor) -> Tensor:


        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj_unnormalized, adj = self.get_propagation_matrix(out, edge_index, n_nodes)

        for i in range(self.num_iterations):

            out = torch.sparse.mm(adj, out)
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes):

        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        edge_weight_tmp = torch.where(edge_weight > 0, 1, 0).type_as(edge_weight)

        adj_unnormalized = torch.sparse.FloatTensor(edge_index, values=edge_weight_tmp, size=(n_nodes, n_nodes)).to(edge_index.device)
        adj_normalzied = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj_unnormalized, adj_normalzied
