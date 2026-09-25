"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    'Given an edge_index, return the same edge_index and edge weights computed as'


    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    device = edge_weight.device
    deg = torch.zeros(n_nodes, dtype=edge_weight.dtype, device=device)
    col = col.to(device)
    edge_weight = edge_weight.to(device)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def get_row_normalized_adjacency(edge_index, n_nodes):
    'Given an edge_index, return the same edge_index and edge weights computed as'


    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    device = edge_weight.device
    deg = torch.zeros(n_nodes, dtype=edge_weight.dtype, device=device)
    col = col.to(device)
    edge_weight = edge_weight.to(device)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DA = deg_inv_sqrt[row] * edge_weight

    return edge_index, DA
