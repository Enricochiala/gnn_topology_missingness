"""Load the static classification dataset with the original graph preprocessing."""

from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import degree, to_networkx

DATASETS = ('tadpole',)


def canonicalize_lap_pe(data):
    data.pe = data.pe.float()
    for j in range(data.pe.size(1)):
        col = data.pe[:, j]
        nz = (col.abs() > 1e-8).nonzero(as_tuple=False)
        if nz.numel() > 0 and col[nz[0, 0]].item() < 0:
            data.pe[:, j] = -col
    data.pe = (data.pe - data.pe.mean(0)) / data.pe.std(0, unbiased=False).clamp_min(1e-8)
    return data


def load_dataset(name, data_dir, *, prepared=False, pe_dim=8, recompute_pe=False):
    path = Path(data_dir) / f'{name}.pt'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: see data/README.md for the input format')
    data = torch.load(path, weights_only=False, map_location='cpu')
    if name == 'synthetic_sbm' and hasattr(data, 'data'):
        data = data[0]
    data = data.clone().cpu()
    if not prepared and name == 'electric':
        deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
        graph = to_networkx(data, to_undirected=True)
        clustering = torch.tensor([nx.clustering(graph, i) for i in range(data.num_nodes)],
                                  dtype=torch.float).view(-1, 1)
        data.x = torch.cat([data.x, deg, clustering], dim=1)
    if name == 'tadpole':
        data.num_classes = 3
    elif getattr(data, 'num_classes', None) is None:
        data.num_classes = int(data.y.max()) + 1
    if data.x.ndim != 2 or not torch.is_floating_point(data.x) or not torch.isfinite(data.x).all():
        raise ValueError('Controlled missingness requires a complete, finite floating-point X')
    if data.y.ndim != 1 or len(data.y) != data.num_nodes:
        raise ValueError('Expected one integer class label per node')
    if data.y.dtype != torch.long or (data.y < 0).any() or (data.y >= data.num_classes).any():
        raise ValueError('Expected int64 class labels in the dataset class range')
    edges = data.edge_index
    if edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long:
        raise ValueError('Expected int64 edge_index with shape [2, E]')
    if edges.numel() and (edges.min() < 0 or edges.max() >= data.num_nodes):
        raise ValueError('Edge indices outside the node range')
    metadata = {'pe_recomputed': False, 'pe_transform_seconds': None}
    if recompute_pe or getattr(data, 'pe', None) is None:
        if prepared and not recompute_pe:
            raise ValueError('Prepared inputs must include the frozen pe tensor')
        if not 1 <= pe_dim < data.num_nodes - 1:
            raise ValueError('PE dimension must be between 1 and num_nodes - 2')
        start = time.perf_counter()
        from temporal_data import topology_pe
        data.pe = torch.from_numpy(topology_pe(data.edge_index.numpy(), data.num_nodes, pe_dim))
        metadata.update(pe_recomputed=True, pe_transform_seconds=time.perf_counter() - start)
    if data.pe.shape != (data.num_nodes, pe_dim):
        raise ValueError('Stored PE dimension differs; use the matching --pe-dim or --recompute-pe')
    if not torch.isfinite(data.pe).all():
        raise ValueError('Non-finite PE')
    if not prepared or recompute_pe or not getattr(data, 'pe_standardized', False):
        canonicalize_lap_pe(data)
        data.pe_standardized = True
    data.num_features = data.x.size(1)
    # Preserve the classification graph operator used by the experiment.
    # GCNConv handles normalization internally for its own convolution.
    data.adj = torch.sparse_coo_tensor(edges, torch.ones(edges.size(1)),
                                     (data.num_nodes, data.num_nodes)).coalesce()
    for key, value in data:
        if torch.is_tensor(value) and value.dtype == torch.float64:
            data[key] = value.float()
    return data, metadata
