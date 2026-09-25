"""
Copyright 2023 Daeho Um
SPDX-License-Identifier: Apache-2.0
"""
import random
import numpy as np
import torch
import torch_geometric.utils
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter as pyg_scatter


def scatter_add(src, index, dim_size):
    'Drop-in replacement per torch_scatter.scatter_add.'


    return pyg_scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")


def fisf(edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None, gamma=None, edge_weight=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    propagation_model = FISF(num_iterations=num_iterations, alpha = alpha, beta=beta, gamma=gamma)
    return propagation_model.propagate(
        x=X, edge_index=edge_index, mask=feature_mask,
        mask_type=mask_type, edge_weight=edge_weight,
    )

class FISF(torch.nn.Module):
    def __init__(self, num_iterations: int, alpha: float, beta: float, gamma:float):
        super(FISF, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor, mask_type: str, edge_weight: OptTensor = None) -> Tensor:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        nv = x.shape[0]
        feat_dim = x.shape[1]

        out = x
        if mask_type == 'structural':
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type)
            adj_c = self.compute_edge_weight_c(edge_index, f_n2d, nv, edge_weight=edge_weight)
            if mask is not None:
                out = torch.zeros_like(x)/3.16
                out[mask] = x[mask]

            for _ in range(self.num_iterations):

                out = torch.sparse.mm(adj_c, out)
                out[mask] = x[mask]
            f_n2d = f_n2d.repeat(feat_dim,1)
        else:
            out = torch.zeros_like(x)/3.16
            if mask is not None:
                out[mask] = x[mask]
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, feat_dim = feat_dim)
            for i in range(feat_dim):
                adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], nv, edge_weight=edge_weight)
                for _ in range(self.num_iterations):
                    out[:,i] = torch.sparse.mm(adj_c, out[:,i].reshape(-1,1)).reshape(-1)
                    out[mask[:,i],i] = x[mask[:,i],i]

        low_var_index = torch.topk(torch.var(out, dim=0), int(feat_dim*self.gamma), largest=False).indices
        low_var_mask = self.index_to_mask(low_var_index, feat_dim)
        high_var_mask = ~low_var_mask
        high_var_index = torch.nonzero(high_var_mask).squeeze()

        print('low channel num: '+str(int(torch.sum(low_var_mask))) +', high channel num: '+str(int(torch.sum(high_var_mask))))
        mask_de_max = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)

        if torch.numel(low_var_index) > 0:
            for j, i in enumerate(low_var_index):
                rand_index = np.random.choice(nv, 1, replace=False)
                idx_max = rand_index[0]
                x[idx_max, i], mask[idx_max, i] = float(torch.rand(1).numpy()[0]), True
                mask_de_max[idx_max, i] = True

        nv = x.shape[0]
        feat_dim = x.shape[1]
        if mask_type == 'structural':
            out = torch.zeros_like(x)/3.16
            if mask is not None:
                out[mask] = x[mask]
            if torch.numel(high_var_index) > 1:
                pre = high_var_index[0]
            elif torch.numel(high_var_index) == 1:
                pre = high_var_index
            else:
                pre = None
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, pre)
            f_n2d_max = self.compute_f_n2d(edge_index, mask_de_max, 'virtual', feat_dim=feat_dim,
                                           virtual_idx=low_var_index)

            pc_com = (self.alpha ** f_n2d.repeat(feat_dim, 1)) * (self.beta ** f_n2d_max)
            adj_c_pre = self.compute_edge_weight_c(edge_index, f_n2d, nv, edge_weight=edge_weight)
            if torch.numel(low_var_index) > 0:
                for i in low_var_index:
                    adj_c = self.compute_edge_weight_from_pc(edge_index, pc_com[i], nv, edge_weight=edge_weight)
                    for _ in range(self.num_iterations):

                        out[:, i] = torch.sparse.mm(adj_c, out[:, i].reshape(-1, 1)).reshape(-1)

                        out[mask[:, i], i] = x[mask[:, i], i]
            for _ in range(self.num_iterations):

                out[:, high_var_mask] = torch.sparse.mm(adj_c_pre, out[:, high_var_mask])
                out[mask] = x[mask]
        else:
            out = torch.zeros_like(x)/3.16
            if mask is not None:
                out[mask] = x[mask]
            if torch.numel(high_var_index) > 1:
                pre = high_var_index[0]
            elif torch.numel(high_var_index) == 1:
                pre = high_var_index
            else:
                pre = None
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, pre, feat_dim = feat_dim)
            f_n2d_max = self.compute_f_n2d(edge_index, mask_de_max, 'virtual', feat_dim=feat_dim,
                                           virtual_idx=low_var_index)
            pc_com = (self.alpha ** f_n2d) * (self.beta ** f_n2d_max)
            if torch.numel(low_var_index) > 0:
                for i in low_var_index:
                    adj_c = self.compute_edge_weight_from_pc(edge_index, pc_com[i], nv, edge_weight=edge_weight)
                    for _ in range(self.num_iterations):

                        out[:, i] = torch.sparse.mm(adj_c, out[:, i].reshape(-1, 1)).reshape(-1)

                        out[mask[:, i], i] = x[mask[:, i], i]
            if torch.numel(high_var_index) > 0:
                for i in high_var_index:
                    adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], nv, edge_weight=edge_weight)
                    for _ in range(self.num_iterations):
                        out[:, i] = torch.sparse.mm(adj_c, out[:, i].reshape(-1, 1)).reshape(-1)
                        out[mask[:, i], i] = x[mask[:, i], i]
        return out

    def compute_f_n2d(self, edge_index, feature_mask, mask_type, pre: OptTensor = None, feat_dim: OptTensor = None,
                      virtual_idx=None):
        nv = feature_mask.shape[0]
        if mask_type == 'structural':
            len_v_0tod_list = []
            f_n2d = torch.zeros(nv, dtype=torch.int, device=feature_mask.device)
            if pre == None:
                v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
            else:
                v_0 = torch.nonzero(feature_mask[:, pre]).view(-1)
            len_v_0tod_list.append(len(v_0))
            v_0_to_now = v_0
            f_n2d[v_0] = 0
            d = 1
            while True:
                v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                if len(v_d) == 0:
                    break
                f_n2d[v_d] = d
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                len_v_0tod_list.append(len(v_d))
                d += 1
        elif mask_type == 'uniform':
            f_n2d = torch.zeros(feat_dim, nv, device=feature_mask.device)
            for i in range(feat_dim):
                v_0 = torch.nonzero(feature_mask[:, i]).view(-1)
                v_0_to_now = v_0
                f_n2d[i, v_0] = 0
                d = 1
                while True:
                    v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
        elif mask_type == 'virtual':
            f_n2d = torch.zeros(feat_dim, nv, device=feature_mask.device)
            for i in virtual_idx:
                v_0 = torch.nonzero(feature_mask[:, i]).view(-1)
                v_0_to_now = v_0
                f_n2d[i, v_0] = 0
                d = 1
                while True:
                    v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
        return f_n2d

    def compute_edge_weight_c(self, edge_index, f_n2d, n_nodes, edge_weight=None):
        row, col = edge_index[0], edge_index[1]
        d_row = f_n2d[row]
        d_col = f_n2d[col]
        edge_weight_c = (self.alpha ** (d_col - d_row + 1)).to(edge_index.device)
        if edge_weight is not None:
            edge_weight_c = edge_weight_c * edge_weight.to(edge_index.device, dtype=edge_weight_c.dtype)
        deg_W = scatter_add(edge_weight_c, row, dim_size= f_n2d.shape[0])
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values= A_Dinv, size=[n_nodes, n_nodes]).to(edge_index.device)
        return adj

    def compute_edge_weight_from_pc(self, edge_index, pc, n_nodes, edge_weight=None):
        row, col = edge_index[0], edge_index[1]
        pc_row = pc[row]
        pc_col = pc[col]
        edge_weight_c = (self.alpha *pc_col/pc_row).to(edge_index.device)
        if edge_weight is not None:
            edge_weight_c = edge_weight_c * edge_weight.to(edge_index.device, dtype=edge_weight_c.dtype)
        deg_W = scatter_add(edge_weight_c, row, dim_size= pc.shape[0])
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values= A_Dinv, size=[n_nodes, n_nodes]).to(edge_index.device)
        return adj

    def index_to_mask(self, index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask
