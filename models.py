import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv
import numpy as np
import torch.nn as nn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import gaussian_kde, norm, entropy
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree, k_hop_subgraph
from torch.nn import ModuleList, Linear, BatchNorm1d

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import SAGEConv, GATConv, GINConv, GCN2Conv
import random
from torch.nn import Sequential, Linear, ReLU
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch_geometric.typing')
warnings.filterwarnings("ignore")



def _default_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _default_torch_device()


def _as_gmm_init_features(init_x):
    if init_x is None:
        return None
    if hasattr(init_x, "detach"):
        return init_x.detach().float().cpu().numpy()
    return np.asarray(init_x, dtype=np.float32)


def ex_relu(mu, sigma):

    sigma = torch.clamp(sigma, min=1e-10)
    sqrt_sigma = torch.sqrt(sigma)
    w = mu / sqrt_sigma
    

    pdf = torch.exp(-0.5 * w**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + torch.erf(w / np.sqrt(2)))
    

    out = mu * cdf + sqrt_sigma * pdf
    return torch.where(torch.isnan(out), F.relu(mu), out)

def init_gmm(features, n_components):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture

    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float64)


    all_nan_cols = np.isnan(features).all(axis=0)
    if all_nan_cols.any():
        features = features.copy()
        features[:, all_nan_cols] = 0.0
    try:
        imp = SimpleImputer(
            missing_values=np.nan,
            strategy="mean",
            keep_empty_features=True,
        )
    except TypeError:
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    init_x = imp.fit_transform(features)
    init_x = np.asarray(init_x, dtype=np.float64)

    n_samples = init_x.shape[0]
    n_unique = np.unique(init_x, axis=0).shape[0]
    k = int(min(n_components, n_samples, n_unique))
    k = max(k, 1)

    for reg in [1e-3, 1e-2, 1e-1]:
        try:
            return GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=reg,
                random_state=0,
                max_iter=300,
                n_init=3
            ).fit(init_x)
        except ValueError:
            continue

    return GaussianMixture(
        n_components=1,
        covariance_type="diag",
        reg_covar=1e-1,
        random_state=0,
        max_iter=300,
        n_init=3
    ).fit(init_x)


def _pad_gmm_params(gmm, n_components, in_features):
    weights = np.asarray(gmm.weights_, dtype=np.float32)
    means = np.asarray(gmm.means_, dtype=np.float32).reshape(-1, in_features)
    covariances = np.asarray(gmm.covariances_, dtype=np.float32).reshape(-1, in_features)

    k = means.shape[0]
    if k < n_components:
        reps = int(np.ceil(n_components / max(k, 1)))
        means = np.tile(means, (reps, 1))[:n_components]
        covariances = np.tile(covariances, (reps, 1))[:n_components]
        weights = np.tile(weights, reps)[:n_components]
    elif k > n_components:
        means = means[:n_components]
        covariances = covariances[:n_components]
        weights = weights[:n_components]

    weights = weights / max(weights.sum(), 1e-12)
    covariances = np.maximum(covariances, 1e-10)
    return weights, means, covariances


class GCNFull(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, num_classes))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPFull(torch.nn.Module):
    'Node-wise MLP with the same depth/width protocol as ``GCNFull``.'

    def __init__(self, in_channels, hidden_channels, num_classes=2,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        if num_layers == 1:
            self.layers.append(torch.nn.Linear(in_channels, num_classes))
        else:
            self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(torch.nn.Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        del edge_index
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNmfConv(nn.Module):
    def __init__(self, in_features, out_features, data, n_components, dropout):
        super(GCNmfConv, self).__init__()
        self.in_features = in_features
        self.n_components = n_components
        self.dropout = dropout
        

        self.logp = Parameter(torch.Tensor(n_components))
        self.means = Parameter(torch.Tensor(n_components, in_features))
        self.logvars = Parameter(torch.Tensor(n_components, in_features))
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))


        self.register_buffer('adj2', torch.mul(data.adj, data.adj), persistent=False)
        self.features = data.x.cpu().numpy()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.bias.data.zero_()
        gmm = init_gmm(self.features[:, :self.in_features], self.n_components)
        weights, means, covariances = _pad_gmm_params(
            gmm, self.n_components, self.in_features
        )
        param_device = self.logp.device
        self.logp.data = torch.log(torch.tensor(weights, device=param_device, dtype=torch.float32) + 1e-10)
        self.means.data = torch.tensor(means, device=param_device, dtype=torch.float32)
        self.logvars.data = torch.log(torch.tensor(covariances, device=param_device, dtype=torch.float32) + 1e-10)

    def forward(self, x, adj, edge_index):

        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars).clamp(min=1e-10)


        mean_mat = torch.where(x_isnan, self.means.unsqueeze(1).expand_as(x_imp), x_imp)
        var_mat = torch.where(x_isnan, variances.unsqueeze(1).expand_as(x_imp), torch.zeros_like(x_imp))


        tx = torch.matmul(mean_mat, self.weight) + self.bias
        tv = torch.matmul(var_mat, self.weight**2)
        

        conv_x = torch.stack([torch.spmm(adj, tx_k) for tx_k in tx])
        conv_v = torch.stack([torch.spmm(self.adj2, tv_k) for tv_k in tv])
        

        expected_x = ex_relu(conv_x, conv_v)


        dist = torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)
        log_prob = self.logp.unsqueeze(1) - 0.5 * dist
        gamma = torch.softmax(log_prob, dim=0)
        
        return torch.sum(expected_x * gamma.unsqueeze(2), dim=0)


class PEMixGMMConv(nn.Module):
    'PEMix probabilistic layer: PE affects mixture responsibilities, not propagation.'


    def __init__(self, x_features, pe_features, out_features, data, n_components, dropout):
        super(PEMixGMMConv, self).__init__()
        self.x_features = int(x_features)
        self.pe_features = int(pe_features)
        self.in_features = self.x_features + self.pe_features
        self.n_components = n_components
        self.dropout = dropout

        self.logp = Parameter(torch.Tensor(n_components))
        self.means = Parameter(torch.Tensor(n_components, self.in_features))
        self.logvars = Parameter(torch.Tensor(n_components, self.in_features))
        self.weight = Parameter(torch.Tensor(self.x_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.register_buffer('adj2', torch.mul(data.adj, data.adj), persistent=False)
        self.features = data.x.cpu().numpy()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.bias.data.zero_()
        gmm = init_gmm(self.features[:, :self.in_features], self.n_components)
        weights, means, covariances = _pad_gmm_params(
            gmm, self.n_components, self.in_features
        )
        param_device = self.logp.device
        self.logp.data = torch.log(torch.tensor(weights, device=param_device, dtype=torch.float32) + 1e-10)
        self.means.data = torch.tensor(means, device=param_device, dtype=torch.float32)
        self.logvars.data = torch.log(torch.tensor(covariances, device=param_device, dtype=torch.float32) + 1e-10)

    def responsibilities(self, z):
        """Eq. (12): normalized diagonal Gaussian on observed features and PE."""
        observed = ~torch.isnan(z)
        logvars = self.logvars.clamp(-14, 14)
        diff = torch.where(observed[None], torch.nan_to_num(z)[None] - self.means[:, None], 0.)
        terms = diff.square() * torch.exp(-logvars[:, None]) + logvars[:, None]
        terms = torch.where(observed[None], terms, 0.)
        # The observed-coordinate 2*pi constant is identical across components.
        return torch.softmax(F.log_softmax(self.logp, 0)[:, None] - .5 * terms.sum(-1), dim=0)

    def forward(self, x, adj, edge_index):
        x_orig = x[:, :self.x_features]
        x_imp = x_orig.repeat(self.n_components, 1, 1)

        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars.clamp(-14, 14))

        means_x = self.means[:, :self.x_features]
        vars_x = variances[:, :self.x_features]
        mean_x = torch.where(x_isnan, means_x.unsqueeze(1).expand_as(x_imp), x_imp)
        var_x = torch.where(x_isnan, vars_x.unsqueeze(1).expand_as(x_imp), torch.zeros_like(x_imp))

        tx = torch.matmul(mean_x, self.weight) + self.bias
        tv = torch.matmul(var_x, self.weight**2)

        conv_x = torch.stack([torch.spmm(adj, tx_k) for tx_k in tx])
        conv_v = torch.stack([torch.spmm(self.adj2, tv_k) for tv_k in tv])
        expected_x = ex_relu(conv_x, conv_v)

        gamma = self.responsibilities(x[:, :self.in_features])

        return torch.sum(expected_x * gamma.unsqueeze(2), dim=0)


class GCNmfConv_MI(GCNmfConv):
    'GCNmfConv with task-aware MI-weighted GMM responsibilities.'


    def __init__(self, in_features, out_features, data, n_components, dropout, feature_weights=None):
        self._init_feature_weights = feature_weights
        super().__init__(in_features, out_features, data, n_components, dropout)

    def reset_parameters(self):
        super().reset_parameters()
        if self._init_feature_weights is None:
            fw = torch.ones(self.in_features, dtype=torch.float)
        else:
            fw = torch.as_tensor(self._init_feature_weights, dtype=torch.float).view(-1)
            if fw.numel() != self.in_features:
                raise ValueError(f"feature_weights has {fw.numel()} dims, expected {self.in_features}")
            fw = torch.nan_to_num(fw, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

            fw = fw / fw.mean().clamp(min=1e-8)
        self.register_buffer('feature_weights', fw.to(self.logp.device), persistent=False)

    def forward(self, x, adj, edge_index):
        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars).clamp(min=1e-10)

        mean_mat = torch.where(x_isnan, self.means.unsqueeze(1).expand_as(x_imp), x_imp)
        var_mat = torch.where(x_isnan, variances.unsqueeze(1).expand_as(x_imp), torch.zeros_like(x_imp))

        tx = torch.matmul(mean_mat, self.weight) + self.bias
        tv = torch.matmul(var_mat, self.weight**2)

        conv_x = torch.stack([torch.spmm(adj, tx_k) for tx_k in tx])
        conv_v = torch.stack([torch.spmm(self.adj2, tv_k) for tv_k in tv])
        expected_x = ex_relu(conv_x, conv_v)


        sq_maha = torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1)
        dist = torch.sum(sq_maha * self.feature_weights.view(1, 1, -1), dim=2)
        log_prob = self.logp.unsqueeze(1) - 0.5 * dist
        gamma = torch.softmax(log_prob, dim=0)
        return torch.sum(expected_x * gamma.unsqueeze(2), dim=0)

class GCNConv_(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCNConv_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x
    
    
class GCNmf(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5, n_components=5,
                 init_x=None):
        super(GCNmf, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GCNmfConv(nfeat, nhid, data, n_components, dropout)
        init_features = _as_gmm_init_features(init_x)
        if init_features is not None:
            self.gc1.features = init_features
            self.gc1.reset_parameters()
        self.gc2 = GCNConv_(nhid, nclass, dropout)
        self.gc1test = GCNConv(nfeat, nhid)
        self.gc2test = GCNConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj, edge_index):


        x = self.gc1(x, adj, edge_index)

        
        x = self.gc2test(x, edge_index)

        return F.log_softmax(x, dim=1)


def get_symmetrically_normalized_adjacency(edge_index, n_nodes, edge_weight=None):
    'Given an edge_index, return the same edge_index and edge weights computed as'


    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    else:
        edge_weight = edge_weight.to(edge_index.device)
    row, col = edge_index[0], edge_index[1]

    device = edge_index.device if isinstance(edge_index, torch.Tensor) else edge_weight.device
    deg = torch.zeros(n_nodes, device=device)
    col = col.to(device)
    edge_weight = edge_weight.to(device)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, DAD

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x, edge_index: Adj, mask, test_mask, edge_weight=None):


        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]
        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(
            out, edge_index, n_nodes,
            no_propagate_mask=test_mask,
            edge_weight=edge_weight,
        )

        for _ in range(self.num_iterations):

            adj = adj.to(out.device)
            out = torch.sparse.mm(adj, out)


            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes, no_propagate_mask=None, edge_weight=None):
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(
            edge_index, n_nodes=n_nodes, edge_weight=edge_weight
        )

        if no_propagate_mask is not None:

            source_nodes = edge_index[1]
            edge_mask = ~no_propagate_mask[source_nodes]
            edge_index = edge_index[:, edge_mask]
            edge_weight = edge_weight[edge_mask]

        adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([n_nodes, n_nodes])).to(edge_index.device)
        return adj
    
    
def get_conv(conv_type, input_dim, output_dim):
    conv_type = conv_type.lower()
    if conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gcn":
        return GCNConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "cheb":
        return ChebConv(input_dim, output_dim, K=4)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")
    
    
class GNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, conv_type="GCN", jumping_knowledge=False,
    ):
        super(GNN, self).__init__()

        self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        if jumping_knowledge:
            self.lin = Linear(hidden_dim, num_classes)
            self.jump = JumpingKnowledge(mode="max", channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge

    def forward(self, x, edge_index=None, adjs=None, full_batch=True):
        return self.forward_full_batch(x, edge_index) if full_batch else self.forward_sampled(x, adjs)

    def forward_full_batch(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)

    def forward_sampled(self, x, adjs):


        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=1)

    def inference(self, x_all, inference_loader, device):
        'Get embeddings for all nodes to be used in evaluation'


        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


def pcfi(edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None, edge_weight=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    propagation_model = PCFI(num_iterations=num_iterations, alpha = alpha, beta=beta)
    return propagation_model.propagate(
        x=X, edge_index=edge_index, mask=feature_mask,
        mask_type=mask_type, edge_weight=edge_weight,
    )

class PCFI(torch.nn.Module):
    def __init__(self, num_iterations: int, alpha: float, beta: float):
        super(PCFI, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

    def propagate(self, x, edge_index, mask, mask_type, edge_weight = None):
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
                out = torch.zeros_like(x)
                out[mask] = x[mask]

            for _ in range(self.num_iterations):

                adj_c = adj_c.to(out.device)
                out = torch.sparse.mm(adj_c, out)
                out[mask] = x[mask]
            f_n2d = f_n2d.repeat(feat_dim,1)
        else:
            out = torch.zeros_like(x)
            if mask is not None:
                out[mask] = x[mask]
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, feat_dim)
            print('\n ==== propagation on {feat_dim} channels ===='.format(feat_dim=feat_dim))
            for i in range(feat_dim):
                adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], nv, edge_weight=edge_weight)
                for _ in range(self.num_iterations):
                    out[:,i] = torch.sparse.mm(adj_c, out[:,i].reshape(-1,1)).reshape(-1)
                    out[mask[:,i],i] = x[mask[:,i],i]
        cor = torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
        f_n2d = f_n2d.to(out.device)
        a_1 = (self.alpha ** f_n2d.T) * (out - torch.mean(out, dim=0))
        a_2 = torch.matmul(a_1, cor)
        out_1 = self.beta * (1 - (self.alpha ** f_n2d.T)) * a_2
        out = out + out_1
        return out

    def compute_f_n2d(self, edge_index, feature_mask, mask_type, feat_dim: OptTensor = None):
        nv = feature_mask.shape[0]
        if mask_type == 'structural':
            len_v_0tod_list = []
            f_n2d = torch.zeros(nv, dtype = torch.int)
            v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
            len_v_0tod_list.append(len(v_0))
            v_0_to_now = v_0
            f_n2d[v_0] = 0
            d = 1
            while True:
                v_d_hop_sub = k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                if len(v_d) == 0:
                    break
                f_n2d[v_d] = d
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                len_v_0tod_list.append(len(v_d))
                d += 1
        else:
            f_n2d = torch.zeros(feat_dim, nv)
            print('\n ==== compute f_n2d for {feat_dim} channels ===='.format(feat_dim=feat_dim))

            for i in range(feat_dim):
                v_0 = torch.nonzero(feature_mask[:,i]).view(-1)
                v_0_to_now = v_0
                f_n2d[i, v_0] = 0
                d=1
                while True:
                    v_d_hop_sub = k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
            print('\n ====== f_n2d is computed ======'.format(feat_dim=feat_dim))
        return f_n2d

  
    def compute_edge_weight_c(self, edge_index, f_n2d, n_nodes, edge_weight=None):

        row, col = edge_index[0], edge_index[1]

        f_n2d = f_n2d.to(edge_index.device)
        d_row = f_n2d[row]
        d_col = f_n2d[col]
        edge_weight_c = (self.alpha ** (d_col - d_row + 1)).to(edge_index.device)
        if edge_weight is not None:
            edge_weight_c = edge_weight_c * edge_weight.to(edge_index.device)

        device = edge_weight_c.device
        deg_W = torch.zeros(n_nodes, dtype=edge_weight_c.dtype, device=device)
        row = row.to(device)
        deg_W.index_add_(0, row, edge_weight_c)

        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values= A_Dinv, size=[n_nodes, n_nodes]).to(edge_index.device)

        return adj


        
    








    



    
class PEMix(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.0, n_components=5, pe_dim=8,
                 init_x=None):
        super(PEMix, self).__init__()
        self.x_channels = data.num_features
        self.pe_dim = pe_dim
        self.in_channels = self.x_channels + self.pe_dim
        

        from torch_geometric.data import Data
        init_features = _as_gmm_init_features(init_x)
        if init_features is None:
            fake_x = torch.zeros((data.num_nodes, self.in_channels))
        else:
            fake_x = torch.as_tensor(init_features, dtype=torch.float32)
            if fake_x.size(1) != self.in_channels:
                raise ValueError(
                    f"init_x has {fake_x.size(1)} dims, expected {self.in_channels}"
                )
        fake_data = Data(x=fake_x, edge_index=data.edge_index.cpu())
        if hasattr(data, 'adj'): fake_data.adj = data.adj.cpu()


        self.gc1 = PEMixGMMConv(
            self.x_channels, self.pe_dim, nhid, fake_data, n_components, dropout
        )
        self.classifier = GCNConv(nhid, int(data.y.max() + 1))
        self.dropout = dropout

    def forward(self, x, adj, edge_index): 
        x = self.gc1(x, adj, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x, edge_index) 


class PEMixFull(nn.Module):
    'Early-fusion ablation in which PE is also transformed and propagated.'

    def __init__(self, data, nhid=16, dropout=0.5, n_components=5, pe_dim=8,
                 init_x=None):
        super(PEMixFull, self).__init__()
        self.in_channels = data.num_features + pe_dim

        from torch_geometric.data import Data
        init_features = _as_gmm_init_features(init_x)
        if init_features is None:
            fake_x = torch.zeros((data.num_nodes, self.in_channels))
        else:
            fake_x = torch.as_tensor(init_features, dtype=torch.float32)
            if fake_x.size(1) != self.in_channels:
                raise ValueError(
                    f"init_x has {fake_x.size(1)} dims, expected {self.in_channels}"
                )
        fake_data = Data(x=fake_x, edge_index=data.edge_index.cpu())
        if hasattr(data, 'adj'):
            fake_data.adj = data.adj.cpu()

        self.gc1 = GCNmfConv(self.in_channels, nhid, fake_data, n_components, dropout)
        self.classifier = GCNConv(nhid, int(data.y.max() + 1))
        self.dropout = dropout

    def forward(self, x, adj, edge_index):
        x = self.gc1(x, adj, edge_index)
        x = F.relu(x)
        return self.classifier(x, edge_index)


class GCNmf_LF(nn.Module):
    'GCNmf con Late Fusion dei PE.'


    def __init__(self, data, nhid=16, dropout=0.5, n_components=5, pe_dim=8,
                 init_x=None):
        super(GCNmf_LF, self).__init__()

        nfeat = data.num_features
        nclass = int(data.y.max() + 1) if not hasattr(data, 'num_classes') else data.num_classes
 
        self.pe_dim = pe_dim
        self.dropout = dropout
 

        if hasattr(data, 'pe') and data.pe is not None:
            self.register_buffer('pe', data.pe.float(), persistent=False)
        else:
            raise ValueError("GCNmf_LF richiede data.pe (Laplacian PE)")
 

        self.gc1 = GCNmfConv(nfeat, nhid, data, n_components, dropout)
        init_features = _as_gmm_init_features(init_x)
        if init_features is not None:
            self.gc1.features = init_features
            self.gc1.reset_parameters()
 

        self.classifier = GCNConv(nhid + pe_dim, nclass)
 
    def forward(self, x, adj, edge_index):
        "x: [N, F] con NaN (solo X originali, NIENTE PE concatenati nell'input)"


        h_X = self.gc1(x, adj, edge_index)
        h_X = F.relu(h_X)
        h_X = F.dropout(h_X, p=self.dropout, training=self.training)
 

        h_full = torch.cat([h_X, self.pe], dim=1)
 

        out = self.classifier(h_full, edge_index)
 
        return F.log_softmax(out, dim=1)  


