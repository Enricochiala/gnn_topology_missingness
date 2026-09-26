import os
import sys
import warnings
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import ks_2samp
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import (
    from_networkx,
    to_scipy_sparse_matrix,
    to_undirected,
    subgraph,
    k_hop_subgraph,
    degree,
)


from tqdm import tqdm

from models import *
from fisf import fisf, FISF


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.typing")
warnings.filterwarnings("ignore")


seeds=[1, 43, 15, 118, 222]
def _default_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _default_torch_device()


def _clone_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _metric_from_logits(out, mask, y, metric="f1"):
    'Compute validation/test metric from model outputs.'


    metric = (metric or "f1").lower()
    y_true = y[mask].detach().cpu().numpy()
    out_m = out[mask]

    if metric == "f1":
        pred = out_m.argmax(dim=1).detach().cpu().numpy()
        return float(f1_score(y_true, pred, average="macro"))

    if metric in ("accuracy", "acc"):
        pred = out_m.argmax(dim=1).detach().cpu().numpy()
        return float(accuracy_score(y_true, pred))

    if metric in ("rocauc", "auc", "roc_auc"):

        if len(np.unique(y_true)) < 2:
            return float("nan")
        probs = torch.softmax(out_m, dim=1).detach().cpu().numpy()
        try:
            if probs.shape[1] == 2:
                return float(roc_auc_score(y_true, probs[:, 1]))
            return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            return float("nan")

    raise ValueError(f"Unknown metric: {metric}")


def _mean_or_nan(values):
    if len(values) == 0:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _rocauc_from_logits(out, mask, y):
    return _metric_from_logits(out, mask, y, "rocauc")

def fill_nan_with_col_mean_split(X, train_val_mask, test_mask):
    'Fill NaNs using column means from the supplied fitting rows.'


    X_filled = X.clone()
    nan_mask = torch.isnan(X_filled)


    col_means = []
    for j in range(X.shape[1]):
        col = X[train_val_mask, j]
        col_no_nan = col[~torch.isnan(col)]
        if len(col_no_nan) > 0:
            col_mean = col_no_nan.mean()
        else:
            col_mean = torch.tensor(0.0, device=X.device)
        col_means.append(col_mean)


    for j in range(X.shape[1]):
        X_filled[nan_mask[:, j], j] = col_means[j]

    return X_filled




def evaluate_gcnmf(
    data,
    max_epochs=1000,
    patience=40,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        data = data.to(device)
        y = data.y
        x = data.masks[seed]["X_incomp"].to(device)
        edge_index = data.edge_index.to(device)
        adj = data.adj.to(device)

        mixture_data = data.clone()
        mixture_data.x = x
        model = GCNmf(mixture_data, nhid=16, dropout=0.0, n_components=5).to(device)


        if hasattr(model, "gc1") and hasattr(model.gc1, "features"):
            model.gc1.features = x.detach().cpu().numpy()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        best_val_f1 = -float('inf')
        best_weights = None
        patience_counter = 0
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, adj, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, adj, edge_index)
                val_f1 = _metric_from_logits(out_val, val_mask, y, metric)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out = model(x, adj, edge_index)
            test_pred = out[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu(), test_pred.cpu())
            test_loss = F.cross_entropy(out[test_mask], y[test_mask]).item()
            test_f1 = _metric_from_logits(out, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out, test_mask, y)

            accs.append(test_acc)
            f1s.append(test_f1)
            rocaucs.append(test_rocauc)
            losses.append(test_loss)

    return (
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    )


def evaluate_gcn(
    data,
    hidden_channels=128,
    max_epochs=500,
    patience=50,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    method=None,
    mod=None,
    metric="f1",
    n_components=5,
):
    'Training loop per modelli custom-input (gcnmi, PEMix, PEMix_full, gcnmf_lf,'


    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask   = data.masks[seed]["val_mask"].to(device)
        test_mask  = data.masks[seed]["test_mask"].to(device)

        data = data.to(device)
        y = data.y.to(device)
        x = data.masks[seed]["X_incomp"].clone().to(device)
        edge_index = data.edge_index.to(device)


        needs_nan = (mod in ("PEMix", "PEMix_full", "gcnmf_lf",
                             "PEMix_gated", "PEMix_hybrid"))
        if not needs_nan:
            train_val_mask = train_mask
            if torch.isnan(x).any():
                x = fill_nan_with_col_mean_split(x, train_val_mask, test_mask)


        if mod is None or mod == "gnnmim":
            lr = 0.01
            hidden_channels = 128
            model = GCNFull(
                x.size(1), hidden_channels,
                num_classes=data.num_classes, num_layers=2
            ).to(device)

        elif mod == "PEMix":


            lr = 0.005
            hidden_channels = 16
            pe_dim = x.size(1) - data.num_features if hasattr(data, 'num_features') else 8
            model = PEMix(data, nhid=hidden_channels, dropout=0.0,
                             n_components=n_components, pe_dim=pe_dim,
                             init_x=x).to(device)

        elif mod in ("PEMix_full", "PEMix_gated", "PEMix_hybrid"):


            lr = 0.005
            hidden_channels = 16
            pe_dim = x.size(1) - data.num_features if hasattr(data, 'num_features') else 8
            model = PEMixFull(data, nhid=hidden_channels, dropout=0.5,
                                  n_components=n_components, pe_dim=pe_dim,
                                  init_x=x).to(device)

        elif mod == "gcnmf_lf":


            lr = 0.005
            hidden_channels = 16
            pe_dim = data.pe.size(1) if hasattr(data, 'pe') and data.pe is not None else 8
            model = GCNmf_LF(data, nhid=hidden_channels, dropout=0.5,
                             n_components=n_components, pe_dim=pe_dim,
                             init_x=x).to(device)

        elif mod in ("gnnmim_pe", "gcnpe"):

            lr = 0.01
            hidden_channels = 128
            model = GCNFull(
                x.size(1), hidden_channels,
                num_classes=data.num_classes, num_layers=2
            ).to(device)

        elif mod == "peonly":


            lr = 0.005
            hidden_channels = 16
            model = GCNFull(
                x.size(1), hidden_channels,
                num_classes=data.num_classes, num_layers=2, dropout=0.5
            ).to(device)

        elif mod == "mlppe":


            lr = 0.005
            hidden_channels = 16
            model = MLPFull(
                x.size(1), hidden_channels,
                num_classes=data.num_classes, num_layers=2, dropout=0.5
            ).to(device)

        else:

            lr = 0.01
            hidden_channels = 128
            model = GCNFull(
                x.size(1), hidden_channels,
                num_classes=data.num_classes, num_layers=2
            ).to(device)


        class_counts = torch.bincount(y[train_mask])
        weight = 1.0 / class_counts.float().clamp(min=1.0)
        weight = weight / weight.sum()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_f1 = -float('inf')
        best_weights = None
        patience_counter = 0


        def _forward(m, x_in):
            if mod in ("PEMix", "PEMix_full", "gcnmf_lf",
                       "PEMix_gated", "PEMix_hybrid"):
                return m(x_in, data.adj.to(device), edge_index)
            return m(x_in, edge_index)


        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = _forward(model, x)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = _forward(model, x)
                val_f1 = _metric_from_logits(out_val, val_mask, y, metric)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break


        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = _forward(model, x)
            valid_test_mask = test_mask & (y != -1)
            test_pred = out_test[valid_test_mask].argmax(dim=1)
            test_acc = accuracy_score(
                y[valid_test_mask].cpu(), test_pred.cpu()
            )
            test_loss = F.cross_entropy(
                out_test[valid_test_mask], y[valid_test_mask]
            ).item()
            test_f1 = _metric_from_logits(out_test, valid_test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, valid_test_mask, y)

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_f1)
            rocaucs.append(test_rocauc)


    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    )

def feature_propagation(edge_index, X, feature_mask, num_iterations, test_mask, edge_weight=None):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    return propagation_model.propagate(
        x=X, edge_index=edge_index, mask=feature_mask,
        test_mask=test_mask, edge_weight=edge_weight,
    )


def filling(filling_method, edge_index, X, feature_mask, num_iterations=None, test_mask=None, edge_weight=None):
    X_reconstructed = feature_propagation(
        edge_index, X, feature_mask, num_iterations, test_mask,
        edge_weight=edge_weight,
    )
    return X_reconstructed


def evaluate_fp(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    accs, losses, f1s, rocaucs = [], [], [], []

    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        missing_feature_mask = ~torch.isnan(x_incomp)

        filling_method = "feature_propagation"
        num_iterations = 40
        num_layers = 2
        dropout = 0.5

        filled_features = filling(
            filling_method,
            data.edge_index.to(device),
            x_incomp,
            missing_feature_mask,
            num_iterations,
            test_mask=data.masks[seed]["test_mask"].to(device),
        )

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=num_layers,
            hidden_dim=hidden_channels,
            dropout=dropout,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.where(missing_feature_mask, x_incomp, filled_features).to(device)

        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_f1 = -float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_f1 = _metric_from_logits(out_val, val_mask, y, metric)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_score = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_score)
            rocaucs.append(test_rocauc)

    return (
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    )


def evaluate_pcfi(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    x_original=None,
    metric="f1",
):
    accs, losses, f1s, rocaucs = [], [], [], []

    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        missing_feature_mask = ~torch.isnan(x_incomp)

        mask_type = "structural"
        num_iterations = 40
        alpha, beta = 0.9, 1.0
        dropout = 0.5

        filled_features = pcfi(
            data.edge_index.to(device),
            x_incomp,
            missing_feature_mask,
            num_iterations,
            mask_type,
            alpha,
            beta,
        )

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=dropout,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        x = filled_features.to(device)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_f1 = -float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_f1 = _metric_from_logits(out_val, val_mask, y, metric)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_f1 = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_f1)
            rocaucs.append(test_rocauc)

            if x_original is not None:
                out_orig = model(x_original.to(device), edge_index)
                _ = out_orig[test_mask]

    return (
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    )




def _safe_train_val_test_split(y, seed, test_size=0.2, val_size=0.125):
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    idx = np.arange(len(y_np))

    def _stratify_or_none(labels, holdout_size):
        _, counts = np.unique(labels, return_counts=True)
        n_classes = len(counts)
        n_holdout = int(np.ceil(len(labels) * holdout_size))
        n_train = len(labels) - n_holdout
        if n_classes <= 1 or counts.min() < 2:
            return None
        if n_holdout < n_classes or n_train < n_classes:
            return None
        return labels

    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=_stratify_or_none(y_np, test_size),
        random_state=seed,
    )
    y_train_val = y_np[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_size,
        stratify=_stratify_or_none(y_train_val, val_size),
        random_state=seed,
    )
    return idx_train, idx_val, idx_test


def umcar(data, p_miss_dict, seeds):
    masks = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        idx_train, idx_val, idx_test = _safe_train_val_test_split(data.y, seed)

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        n, d = data.x.shape
        mask = torch.zeros(n, d, dtype=torch.bool)

        for mask_name, node_mask, p in [('train', train_mask | val_mask, p_miss_dict['train']),
                                        ('test', test_mask, p_miss_dict['test'])]:
            selected_nodes = torch.where(node_mask)[0]
            temp_mask = torch.rand(len(selected_nodes), d) < p
            mask[selected_nodes] = temp_mask

        for j in range(d):
            if mask[:, j].all():
                i = torch.randint(0, n, (1,))
                mask[i, j] = False

        X_incomp = data.x.clone()
        X_incomp[mask] = float('nan')

        masks[seed] = {
            'X_incomp': X_incomp,
            'mask': mask,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

    return masks




def _pe_edge_weight(edge_index, pe, eps=1e-6):
    'RBF edge weights from Laplacian PE distances.'
    pe = pe.float()
    row, col = edge_index[0], edge_index[1]
    dist2 = (pe[row] - pe[col]).pow(2).sum(dim=1)
    positive = dist2[dist2 > 0]
    if positive.numel() == 0:
        return torch.ones(edge_index.size(1), dtype=pe.dtype, device=pe.device)
    bandwidth = positive.median().clamp(min=eps)
    return torch.exp(-dist2 / (2.0 * bandwidth)).clamp(min=eps)


def evaluate_fppe(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    'FP where PE modulates propagation weights; PE is not fed to the classifier.'
    if not hasattr(data, "pe") or data.pe is None:
        raise ValueError("fppe requires data.pe")

    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    edge_index = data.edge_index.to(device)
    pe_edge_weight = _pe_edge_weight(edge_index, data.pe.to(device))

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        feature_mask = ~torch.isnan(x_incomp)
        filled = filling(
            "feature_propagation",
            edge_index,
            x_incomp,
            feature_mask,
            40,
            test_mask=data.masks[seed]["test_mask"].to(device),
            edge_weight=pe_edge_weight,
        )
        x = torch.where(feature_mask, x_incomp, filled).to(device)

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=0.5,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        y = data.y.to(device)
        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_score, best_weights, patience_counter = -1.0, None, 0
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_score = _metric_from_logits(out_val, val_mask, y, metric)
            if val_score > best_val_score:
                best_val_score = val_score
                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_score = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)

        accs.append(test_acc)
        losses.append(test_loss)
        f1s.append(test_score)
        rocaucs.append(test_rocauc)

    return [
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    ]


def evaluate_pcfipe(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    'PCFI where PE modulates propagation weights; PE is not fed to the classifier.'
    if not hasattr(data, "pe") or data.pe is None:
        raise ValueError("pcfipe requires data.pe")

    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    edge_index = data.edge_index.to(device)
    pe_edge_weight = _pe_edge_weight(edge_index, data.pe.to(device))

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        feature_mask = ~torch.isnan(x_incomp)
        x = pcfi(
            edge_index,
            x_incomp,
            feature_mask,
            num_iterations=40,
            mask_type="structural",
            alpha=0.9,
            beta=1.0,
            edge_weight=pe_edge_weight,
        ).to(device)

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=0.5,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        y = data.y.to(device)
        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_score, best_weights, patience_counter = -1.0, None, 0
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_score = _metric_from_logits(out_val, val_mask, y, metric)
            if val_score > best_val_score:
                best_val_score = val_score
                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_score = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)

        accs.append(test_acc)
        losses.append(test_loss)
        f1s.append(test_score)
        rocaucs.append(test_rocauc)

    return [
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    ]


def evaluate_fp_pe(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    'FP baseline + PE concatenated AFTER the imputation, BEFORE the GNN.'
    accs, f1s, rocaucs = [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    pe = data.pe.to(device).float()
    pe_dim = pe.size(1)

    for seed in seed_keys:
        torch.manual_seed(seed); np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        feature_mask = ~torch.isnan(x_incomp)
        filled = filling("feature_propagation",
                         data.edge_index.to(device), x_incomp, feature_mask, 40,
                         test_mask=data.masks[seed]["test_mask"].to(device))
        x_filled = torch.where(feature_mask, x_incomp, filled).to(device)
        x_aug = torch.cat([x_filled, pe], dim=1)

        model = GNN(
            num_features=x_aug.shape[1],
            num_classes=int(data.y.max() + 1),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=0.5,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        edge_index = data.edge_index.to(device)
        y          = data.y.to(device)
        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask   = data.masks[seed]["val_mask"].to(device)
        test_mask  = data.masks[seed]["test_mask"].to(device)

        best_val_score, best_w, pc = -1.0, None, 0
        for epoch in range(max_epochs):
            model.train(); opt.zero_grad()
            out = model(x_aug, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                out_val = model(x_aug, edge_index)
                vs = _metric_from_logits(out_val, val_mask, y, metric)
            if vs > best_val_score:
                best_val_score = vs
                best_w = _clone_state_dict(model)
                pc = 0
            else:
                pc += 1
                if pc >= patience: break

        if best_w is not None:
            model.load_state_dict(best_w)
        model.eval()
        with torch.no_grad():
            out = model(x_aug, edge_index)
            tp = out[test_mask].argmax(1)
            test_score = _metric_from_logits(out, test_mask, y, metric)
            test_acc = (tp == y[test_mask]).float().mean().item()
            test_rocauc = _rocauc_from_logits(out, test_mask, y)
        accs.append(test_acc); f1s.append(test_score); rocaucs.append(test_rocauc)

    return [
        float(np.mean(accs)),
        float("nan"),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    ]


def evaluate_pcfi_pe(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    metric="f1",
):
    'PCFI + late-fusion PE.'
    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    pe = data.pe.to(device).float()

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)
        missing_feature_mask = ~torch.isnan(x_incomp)

        mask_type      = "structural"
        num_iterations = 40
        alpha, beta    = 0.9, 1.0
        dropout        = 0.5

        filled_features = pcfi(
            data.edge_index.to(device),
            x_incomp,
            missing_feature_mask,
            num_iterations,
            mask_type,
            alpha,
            beta,
        )

        x_aug = torch.cat([filled_features.to(device), pe], dim=1)

        model = GNN(
            num_features=x_aug.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=dropout,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        edge_index = data.edge_index.to(device)
        y          = data.y.to(device)
        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask   = data.masks[seed]["val_mask"].to(device)
        test_mask  = data.masks[seed]["test_mask"].to(device)

        best_val_score, best_weights, patience_counter = -1.0, None, 0
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out  = model(x_aug, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val  = model(x_aug, edge_index)
                val_score = _metric_from_logits(out_val, val_mask, y, metric)
            if val_score > best_val_score:
                best_val_score = val_score
                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            out_test  = model(x_aug, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc  = accuracy_score(y[test_mask].cpu().numpy(),
                                       test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_score = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)

        accs.append(test_acc); losses.append(test_loss); f1s.append(test_score); rocaucs.append(test_rocauc)

    return [
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    ]


def _evaluate_fisf_common(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_pe=False,
    metric="f1",
    num_iterations=50,
    mask_type="uniform",
    alpha=0.5,
    beta=0.1,
    gamma=0.5,
):
    'Evaluate FISF+ imputation followed by the same GCN classifier.'


    accs, losses, f1s, rocaucs = [], [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    pe_edge_weight = None
    if use_pe:
        if not hasattr(data, "pe") or data.pe is None:
            raise ValueError("data.pe not found; compute Laplacian PE before evaluate_fisf_pe.")
        pe_edge_weight = _pe_edge_weight(data.edge_index.to(device), data.pe.to(device))

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"].to(device)

        observed_mask = (~torch.isnan(x_incomp)).bool().to(device)
        x_zero = torch.nan_to_num(x_incomp, nan=0.0).float()

        edge_index = data.edge_index.to(device)
        if x_zero.std(dim=0).max() < 1e-8:


            x_fisf = x_zero
        else:
            x_fisf = fisf(
                edge_index=edge_index,
                X=x_zero,
                feature_mask=observed_mask,
                num_iterations=num_iterations,
                mask_type=mask_type,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                edge_weight=pe_edge_weight,
            ).float()


        x = torch.nan_to_num(x_fisf, nan=0.0, posinf=0.0, neginf=0.0)

        model = GNN(
            num_features=x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=0.5,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        y = data.y.to(device)
        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val = -1.0
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_score = _metric_from_logits(out_val, val_mask, y, metric)

            if val_score > best_val:
                best_val = val_score
                best_weights = _clone_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_score = _metric_from_logits(out_test, test_mask, y, metric)
            test_rocauc = _rocauc_from_logits(out_test, test_mask, y)
            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_score)
            rocaucs.append(test_rocauc)

    return (
        float(np.mean(accs)),
        float(np.mean(losses)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
        _mean_or_nan(rocaucs),
    )


def evaluate_fisf(data, hidden_channels=16, max_epochs=1000, patience=30,
                   seeds=None, device="cuda" if torch.cuda.is_available() else "cpu",
                   metric="f1"):
    return _evaluate_fisf_common(
        data, hidden_channels=hidden_channels, max_epochs=max_epochs,
        patience=patience, seeds=seeds, device=device, use_pe=False,
        metric=metric,
    )


def evaluate_fisf_pe(data, hidden_channels=16, max_epochs=1000, patience=30,
                      seeds=None, device="cuda" if torch.cuda.is_available() else "cpu",
                      metric="f1"):
    'FISF where PE modulates imputation weights; PE is not fed to the GCN.'
    return _evaluate_fisf_common(
        data, hidden_channels=hidden_channels, max_epochs=max_epochs,
        patience=patience, seeds=seeds, device=device, use_pe=True,
        metric=metric,
    )
