"""Exact distance acceleration for retained PCFI/FISF implementations.

Propagation/update equations stay in models.py and fisf.py. A shortest-path
lookup replaces repeated k-hop subgraph searches. Legacy unreachable-node
sentinel zero is retained, including for entirely missing channels.
"""
from functools import lru_cache
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
from models import PCFI
from fisf import FISF


class DistanceLookup:
    def __init__(self,edges,n):
        edges=np.asarray(edges)
        adjacency=sp.csr_matrix((np.ones(edges.shape[1]),edges),shape=(n,n))
        self.distances=shortest_path(adjacency,directed=True,unweighted=True)
        self.n=n

    @lru_cache(maxsize=4096)
    def column(self,key):
        observed=np.frombuffer(key,dtype=np.bool_)
        if observed.any():
            out=self.distances[:,observed].min(1)
            out=np.where(np.isfinite(out),out,0.)
        else:out=np.zeros(self.n)
        return out.astype('float32')

    def features(self,mask,columns=None):
        a=mask.detach().cpu().numpy()
        out=np.zeros((a.shape[1],a.shape[0]),dtype='float32')
        columns=range(a.shape[1]) if columns is None else columns
        for j in columns:
            j=int(j);out[j]=self.column(np.ascontiguousarray(a[:,j]).tobytes())
        return torch.as_tensor(out,device=mask.device)


class FastPCFI(PCFI):
    def __init__(self,lookup,*args):super().__init__(*args);self.lookup=lookup
    def compute_f_n2d(self,edge_index,feature_mask,mask_type,feat_dim=None):
        if mask_type=='structural':return self.lookup.features(feature_mask,[0])[0]
        return self.lookup.features(feature_mask)


class FastFISF(FISF):
    def __init__(self,lookup,*args):super().__init__(*args);self.lookup=lookup
    def compute_f_n2d(self,edge_index,feature_mask,mask_type,pre=None,feat_dim=None,virtual_idx=None):
        if mask_type=='structural':
            j=0 if pre is None else int(pre)
            return self.lookup.features(feature_mask,[j])[j]
        return self.lookup.features(feature_mask,virtual_idx if mask_type=='virtual' else None)


def _diffuse(x,mask,edge,weights,distance,alpha,iterations,virtual_distance=None,beta=None):
    """Channel-specific sparse multiplication, vectorized over feature columns."""
    row,col=edge
    delta=distance[col]-distance[row]
    edge_values=torch.pow(alpha,delta+1)
    if virtual_distance is not None:
        # Ratio of alpha^d * beta^virtual_d in original FISF, in a numerically
        # stable form which never materializes vanishing global confidences.
        edge_values=edge_values*torch.pow(beta,virtual_distance[col]-virtual_distance[row])
    if weights is not None:edge_values=edge_values*weights[:,None]
    degree=torch.zeros_like(x).index_add_(0,row,edge_values)
    norm=edge_values/degree[row].clamp(min=torch.finfo(x.dtype).tiny)
    out=torch.where(mask,x,0.)
    for _ in range(iterations):
        out=torch.zeros_like(out).index_add_(0,row,norm*out[col])
        out=torch.where(mask,x,out)
    return out


def _pcfi_propagate(self,x,edge_index,mask,mask_type,edge_weight=None):
    if mask_type!='uniform':return PCFI.propagate(self,x,edge_index,mask,mask_type,edge_weight)
    distance=self.lookup.features(mask).T
    out=_diffuse(x,mask,edge_index,edge_weight,distance,self.alpha,self.num_iterations)
    correlation=torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
    confidence=torch.pow(self.alpha,distance)
    return out+self.beta*(1-confidence)*((confidence*(out-out.mean(0)))@correlation)


def _fisf_propagate(self,x,edge_index,mask,mask_type,edge_weight=None):
    if mask_type!='uniform':return FISF.propagate(self,x,edge_index,mask,mask_type,edge_weight)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    n,d=x.shape
    distance=self.lookup.features(mask).T
    out=_diffuse(x,mask,edge_index,edge_weight,distance,self.alpha,self.num_iterations)
    low=torch.topk(torch.var(out,dim=0),int(d*self.gamma),largest=False).indices
    virtual=torch.zeros_like(mask)
    for j in low.cpu().tolist():
        source=int(np.random.choice(n,1,replace=False)[0])
        x[source,j]=float(torch.rand(1).item());mask[source,j]=True;virtual[source,j]=True
    distance=self.lookup.features(mask).T
    virtual_distance=self.lookup.features(virtual,low.cpu().tolist()).T
    return _diffuse(x,mask,edge_index,edge_weight,distance,self.alpha,self.num_iterations,virtual_distance,self.beta)


# Retain the inherited configuration/public interfaces. Only execution is
# vectorized; the channel updates and virtual-feature mechanism are unchanged.
FastPCFI.propagate=_pcfi_propagate
FastFISF.propagate=_fisf_propagate
