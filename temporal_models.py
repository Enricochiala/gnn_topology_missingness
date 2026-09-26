"""Static repository architectures adapted to continuous multi-horizon targets.

PEMix uses the normalized observed-coordinate density in Eq. (12).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models import GCNFull, GCNmfConv, PEMixGMMConv, ex_relu


# Shared implementation for classification and forecasting.
GaussianPEMix = PEMixGMMConv


class SpatialBatch:
    def __init__(self,edges,weights,n,device):
        self.edges=torch.as_tensor(edges,dtype=torch.long,device=device)
        self.weights=torch.as_tensor(weights,dtype=torch.float32,device=device)
        self.n=n; self.device=device; self.cache={}
    def get(self,b):
        if b not in self.cache:
            edge=(self.edges[:,None,:]+torch.arange(b,device=self.device)[None,:,None]*self.n).reshape(2,-1)
            weight=self.weights.repeat(b)
            e,w=gcn_norm(edge,weight,b*self.n,add_self_loops=True)
            # sparse MM uses receiving rows, PyG edge_index uses source rows.
            adj=torch.sparse_coo_tensor(e.flip(0),w,(b*self.n,b*self.n)).coalesce()
            self.cache[b]=(edge,weight,adj)
        return self.cache[b]


class StaticRegressor(nn.Module):
    def __init__(self,name,features,outputs,graph,pe,init_x,config,embedding=None):
        super().__init__()
        from experiments import MAIN_MODELS
        if name not in MAIN_MODELS: raise ValueError(f'Unknown model: {name}')
        self.name=name; self.graph=graph; self.outputs=outputs
        hidden=config['hidden']; dropout=config['dropout']; self.dropout=dropout
        if name=='PEMix' and pe.shape[1]<config['q']:
            raise ValueError('Insufficient prepared positional-encoding dimensions')
        self.register_buffer('pe',torch.as_tensor(pe[:,:config.get('q',8)],device=graph.device))
        if name in ('gcnmf','PEMix'):
            q=self.pe.shape[1] if name=='PEMix' else 0
            fake=Data(x=torch.as_tensor(init_x),edge_index=graph.edges.cpu())
            fake.adj=graph.get(1)[2].cpu()
            if name=='gcnmf':
                self.first=GaussianPEMix(features,0,hidden,fake,config.get('k',5),dropout) if config.get('likelihood')=='gaussian' else GCNmfConv(features,hidden,fake,config.get('k',5),dropout)
            else:
                if config.get('likelihood') != 'gaussian':
                    raise ValueError('PEMix requires the normalized Gaussian likelihood')
                self.first=PEMixGMMConv(features,q,hidden,fake,config.get('k',5),dropout)
            self.final=GCNConv(hidden,outputs)
        else:
            self.gcn=GCNFull(features,hidden,outputs,num_layers=2,dropout=dropout)

    def forward(self,x):
        b,n,d=x.shape; edge,weight,adj=self.graph.get(b); flat=x.reshape(b*n,d)
        if self.name in ('gcnmf','PEMix'):
            if self.name=='PEMix': flat=torch.cat([flat,self.pe.repeat(b,1)],1)
            self.first.adj2=adj.square()
            z=self.first(flat,adj,edge)
            # Apply the configured dropout before the final graph-convolution layer.
            z=F.dropout(z,p=self.dropout,training=self.training)
            out=self.final(z,edge,weight)
        else:
            out=flat
            for i,conv in enumerate(self.gcn.convs):
                out=conv(out,edge,weight)
                if i<len(self.gcn.convs)-1:
                    out=F.dropout(F.relu(out),p=self.dropout,training=self.training)
        return out.reshape(b,n,self.outputs)

