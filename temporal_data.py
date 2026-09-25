"""Audited temporal sources and causal static-window representation.

No imputations or learned graph construction happen in this module. A node is a
sensor; feature j = lag * channels + channel. Targets are strictly future data.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh
import torch

DATASETS = ('engrad', 'aqi', 'pv_us', 'graphmso', 'metr_la', 'pems_bay')
WINDOWS = {'engrad': (24, 6), 'aqi': (24, 6), 'pv_us': (72, 6),
           'graphmso': (72, 36), 'metr_la': (24, 12), 'pems_bay': (24, 12)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def haversine(coords):
    # Preserve the official coordinate dtype and sklearn haversine arithmetic.
    from sklearn.metrics.pairwise import haversine_distances
    radians=np.radians(np.asarray(coords))
    return (haversine_distances(radians)*6371.0088).astype(radians.dtype)


def geographic_graph(dist,theta=None,knn=None,connect=True,connection_theta=None):
    theta=np.std(dist) if theta is None else theta
    sim=np.exp(-np.square(dist/theta))
    adj=sim.copy()
    if knn is not None:
        # Exactly TSL top_k: partition first, then threshold and remove diagonal.
        adj=adj-np.diag([np.inf]*len(adj)).astype(adj.dtype)
        non_topk=np.argpartition(adj,-knn)[:,:-knn]
        adj[np.arange(len(adj))[:,None],non_topk]=0
    adj[adj<.1]=0;np.fill_diagonal(adj,0)
    adj=np.maximum(adj,adj.T)
    # HD-TTS connects using get_similarity() defaults, not necessarily the
    # explicit theta used to construct the thresholded initial adjacency.
    sim=np.exp(-np.square(dist/(theta if connection_theta is None else connection_theta)))
    np.fill_diagonal(sim,0);added=[]
    while connect:
        count,comp=connected_components(sp.csr_matrix(adj),directed=False)
        if count==1:break
        store=[(0.,None,None) for _ in range(count)]
        for i in range(count-1):
            src=np.flatnonzero(comp==i)
            for j in range(i+1,count):
                dst=np.flatnonzero(comp==j)
                sub=sim[np.ix_(src,dst)]
                r,c=np.unravel_index(sub.argmax(),sub.shape)
                value=sub[r,c];r,c=int(src[r]),int(dst[c])
                if value>store[i][0]:store[i]=(value,r,c)
                if value>store[j][0]:store[j]=(value,c,r)
        links=[(r,c) for value,r,c in store if value>0]
        if not links:break
        for r,c in links:adj[r,c]=adj[c,r]=.1;added.append((r,c))
    return adj,added


def graphmso():
    # Exact HD-TTS v0.3 generating process, seed 123, knn3, order 3,
    # max_neighbors 5; no original artificially injected point/block mask.
    n, t, seed = 100, 10000, 123
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), 3)
    cols = np.concatenate([rng.choice(np.delete(np.arange(n), i), 3, replace=False) for i in range(n)])
    a = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n))
    kernel = a + a@a + a@a@a
    kernel.setdiag(0); kernel.eliminate_zeros(); kernel.data[:] = 1
    kernel = kernel.toarray()
    rng = np.random.default_rng(seed+401)
    for i in range(n):
        nnz = np.flatnonzero(kernel[i])
        if len(nnz)>5: kernel[i, rng.choice(nnz, len(nnz)-5, replace=False)] = 0
    signal = np.sin(np.exp(-np.arange(n)/n)[:,None] * np.arange(t)[None,:])
    values = (signal + kernel@signal).T[...,None].astype('float32')
    # HD-TTS publishes a receiving-row adjacency. Convert to PyG source->target.
    return values, a.toarray().T, np.arange(t), ['signal'], [str(i) for i in range(n)]


def load_raw(name, root):
    root = Path(root); info = {}; extra = {}
    if name == 'graphmso':
        x, adj, times, channels, nodes = graphmso()
        mask = np.isfinite(x)
    elif name == 'engrad':
        path = root/'engrad.h5'; df = pd.read_hdf(path, 'data')
        meta = pd.read_hdf(path, 'metadata')
        nodes = list(df.columns.get_level_values(0).unique())
        channels = list(df.columns.get_level_values(1).unique())
        df = df.reindex(columns=pd.MultiIndex.from_product([nodes,channels]))
        x = df.to_numpy(dtype='float32').reshape(len(df),len(nodes),len(channels))
        # Preserve valid zero radiation, as in the main complete-data task.
        x[:,:,channels.index('precipitation')] /= 10
        mask = np.isfinite(x); times = df.index.asi8
        dist = haversine(meta.loc[nodes,['lat','lon']])
        adj, links = geographic_graph(dist,50,8,connection_theta=np.std(dist))
        info.update(precipitation_unit='cm',zero_radiation_is_observed=True, added_edges=links)
    elif name == 'aqi':
        path = root/'aqi/full437.h5'; df = pd.read_hdf(path,'pm25')
        nodes = list(df.columns); channels = ['pm25']
        meta = pd.read_hdf(path,'stations').loc[nodes]
        x = df.to_numpy(dtype='float32')[...,None]; mask = np.isfinite(x); times=df.index.asi8
        dist=haversine(meta[['latitude','longitude']])
        theta=float(np.std(dist[:36,:36]))
        adj, links = geographic_graph(dist,theta)
        info['kernel_theta_km']=theta
        info.update(added_edges=links, ignores_imputation_evaluation_mask=True)
    elif name == 'pv_us':
        path = root/'pv_us.h5'
        df = pd.read_hdf(path,'actual').sort_index(axis=1, level=0)
        meta = pd.read_hdf(path,'metadata').sort_index()
        # Read only the literal timezone mapping from the official loader.
        tree = ast.parse(Path('research/sources/hdtts/lib/datasets/pv_us.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        mapping = ast.literal_eval(next(n.value for n in cls.body if isinstance(n,ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id=='tz_mapper' for t in n.targets)))
        zones = meta.state.map(mapping)
        if zones.isna().any(): raise ValueError('Unmapped PV timezone')
        target_tz = zones.mode().iloc[0]
        pieces=[]
        for tz in zones.unique():
            ids = meta.index[zones==tz]
            pieces.append(df.loc[:,ids].tz_localize(tz).tz_convert(target_tz))
        df = pd.concat(pieces,axis=1).sort_index(axis=1,level=0)
        # Keep absent timezone-boundary observations missing, never backfill.
        df = df.resample('20min').mean()
        nodes = list(df.columns.get_level_values(0))
        broken = nodes[485]
        df = df.drop(columns=broken,level=0); nodes.pop(485)
        channels=['power_mw']; x=df.to_numpy(dtype='float32')[...,None]
        mask=np.isfinite(x)&(x>0); times=df.index.asi8
        adj, links = geographic_graph(haversine(meta.loc[nodes,['lat','lon']]),150,8)
        info.update(added_edges=links, excluded_broken_node=broken,
                    zero_policy='original benchmark excludes nighttime zeros; not sensor missingness', timezone=target_tz)
    else:
        path=root/name/f'{name}.h5'; df=pd.read_hdf(path)
        df=df.reindex(pd.date_range(df.index[0],df.index[-1],freq='5min'))
        nodes=list(df.columns); channels=['speed_mph']
        x=df.to_numpy(dtype='float32')[...,None]
        mask=np.isfinite(x)&(x!=0); times=df.index.asi8
        distances=pd.read_csv(root/name/f'distances_{"la" if name=="metr_la" else "bay"}.csv')
        ids={int(v):i for i,v in enumerate(nodes)}
        dist=np.full((len(nodes),len(nodes)),np.inf,dtype=np.float32)
        for source,target,value in distances.itertuples(index=False,name=None):
            if int(source) in ids and int(target) in ids: dist[ids[int(source)],ids[int(target)]]=value
        adj=np.exp(-(dist/dist[np.isfinite(dist)].std())**2)
        adj[adj<.1]=0; np.fill_diagonal(adj,0)
        adj=adj.T # TSL adjacency -> PyG source/destination convention.
        info.update(zero_policy='zero speed treated as missing by official loader', directed=True)
    edges=np.array(np.nonzero(adj),dtype='int64')
    extra.update(values=x,observed=mask, timestamps=np.asarray(times,dtype='int64'),
                 edge_index=edges,edge_weight=adj[tuple(edges)].astype('float32'),
                 channels=np.array(channels,dtype='str'), node_ids=np.array(nodes,dtype='str'))
    count,_=connected_components(sp.csr_matrix(adj),directed=False)
    info.update(dataset=name,shape=list(x.shape),original_nonfinite_fraction=float((~np.isfinite(x)).mean()),
                benchmark_missing_fraction=float((~mask).mean()),zero_fraction=float((x==0).mean()),
                directed_edges=int(edges.shape[1]),components=int(count),details=WINDOWS[name],
                first_timestamp=str(pd.Timestamp(times[0])) if name!='graphmso' else str(times[0]),
                last_timestamp=str(pd.Timestamp(times[-1])) if name!='graphmso' else str(times[-1]))
    return extra,info


def temporal_split(timestamps, name):
    """Strict chronological holdout; no future dates used in fitting.

    EngRAD: through September 2019 train, October-December 2019 validation,
    2020 test. Other datasets: 70/10/20 chronological timestamps.
    Windows crossing any boundary are excluded in their entirety.
    """
    if name=='engrad':
        a=int(np.searchsorted(timestamps,pd.Timestamp('2019-10-01').value))
        b=int(np.searchsorted(timestamps,pd.Timestamp('2020-01-01').value))
    else: a,b=int(.7*len(timestamps)),int(.8*len(timestamps))
    return ((0,a),(a,b),(b,len(timestamps)))


def origins(bounds,window,horizon,stride):
    return [np.arange(a+window,b-horizon+1,stride,dtype='int64') for a,b in bounds]


def topology_pe(edges,n,k):
    """Fixed symmetric spatial Laplacian, exclude ALL nullspace eigenvectors.

    For directed graphs PE uses the undirected support, while model edges retain
    their direction. Standardization uses graph geometry, never feature values.
    """
    a=sp.coo_matrix((np.ones(edges.shape[1]),edges),shape=(n,n)).tocsr()
    a=a.maximum(a.T); a.setdiag(0); a.eliminate_zeros(); a.data[:]=1
    lap=sp.csgraph.laplacian(a,normed=True).toarray()
    vals,vec=eigh(lap)
    vec=vec[:,vals>1e-8][:,:k]
    if vec.shape[1]!=k: raise ValueError('Insufficient nontrivial Laplacian eigenvectors')
    for j in range(k):
        pivot=np.argmax(np.abs(vec[:,j]))
        if vec[pivot,j]<0: vec[:,j]*=-1
    vec=(vec-vec.mean(0))/np.maximum(vec.std(0),1e-8)
    return vec.astype('float32')


def artificial_mask(edges,shape,seed,rate,mechanism,cache_dir=None):
    t,n,c=shape
    if not np.isfinite(rate) or not 0<=rate<=1: raise ValueError('Invalid missing rate')
    if mechanism=='natural':
        if rate!=0: raise ValueError('Natural missingness forbids additional injection')
        return np.zeros(shape,dtype=bool)
    if mechanism=='UMCAR':
        return np.random.default_rng(seed).random(shape)<rate
    if mechanism=='RT':
        from missingness_shared_seed import rtmar_shared_seed_mask
        mask=rtmar_shared_seed_mask(edges,n,t*c,rate,seed,cache_dir=cache_dir)
        return mask.numpy().reshape(n,t,c).transpose(1,0,2)
    raise ValueError('Only RT and UMCAR are supported')


@dataclass
class StaticWindows:
    values: np.ndarray
    observed: np.ndarray
    input_observed: np.ndarray
    splits: list
    window: int
    horizon: int
    center: np.ndarray
    scale: np.ndarray
    calendar: np.ndarray = None

    @classmethod
    def build(cls,data,name,seed,rate,mechanism,stride=None,mask_cache=None):
        x=data['values']; obs=data['observed'].astype(bool)
        if name in ('aqi','pv_us','metr_la') and mechanism!='natural':
            raise ValueError(f'Additional injection forbidden for {name}')
        w,h=WINDOWS[name]
        bounds=temporal_split(data['timestamps'],name)
        observed=obs & ~artificial_mask(data['edge_index'],x.shape,seed,rate,mechanism,mask_cache)
        train=np.where(observed[:bounds[0][1]],x[:bounds[0][1]],np.nan)
        center=np.nanmean(train,axis=(0,1)); scale=np.nanstd(train,axis=(0,1))
        center=np.nan_to_num(center,nan=0); scale=np.where(np.isfinite(scale)&(scale>1e-6),scale,1)
        calendar=None
        if name!='graphmso':
            dates=pd.to_datetime(data['timestamps'],utc=True)
            day=(dates.hour.to_numpy()+dates.minute.to_numpy()/60)/24
            year=(dates.dayofyear.to_numpy()-1+day)/np.where(dates.is_leap_year,366,365)
            covariates=[np.sin(2*np.pi*day),np.cos(2*np.pi*day),
                        np.sin(2*np.pi*year),np.cos(2*np.pi*year)]
            if name in ('aqi','metr_la','pems_bay'):
                covariates.extend(np.eye(7)[dates.dayofweek].T)
            calendar=np.stack(covariates,axis=-1).astype('float32')
        return cls(x,obs,observed,origins(bounds,w,h,h if stride is None else stride),w,h,center,scale,calendar)

    def get(self,indices):
        # B,N,W*C, preserving lag order without aggregation.
        lag=np.asarray(indices)[:,None]+np.arange(-self.window,0)[None,:]
        future=np.asarray(indices)[:,None]+np.arange(self.horizon)[None,:]
        x=(self.values[lag]-self.center)/self.scale
        x=np.where(self.input_observed[lag],x,np.nan)
        x=x.transpose(0,2,1,3).reshape(len(indices),self.values.shape[1],-1)
        if self.calendar is not None:
            calendar=np.broadcast_to(self.calendar[np.asarray(indices)][:,None],(len(indices),self.values.shape[1],self.calendar.shape[1]))
            x=np.concatenate([x,calendar],axis=-1)
        y=self.values[future].transpose(0,2,1,3).reshape(len(indices),self.values.shape[1],-1)
        valid=self.observed[future].transpose(0,2,1,3).reshape(y.shape)
        return x.astype('float32'),y.astype('float32'),valid


def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',type=Path,default=Path('data/temporal/raw'))
    p.add_argument('--out',type=Path,default=Path('data/temporal/prepared_v2'))
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS)); args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); reports=[]
    for name in args.datasets:
        data,report=load_raw(name,args.raw)
        data['pe']=topology_pe(data['edge_index'],data['values'].shape[1],16)
        path=args.out/f'{name}.npz'; np.savez_compressed(path,**data)
        report['prepared_sha256']=sha256(path)
        report['split_bounds']=temporal_split(data['timestamps'],name)
        report['windows_stride_horizon']=[len(v) for v in origins(report['split_bounds'],*WINDOWS[name],WINDOWS[name][1])]
        (args.out/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report); print(json.dumps(report),flush=True)
    (args.out/'audit.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__': main()
