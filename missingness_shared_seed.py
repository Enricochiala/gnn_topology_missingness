"""Exact efficient adapter of the compact shared-source RT.

Independent feature/component growth admits full growth orders. A native heap
updates frontier neighbour counts, preserving the supplied greedy choice and
NumPy random streams exactly. Component quotas are recomputed for every rate:
the supplied largest-remainder allocation does NOT guarantee nested masks.
"""
import ctypes
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE=Path(__file__).resolve().parent
_LIB=None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def graph_info(edge_index,n,d,seed):
    if n<1 or d<1 or int(n)!=n or int(d)!=d or n>65535:
        raise ValueError('Require integer n,d >= 1 and n <= 65535')
    if int(seed)!=seed or seed<0:raise ValueError('seed must be a nonnegative integer')
    edges=torch.as_tensor(edge_index).detach().cpu().numpy()
    if edges.ndim!=2 or edges.shape[0]!=2 or not np.issubdtype(edges.dtype,np.integer):
        raise ValueError('edge_index must be integer [2,E]')
    if edges.size and (edges.min()<0 or edges.max()>=n):raise ValueError('Invalid node index')
    adj=coo_matrix((np.ones(edges.shape[1]),(edges[0],edges[1])),shape=(n,n)).tocsr()
    adj=adj.maximum(adj.T);adj.setdiag(0);adj.eliminate_zeros();adj.data[:]=1;adj.sort_indices()
    count,labels=connected_components(adj,directed=False,return_labels=True)
    components=[np.flatnonzero(labels==c) for c in range(count)]
    return adj,labels,np.asarray([len(c) for c in components],dtype=np.int64),components


def streams(seed):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(4)]


def component_quotas(n,d,rate,seed,sizes):
    mu=Decimal(str(rate))
    if not mu.is_finite() or not 0<=mu<=1:raise ValueError('finite mu in [0,1] required')
    total=int((mu*Decimal(n*d)).to_integral_value(rounding=ROUND_FLOOR))
    quota_rng,_,_,component_rng=streams(seed)
    priority=quota_rng.permutation(d)
    q,remainder=divmod(total,d)
    feature=np.full(d,q,dtype=np.int64);feature[priority[:remainder]]+=1
    active=np.flatnonzero(feature>0)
    quotas=np.zeros((d,len(sizes)),dtype=np.int64)
    # Same RNG calls as the reference: zero-feature quotas consume no draws.
    exact=feature[active,None]*sizes[None,:]/n
    floor=np.floor(exact)
    base=np.minimum(floor.astype(np.int64),sizes[None,:])
    left=feature[active]-base.sum(axis=1)
    tie=component_rng.random((len(active),len(sizes)))
    order=np.lexsort((tie,exact-floor),axis=1)[:,::-1]
    for position in range(len(sizes)):
        chosen=np.flatnonzero(left>position)
        base[chosen,order[chosen,position]]+=1
    quotas[active]=base
    if int(quotas.sum())!=total or np.any(quotas>sizes[None,:]):raise RuntimeError('Invalid component budget')
    return quotas,total


def native():
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(HERE/'shared_seed_growth.so'))
        fn=_LIB.shared_seed_growth
        ptr64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        ptr16=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')
        fn.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ptr64,ptr64,ptr64,ptr64,ptr16,ptr16,ctypes.c_int]
        fn.restype=ctypes.c_int
    return _LIB.shared_seed_growth


def make_ranks(adj,components,sizes,n,d,seed,path,threads=4,chunk=1024):
    _,source_rng,tie_rng,_=streams(seed)
    sources=np.asarray([int(source_rng.choice(nodes)) for nodes in components],dtype=np.int64)
    ranks=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=(d,n))
    indptr=np.ascontiguousarray(adj.indptr,dtype=np.int64)
    indices=np.ascontiguousarray(adj.indices,dtype=np.int64)
    order=np.arange(n,dtype=np.uint16)
    for start in range(0,d,chunk):
        size=min(chunk,d-start)
        ties=np.empty((size,n),dtype=np.uint16)
        for j in range(size):ties[j,tie_rng.permutation(n)]=order
        result=native()(n,size,len(sizes),indptr,indices,sources,sizes,ties,ranks[start:start+size],threads)
        if result:raise RuntimeError('Native frontier exhausted unexpectedly')
    ranks.flush()
    return sources


def _mask_from_ranks(ranks,labels,sizes,n,d,rate,seed):
    quotas,total=component_quotas(n,d,rate,seed,sizes)
    mask=np.empty((n,d),dtype=bool)
    for start in range(0,d,4096):
        end=min(d,start+4096)
        mask[:,start:end]=(ranks[start:end]<quotas[start:end,labels]).T
    if int(mask.sum())!=total:raise RuntimeError('Mask differs from exact global budget')
    return torch.from_numpy(mask)


def rtmar_shared_seed_mask(edge_index,n,d,mu,seed=1,cache_dir=None,threads=4):
    adj,labels,sizes,components=graph_info(edge_index,n,d,seed)
    # Validate rate even when a shortcut is possible.
    rate=Decimal(str(mu))
    if not rate.is_finite() or not 0<=rate<=1:raise ValueError('finite mu in [0,1] required')
    if rate==0:return torch.zeros((n,d),dtype=torch.bool)
    if rate==1:return torch.ones((n,d),dtype=torch.bool)
    if cache_dir is None:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ranks.npy'
            make_ranks(adj,components,sizes,n,d,seed,path,threads)
            return _mask_from_ranks(np.load(path,mmap_mode='r'),labels,sizes,n,d,mu,seed)
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    source_hashes={name:digest(HERE/name) for name in ('missingness_shared_seed.py','shared_seed_growth.cpp','shared_seed_growth.so','shared_seed_reference.py')}
    key=hashlib.sha256(adj.indptr.astype(np.int64).tobytes()+adj.indices.astype(np.int64).tobytes()+json.dumps([n,d,int(seed),source_hashes],sort_keys=True).encode()).hexdigest()
    rankpath=cache/f'{key}.shared_ranks.npy';manifest=cache/f'{key}.json'
    import fcntl
    with (cache/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not manifest.exists():
            pending=cache/f'{key}.pending.npy'
            sources=make_ranks(adj,components,sizes,n,d,seed,pending,threads)
            pending.replace(rankpath)
            meta=dict(algorithm='RT',rank_sha256=digest(rankpath),shape=[d,n],seed=int(seed),component_sources=sources.tolist(),component_sizes=sizes.tolist(),source_hashes=source_hashes)
            tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(manifest)
        meta=json.loads(manifest.read_text())
        if digest(rankpath)!=meta['rank_sha256']:raise ValueError('Shared-seed rank cache was modified')
    return _mask_from_ranks(np.load(rankpath,mmap_mode='r'),labels,sizes,n,d,mu,seed)
