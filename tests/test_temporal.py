import unittest
from unittest.mock import patch
import numpy as np
import torch
from temporal_data import StaticWindows,WINDOWS,artificial_mask,topology_pe
from temporal_models import GaussianPEMix,SpatialBatch,StaticRegressor
from temporal_run import reference_config,gmm_init,make_inputs
from torch_geometric.data import Data
from tempfile import TemporaryDirectory
from pathlib import Path

class TemporalTests(unittest.TestCase):
    def setUp(self):
        n=12;t=200
        self.edges=np.stack([np.r_[np.arange(n),np.roll(np.arange(n),1)],np.r_[np.roll(np.arange(n),1),np.arange(n)]])
        rng=np.random.default_rng(73)
        self.data=dict(values=rng.normal(size=(t,n,1)).astype('float32'),observed=np.ones((t,n,1),bool),
                       timestamps=np.arange(t),edge_index=self.edges,edge_weight=np.ones(self.edges.shape[1],dtype='float32'))

    def test_causal_features_and_statistics(self):
        with patch.dict(WINDOWS,{'graphmso':(8,3)}):
            first=StaticWindows.build(self.data,'graphmso',1,.5,'RT')
            changed={k:v.copy() for k,v in self.data.items()}
            changed['values'][140:]=1e8
            second=StaticWindows.build(changed,'graphmso',1,.5,'RT')
            np.testing.assert_array_equal(first.center,second.center)
            np.testing.assert_array_equal(first.scale,second.scale)
            for a,b in zip(first.get(first.splits[0]),second.get(second.splits[0])):np.testing.assert_array_equal(a,b)
            origin=first.splits[2][0]
            x_before=first.get([origin])[0]
            changed={k:v.copy() for k,v in self.data.items()};changed['values'][origin:]=-1e9
            after=StaticWindows.build(changed,'graphmso',1,.5,'RT')
            np.testing.assert_array_equal(x_before,after.get([origin])[0])
            # Entire window+horizon support lies in one split.
            supports=[set(np.concatenate([np.arange(v-8,v+3) for v in split])) for split in first.splits]
            self.assertFalse(supports[0]&supports[1]);self.assertFalse(supports[1]&supports[2])

    def test_natural_no_injection_and_no_target_imputation(self):
        self.data['observed'][::2,0]=False;self.data['values'][::2,0]=np.nan
        with patch.dict(WINDOWS,{'aqi':(8,3)}):
            w=StaticWindows.build(self.data,'aqi',1,0,'natural')
            np.testing.assert_array_equal(w.observed,w.input_observed)
            _,y,mask=w.get(w.splits[0]);self.assertTrue(np.isnan(y[~mask]).all())
            with self.assertRaises(ValueError):StaticWindows.build(self.data,'aqi',1,.5,'UMCAR')

    def test_rt_exact_budget_connected_graph(self):
        small=artificial_mask(self.edges,(9,12,2),9,.2,'RT')
        large=artificial_mask(self.edges,(9,12,2),9,.8,'RT')
        self.assertEqual(small.sum(),43);self.assertEqual(large.sum(),172)
        self.assertFalse(artificial_mask(self.edges,(9,12,2),9,0,'RT').any())

    def test_pemix_matches_normalized_density_and_gradients(self):
        z=torch.tensor([[1.,float('nan'),.2],[float('nan'),float('nan'),-.1],[.5,2.,.7]])
        eye=torch.eye(3).to_sparse();data=Data(x=z,adj=eye)
        layer=GaussianPEMix(2,1,4,data,2,0.)
        with torch.no_grad():
            layer.means.copy_(torch.tensor([[0.,1.,0.],[1.,0.,.5]]))
            layer.logvars.copy_(torch.log(torch.tensor([[1.,2.,.5],[3.,.2,2.]])))
            layer.logp.copy_(torch.log(torch.tensor([.3,.7])))
        expected=[]
        for row in z:
            valid=~row.isnan()
            dist=torch.distributions.Normal(layer.means[:,valid],torch.exp(.5*layer.logvars[:,valid]))
            expected.append(torch.softmax(layer.logp+dist.log_prob(row[valid]).sum(-1),0))
        torch.testing.assert_close(layer.responsibilities(z),torch.stack(expected,1))
        self.assertEqual(tuple(layer.weight.shape),(2,4)) # PE cannot enter W.
        out=layer(z,eye,torch.tensor([[0,1,2],[0,1,2]]));out.sum().backward()
        self.assertTrue(torch.isfinite(out).all())
        for p in layer.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_spatial_batches_do_not_exchange_future_windows(self):
        graph=SpatialBatch(self.edges,np.ones(self.edges.shape[1]),12,'cpu')
        edge,_,_=graph.get(3)
        self.assertTrue(torch.equal(edge[0]//12,edge[1]//12))
        pe=topology_pe(self.edges,12,4)
        x=np.random.default_rng(1).normal(size=(3,12,6)).astype('float32')
        config=dict(reference_config('gnnzero'),q=4)
        model=StaticRegressor('gnnzero',6,2,graph,pe,x.reshape(-1,6),config).eval()
        a=model(torch.from_numpy(x)).detach();x[1:]=1e6
        b=model(torch.from_numpy(x)).detach();torch.testing.assert_close(a[0],b[0])
        with self.assertRaises(ValueError):StaticRegressor('not_a_model',6,2,graph,pe,x.reshape(-1,6),config)

    def test_tuning_does_not_materialize_test_windows(self):
        with patch.dict(WINDOWS,{'graphmso':(8,3)}):
            w=StaticWindows.build(self.data,'graphmso',1,0,'UMCAR')
            real_get=w.get
            def audited_get(ids):
                self.assertTrue(max(ids)<160,'Tuning requested test windows')
                return real_get(ids)
            with patch.object(w,'get',side_effect=audited_get),TemporaryDirectory() as tmp:
                arrays,_=make_inputs(w,w.splits,'gnnmi',self.data,Path(tmp),'tune')
                self.assertEqual(len(arrays),2)

class ImputationEquivalenceTests(unittest.TestCase):
    def test_accelerated_distances_and_outputs_match_original(self):
        from models import PCFI
        from fisf import FISF
        from temporal_imputation import DistanceLookup,FastPCFI,FastFISF
        edge=torch.tensor([[0,1,1,2,3,4],[1,0,2,3,4,3]]) # directed + isolated node 5
        x=torch.arange(30,dtype=torch.float32).reshape(6,5)/10
        observed=torch.rand((6,5),generator=torch.Generator().manual_seed(7))>.6
        observed[:,0]=False
        lookup=DistanceLookup(edge.numpy(),6)
        for original,fast in ((PCFI(3,.9,1.),FastPCFI(lookup,3,.9,1.)),
                              (FISF(3,.5,.1,.5),FastFISF(lookup,3,.5,.1,.5))):
            torch.testing.assert_close(original.compute_f_n2d(edge,observed,'uniform',feat_dim=5),
                                       fast.compute_f_n2d(edge,observed,'uniform',feat_dim=5))
            np.random.seed(0)
            a=original.propagate(x.clone(),edge,observed.clone(),'uniform')
            np.random.seed(0)
            b=fast.propagate(x.clone(),edge,observed.clone(),'uniform')
            torch.testing.assert_close(a,b)

    def test_cached_rtmar_is_bitwise_identical(self):
        edges=np.array([[0,1,1,2,2,3],[1,0,2,1,3,2]])
        with TemporaryDirectory() as tmp:
            for rate in (.0,.2,.7,1.):
                a=artificial_mask(edges,(11,4,2),13,rate,'RT')
                b=artificial_mask(edges,(11,4,2),13,rate,'RT',Path(tmp))
                np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
