import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from missingness_shared_seed import rtmar_shared_seed_mask,graph_info
from shared_seed_reference import rtmar_shared_seed_mask as reference
from temporal_data import artificial_mask, StaticWindows


class SharedSeedTests(unittest.TestCase):
    def test_exact_reference_random_graphs(self):
        for case in range(24):
            rng=np.random.default_rng(case)
            n=1+case%15;d=1+case%11
            a=rng.random((n,n))<.13
            edges=np.asarray(np.nonzero(a),dtype=np.int64)
            for rate in (0,.001,.07,.1,.37,.5,.9,.99,1.):
                expected=reference(edges,n,d,rate,seed=case)
                actual=rtmar_shared_seed_mask(edges,n,d,rate,seed=case,threads=2)
                self.assertTrue(torch.equal(actual,expected),(case,rate))

    def test_disconnected_shared_sources_and_cache(self):
        edges=np.array([[0,1,2,3,5,6,7],[1,2,3,4,6,7,8]],dtype=np.int64)
        n,d=10,31
        with tempfile.TemporaryDirectory() as directory:
            for rate in (.03,.1,.35,.6,.99):
                result=rtmar_shared_seed_mask(edges,n,d,rate,seed=43,cache_dir=directory,threads=2)
                self.assertTrue(torch.equal(result,reference(edges,n,d,rate,seed=43)))
                adj,labels,sizes,components=graph_info(edges,n,d,43)
                meta=json.loads(next(Path(directory).glob('*.json')).read_text())
                for j in range(d):
                    for c,nodes in enumerate(components):
                        selected=nodes[result[nodes,j].numpy()]
                        if len(selected):
                            self.assertIn(meta['component_sources'][c],selected)
                            self.assertEqual(connected_components(adj[selected][:,selected],directed=False,return_labels=False),1)
            self.assertEqual(len(list(Path(directory).glob('*.shared_ranks.npy'))),1)
            rank=next(Path(directory).glob('*.shared_ranks.npy'))
            with rank.open('r+b') as stream:
                stream.seek(-1,2);stream.write(b'Z')
            with self.assertRaisesRegex(ValueError,'cache was modified'):
                rtmar_shared_seed_mask(edges,n,d,.5,43,cache_dir=directory)

    def test_raw_time_channel_mapping(self):
        edges=np.array([[0,1,2],[1,2,3]],dtype=np.int64)
        shape=(7,4,3)
        actual=artificial_mask(edges,shape,15,.37,'RT')
        expected=reference(edges,4,21,.37,15).numpy().reshape(4,7,3).transpose(1,0,2)
        np.testing.assert_array_equal(actual,expected)

    def test_chronological_fit_and_unmasked_targets(self):
        rng=np.random.default_rng(0)
        values=rng.normal(size=(1000,4,1)).astype('float32')
        data=dict(values=values,observed=np.ones_like(values,dtype=bool),timestamps=np.arange(1000),edge_index=np.array([[0,1,2],[1,2,3]],dtype=np.int64))
        with tempfile.TemporaryDirectory() as directory:
            a=StaticWindows.build(data,'graphmso',15,.5,'RT',mask_cache=directory)
            poisoned={**data,'values':values.copy()};poisoned['values'][700:]=1e10
            b=StaticWindows.build(poisoned,'graphmso',15,.5,'RT',mask_cache=directory)
            np.testing.assert_array_equal(a.center,b.center);np.testing.assert_array_equal(a.scale,b.scale)
            np.testing.assert_array_equal(a.input_observed,b.input_observed)
            x,y,valid=a.get(a.splits[0][:1])
            self.assertTrue(valid.all())
            self.assertEqual(int((~a.input_observed).sum()),2000)
            self.assertTrue(np.isnan(x).any())

    def test_invalid_inputs(self):
        edges=np.empty((2,0),dtype=np.int64)
        for rate in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):rtmar_shared_seed_mask(edges,3,7,rate)
        with self.assertRaises(ValueError):rtmar_shared_seed_mask(edges,3,7,.5,-1)
        with self.assertRaises(ValueError):rtmar_shared_seed_mask(np.array([[0.],[1.]]),3,7,.5)

if __name__=='__main__':unittest.main()
