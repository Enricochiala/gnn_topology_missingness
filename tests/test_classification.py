import unittest
import torch
from torch_geometric.data import Data
from experiments import MAIN_MODELS, MECHANISMS, make_entries
from shared_seed_reference import rtmar_shared_seed_mask
from extended_search import pemix_candidates

class ClassificationTests(unittest.TestCase):
    def test_paired_splits_and_rt_reference(self):
        n=60
        edge=torch.stack([torch.arange(n),torch.arange(n).roll(1)])
        data=Data(x=torch.ones(n,4),y=torch.arange(n)%3,edge_index=edge)
        a=make_entries(data,43,.5,MECHANISMS)
        expected=rtmar_shared_seed_mask(edge,n,4,.5,43)
        self.assertTrue(torch.equal(a['RT']['mask'],expected))
        for split in ['train_mask','val_mask','test_mask']:
            self.assertTrue(torch.equal(a['RT'][split],a['UMCAR'][split]))
        changed=data.clone();changed.x*=100;changed.y=changed.y.roll(1)
        b=make_entries(changed,43,.5,('RT',))
        self.assertTrue(torch.equal(a['RT']['mask'],b['RT']['mask']))

    def test_nine_main_methods_and_search(self):
        self.assertEqual(len(MAIN_MODELS),9)
        self.assertEqual(set(MECHANISMS),{'RT','UMCAR'})
        self.assertEqual(len(pemix_candidates()),48)

if __name__=='__main__':unittest.main()
