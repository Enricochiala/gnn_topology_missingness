import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch_geometric.data import Data
from protocol import RATES, PRESETS, load_preset, select_curves, validate_selection
from models import PEMix, PEMixGMMConv
from temporal_models import GaussianPEMix
from paper_data import load_dataset
from curve_summary import export_auc


class ProtocolTests(unittest.TestCase):
    def records(self):
        # Per-rate winners differ; candidate 1 wins the integrated curve.
        rows=[]
        for candidate, scores in enumerate(([1.,100.,1.], [3.,3.,3.])):
            for rate, score in zip([0.,.5,.99],scores):
                rows.append(dict(dataset='graphmso',mechanism='RT',model='PEMix',seed=2026,
                    rate=rate,candidate=candidate,config={'hidden':16+candidate,'likelihood':'gaussian'},
                    status='ok',validation_mae=score,mae=10000. if candidate==1 else 0.))
        return rows

    def test_auc_selection_is_fixed_and_ignores_test_scores(self):
        selected=select_curves(self.records(),[0.,.5,.99])
        self.assertEqual({v['candidate'] for v in selected.values()},{1})
        self.assertAlmostEqual(next(iter(selected.values()))['selection_score'],2.97)
        validate_selection(selected,['graphmso'],['RT'],[0.,.5,.99],['PEMix'])

    def test_failed_incomplete_or_changing_candidates_rejected(self):
        for failure in ('failed','missing','different'):
            rows=self.records()
            if failure=='failed':rows[0]['status']='failed'
            elif failure=='missing':rows.pop()
            else:rows[0]['config']={'hidden':999}
            with self.assertRaises(ValueError):select_curves(rows,[0.,.5,.99])

    def test_natural_uses_validation_mae(self):
        rows=[dict(row,dataset='aqi',mechanism='natural') for row in self.records() if row['rate']==0.]
        selected=select_curves(rows,RATES)
        self.assertEqual(next(iter(selected.values()))['candidate'],0)
        self.assertEqual(next(iter(selected.values()))['selection_criterion'],'validation_mae')

    def test_bundled_presets_fixed_at_every_rate(self):
        for path in PRESETS.glob('*.json'):
            preset=json.loads(path.read_text())
            selection=load_preset(path,preset['input_hashes'],RATES,list(preset['models']))
            validate_selection(selection,[preset['dataset']],[preset['mechanism']],RATES,list(preset['models']))
            with self.assertRaises(ValueError):load_preset(path,{},RATES,['PEMix'])

    def test_rate_specific_selection_cannot_bypass_validation_with_subset(self):
        selected=select_curves(self.records(),[0.,.5,.99])
        selected=copy.deepcopy(selected)
        selected['graphmso/RT/0.99/PEMix']['config']['hidden']=999
        with self.assertRaises(ValueError):validate_selection(selected,['graphmso'],['RT'],[0.],['PEMix'])

    def test_classification_uses_same_density_as_forecasting(self):
        self.assertIs(GaussianPEMix,PEMixGMMConv)
        data=Data(x=torch.ones(4,2),y=torch.tensor([0,1,0,1]),edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]))
        data.adj=torch.eye(4).to_sparse()
        model=PEMix(data,pe_dim=1,init_x=torch.randn(4,3),n_components=2)
        self.assertIsInstance(model.gc1,PEMixGMMConv)
        self.assertEqual(model.dropout,0.)
        z=torch.full((4,3),float('nan'))
        z[:,-1]=torch.tensor([-1.,0.,1.,2.])
        with torch.no_grad():
            model.gc1.means.copy_(torch.tensor([[0.,0.,-1.],[0.,0.,1.]]))
            model.gc1.logvars.zero_();model.gc1.logp.zero_()
        gamma=model.gc1.responsibilities(z)
        expected=torch.softmax(-.5*(z[:,-1][None]-model.gc1.means[:,-1,None]).square(),dim=0)
        torch.testing.assert_close(gamma,expected)
        model.gc1(z,data.adj,data.edge_index).sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_prepared_pe_is_standardized_and_resume_stable(self):
        with TemporaryDirectory() as directory:
            data=Data(x=torch.ones(8,2),y=torch.arange(8)%3,
                      edge_index=torch.stack([torch.arange(8),torch.arange(8).roll(1)]),pe=torch.randn(8,2))
            torch.save(data,Path(directory)/'tadpole.pt')
            first,_=load_dataset('tadpole',directory,prepared=True,pe_dim=2)
            torch.testing.assert_close(first.pe.mean(0),torch.zeros(2),atol=1e-6,rtol=0)
            torch.testing.assert_close(first.pe.std(0,unbiased=False),torch.ones(2))
            torch.save(first,Path(directory)/'tadpole.pt')
            second,_=load_dataset('tadpole',directory,prepared=True,pe_dim=2)
            self.assertTrue(torch.equal(first.pe,second.pe))
            self.assertTrue(torch.isfinite(second.adj.values()).all())

    def test_gcnmf_initialization_never_receives_hidden_complete_values(self):
        from unittest.mock import patch
        from models import init_gmm
        from experiments import make_entries, evaluate
        n=30
        edges=torch.stack([torch.arange(n),torch.arange(n).roll(1)])
        data=Data(x=torch.randn(n,3),y=torch.arange(n)%3,edge_index=edges,num_classes=3)
        data.adj=torch.eye(n).to_sparse()
        entry=make_entries(data,1,.5,['UMCAR'])['UMCAR']
        with patch('models.init_gmm',wraps=init_gmm) as init:
            evaluate(data,entry,1,'gcnmf',max_epochs=1)
        self.assertGreater(init.call_count,0)
        for call in init.call_args_list:
            actual=np.asarray(call.args[0])
            np.testing.assert_array_equal(np.isnan(actual),entry['mask'].numpy())
            np.testing.assert_allclose(actual,entry['X_incomp'].numpy(),equal_nan=True)

    def test_auc_not_normalized_and_incomplete_groups_have_no_score(self):
        import pandas as pd
        with TemporaryDirectory() as directory:
            rows=[dict(dataset='graphmso',mechanism='RT',model='PEMix',seed=1,rate=rate,status='ok',mae=2.) for rate in RATES]
            export_auc(rows,{'seeds':[1]},directory,'rate','mae')
            self.assertAlmostEqual(pd.read_csv(Path(directory)/'auc.csv').auc_mean.iloc[0],1.98)
            export_auc(rows[:-1],{'seeds':[1]},directory,'rate','mae')
            self.assertTrue(np.isnan(pd.read_csv(Path(directory)/'auc.csv').auc_mean.iloc[0]))

if __name__=='__main__':unittest.main()
