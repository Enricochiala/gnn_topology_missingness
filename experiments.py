"""Paper model dispatch and paired masking, without changing the evaluators."""

import numpy as np
import torch

import utils
from missingness_shared_seed import rtmar_shared_seed_mask

MAIN_MODELS = ('gnnzero', 'gnnmi', 'gnnmedian', 'gnnmim', 'fp', 'pcfi', 'fisf',
               'gcnmf', 'PEMix')
MECHANISMS = ('UMCAR', 'RT')


def make_entries(data, seed, mu, mechanisms, plans=None):
    """Paired feature-level masks; RT is the shared-seed compact-growth variant."""
    if not np.isfinite(mu) or not 0 <= mu <= 1:
        raise ValueError('mu must be finite and in [0, 1]')
    baseline = utils.umcar(data, {'train': mu, 'test': mu}, [seed])[seed]
    entries = {}
    for mechanism in mechanisms:
        if mechanism not in MECHANISMS:
            raise ValueError(f'Unknown mechanism: {mechanism}')
        entry = {k: v.clone() for k, v in baseline.items()}
        if mechanism == 'RT':
            mask = rtmar_shared_seed_mask(data.edge_index, *data.x.shape, mu, seed)
            entry.update(mask=mask, X_incomp=data.x.clone())
            entry['X_incomp'][mask] = float('nan')
        assert torch.equal(torch.isnan(entry['X_incomp']), entry['mask'])
        entries[mechanism] = entry
    return entries


def evaluate(data, original, seed, model, *, device='cpu', n_components=5,
             max_epochs=None, patience=None):
    if model not in MAIN_MODELS:
        raise ValueError(f'Unknown model: {model}')
    data = data.clone()
    entry = {k: v.clone() for k, v in original.items()}
    data.masks = {seed: entry}
    if model in ('gnnzero', 'gnnmi', 'gnnmedian', 'gnnmim', 'PEMix'):
        x = entry['X_incomp']
        mod = model
        if model in ('gnnzero', 'gnnmi', 'gnnmedian'):
            mod = None
            if model == 'gnnzero':
                x = torch.nan_to_num(x, nan=0.)
            elif model == 'gnnmedian':
                for j in range(x.size(1)):
                    values = x[entry['train_mask'], j]
                    values = values[~torch.isnan(values)]
                    x[torch.isnan(x[:, j]), j] = values.median() if values.numel() else 0.
        elif model == 'PEMix':
            x = torch.cat([x, data.pe], dim=1)
        elif model == 'gnnmim':
            x = torch.cat([torch.nan_to_num(x, nan=0.), entry['mask'].float()], dim=1)
        entry['X_incomp'] = x
        return utils.evaluate_gcn(data, mod=mod, seeds=[seed], metric='f1',
            device=device, n_components=n_components,
            max_epochs=500 if max_epochs is None else max_epochs,
            patience=50 if patience is None else patience)
    kwargs = dict(max_epochs=1000 if max_epochs is None else max_epochs,
                  patience=40 if patience is None else patience, seeds=[seed], device=device)
    if model == 'gcnmf':
        return utils.evaluate_gcnmf(data, metric='f1', **kwargs)
    return getattr(utils, f'evaluate_{model}')(data, hidden_channels=64, metric='f1', **kwargs)
