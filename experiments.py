"""Paper model dispatch and paired masking, without changing the evaluators."""

import numpy as np
import torch

import utils
from missingness_shared_seed import rtmar_shared_seed_mask

MAIN_MODELS = ('gnnzero', 'gnnmi', 'gnnmedian', 'gnnmim', 'fp', 'pcfi', 'fisf',
               'gcnmf', 'gcnmf_pe')
ABLATIONS = ('gcnmf_lf', 'gcnmf_pe_full', 'gcnpe', 'gnnmim_pe', 'peonly', 'mlppe',
             'gcnmf_pe_nofeatures', 'gcnmf_pe_nofeatures_shuffledpe',
             'gcnmf_pe_nofeatures_zerope', 'fp_pe', 'pcfi_pe', 'fppe', 'pcfipe', 'fisf_pe')
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
    if model not in MAIN_MODELS + ABLATIONS:
        raise ValueError(f'Unknown model: {model}')
    data = data.clone()
    entry = {k: v.clone() for k, v in original.items()}
    data.masks = {seed: entry}
    custom = ('gnnzero', 'gnnmi', 'gnnmedian', 'gnnmim', 'gcnmf_pe',
              'gcnmf_lf', 'gcnmf_pe_full', 'gcnpe', 'gnnmim_pe', 'peonly', 'mlppe',
              'gcnmf_pe_nofeatures', 'gcnmf_pe_nofeatures_shuffledpe',
              'gcnmf_pe_nofeatures_zerope')
    if model in custom:
        x, pe = entry['X_incomp'], data.pe
        mod = model
        if model in ('gnnzero', 'gnnmi', 'gnnmedian'):
            mod = None
            if model == 'gnnzero':
                x = torch.nan_to_num(x, nan=0.)
            elif model == 'gnnmedian':
                train_val = entry['train_mask'] | entry['val_mask']
                for j in range(x.size(1)):
                    values = x[train_val, j]
                    values = values[~torch.isnan(values)]
                    x[torch.isnan(x[:, j]), j] = values.median() if values.numel() else 0.
        elif model.startswith('gcnmf_pe_nofeatures'):
            x = torch.full_like(x, float('nan'))
            if model.endswith('shuffledpe'):
                gen = torch.Generator().manual_seed(int(seed))
                pe = pe[torch.randperm(pe.size(0), generator=gen)]
            elif model.endswith('zerope'):
                pe = torch.zeros_like(pe)
            x, mod = torch.cat([x, pe], dim=1), 'gcnmf_pe'
        elif model in ('gcnmf_pe', 'gcnmf_pe_full'):
            x = torch.cat([x, pe], dim=1)
        elif model == 'gnnmim':
            x = torch.cat([torch.nan_to_num(x, nan=0.), entry['mask'].float()], dim=1)
        elif model == 'gnnmim_pe':
            x = torch.cat([torch.nan_to_num(x, nan=0.), pe, entry['mask'].float()], dim=1)
        elif model == 'gcnpe':
            x = torch.cat([torch.nan_to_num(x, nan=0.), pe], dim=1)
        elif model in ('peonly', 'mlppe'):
            x = pe.clone()
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
