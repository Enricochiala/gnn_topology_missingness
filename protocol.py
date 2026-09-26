"""Fixed configurations and validation-only selection over complete curves."""
import copy
import json
from pathlib import Path

import numpy as np

RATES = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
CONTROLLED = ('engrad', 'pems_bay', 'graphmso')
PRESETS = Path(__file__).resolve().parent / 'configs' / 'presets'


def load_preset(path, input_hashes, rates, models):
    preset = json.loads(Path(path).read_text())
    if preset.get('schema_version') != 2:
        raise ValueError('Expected a fixed-curve configuration (schema_version=2)')
    dataset, mechanism = preset['dataset'], preset['mechanism']
    if preset['input_hashes'].get(dataset) != input_hashes.get(dataset):
        raise ValueError('Preset input fingerprint mismatch')
    result = {}
    for model in models:
        config = preset['models'][model]
        if model == 'PEMix' and config['likelihood'] != 'gaussian':
            raise ValueError('PEMix requires the normalized likelihood in Eq. (12)')
        for rate in rates if dataset in CONTROLLED else [0.]:
            result[f'{dataset}/{mechanism}/{float(rate)}/{model}'] = {'config': copy.deepcopy(config)}
    return result


def validate_selection(selection, datasets, mechanisms, rates, models):
    by_curve = {}
    for key, value in selection.items():
        dataset, mechanism, rate, model = key.split('/')
        curve = (dataset, mechanism, model)
        if curve in by_curve and by_curve[curve] != value['config']:
            raise ValueError('Configurations must remain fixed across the missingness curve')
        by_curve[curve] = value['config']
    for dataset in datasets:
        for mechanism in mechanisms if dataset in CONTROLLED else ['natural']:
            for model in models:
                configs = []
                for rate in rates if dataset in CONTROLLED else [0.]:
                    key = f'{dataset}/{mechanism}/{float(rate)}/{model}'
                    if key not in selection:
                        raise ValueError(f'Missing configuration: {key}')
                    config = selection[key]['config']
                    if model == 'PEMix' and config['likelihood'] != 'gaussian':
                        raise ValueError('PEMix requires the normalized likelihood in Eq. (12)')
                    configs.append(config)
                if any(config != configs[0] for config in configs):
                    raise ValueError('Configurations must remain fixed across the missingness curve')


def select_curves(records, rates):
    """Minimize validation MAE AUC; natural missingness minimizes validation MAE.

    Incomplete candidates, duplicates, failed fits, and mixed configurations are
    errors. Test metrics never enter the selection rule.
    """
    groups = {}
    for row in records:
        if row['status'] != 'ok':
            raise ValueError('Cannot select from a campaign with failed fits')
        if not np.isfinite(row['validation_mae']):
            raise ValueError('Non-finite validation score')
        key = (row['dataset'], row['mechanism'], row['model'])
        groups.setdefault(key, {}).setdefault(row['candidate'], []).append(row)
    chosen = {}
    for (dataset, mechanism, model), candidates in groups.items():
        expected = sorted(rates if dataset in CONTROLLED else [0.])
        scored = []
        for candidate, rows in candidates.items():
            rows = sorted(rows, key=lambda row: row['rate'])
            if [row['rate'] for row in rows] != expected:
                raise ValueError('Every candidate must cover every requested rate exactly once')
            if any(row['config'] != rows[0]['config'] for row in rows):
                raise ValueError('A candidate changed configuration across rates')
            values = [row['validation_mae'] for row in rows]
            score = float(np.trapezoid(values, expected)) if len(expected) > 1 else float(values[0])
            scored.append((score, candidate, rows[0]['config']))
        score, candidate, config = min(scored, key=lambda item: (item[0], item[1]))
        criterion = 'validation_mae_auc' if dataset in CONTROLLED else 'validation_mae'
        for rate in expected:
            chosen[f'{dataset}/{mechanism}/{float(rate)}/{model}'] = dict(
                config=copy.deepcopy(config), candidate=candidate, selection_criterion=criterion,
                selection_score=score, selection_rates=expected)
    return chosen
