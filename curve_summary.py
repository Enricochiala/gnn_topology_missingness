"""Trapezoidal areas on complete per-seed controlled-missingness curves."""
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import RATES


def export_auc(rows, config, root, rate_key, metric):
    per_seed = []
    grouped = {}
    for row in rows:
        if row['mechanism'] == 'natural':
            continue
        key = (row['dataset'], row['mechanism'], row['model'], row['seed'])
        grouped.setdefault(key, []).append(row)
    for key, curve in sorted(grouped.items()):
        curve = sorted(curve, key=lambda row: row[rate_key])
        complete = [row[rate_key] for row in curve] == RATES and all(
            row['status'] == 'ok' and row.get(metric) is not None and np.isfinite(row[metric]) for row in curve)
        record = dict(zip(('dataset', 'mechanism', 'model', 'seed'), key))
        record.update(complete=complete, pilot=config.get('pilot', False),
                      auc=float(np.trapezoid([row[metric] for row in curve], RATES)) if complete else None)
        per_seed.append(record)
    fields = ['dataset', 'mechanism', 'model', 'seed', 'complete', 'pilot', 'auc']
    df = pd.DataFrame(per_seed, columns=fields)
    df.to_csv(Path(root)/'auc_per_seed.csv', index=False)
    summary = []
    for key, group in df.groupby(['dataset', 'mechanism', 'model']):
        complete = bool(group.complete.all() and set(group.seed) == set(config['seeds']))
        record = dict(zip(('dataset', 'mechanism', 'model'), key))
        record.update(complete=complete, pilot=config.get('pilot', False),
                      auc_mean=float(group.auc.mean()) if complete else None,
                      auc_sd=float(group.auc.std(ddof=1)) if complete and len(group)>1 else None)
        summary.append(record)
    pd.DataFrame(summary, columns=['dataset','mechanism','model','complete','pilot','auc_mean','auc_sd']).to_csv(
        Path(root)/'auc.csv', index=False)
