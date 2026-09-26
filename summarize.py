"""Export per-model CSV summaries and macro-F1 curves; never average across models."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows:
        raise ValueError('No records to export')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.folder / 'per_seed.jsonl').read_text().splitlines() if line.strip()]
    write_csv(args.folder / 'per_seed.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['model'], row['mechanism'], row['mu'])].append(row)
    config = json.loads((args.folder / 'config.json').read_text())
    from curve_summary import export_auc
    export_auc(rows,config,args.folder,'mu','f1')
    summary = []
    for key, attempts in sorted(groups.items()):
        row = dict(zip(('dataset', 'model', 'mechanism', 'mu'), key))
        valid = [r for r in attempts if r['status'] == 'ok']
        row.update(n_attempted=len(attempts), n_successful=len(valid), n_failed=len(attempts)-len(valid))
        complete = len(valid) == len(config['seeds']) and {r['seed'] for r in valid} == set(config['seeds'])
        row.update(complete=complete, pilot=config.get('pilot', False))
        for metric in ('f1', 'accuracy', 'roc_auc', 'actual_missingness', 'fully_missing_rows'):
            values = [r[metric] for r in (attempts if metric in ('actual_missingness', 'fully_missing_rows') else valid)
                      if r.get(metric) is not None]
            if metric in ('f1', 'accuracy', 'roc_auc') and not complete:
                values = []
            row[f'{metric}_mean'] = float(np.mean(values)) if values else None
            row[f'{metric}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_csv(args.folder / 'summary.csv', summary)
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        config = json.loads((args.folder / 'config.json').read_text())
        models = config['models']
        mechanisms = config['mechanisms']
        for dataset in config['datasets']:
            for offset in range(0, len(models), 3):
                selected = models[offset:offset+3]
                fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 3.7), squeeze=False)
                for ax, model in zip(axes[0], selected):
                    for mechanism in mechanisms:
                        points = sorted([r for r in summary if r['dataset'] == dataset and r['model'] == model
                                         and r['mechanism'] == mechanism], key=lambda r: r['mu'])
                        x = np.array([r['mu'] for r in points])
                        # Incomplete seed groups remain gaps instead of looking like complete results.
                        y = np.array([r['f1_mean'] if r['n_successful'] == len(config['seeds']) else np.nan for r in points], dtype=float)
                        sd = np.array([r['f1_sd'] if r['f1_sd'] is not None else np.nan for r in points])
                        line, = ax.plot(x, y, 'o-', label=mechanism, ms=4)
                        ax.fill_between(x, y-sd, y+sd, alpha=.12, color=line.get_color())
                    ax.set(title=f'{dataset}: {model}', xlabel='Missing fraction', ylabel='Macro-F1', ylim=(0, 1.02))
                    ax.grid(alpha=.2)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), fontsize=8)
                fig.tight_layout(rect=(0, .12 if len(labels) > 4 else .08, 1, 1))
                stem = args.folder / f'{dataset}_f1_{offset//3+1}'
                fig.savefig(stem.with_suffix('.pdf'))
                fig.savefig(stem.with_suffix('.png'), dpi=160)
                plt.close(fig)
    print(f'Exported {len(rows)} attempts and {len(summary)} groups to {args.folder}')


if __name__ == '__main__':
    main()
