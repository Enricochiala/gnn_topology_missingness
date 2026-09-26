"""Run reproducible node-classification comparisons and save each attempted fit."""

import argparse
from contextlib import chdir, redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from paper_data import DATASETS, load_dataset
from experiments import MAIN_MODELS, MECHANISMS, evaluate, make_entries

ROOT = Path(__file__).resolve().parent
KEYS = ('dataset', 'model', 'seed', 'mu', 'mechanism')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    header = f'{tuple(value.shape)}:{value.dtype}:'.encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--prepared', action='store_true', help='Input already includes preprocessing and frozen PE')
    parser.add_argument('--models', nargs='+', choices=MAIN_MODELS + ('all',), default=['all'])
    parser.add_argument('--mechanisms', nargs='+', choices=MECHANISMS, default=['UMCAR', 'RT'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 43, 15, 118, 222])
    parser.add_argument('--rates', nargs='+', type=float, default=[0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
    parser.add_argument('--pe-dim', type=positive, default=8)
    parser.add_argument('--recompute-pe', action='store_true')
    parser.add_argument('--n-components', type=positive, default=5, help='For PEMix; plain GCNmf remains K=5')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-epochs', type=positive, default=None, help='Optional override; omit for paper defaults')
    parser.add_argument('--patience', type=positive, default=None)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.models = list(dict.fromkeys(m for model in args.models
                                   for m in (MAIN_MODELS if model == 'all' else [model])))
    for key in ('datasets', 'mechanisms', 'seeds', 'rates'):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    if any(s < 0 for s in args.seeds) or any(not np.isfinite(p) or not 0 <= p <= 1 for p in args.rates):
        parser.error('Nonnegative seeds and finite rates in [0, 1] are required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('out', 'data_dir')}
    config['pilot'] = any(v is not None for v in (args.max_epochs, args.patience)) or args.pe_dim != 8 or args.n_components != 5
    config['source_hashes'] = {p.name: sha256(p) for p in sorted(ROOT.glob('*.py'))}
    config['input_hashes'] = {d: sha256(args.data_dir / f'{d}.pt') for d in args.datasets}
    config['versions'] = {n: importlib.metadata.version(n) for n in
                         ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn',
                          'networkx', 'pandas', 'threadpoolctl')}
    config['python'] = platform.python_version()
    config['machine'] = platform.platform()
    config_file = args.out / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            parser.error('Output belongs to a different configuration/code/input/environment; use a new --out')
    else:
        if any(args.out.iterdir()):
            parser.error('New output directory must be empty')
        write_json(config_file, config)
    for part in ('inputs', 'masks', 'logs'):
        (args.out / part).mkdir(exist_ok=True)
    records_file = args.out / 'per_seed.jsonl'
    rows = [json.loads(line) for line in records_file.read_text().splitlines() if line.strip()] if records_file.exists() else []
    done = {tuple(r[k] for k in KEYS) for r in rows}
    if len(done) != len(rows):
        raise ValueError('Duplicate result records')
    failed = sum(r['status'] != 'ok' for r in rows)
    with threadpool_limits(limits=1):
        for dataset in args.datasets:
            frozen = args.out / 'inputs' / f'{dataset}.pt'
            if frozen.exists():
                data, _ = load_dataset(dataset, frozen.parent, prepared=True, pe_dim=args.pe_dim)
            else:
                data, metadata = load_dataset(dataset, args.data_dir, prepared=args.prepared,
                    pe_dim=args.pe_dim, recompute_pe=args.recompute_pe)
                torch.save(data, frozen)
                write_json(frozen.with_suffix('.json'), dict(**metadata,
                    tensors={k: tensor_hash(getattr(data, k)) for k in ('x', 'y', 'edge_index', 'pe')}))
            tensor_metadata = json.loads(frozen.with_suffix('.json').read_text())['tensors']
            for key, digest in tensor_metadata.items():
                if tensor_hash(getattr(data, key)) != digest:
                    raise ValueError(f'Frozen input changed: {dataset}/{key}')
            for seed in args.seeds:
                plans = {}
                for mu in args.rates:
                    entries = make_entries(data, seed, mu, args.mechanisms, plans)
                    for mechanism, entry in entries.items():
                        name = f'{dataset}_{seed}_{mu}_{mechanism}'
                        mask_file = args.out / 'masks' / f'{name}.npz'
                        arrays = {k: v.numpy() for k, v in entry.items() if k != 'X_incomp'}
                        if mask_file.exists():
                            with np.load(mask_file) as previous:
                                for key, values in arrays.items():
                                    np.testing.assert_array_equal(previous[key], values)
                        else:
                            np.savez_compressed(mask_file, **arrays)
                        mask = entry['mask']
                        for model in args.models:
                            key = (dataset, model, seed, mu, mechanism)
                            if key in done:
                                continue
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            random.seed(seed)
                            record = dict(zip(KEYS, key))
                            record.update(pilot=config['pilot'], mask_hash=tensor_hash(mask), pe_hash=tensor_metadata['pe'],
                                split_hash=tensor_hash(torch.stack([entry[k] for k in ('train_mask', 'val_mask', 'test_mask')])),
                                actual_missingness=float(mask.float().mean()),
                                fully_missing_rows=float(mask.all(1).float().mean()))
                            start = time.perf_counter()
                            with (args.out / 'logs' / f'{name}_{model}.log').open('w') as log, redirect_stdout(log), redirect_stderr(log), chdir(args.out):
                                try:
                                    values = evaluate(data, entry, seed, model, device=args.device,
                                        n_components=args.n_components, max_epochs=args.max_epochs, patience=args.patience)
                                    f1, acc = float(values[2]), float(values[0])
                                    if not np.isfinite([f1, acc]).all() or not 0 <= f1 <= 1 or not 0 <= acc <= 1:
                                        raise ValueError('Non-finite or out-of-range score')
                                    auc = float(values[4]) if len(values) > 4 and np.isfinite(values[4]) else None
                                    record.update(status='ok', f1=f1, accuracy=acc, roc_auc=auc, error=None)
                                except Exception as exc:
                                    traceback.print_exc()
                                    failed += 1
                                    record.update(status='failed', f1=None, accuracy=None, roc_auc=None,
                                                  error=f'{type(exc).__name__}: {exc}')
                            record['seconds'] = time.perf_counter() - start
                            with records_file.open('a') as handle:
                                handle.write(json.dumps(record, allow_nan=False) + '\n')
                                handle.flush()
                            done.add(key)
                            print(f'{dataset} seed={seed} mu={mu} {mechanism} {model}: '
                                  f'{record["status"]}, F1={record["f1"]}', flush=True)
    expected = len(args.datasets) * len(args.models) * len(args.seeds) * len(args.rates) * len(args.mechanisms)
    if len(done) != expected:
        raise RuntimeError(f'Unexpected result count: {len(done)} != {expected}')
    write_json(args.out / 'COMPLETE.json', {'attempted': len(done), 'failed': failed,
                                           'pilot': config['pilot'], 'successful': len(done) - failed})
    print(f'Completed {len(done)} attempts; {failed} failures. Saved in {args.out}')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
