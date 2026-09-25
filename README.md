# GNNs with missing features

Anonymous research code for node classification and static-graph forecasting.
The original flat repository layout and baseline model implementations are
retained. The temporal adapter represents each sensor as a node and its past
observations as node features, predicting future observations with a static GNN.

## Included methods and missingness

`--models all` selects exactly these nine methods:

| Argument | Method |
|---|---|
| `gnnzero` | GNN-Zero |
| `gnnmi` | GNN with mean imputation |
| `gnnmedian` | GNN with median imputation |
| `gnnmim` | GNN with a missingness indicator |
| `fp` | Feature Propagation |
| `pcfi` | PCFI |
| `fisf` | FISF |
| `gcnmf` | GCNmf |
| `gcnmf_pe` | PEMix / SPAR / GCNmf-PE |

Only **RT** and **UMCAR** can be injected. RT is the **shared-seed,
feature-level compact-growth mechanism**: features share a source node within
each connected component, with feature-specific growth tie-breaking. Its global
budget is exactly `floor(rate * nodes * features)`. Component allocation can
make masks non-nested across rates. The reference Python implementation is
retained and tested against the C++ accelerator. RT never reads feature values
or labels. The older node-level and independently seeded RT variants are absent.

UMCAR samples independent feature entries. The classification path retains the
original safeguard that leaves at least one observed entry in each feature;
the forecasting path uses Bernoulli masks without this safeguard. For temporal
RT, features of the missingness matrix index raw `(timestamp, channel)` pairs,
not repeatedly generated overlapping windows. `True` means missing.

AQI, PV-US and METR-LA automatically use their **natural** missingness, with no
additional injection, regardless of the artificial-mechanism CLI arguments.

## Installation

Tested on Linux with Python 3.13 and the package versions in `requirements.txt`.
The shared-seed accelerator requires a C++17 compiler and OpenMP (`g++` on Linux).
No remote service, account, absolute data path or dataset download is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python build_native.py
python verify_data.py
python -m unittest discover -s tests -v
```

The commands below run from the repository root. CPU is the default. To use a
GPU, install the PyTorch build appropriate for the available CUDA runtime and
pass `--device cuda`. Do not copy a virtual environment between machines.
The native accelerator is built locally, not shipped as a machine-specific binary.

## Bundled data

The archive includes the exact prepared temporal arrays used in the experiments,
including frozen graph PE, and a tensor-only TADPOLE graph. Checksums, dimensions,
and preprocessing notes are in `data/MANIFEST.json` and `data/README.md`.

| Dataset | Task | Nodes | Physical channels | History / horizon | Missingness |
|---|---|---:|---:|---:|---|
| TADPOLE | 3-class node classification | 555 | 15 node features | n/a | injected RT / UMCAR |
| EngRAD | Forecasting | 487 | 5 | 24 / 6 | injected RT / UMCAR |
| GraphMSO | Forecasting | 100 | 1 | 72 / 36 | injected RT / UMCAR |
| PEMS-BAY | Forecasting | 325 | 1 | 24 / 12 | injected RT / UMCAR over original observations |
| AQI | Forecasting | 437 | 1 | 24 / 6 | natural |
| PV-US | Forecasting | 1081 | 1 | 72 / 6 | natural |
| METR-LA | Forecasting | 207 | 1 | 24 / 12 | natural |

Relational experiments use a different graph construction and are not part of
these seven-dataset GNN campaigns.

## Quick installation checks

These commands use very short training and are **not paper results**.

```bash
python run.py --datasets tadpole --data-dir data --prepared \
  --models all --mechanisms RT UMCAR --rates 0.5 --seeds 1 \
  --max-epochs 2 --patience 1 --out outputs/check_tadpole

python temporal_run.py --datasets graphmso --models all \
  --mechanisms RT UMCAR --rates 0.5 --seeds 1 \
  --epochs 1 --max-windows 4 --out outputs/check_forecasting

python summarize.py outputs/check_tadpole --plots
python temporal_summarize.py outputs/check_forecasting
```

## Full classification experiment

```bash
python run.py --datasets tadpole --data-dir data --prepared --models all \
  --mechanisms RT UMCAR --device cuda --out outputs/tadpole
python summarize.py outputs/tadpole --plots
```

Defaults are the eleven rates `0, .1, .2, ..., .9, .99` and five seeds
`1, 43, 15, 118, 222`. Classification retains the original evaluators and their
fixed hyperparameters. The transductive preprocessing and train/validation/test
node splits are unchanged; mean/median imputation uses the original
train-plus-validation feature pool. There is no new automatic classification
tuning loop. TADPOLE PE is computed from graph topology, sign-canonicalized and
bundled; the historical spectral basis of external experiments is not asserted
to be bitwise identical.

## Full forecasting campaign with validation tuning

```bash
python temporal_campaign.py --datasets engrad graphmso pems_bay aqi pv_us metr_la \
  --models all --mechanisms RT UMCAR --device cuda --out outputs/forecasting
```

This sequential launcher runs tuning and testing for each dataset, stopping on
an error. Run the same command to resume completed fits and saved checkpoints.
The default search has 12 candidates **per model, mechanism and missing rate**.
Use `--extended-pemix` for the 48-candidate PEMix grid (other methods remain at 12).
An extended search can be restricted to the desired dataset/model/mechanism:

```bash
python temporal_campaign.py --datasets engrad --models gcnmf_pe \
  --mechanisms UMCAR --extended-pemix --device cuda \
  --out outputs/engrad_umcar_extended
```

For direct control over each phase:

```bash
python temporal_run.py --datasets aqi --models all --phase tune --seeds 2026 \
  --device cuda --out outputs/aqi/tune
python temporal_run.py --datasets aqi --models all --phase test \
  --selection outputs/aqi/tune --device cuda --out outputs/aqi/test
python temporal_summarize.py outputs/aqi/test
```

Tuning uses seed 2026 and validation MAE; it does not materialize test windows.
Selection is **per rate**, not one fixed configuration across a complete curve.
Final evaluation uses five independent training/masking seeds. GPU and library
versions can affect numerical results. Original training maxima (500 or 1000)
and validation early-stopping patience (50 or 40) are retained. Adam uses zero
weight decay in forecasting. Batch size defaults to 32.

The `legacy` and normalized `gaussian` likelihood implementations are retained
for consistency with completed searches; both GCNmf and PEMix can use the latter.
PE conditions PEMix responsibilities and is not concatenated to the downstream
hidden representation. `--phase reference` uses historical defaults; it does
not use validation-selected or extended-search parameters.

## Reuse completed hyperparameter selections

`configs/presets/` includes the completed per-rate selections, with prepared-data
fingerprints. This avoids repeating the search:

```bash
python temporal_run.py --datasets engrad --models all --mechanisms RT \
  --phase test --preset configs/presets/engrad_RT.json \
  --device cuda --out outputs/engrad_rt_selected

python temporal_run.py --datasets pv_us --models all --phase test \
  --preset configs/presets/pv_us_natural.json \
  --device cuda --out outputs/pvus_selected
```

Supply only the dataset and artificial mechanism covered by a preset. EngRAD RT
includes the completed extended PEMix selection (48 candidates), with baseline
selections from their original searches (12 candidates). EngRAD UMCAR contains
the completed original search only; its later extended search was still running
when this archive was assembled. Presets are **hyperparameters**, not checkpoints
or guarantees of reproducing identical floating-point scores. A supplied preset
is held fixed; test performance is not used for reselection.

## Temporal protocol and outputs

Nodes and graph edges remain fixed. The feature vector contains a history of
sensor observations plus known calendar covariates. Chronological train,
validation and test intervals are separated **before** creating windows; a
window and its complete prediction horizon stay within one interval. Default
stride equals the prediction horizon. Scaling and imputation statistics use
observed training inputs after masking. PE uses graph topology only. Artificial
masks affect inputs; targets retain their original observation mask. Naturally
missing targets are excluded from loss and metrics, not imputed.

Each run writes `config.json`, `per_seed.jsonl`, `COMPLETE.json`, masks, logs and,
for forecasting, checkpoints. Tuning also writes `selection.json`. Summarizers
produce CSVs, plots and (for forecasting) a Markdown report. All failed fits are
recorded and produce a nonzero exit code. Do not treat a run with failed fits as
a completed comparison. Means and sample standard deviations require all
configured seeds. Forecast MAE and RMSE are in original units. Rate-curve AUC
is an integral of the error curve, **not ROC AUC**.

Use a distinct `--out` for different configurations. Source, data, environment
and configuration checks prevent silently appending incompatible runs. Cache
and output folders are created locally and are not shipped. Models and
optimizers are unaffected by packaging changes. See `VALIDATION.md` for the
actual release checks and `THIRD_PARTY_NOTICES.md` for retained attributions.

## Layout

- `models.py`, `utils.py`, `layers.py`, `fisf.py`, `filling_strategies.py`: retained model/evaluation code.
- `experiments.py`, `run.py`, `paper_data.py`, `summarize.py`: classification pipeline.
- `temporal_*.py`: static forecasting adapter, runner and summaries.
- `missingness_shared_seed.py`, `shared_seed_reference.py`, `shared_seed_growth.cpp`: RT.
- `configs/`: standard completed selections and extended PEMix search space.
- `data/`: bundled prepared datasets and checksums.
- `tests/`: missingness, reference equivalence and temporal leakage checks.
