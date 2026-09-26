# Supplementary code

PEMix and the nine-method comparisons for node classification and static-graph forecasting.
The method, experimental protocol, and hyperparameters are described in the paper,
particularly Section 4 and Appendices B and E. The runners load the corresponding
configurations automatically; no hyperparameters need to be copied into commands.

## Installation

Use Python 3.13 on Linux or macOS and run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_native.py
```

RT requires a C++17 compiler. The build enables OpenMP when available and otherwise
builds the same algorithm in serial mode. `CXX` can select a compiler; use
`python build_native.py --no-openmp` to explicitly select the serial build.
For GPU execution, install a compatible PyTorch build and use `--device cuda`.

## Data

See [data/README.md](data/README.md) for input formats and preparation.
GraphMSO is generated locally without a download:

```bash
python temporal_data.py --datasets graphmso
```

## Node classification

Place the prepared TADPOLE graph at `data/tadpole.pt`, then run:

```bash
python run.py --datasets tadpole --data-dir data --prepared \
  --models all --mechanisms RT UMCAR --device cpu --out outputs/tadpole
python summarize.py outputs/tadpole --plots
```

Use `--models PEMix` to run PEMix alone. Network and optimization settings follow
Appendix E, Table 26; PEMix uses the responsibilities in Eq. (12).

## Forecasting

```bash
python temporal_run.py --datasets graphmso --models all \
  --mechanisms RT UMCAR --device cpu --out outputs/graphmso
python temporal_summarize.py outputs/graphmso
```

The default test phase automatically loads the dataset/mechanism configuration
from `configs/presets/`. Each configuration is fixed across the missingness curve.
See Appendix E, Tables 27–38, for model settings and Eq. (12) for PEMix responsibilities.
Use `--models PEMix` to run PEMix alone.

The other supported datasets are `engrad`, `pems_bay`, `aqi`, `pv_us`, and `metr_la`.
For the latter three, the runner preserves natural missingness and injects no
additional mask. Prepare the relevant raw files before changing `--datasets`.

To perform a new validation search and then evaluate its selected configuration:

```bash
python temporal_campaign.py --datasets graphmso --models all \
  --device cpu --out outputs/graphmso_search
```

The search evaluates the same number of candidates per method. Controlled experiments
select one configuration per method and mechanism by validation MAE AUC over the
full curve. Natural-missingness experiments select by validation MAE. The tuning
seed is separate from the evaluation seeds, and tuning does not evaluate the test set.

## Verification and outputs

```bash
python -m unittest discover -s tests -v
python temporal_run.py --datasets graphmso --models all --rates 0.5 \
  --seeds 1 --epochs 2 --max-windows 2 --out outputs/smoke
```

Reduced runs are marked as pilots. Every attempted fit is saved in `per_seed.jsonl`;
errors are retained in the records and detailed logs. Summarizers export per-rate
means and sample standard deviations, plus `auc.csv` and `auc_per_seed.csv` for
complete controlled curves. Areas use the trapezoidal rule without normalization.
Incomplete groups have no aggregate score.

Repeating an identical command resumes its completed fits. Source, input, configuration,
and environment fingerprints must match; use a new output directory after changes.
