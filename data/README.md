# Data preparation

Datasets are not bundled. Keep their original access and redistribution terms.

## TADPOLE

`run.py --data-dir data --prepared` expects `data/tadpole.pt`, a serialized
`torch_geometric.data.Data` with:

- `x`: finite floating-point node features, shape `[N, D]`;
- `y`: integer class labels in `{0, 1, 2}`, shape `[N]`;
- `edge_index`: int64 source/destination graph edges, shape `[2, E]`;
- `pe`: floating-point positional encodings, shape `[N, q]`, with `q` from Appendix E.

Use the graph preprocessing described by the dataset source and the paper. The
runner standardizes PE using graph geometry, creates paired stratified splits,
and generates missingness masks. Without `--prepared`, it computes nontrivial
normalized-Laplacian eigenvectors from the stored graph. Original complete values
are not supplied to the model after masking.

## Forecasting

GraphMSO needs no external input:

```bash
python temporal_data.py --datasets graphmso
```

For the other datasets, [temporal/SOURCES.json](temporal/SOURCES.json) records the
raw download locations and checksums. Download and extract into this structure:

```text
data/temporal/raw/
  engrad.h5
  pv_us.h5
  aqi/full437.h5
  metr_la/metr_la.h5
  metr_la/distances_la.csv
  pems_bay/pems_bay.h5
  pems_bay/distances_bay.csv
```

Prepare whichever datasets are present:

```bash
python temporal_data.py --datasets engrad pems_bay aqi pv_us metr_la
```

The preparation command writes `.npz` arrays and JSON metadata under
`data/temporal/prepared_v2/`. It includes the PV-US timezone mapping locally and
requires no checkout of another repository. `--raw` and `--out` override input
and output directories.

Each archive contains `values` and `observed` with shape `[T, N, C]`, chronological
`timestamps`, source/destination `edge_index`, `edge_weight`, fixed topology-derived
`pe`, `channels`, and `node_ids`. JSON sidecars describe preprocessing and split
boundaries. Bundled forecasting configurations check the prepared input fingerprint.
The chronological windowing and missing-target treatment follow Appendix B.
