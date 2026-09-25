# Release validation

Checks run on CPU with the exact direct dependency versions in
`requirements.txt` (Python 3.13). No scientific campaigns were relaunched.
These checks establish executable paths and basic invariants, not full
five-seed score reproducibility. A clean dependency download/install and GPU
execution were not repeated for this packaging task.

- 15 unit tests pass: RT reference equivalence on random/disconnected graphs,
  exact budgets, shared component sources, cache integrity, paired classification
  splits, time/channel indexing, chronological separation, training-only scaler
  statistics, no test-window materialization during tuning, Gaussian mixture
  responsibilities and gradients, disconnected batched graph isolation, and
  accelerated PCFI/FISF equivalence to the original implementations.
- All nine main models complete 2-epoch classification checks on the bundled
  TADPOLE graph under both RT and UMCAR: 18 successful fits.
- All nine main models complete 1-epoch forecasting checks on each of the six
  bundled temporal datasets: natural masks for AQI/PV-US/METR-LA and UMCAR for
  EngRAD/GraphMSO/PEMS-BAY. All nine also complete RT on GraphMSO: 63 successful
  fits. Pilots use 4 or 8 windows per split, not the full dataset for training.
- The initial 4-window PV-US pilot correctly stopped because its test targets
  were all unobserved nighttime values. Expanding to 8 predetermined,
  chronologically spaced windows gives observed targets in every split and all
  nine methods complete. No mask, target value or model was changed.
- A 12-candidate PEMix pilot search, validation selection, final test, resume
  without duplicated records, and supplied-preset evaluation all complete.
  The selection/resume/preset checks were repeated after the final provenance
  guard changes. The extended grid contains exactly 48 candidates.
- Classification and forecasting CSV/plot/report generation completes.
- All bundled dataset SHA-256 checks pass. The temporal files retain their
  original bytes. TADPOLE retains x/y/edges and adds frozen topology PE, removing
  unused auxiliary fields and historical split masks.
- Retained definitions in `models.py`, `utils.py`, `fisf.py` and
  `temporal_imputation.py` match the originals by AST comparison; see
  `IMPLEMENTATION_CHECK.json`. Packaging removes obsolete branches and adapts
  launchers/missingness dispatch; it does not rewrite baseline computations.

The ZIP uses uniform member timestamps, relative paths and no archive comments,
UID/GID metadata, symlinks, virtual environments, compiled native objects,
bytecode, outputs, caches, checkpoints, history or private infrastructure files.
Project-specific names, accounts, email addresses, server paths and notification
endpoints are scanned out of shipped source, JSON and serialized metadata.
Third-party legal attributions and public dataset provenance are retained.
The archive is additionally extracted into a new folder and checked for data
integrity, native compilation, unit tests and a short classification run.
