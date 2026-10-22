# Data Notes

This repository does not ship real FeTS27 medical data.

Use `python -m fets27_challenge.cli prepare-assets --data-root ./data/toy` to create a tiny synthetic dataset for smoke tests.

For real runs, point `--data-root` to the organizer-provided data directory that follows the layout documented in the root `README.md`.

To build that layout from the released training tree and the official partition,
run `python scripts/prepare_partition_datalists.py` with
`--partition-csv Task1_Segmentation_Partition.csv` and
`--training-dir fets2027-data/training`. Training masks label the enhancing
tumor as `3` (the older label `4` was remapped to `3`). Legacy masks that still
use `4` for the enhancing tumor are accepted and treated as ET with a one-time
warning (BraTS 2023 convention). The released `validation/` split has no masks,
so the helper reserves labeled training cases for local validation.

