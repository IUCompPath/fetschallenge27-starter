# Scripts

The main user-facing entrypoints live in:

- `setup.sh`
- `setup.ps1`
- `python -m fets27_challenge.cli ...`

Organizer-side helpers in this folder:

- `prepare_partition_datalists.py` - builds the runtime `<data-root>/glioma`
  layout from `Task1_Segmentation_Partition.csv`. It reads the `ID,Site_ID`
  partition, writes one `datalist/site-<id>.json` per site plus `site-All.json`,
  and holds out `--validation-count` labeled training cases as the shared local
  validation list because the released `validation/` split has no masks.

  ```bash
  python scripts/prepare_partition_datalists.py \
    --partition-csv /path/to/Task1_Segmentation_Partition.csv \
    --training-dir  /path/to/fets2027-data/training \
    --data-root     ./data/fets2027 \
    --validation-count 16
  ```

  Add `--file-mode copy` to copy NIfTI files into `glioma/dataset/` instead of
  referencing them in place.


