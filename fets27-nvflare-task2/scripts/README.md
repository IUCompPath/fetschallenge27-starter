# Scripts

The main user-facing entrypoints live in:

- `setup.sh`
- `setup.ps1`
- `python -m fets27_classification.cli ...`

Organizer-side helpers in this folder:

- `prepare_partition_datalists.py` - builds the runtime `<data-root>/brain_tumor`
  layout from the released Task 2 files under `fets2027-data/`. It reads the
  `ID,Site_ID,Label` training partition plus the `ID,Label` validation labels,
  writes one `datalist/site-<id>.json` per training site plus `site-All.json`,
  and embeds the *official* validation split in every site datalist so all
  simulated clients evaluate the same cases.

  ```bash
  python scripts/prepare_partition_datalists.py \
    --training-partition-csv /path/to/fets2027-data/Task2_Classification_Training_Partition.csv \
    --validation-labels-csv  /path/to/fets2027-data/Task2_Classification_Validation_Labels.csv \
    --training-dir  /path/to/fets2027-data/training \
    --validation-dir /path/to/fets2027-data/validation \
    --data-root     ./data/fets2027_task2
  ```

  Add `--file-mode copy` to copy the four modality NIfTI files into
  `brain_tumor/dataset/` instead of referencing them in place.
