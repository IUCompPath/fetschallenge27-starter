# Data Notes

This repository does not ship real FeTS27 medical data.

Use `python -m fets27_classification.cli prepare-assets --data-root ./data/toy`
to create a tiny synthetic dataset for smoke tests.

For real runs, build the runtime data root from the released Task 2 files under
`fets2027-data/`:

```bash
python scripts/prepare_partition_datalists.py \
  --training-partition-csv /path/to/fets2027-data/Task2_Classification_Training_Partition.csv \
  --validation-labels-csv  /path/to/fets2027-data/Task2_Classification_Validation_Labels.csv \
  --training-dir  /path/to/fets2027-data/training \
  --validation-dir /path/to/fets2027-data/validation \
  --data-root ./data/fets2027_task2
```

The helper reads `ID,Site_ID,Label` for training and `ID,Label` for the official
shared validation split, maps `glioma -> 0` and `meningioma -> 1`, and writes
`<data-root>/brain_tumor/datalist/site-<Site_ID>.json` plus `site-All.json`. It
never writes inside the source data tree; by default it references the four
NIfTI modalities in place. Add `--file-mode copy` to copy them into
`brain_tumor/dataset/` instead.

Classification cases use only `t1c`, `t1n`, `t2f`, and `t2w`; segmentation masks
are never read (the released validation split has none).
