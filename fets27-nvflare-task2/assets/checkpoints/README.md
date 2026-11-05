# Baseline Checkpoints

FeTS27 does not ship the real task-2 baseline weights in this repository.

Run:

```bash
python -m fets27_classification.cli prepare-assets --data-root ./data/toy
```

to generate a deterministic, randomly initialized `brain_tumor_baseline.pt`
(the locked ViT-Small/16 architecture) plus a tiny synthetic toy dataset.

`run-local`/`run-official` load this checkpoint as the server's `initial_ckpt`
when it exists and otherwise start from a fresh model. Generated checkpoints are
listed in `.gitignore` and must not be committed.
