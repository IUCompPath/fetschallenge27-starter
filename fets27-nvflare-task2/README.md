# FeTS27 Task 2 NVFLARE Classification Challenge

This repository is a simulator-first NVFLARE baseline for FeTS27 Task 2, a
four-modality brain-tumor classification task:

- label `0` = **glioma**, label `1` = **meningioma**
- four MRI modalities per case: `t1c`, `t1n`, `t2f`, `t2w`
- segmentation masks are never read (the released validation split has none)

Participants are expected to edit exactly two files:

- `participant/aggregator.py`
- `participant/client.py`, which exposes `local_train()`

Everything else should be treated as organizer-controlled unless you are
explicitly maintaining the runtime.

## What This Repo Does

- Builds the runtime data root from the released Task 2 partition and validation
  labels under `fets2027-data/`
- Runs federated training for the `brain_tumor` cohort with the NVFLARE simulator
- Starts the server from a locked GroupNorm ResNet-50 classification checkpoint
- Loads the participant-defined server aggregator
- Provides per-site default training hyperparameters to `local_train()`
- Scores the best global checkpoint with the locked evaluator

## Metrics

The locked validation metric that selects the best global checkpoint and defines
the Task 2 score is **balanced accuracy at a fixed 0.5 decision threshold**.

The evaluator reports the full metric set for every run:

| Metric | Definition |
|---|---|
| `balanced_accuracy` | mean of sensitivity and specificity at threshold 0.5 (key metric) |
| `accuracy` | fraction of correctly classified validation cases |
| `auroc` | rank-based area under the ROC curve |
| `macro_f1` | unweighted mean of the positive and negative F1 scores |
| `sensitivity` | recall of the meningioma class |
| `specificity` | recall of the glioma class |
| `mcc` | Matthews correlation coefficient |
| `confusion_matrix` | `[[tn, fp], [fn, tp]]` |

MCC is undefined when any confusion-matrix marginal is zero (for example a
constant predictor or a single-class validation pool); the metric reports `0.0`
in that degenerate case so rankings stay finite. The pilot's
validation-tuned "calibrated" threshold is deliberately **not** used: the
validation split is the ranking split, so tuning a threshold on it would leak.

## Quick Start

This section reflects the WSL flow that was verified in this workspace.

### 1. Create the conda environment

```bash
conda create -y -n nvflare python=3.10
conda activate nvflare
```

### 2. Install runtime dependencies

Install PyTorch first for your machine. In this WSL setup we used:

```bash
pip install torch torchvision
```

Then install the challenge package:

```bash
pip install -r requirements.txt
pip install -e .
```

You can also run:

```bash
./setup.sh
```

On Windows PowerShell:

```powershell
.\setup.ps1
```

If you would rather not install the package (for example in a shared
checkout), prefix every `python -m fets27_classification.cli ...` command with
`PYTHONPATH=src `.

### 3. Prepare the baseline assets

This writes the deterministic baseline checkpoint used by the simulator plus a
tiny synthetic data root for smoke tests:

```bash
python -m fets27_classification.cli prepare-assets --data-root ./data/toy
```

### 4. Build the runtime data root from `fets2027-data/`

The released Task 2 files are:

```text
fets2027-data/
  Task2_Classification_Training_Partition.csv    # ID,Site_ID,Label
  Task2_Classification_Validation_Labels.csv     # ID,Label
  training/FETS2027_Tr_*/                        # t1c,t1n,t2f,t2w[,seg]
  validation/FETS2027_Va_*/                      # t1c,t1n,t2f,t2w
```

Convert them into the runtime layout:

```bash
python scripts/prepare_partition_datalists.py \
  --training-partition-csv /path/to/fets2027-data/Task2_Classification_Training_Partition.csv \
  --validation-labels-csv  /path/to/fets2027-data/Task2_Classification_Validation_Labels.csv \
  --training-dir  /path/to/fets2027-data/training \
  --validation-dir /path/to/fets2027-data/validation \
  --data-root     ./data/fets2027_task2
```

The helper maps `glioma -> 0` and `meningioma -> 1`, writes one
`brain_tumor/datalist/site-<Site_ID>.json` per training site plus `site-All.json`,
and embeds the **official** validation split in every site datalist so all
simulated clients evaluate the same 395 cases. It validates two-way coverage
between the CSVs and the case folders, requires the four modalities, and never
writes inside the released data tree. By default the datalists reference the
source NIfTI files in place; add `--file-mode copy` to copy them into
`brain_tumor/dataset/`.

### 5. Run a local experiment

Single-round local run:

```bash
python -m fets27_classification.cli run-local \
  --data-root ./data/toy \
  --workspace ./workspace/toy \
  --output-dir ./outputs/toy \
  --num-rounds 1 \
  --threads 2 \
  --data-loader-workers 2
```

Notes:

- `--cohort` is optional and defaults to every registered cohort; the only
  registered cohort is `brain_tumor`, so the flag can be omitted.
- `--gpu '[0],[1]'` maps one simulator client to GPU 0 and the other to GPU 1.
  If you want CPU-only execution, omit `--gpu`.
- With a single GPU and many partition sites, pass `--gpu 0 --threads 1` to run
  clients one at a time and avoid GPU memory exhaustion.
- `--data-loader-workers` is the number of worker processes used by each client
  loader; it must be at least 1.
- `--validation-limit N` is a **diagnostic** speed knob that caps client-side
  validation to the first `N` cases. The organizer-side evaluator always scores
  the full validation split, so this cannot change the reported score.
- The real Task 2 partition has 2,332 training cases across 25 sites and 395
  validation cases. Running all 25 sites at 96³ on one GPU (or on CPU) is
  expensive; start with `data/toy` or a subset data root.

### 6. Inspect the results

The local run writes:

- `outputs/toy/local_summary.json`
- `outputs/toy/local_summary.csv`
- `workspace/toy/fets27_brain_tumor/server/simulate_job/app_server/FL_global_model.pt`

TensorBoard logs are written under:

- `workspace/toy/fets27_brain_tumor/server/simulate_job/tb_events`

To watch a run while it is still training:

```bash
tail -f workspace/toy/fets27_brain_tumor/site-*/log.txt
```

For a shorter progress-only view:

```bash
grep -E "official validation complete|starting participant local_train|epoch .*/|accepted client update|Traceback|ERROR" \
  workspace/toy/fets27_brain_tumor/site-*/log.txt
```

Server-side NVFLARE logs are in:

```bash
tail -f workspace/toy/fets27_brain_tumor/server/log.txt
```

## Expected Data Layout

The runtime expects:

```text
<data-root>/
  brain_tumor/
    dataset/
    datalist/
      site-0.json
      site-1.json
      ...
      site-All.json
```

Each `site-<id>.json` defines the `training` cases for that partition site
(entries carry the four modality paths, the integer class label, and the case
subject) plus a shared `validation` list. `<id>` is the `Site_ID` value from
`Task2_Classification_Training_Partition.csv`. Reference datalist examples are
provided under `assets/sample_datalists/`.

## Model

The locked cohort model is a four-channel, two-class 3D ResNet-50 trained from
scratch. MONAI's `resnet50(..., norm=("GROUP", ...))` factory only applies
GroupNorm to the network stem — every bottleneck block stays BatchNorm — so the
locked model replaces all BatchNorm layers with channel-compatible GroupNorm
after construction. That removes running statistics and `num_batches_tracked`
buffers from the parameter payload, which is required for clean weight-difference
(`TransferType.DIFF`) federation across heterogeneous sites.

## NVFLARE Compatibility

The runtime adapts to the installed NVFLARE recipe API. When `FedAvgRecipe`
supports `per_site_config`, every simulated client receives its own resolved
`train_args` (site datalist and hyperparameters). Older releases (for example
2.7.0) accept only one shared `train_args` string; the locked client then
resolves `<datalist_dir>/<site_name>.json` and this site's `site_hparams.yaml`
overrides from the site name reported by the simulator, so per-site data and
hyperparameters still apply. The recipe's `initial_ckpt`, `key_metric`, and
`save_filename` arguments are used only when the installed release provides
them; otherwise the locked checkpoint is loaded into the server model
in-process and the evaluator falls back to `FL_global_model.pt`.

## Participant Workflow

1. Edit `participant/aggregator.py` and/or `participant/client.py`
2. Run locally:

```bash
python -m fets27_classification.cli run-local \
  --data-root ./data/toy \
  --workspace ./workspace/toy \
  --output-dir ./outputs/toy
```

3. Check the metric block in `outputs/toy/local_summary.json`.

By convention only the two participant files are part of a submission; this
repository intentionally ships no manifest or packaging tooling.

## Organizer Workflow

Run the official entrypoint on the hidden data root:

```bash
python -m fets27_classification.cli run-official \
  --data-root /secure/fets27_hidden \
  --workspace ./workspace \
  --output-dir ./outputs/official
```

The official flow uses the same locked evaluator and metric calculation as the
public local flow. The only difference is the data root.

## Local Training Customization

`participant/client.py` exposes one function, `local_train()`. The locked client
runner builds the model and data loaders, receives each global model, computes
the official validation metrics, calls `local_train()`, validates its result, and
sends that result through NVFLARE.

`local_train()` receives the organizer-provided model and training loader plus
the current round, default training hyperparameters, the organizer-computed
inverse-frequency `class_weights` vector for the site's training cases,
persistent per-client `state`, and metadata returned by the server in the
previous round. Participants may replace the baseline optimizer, loss, local
schedule, regularization, and other local optimization behavior inside this
function.

The function must return an NVFLARE `FLModel`:

- `params` must contain the full locally updated state dictionary. The locked
  runner rejects missing, extra, reshaped, or retyped parameters. NVFLARE
  applies the configured `DIFF` transfer after `local_train()` returns. The
  runner accepts equivalent torch or NumPy dtypes and copies all returned
  parameters into detached CPU tensors before NVFLARE computes the difference.
- `meta` may contain string-keyed, NVFLARE-serializable information for the
  participant aggregator, such as update statistics or algorithm state. The
  transport-reserved keys `initial_metrics` and `validate_type` are rejected.
  The baseline reports a positive, finite `NUM_STEPS_CURRENT_ROUND`; the locked
  runner supplies the configured default when that key is omitted.
- `metrics` may contain string-keyed, NVFLARE-serializable participant metrics.
  The locked runner always computes and overwrites the official
  `val_balanced_accuracy` and `val_*` metric values itself.

The organizer-owned runner retains control of `flare.init()`, `flare.receive()`,
official validation, `flare.send()`, and the overall client lifecycle. The
participant aggregator remains fully editable and receives the returned
`FLModel` through `accept_model()`. Metadata returned by `aggregate_model()` is
passed to `local_train()` as `server_meta` in the next round.

The `approx_payload` log messages in the client runner and baseline aggregator
are non-authoritative diagnostics based only on raw tensor storage. They omit
serialization, metadata, transport behavior, and cumulative directional totals,
so they must not be used for official communication accounting or ranking.

## Aggregation Ideas

Reference implementations are available in
`src/fets27_classification/reference_aggregators.py`:

- weighted FedAvg baseline
- coordinate-wise median
- clipped mean

## Hyperparameters

Per-site defaults live in `participant/site_hparams.yaml`. `cache_dataset`
defaults to `0.0`: caching 96³ × 4 float32 cases for the largest site would need
roughly 7 GB, which is not viable. Enable FedProx by setting
`fedproxloss_mu: 0.01`, which reproduces the classifier pilot's FedProx template.
