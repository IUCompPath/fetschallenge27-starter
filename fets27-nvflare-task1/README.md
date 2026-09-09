# FeTS27 Task 1 NVFLARE Challenge

This repository is a simulator-first NVFLARE baseline for FeTS27 Task 1.

Participants are expected to edit exactly two files:

- `participant/aggregator.py`
- `participant/client.py`, which exposes `local_train()`

Everything else should be treated as organizer-controlled unless you are explicitly maintaining the runtime.

## Challenge Tracks

The challenge is planned as two independent open tracks focused on the best
federated task performance:

1. **Open segmentation:** participants may customize both local training and
   server aggregation. This is the only track implemented in this repository.
2. **Open classification:** participants will have the same customization
   surface for a future classification task. Classification data, models,
   training, and evaluation are not implemented yet.

The previously considered aggregation-only track has been removed because it
would duplicate a constrained subset of the open tracks. Organizers will keep
the dataset, splits, official evaluation, initial checkpoint, and federated
lifecycle fixed for each task so submissions remain comparable.

Each track is intended to combine normalized task performance with
communication efficiency:

```text
communication_efficiency = min(1, baseline_bytes / measured_bytes)
ranking_score = 0.95 * task_performance_score + 0.05 * communication_efficiency
```

This repository does not yet implement scoring-grade communication accounting
or the combined ranking score. That work is tracked in
[issue #3](https://github.com/IUCompPath/fetschallenge27-starter/issues/3).

## What This Repo Does

- Runs federated training for the `glioma` cohort with the NVFLARE simulator
- Starts the server from a locked SegResNet baseline checkpoint
- Loads the participant-defined server aggregator
- Provides per-site default training hyperparameters to `local_train()`
- Evaluates the best global checkpoint with the public scorer
- Packages a submission containing only the allowed participant files

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

### 3. Prepare the baseline assets

This writes the deterministic baseline checkpoint used by the simulator:

```bash
python -m fets27_challenge.cli prepare-assets --data-root ./data/toy
```

### 4. Prepare the training-dummy layout

If your FeTS dummy cases are available in WSL at `/mnt/f/Brain/Training-Dummy`, run:

```bash
python -m fets27_challenge.cli prepare-training-dummy \
  --source-root /mnt/f/Brain/Training-Dummy \
  --data-root ./data/training_dummy
```

By default this keeps the original NIfTI files in place and writes datalists under `data/training_dummy/glioma/datalist/`.

If you want to physically copy the dataset into this repo layout instead, add:

```bash
--file-mode copy
```

### 5. Run a local experiment

Single-round local run:

```bash
python -m fets27_challenge.cli run-local \
  --cohort glioma \
  --data-root ./data/training_dummy \
  --workspace ./workspace/training_dummy \
  --output-dir ./outputs/training_dummy \
  --num-rounds 1 \
  --threads 2 \
  --data-loader-workers 2 \
  --gpu '[0],[1]'
```

Notes:

- `--gpu '[0],[1]'` maps one simulator client to GPU 0 and the other to GPU 1.
- If you want CPU-only execution, omit `--gpu`.
- NVFLARE may force one thread per GPU group when multi-GPU simulation is used.
- `--data-loader-workers` is the number of worker processes used by each
  training and validation loader in every client. The simulator default is 2;
  resource use grows with the client, loader, and worker counts.

### 6. Inspect the results

The local run writes:

- `outputs/training_dummy/local_summary.json`
- `outputs/training_dummy/local_summary.csv`
- `workspace/training_dummy/fets27_glioma/server/simulate_job/app_server/FL_global_model.pt`

TensorBoard logs are written under:

- `workspace/training_dummy/fets27_glioma/server/simulate_job/tb_events`

To watch a run while it is still training:

```bash
tail -f \
  workspace/training_dummy/fets27_glioma/site-1/log.txt \
  workspace/training_dummy/fets27_glioma/site-2/log.txt
```

For a shorter progress-only view:

```bash
grep -E "starting validation|validation complete|starting local training|step [0-9]+/|epoch .*complete|prepared update|sending update|Traceback|ERROR" \
  workspace/training_dummy/fets27_glioma/site-*/log.txt
```

Server-side NVFLARE logs are in:

```bash
tail -f workspace/training_dummy/fets27_glioma/server/log.txt
```

You can also open TensorBoard:

```bash
tensorboard --logdir workspace/training_dummy/fets27_glioma/server/simulate_job/tb_events --port 6006
```

Then open `http://localhost:6006`.

If the console stops at `Waiting for result from peer`, that usually means the server is waiting for each client training script to finish and send its model update. Check the per-site `log.txt` files above for validation, step, loss, and update progress. On multi-GPU runs, each client may log `device=cuda:0` even when `--gpu '[0],[1]'` is used, because each client process sees its assigned GPU as local device `cuda:0`.

## Expected Data Layout

The runtime expects:

```text
<data-root>/
  glioma/
    dataset/
    datalist/
      site-1.json
      site-2.json
      site-All.json
```

Reference datalist examples are provided under `assets/sample_datalists/`.

## Participant Workflow

1. Edit `participant/aggregator.py` and/or `participant/client.py`
2. Validate the submission surface:

```bash
python -m fets27_challenge.cli validate-submission
```

3. Run locally:

```bash
python -m fets27_challenge.cli run-local \
  --cohort glioma \
  --data-root ./data/training_dummy \
  --workspace ./workspace/training_dummy \
  --output-dir ./outputs/training_dummy
```

4. Package the submission:

```bash
python -m fets27_challenge.cli package-submission \
  --output ./submission/fets27_task1_submission.zip
```

## Organizer Workflow

Run the official entrypoint on the hidden data root:

```bash
python -m fets27_challenge.cli run-official \
  --data-root /secure/fets27_hidden \
  --workspace ./workspace \
  --output-dir ./outputs/official
```

The official flow uses the same locked evaluator and score calculation as the public local flow.

## Scoring

The currently implemented segmentation evaluation reports:

- Per cohort: mean validation Dice across participating sites using the best global checkpoint
- Overall public score: the `glioma` validation Dice

The evaluator does not rely on TensorBoard summaries. It reloads the selected checkpoint and recomputes the score.

The planned official ranking will combine a task-defined performance score with
communication efficiency as described under [Challenge Tracks](#challenge-tracks).
The classification performance metric will be defined with that task. Until
[issue #3](https://github.com/IUCompPath/fetschallenge27-starter/issues/3) is
implemented, the current Dice results are task-performance outputs only and no
communication-aware ranking score is produced.

## Public vs Hidden Evaluation

Local public evaluation and official hidden evaluation use the same code path:

- same participant file surface
- same organizer-owned client lifecycle and evaluator
- same locked evaluator
- same score aggregation rule

The only difference is the data root and split files.

## Aggregation Ideas

Reference implementations are available in `src/fets27_challenge/reference_aggregators.py`:

- weighted FedAvg baseline
- coordinate-wise median
- clipped mean

## Local Training Customization

`participant/client.py` exposes one function, `local_train()`. The locked
client runner builds the model and data loaders, receives each global model,
computes the official validation metric, calls `local_train()`, validates its
result, and sends that result through NVFLARE.

`local_train()` receives the organizer-provided model and training loader plus
the current round, default training hyperparameters, persistent per-client
`state`, and metadata returned by the server in the previous round. Participants
may replace the baseline optimizer, loss, local schedule, regularization, and
other local optimization behavior inside this function.

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
  The locked runner always computes and sets the official `val_dice` itself.

The organizer-owned runner retains control of `flare.init()`,
`flare.receive()`, official validation, `flare.send()`, and the overall client
lifecycle. The participant aggregator remains fully editable and receives the
returned `FLModel` through `accept_model()`. Metadata returned by
`aggregate_model()` is passed to `local_train()` as `server_meta` in the next
round.

The `approx_payload` log messages in the client runner and baseline aggregator
are non-authoritative diagnostics based only on raw tensor storage. They omit
serialization, metadata, transport behavior, and cumulative directional totals,
so they must not be used for official communication accounting or ranking.

This branch is an exploratory interface. Final challenge validation still
needs explicit resource limits, metadata size/type checks, and isolation of
participant code from other clients' data and side channels.
