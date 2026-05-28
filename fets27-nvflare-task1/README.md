# FeTS27 Task 1 NVFLARE Challenge

This repository is a simulator-first NVFLARE baseline for FeTS27 Task 1.

Participants are expected to edit exactly one file:

- `participant/aggregator.py`

Everything else should be treated as organizer-controlled unless you are explicitly maintaining the runtime.

## What This Repo Does

- Runs federated training for the `glioma` cohort with the NVFLARE simulator
- Starts the server from a locked SegResNet baseline checkpoint
- Loads the participant-defined server aggregator
- Applies locked per-site training hyperparameters
- Evaluates the best global checkpoint with the public scorer
- Packages a submission containing only the allowed participant file

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
  --gpu '[0],[1]'
```

Notes:

- `--gpu '[0],[1]'` maps one simulator client to GPU 0 and the other to GPU 1.
- If you want CPU-only execution, omit `--gpu`.
- NVFLARE may force one thread per GPU group when multi-GPU simulation is used.

### 6. Inspect the results

The local run writes:

- `outputs/training_dummy/local_summary.json`
- `outputs/training_dummy/local_summary.csv`
- `workspace/training_dummy/fets27_glioma/server/simulate_job/app_server/FL_global_model.pt`

TensorBoard logs are written under:

- `workspace/training_dummy/fets27_glioma/server/simulate_job/tb_events`

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

1. Edit `participant/aggregator.py`
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

- Per cohort: mean validation Dice across participating sites using the best global checkpoint
- Overall public score: the `glioma` validation Dice

The evaluator does not rely on TensorBoard summaries. It reloads the selected checkpoint and recomputes the score.

## Public vs Hidden Evaluation

Local public evaluation and official hidden evaluation use the same code path:

- same participant file surface
- same locked client training loop
- same locked evaluator
- same score aggregation rule

The only difference is the data root and split files.

## Aggregation Ideas

Reference implementations are available in `src/fets27_challenge/reference_aggregators.py`:

- weighted FedAvg baseline
- coordinate-wise median
- clipped mean
