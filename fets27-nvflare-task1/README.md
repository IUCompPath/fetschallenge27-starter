# FeTS27 Task 1 NVFLARE Challenge

This repository is a simulator-first NVFLARE baseline for FeTS27 Task 1.

Participants are expected to edit exactly two files:

- `participant/aggregator.py`
- `participant/site_hparams.yaml`

Everything else is organizer-controlled and should be treated as read-only.

## What This Repo Does

- Runs cohort-specific NVFLARE federated training for `glioma`, `meningioma`, and `sub_sahara`
- Initializes the server model from a locked SegResNet baseline
- Loads a participant-defined server aggregator
- Applies participant-defined per-site training hyperparameters
- Evaluates the best global checkpoint with a locked public scorer
- Packages a submission containing only the two allowed participant files

## Environment Setup

Install PyTorch separately for your machine first. Example:

```bash
pip install torch torchvision
```

Then install the challenge dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Or use the provided setup scripts:

```bash
./setup.sh
```

On Windows PowerShell:

```powershell
.\setup.ps1
```

The setup step can also generate toy data and deterministic baseline checkpoints once the required ML packages are installed.

## Expected Data Layout

The real FeTS27 data is distributed outside this repository. The runtime expects:

```text
<data-root>/
  glioma/
    dataset/
    datalist/
      site-1.json
      site-2.json
      site-All.json
  meningioma/
    dataset/
    datalist/
      site-1.json
      site-2.json
      site-All.json
  sub_sahara/
    dataset/
    datalist/
      site-1.json
      site-2.json
      site-All.json
```

Reference datalist examples are provided under `assets/sample_datalists/`.

## Participant Workflow

1. Edit `participant/aggregator.py`
2. Edit `participant/site_hparams.yaml`
3. Validate the submission surface:

```bash
python -m fets27_challenge.cli validate-submission
```

4. Run one cohort locally:

```bash
python -m fets27_challenge.cli run-local --cohort glioma --data-root ./data/toy --workspace ./workspace --output-dir ./outputs/local
```

5. Run all cohorts locally:

```bash
python -m fets27_challenge.cli run-local --cohort all --data-root ./data/toy --workspace ./workspace --output-dir ./outputs/local
```

6. Package the submission:

```bash
python -m fets27_challenge.cli package-submission --output ./submission/fets27_task1_submission.zip
```

## Organizer Workflow

Run the official evaluation entrypoint on the hidden data root:

```bash
python -m fets27_challenge.cli run-official --data-root /secure/fets27_hidden --workspace ./workspace --output-dir ./outputs/official
```

The official command uses the same locked evaluator and scoring formula as the public local flow.

## Scoring

- Per cohort: mean validation Dice across participating sites using the best global checkpoint
- Overall public score: equal-weight macro average across `glioma`, `meningioma`, and `sub_sahara`

The evaluator does not trust TensorBoard summaries. It loads the selected best checkpoint and recomputes the score.

## Public vs Hidden Evaluation

Local public evaluation and official hidden evaluation use the same code path:

- same participant files
- same locked client loop
- same locked evaluator
- same score aggregation rule

The only difference is the data and split files passed to the run.

## Aggregation Ideas

Reference implementations are included in `src/fets27_challenge/reference_aggregators.py`:

- weighted FedAvg baseline
- coordinate-wise median
- clipped mean

Additional notes are in `docs/aggregation_ideas.md`.

## Checkpoints and Toy Data

- `assets/checkpoints/` is where deterministic baseline checkpoints are written
- `data/toy/` can be generated for smoke tests and onboarding

Generate both with:

```bash
python -m fets27_challenge.cli prepare-assets --data-root ./data/toy
```

