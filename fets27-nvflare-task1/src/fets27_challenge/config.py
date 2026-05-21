"""Project-level constants."""

from __future__ import annotations

from pathlib import Path

COHORT_NAMES = ("glioma",)
ALLOWED_HPARAM_KEYS = (
    "learning_rate",
    "batch_size",
    "aggregation_epochs",
    "weight_decay",
    "fedproxloss_mu",
    "cache_dataset",
)
SAMPLE_SITE_NAMES = ("site-1", "site-2")

PARTICIPANT_DIR = Path("participant")
PARTICIPANT_AGGREGATOR_FILE = PARTICIPANT_DIR / "aggregator.py"
PARTICIPANT_HPARAM_FILE = PARTICIPANT_DIR / "site_hparams.yaml"
ALLOWED_EDITABLE_FILES = {
    PARTICIPANT_AGGREGATOR_FILE.as_posix(),
}

IGNORED_VALIDATION_PARTS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "workspace",
    "outputs",
    ".test-artifacts",
}

DEFAULT_KEY_METRIC = "val_dice"
DEFAULT_SAVE_FILENAME = "best_FL_global_model.pt"
DEFAULT_TRANSFER_TYPE = "DIFF"
MANIFEST_FILE = Path("challenge_manifest.json")
