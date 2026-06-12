"""Loading helpers for participant-editable files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .config import PARTICIPANT_AGGREGATOR_FILE


def load_participant_aggregator(repo_root: Path):
    """Load the participant's custom aggregator from the repository.

    This function dynamically imports the participant's aggregator file, instantiates the
    aggregator class or calls the builder function, and verifies that the required interface methods
    (`accept_model`, `aggregate_model`, and `reset_stats`) are callable.

    Args:
        repo_root: Path to the repository root directory.

    Returns:
        An instance of the participant's aggregator.

    Raises:
        ImportError: If the participant aggregator module cannot be loaded.
        AttributeError: If the module does not expose ParticipantAggregator or build_aggregator().
        TypeError: If any of the required methods are missing or not callable.
    """
    module_path = repo_root / PARTICIPANT_AGGREGATOR_FILE
    spec = importlib.util.spec_from_file_location("participant.aggregator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load participant aggregator from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "build_aggregator"):
        aggregator = module.build_aggregator()
    elif hasattr(module, "ParticipantAggregator"):
        aggregator = module.ParticipantAggregator()
    else:
        raise AttributeError(
            "participant/aggregator.py must expose ParticipantAggregator or build_aggregator()."
        )

    for method_name in ("accept_model", "aggregate_model", "reset_stats"):
        if not callable(getattr(aggregator, method_name, None)):
            raise TypeError(
                f"Participant aggregator is missing callable method {method_name}()."
            )
    return aggregator
