"""Local compatibility shims for tests when NVFLARE is not installed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FLMetaKey:
    NUM_STEPS_CURRENT_ROUND = "NUM_STEPS_CURRENT_ROUND"


@dataclass
class FLModel:
    params: dict[str, Any] = field(default_factory=dict)
    params_type: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    current_round: int = 0


class ModelAggregator:
    """Base class for model aggregation."""

    def error(self, message: str):
        """Raise a RuntimeError with the specified message.

        Args:
            message: The error message to raise.
        """
        raise RuntimeError(message)

