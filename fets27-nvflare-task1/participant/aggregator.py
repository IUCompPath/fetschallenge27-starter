"""Participant-editable server aggregation logic.

Edit this file and only this file if you want to change how the server aggregates
client updates.
"""

from __future__ import annotations

import logging
import time

import numpy as np

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
except ModuleNotFoundError:
    from fets27_challenge.compat import FLMetaKey, FLModel, ModelAggregator


LOGGER = logging.getLogger(__name__)


def _value_nbytes(value) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    return int(np.asarray(value).nbytes)


def _summarize_params(params) -> tuple[int, int]:
    if not params:
        return 0, 0
    return len(params), sum(_value_nbytes(value) for value in params.values())


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


class ParticipantAggregator(ModelAggregator):
    """Baseline weighted FedAvg aggregator.

    Participants are expected to keep the public contract the same:
    `accept_model`, `aggregate_model`, and `reset_stats`.
    """

    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None
        self.accepted_updates = 0

    def accept_model(self, model: FLModel):
        accept_start = time.perf_counter()
        # TODO: participants can replace this weighting rule.
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        tensor_count, payload_bytes = _summarize_params(model.params)
        LOGGER.info(
            "[server] accepting client update: tensors=%s approx_payload=%s "
            "weight=%s params_type=%s metrics=%s meta_keys=%s",
            tensor_count,
            _format_bytes(payload_bytes),
            weight,
            model.params_type,
            getattr(model, "metrics", {}),
            sorted(getattr(model, "meta", {}).keys()),
        )

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            LOGGER.error(
                "[server] rejecting client update: expected params_type=%r "
                "received=%r",
                self.params_type,
                model.params_type,
            )
            raise ValueError(
                f"Expected params_type={self.params_type!r} but received {model.params_type!r}."
            )

        for key, value in model.params.items():
            value = np.asarray(value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight
        self.accepted_updates += 1
        LOGGER.info(
            "[server] accepted client update %s: total_weight=%.3f elapsed=%.1fs",
            self.accepted_updates,
            self.total_weight,
            time.perf_counter() - accept_start,
        )

    def aggregate_model(self) -> FLModel:
        aggregate_start = time.perf_counter()
        # TODO: participants can replace the aggregation rule itself.
        if self.total_weight <= 0:
            LOGGER.error("[server] aggregation waiting failed: no accepted updates")
            raise ValueError(
                "No accepted client updates are available for aggregation."
            )

        aggregated = {
            key: value / self.total_weight for key, value in self.weighted_sum.items()
        }
        tensor_count, payload_bytes = _summarize_params(aggregated)
        LOGGER.info(
            "[server] aggregated %s client updates: tensors=%s approx_payload=%s "
            "total_weight=%.3f elapsed=%.1fs",
            self.accepted_updates,
            tensor_count,
            _format_bytes(payload_bytes),
            self.total_weight,
            time.perf_counter() - aggregate_start,
        )
        return FLModel(params=aggregated, params_type=self.params_type)

    def reset_stats(self):
        # TODO: participants can keep per-round state if they also clear it here.
        LOGGER.info(
            "[server] reset aggregation stats: cleared_updates=%s total_weight=%.3f",
            self.accepted_updates,
            self.total_weight,
        )
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None
        self.accepted_updates = 0


def build_aggregator() -> ParticipantAggregator:
    """Stable factory used by the locked runtime."""

    return ParticipantAggregator()
