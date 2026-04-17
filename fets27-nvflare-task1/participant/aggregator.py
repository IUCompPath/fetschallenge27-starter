"""Participant-editable server aggregation logic.

Edit this file and only this file if you want to change how the server aggregates
client updates.
"""

from __future__ import annotations

import numpy as np

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
except ModuleNotFoundError:
    from fets27_challenge.compat import FLMetaKey, FLModel, ModelAggregator


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

    def accept_model(self, model: FLModel):
        # TODO: participants can replace this weighting rule.
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
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

    def aggregate_model(self) -> FLModel:
        # TODO: participants can replace the aggregation rule itself.
        if self.total_weight <= 0:
            raise ValueError(
                "No accepted client updates are available for aggregation."
            )

        aggregated = {
            key: value / self.total_weight for key, value in self.weighted_sum.items()
        }
        return FLModel(params=aggregated, params_type=self.params_type)

    def reset_stats(self):
        # TODO: participants can keep per-round state if they also clear it here.
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None


def build_aggregator() -> ParticipantAggregator:
    """Stable factory used by the locked runtime."""

    return ParticipantAggregator()
