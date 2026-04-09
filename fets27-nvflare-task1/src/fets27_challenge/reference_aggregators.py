"""Read-only reference aggregators for participants and organizers."""

from __future__ import annotations

import numpy as np

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
except ImportError:  # pragma: no cover - local tests use compat shims
    from .compat import FLMetaKey, FLModel, ModelAggregator


class WeightedFedAvgAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None

    def accept_model(self, model: FLModel):
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError("All client models must have the same params_type.")

        for key, value in model.params.items():
            value = np.asarray(value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight <= 0:
            self.error("No accepted models are available.")
            return FLModel(params={})
        return FLModel(
            params={key: value / self.total_weight for key, value in self.weighted_sum.items()},
            params_type=self.params_type,
        )

    def reset_stats(self):
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None


class MedianAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError("All client models must have the same params_type.")
        self.client_models.append({key: np.asarray(value) for key, value in model.params.items()})

    def aggregate_model(self) -> FLModel:
        if not self.client_models:
            self.error("No accepted models are available.")
            return FLModel(params={})

        aggregated = {}
        for key in self.client_models[0]:
            stacked = np.stack([model[key] for model in self.client_models], axis=0)
            aggregated[key] = np.median(stacked, axis=0)
        return FLModel(params=aggregated, params_type=self.params_type)

    def reset_stats(self):
        self.client_models = []
        self.params_type = None


class ClippedMeanAggregator(ModelAggregator):
    def __init__(self, clip_percentile: float = 80.0):
        super().__init__()
        self.clip_percentile = clip_percentile
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError("All client models must have the same params_type.")
        self.client_models.append({key: np.asarray(value) for key, value in model.params.items()})

    def aggregate_model(self) -> FLModel:
        if not self.client_models:
            self.error("No accepted models are available.")
            return FLModel(params={})

        aggregated = {}
        for key in self.client_models[0]:
            stacked = np.stack([model[key] for model in self.client_models], axis=0)
            center = np.mean(stacked, axis=0)
            deltas = stacked - center
            clip_value = np.percentile(np.abs(deltas), self.clip_percentile)
            clipped = np.clip(deltas, -clip_value, clip_value)
            aggregated[key] = np.mean(center + clipped, axis=0)
        return FLModel(params=aggregated, params_type=self.params_type)

    def reset_stats(self):
        self.client_models = []
        self.params_type = None

