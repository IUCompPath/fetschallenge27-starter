"""Read-only reference aggregators for participants and organizers."""

from __future__ import annotations

import numpy as np

from .compat import FLMetaKey, FLModel, ModelAggregator


class WeightedFedAvgAggregator(ModelAggregator):
    """Reference federated averaging aggregator that weights client models by step count."""

    def __init__(self):
        """Initialize the WeightedFedAvgAggregator."""
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None

    def accept_model(self, model: FLModel):
        """Accept a client model and accumulate its weighted parameter values.

        Args:
            model: The FLModel instance containing parameters and metadata.

        Raises:
            ValueError: If the model's params_type does not match previously accepted models.
        """
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
        """Compute the weighted average of all accepted models.

        Returns:
            An FLModel containing the aggregated parameters.
        """
        if self.total_weight <= 0:
            self.error("No accepted models are available.")
            return FLModel(params={})
        return FLModel(
            params={
                key: value / self.total_weight
                for key, value in self.weighted_sum.items()
            },
            params_type=self.params_type,
        )

    def reset_stats(self):
        """Reset the internal aggregation state for a new round."""
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.params_type = None


class MedianAggregator(ModelAggregator):
    """Reference aggregator that computes the element-wise median of client parameters."""

    def __init__(self):
        """Initialize the MedianAggregator."""
        super().__init__()
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        """Accept a client model and store its parameters.

        Args:
            model: The FLModel instance containing parameters.

        Raises:
            ValueError: If the model's params_type does not match previously accepted models.
        """
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError("All client models must have the same params_type.")
        self.client_models.append(
            {key: np.asarray(value) for key, value in model.params.items()}
        )

    def aggregate_model(self) -> FLModel:
        """Compute the element-wise median of all accepted client parameters.

        Returns:
            An FLModel containing the aggregated median parameters.
        """
        if not self.client_models:
            self.error("No accepted models are available.")
            return FLModel(params={})

        aggregated = {}
        for key in self.client_models[0]:
            stacked = np.stack([model[key] for model in self.client_models], axis=0)
            aggregated[key] = np.median(stacked, axis=0)
        return FLModel(params=aggregated, params_type=self.params_type)

    def reset_stats(self):
        """Reset the internal list of accepted client models for a new round."""
        self.client_models = []
        self.params_type = None


class ClippedMeanAggregator(ModelAggregator):
    """Reference aggregator that clips updates to a specified percentile before averaging."""

    def __init__(self, clip_percentile: float = 80.0):
        """Initialize the ClippedMeanAggregator with the specified clip percentile.

        Args:
            clip_percentile: The percentile at which to clip parameter deviations.
        """
        super().__init__()
        self.clip_percentile = clip_percentile
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        """Accept a client model and store its parameters.

        Args:
            model: The FLModel instance containing parameters.

        Raises:
            ValueError: If the model's params_type does not match previously accepted models.
        """
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError("All client models must have the same params_type.")
        self.client_models.append(
            {key: np.asarray(value) for key, value in model.params.items()}
        )

    def aggregate_model(self) -> FLModel:
        """Clip parameter deviations from the mean and compute the average.

        Returns:
            An FLModel containing the clipped and averaged parameters.
        """
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
        """Reset the aggregator state for a new round."""
        self.client_models = []
        self.params_type = None
