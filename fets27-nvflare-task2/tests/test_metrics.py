from __future__ import annotations

import math

import pytest

from fets27_classification.metrics import DEFAULT_THRESHOLD, binary_metrics

LABELS = [0, 0, 1, 1]
PROBABILITIES = [0.1, 0.4, 0.35, 0.8]


def test_binary_metrics_match_hand_computed_values():
    metrics = binary_metrics(LABELS, PROBABILITIES)

    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["mcc"] == pytest.approx(2 / math.sqrt(12))


def test_default_threshold_is_fixed_at_one_half():
    assert DEFAULT_THRESHOLD == 0.5
    at_half = binary_metrics([0], [0.5])
    below_half = binary_metrics([0], [0.4999])

    assert at_half["confusion_matrix"] == [[0, 1], [0, 0]]
    assert below_half["confusion_matrix"] == [[1, 0], [0, 0]]


def test_balanced_accuracy_is_symmetric_for_perfect_and_inverted_models():
    perfect = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    inverted = binary_metrics([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])

    assert perfect["balanced_accuracy"] == pytest.approx(1.0)
    assert perfect["mcc"] == pytest.approx(1.0)
    assert inverted["balanced_accuracy"] == pytest.approx(0.0)
    assert inverted["mcc"] == pytest.approx(-1.0)


def test_mcc_is_zero_for_constant_predictions():
    metrics = binary_metrics([0, 0, 1, 1], [0.4, 0.4, 0.4, 0.4])

    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_mcc_returns_zero_when_a_marginal_is_empty():
    metrics = binary_metrics([1, 1], [0.9, 0.9])

    assert metrics["mcc"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 0], [0, 2]]


def test_auroc_handles_ties_and_single_class():
    tied = binary_metrics([0, 1], [0.5, 0.5])
    single_class = binary_metrics([0, 0], [0.2, 0.8])

    assert tied["auroc"] == pytest.approx(0.5)
    assert math.isnan(single_class["auroc"])
