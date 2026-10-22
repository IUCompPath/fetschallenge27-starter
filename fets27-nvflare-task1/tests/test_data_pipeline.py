from __future__ import annotations

import logging

import numpy as np

from fets27_challenge.segmentation import ConvertToMultiChannelBasedOnBratsEt3d


def _transform() -> ConvertToMultiChannelBasedOnBratsEt3d:
    transform = ConvertToMultiChannelBasedOnBratsEt3d.__new__(
        ConvertToMultiChannelBasedOnBratsEt3d
    )
    transform.keys = ["label"]
    transform._legacy_label_warned = False
    return transform


def test_brats_et3_channel_mapping():
    label = np.array([[0, 1, 2, 3, 4]], dtype=np.float32)
    out = _transform()({"label": label})["label"]

    assert out.shape == (3, 1, 5)
    # TC = 1, 3, or 4; WT = 1, 2, 3, or 4; ET = 3 or 4
    assert out[0].ravel().tolist() == [False, True, False, True, True]
    assert out[1].ravel().tolist() == [False, True, True, True, True]
    assert out[2].ravel().tolist() == [False, False, False, True, True]


def test_brats_treats_legacy_label_four_as_et():
    label = np.array([[4]], dtype=np.float32)
    out = _transform()({"label": label})["label"]

    assert out[0].ravel().tolist() == [True]
    assert out[1].ravel().tolist() == [True]
    assert out[2].ravel().tolist() == [True]


def test_brats_et3_squeezes_singleton_channel():
    label = np.zeros((1, 2, 2, 2), dtype=np.float32)
    label[0, 0, 0, 0] = 3
    out = _transform()({"label": label})["label"]

    assert out.shape == (3, 2, 2, 2)
    assert out[0, 0, 0, 0] == 1
    assert out[2, 0, 0, 0] == 1


def test_brats_warns_once_when_legacy_label_four_present(caplog):
    transform = _transform()
    label = np.array([[4]], dtype=np.float32)

    with caplog.at_level(logging.WARNING):
        transform({"label": label})
        transform({"label": label})

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "BraTS 2023 convention" in warnings[0].getMessage()


def test_brats_does_not_warn_for_label_three(caplog):
    with caplog.at_level(logging.WARNING):
        _transform()({"label": np.array([[3]], dtype=np.float32)})

    assert not caplog.records
