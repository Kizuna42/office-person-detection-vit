"""Unit tests for CoordinateTransformer."""

from __future__ import annotations

import numpy as np
import pytest

from src.transform import CoordinateTransformer


def test_validate_matrix_invalid_shape():
    """3x3以外の行列を渡すと ValueError を送出する。"""

    with pytest.raises(ValueError, match=".*3x3.*"):
        CoordinateTransformer([[1, 0], [0, 1]])


def test_transform_identity():
    """単位行列での変換では座標が変化しない。"""

    transformer = CoordinateTransformer(np.eye(3))
    x, y = transformer.transform((100.0, 200.0))
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(200.0)


def test_transform_with_origin_offset():
    """原点オフセットが正しく適用される。"""

    config = {
        "image_origin_x": 7.0,
        "image_origin_y": 9.0,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)
    result = transformer.transform((10.0, 20.0))
    assert result == pytest.approx((17.0, 29.0))


def test_transform_batch():
    """バッチ変換で複数座標を一括変換できる。"""

    transformer = CoordinateTransformer(np.eye(3))
    points = [(0.0, 0.0), (10.0, 20.0), (30.0, 40.0)]
    results = transformer.transform_batch(points)

    assert len(results) == 3
    for expected, actual in zip(points, results, strict=False):
        assert actual == pytest.approx(expected)


def test_transform_detection():
    """バウンディングボックスから足元座標を計算し変換する。"""

    transformer = CoordinateTransformer(np.eye(3))
    result = transformer.transform_detection((100.0, 200.0, 50.0, 100.0))
    assert result == pytest.approx((125.0, 300.0))


def test_pixel_to_mm_and_mm_to_pixel_round_trip():
    """ピクセルとミリメートル座標の相互変換。"""

    config = {
        "image_x_mm_per_pixel": 10.0,
        "image_y_mm_per_pixel": 20.0,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    mm = transformer.pixel_to_mm((3.0, 4.0))
    assert mm == pytest.approx((30.0, 80.0))

    pixels = transformer.mm_to_pixel(mm)
    assert pixels == pytest.approx((3.0, 4.0))


def test_is_within_bounds():
    """フロアマップの幅・高さによる境界判定を行う。"""

    config = {
        "image_width": 100,
        "image_height": 50,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    assert transformer.is_within_bounds((10.0, 10.0))
    assert not transformer.is_within_bounds((150.0, 10.0))
    assert not transformer.is_within_bounds((10.0, -1.0))
