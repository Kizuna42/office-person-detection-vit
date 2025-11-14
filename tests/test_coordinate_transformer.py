"""Unit tests for CoordinateTransformer."""

from __future__ import annotations

import numpy as np
import pytest

from src.transform import CoordinateTransformer


def test_validate_matrix_invalid_shape():
    """3x3以外の行列を渡すと ValueError を送出する。"""

    with pytest.raises(ValueError, match=r".*3x3.*"):
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


def test_validate_matrix_invalid_type():
    """無効な型の行列を渡すと ValueError を送出する。"""
    with pytest.raises(ValueError, match="ホモグラフィ行列はリストまたはnumpy配列である必要があります"):
        CoordinateTransformer("invalid")


def test_validate_matrix_singular():
    """特異行列（行列式が0に近い）の場合のエラーテスト"""
    # 特異行列（行列式が0）の場合、ValueErrorが発生する
    singular_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="ホモグラフィ行列が特異行列です"):
        CoordinateTransformer(singular_matrix)


def test_transform_edge_cases():
    """座標変換のエッジケーステスト"""
    transformer = CoordinateTransformer(np.eye(3))

    # ゼロ座標
    x, y = transformer.transform((0.0, 0.0))
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)

    # 負の座標
    x, y = transformer.transform((-10.0, -20.0))
    assert x == pytest.approx(-10.0)
    assert y == pytest.approx(-20.0)

    # 非常に大きい座標
    x, y = transformer.transform((10000.0, 20000.0))
    assert x == pytest.approx(10000.0)
    assert y == pytest.approx(20000.0)


def test_transform_batch_empty():
    """空のバッチ変換テスト"""
    transformer = CoordinateTransformer(np.eye(3))
    results = transformer.transform_batch([])
    assert len(results) == 0


def test_transform_batch_single_point():
    """単一点のバッチ変換テスト"""
    transformer = CoordinateTransformer(np.eye(3))
    points = [(100.0, 200.0)]
    results = transformer.transform_batch(points)
    assert len(results) == 1
    assert results[0] == pytest.approx((100.0, 200.0))


def test_transform_detection_edge_cases():
    """検出結果変換のエッジケーステスト"""
    transformer = CoordinateTransformer(np.eye(3))

    # ゼロサイズのバウンディングボックス
    result = transformer.transform_detection((100.0, 200.0, 0.0, 0.0))
    assert result == pytest.approx((100.0, 200.0))

    # 負のサイズ（通常は発生しないが、テスト）
    result = transformer.transform_detection((100.0, 200.0, -10.0, -20.0))
    assert result == pytest.approx((95.0, 180.0))


def test_pixel_to_mm_edge_cases():
    """ピクセル→mm変換のエッジケーステスト"""
    config = {
        "image_x_mm_per_pixel": 10.0,
        "image_y_mm_per_pixel": 20.0,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    # ゼロ座標
    mm = transformer.pixel_to_mm((0.0, 0.0))
    assert mm == pytest.approx((0.0, 0.0))

    # 負の座標
    mm = transformer.pixel_to_mm((-5.0, -10.0))
    assert mm == pytest.approx((-50.0, -200.0))


def test_mm_to_pixel_edge_cases():
    """mm→ピクセル変換のエッジケーステスト"""
    config = {
        "image_x_mm_per_pixel": 10.0,
        "image_y_mm_per_pixel": 20.0,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    # ゼロ座標
    pixels = transformer.mm_to_pixel((0.0, 0.0))
    assert pixels == pytest.approx((0.0, 0.0))

    # 負の座標
    pixels = transformer.mm_to_pixel((-50.0, -200.0))
    assert pixels == pytest.approx((-5.0, -10.0))


def test_is_within_bounds_edge_cases():
    """境界判定のエッジケーステスト"""
    config = {
        "image_width": 100,
        "image_height": 50,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    # 境界上の点
    assert transformer.is_within_bounds((0.0, 0.0))
    assert transformer.is_within_bounds((99.0, 49.0))  # 境界内（width-1, height-1）

    # 境界外の点
    assert not transformer.is_within_bounds((100.0, 50.0))  # 境界上（width, height）
    assert not transformer.is_within_bounds((-1.0, 0.0))
    assert not transformer.is_within_bounds((0.0, -1.0))

    # 非常に大きい座標
    assert not transformer.is_within_bounds((10000.0, 10000.0))


def test_is_within_bounds_no_config():
    """設定がない場合の境界判定テスト"""
    transformer = CoordinateTransformer(np.eye(3))
    # 設定がない場合は常にTrueを返す
    assert transformer.is_within_bounds((100.0, 200.0))
    assert transformer.is_within_bounds((-100.0, -200.0))


def test_transform_with_scale():
    """スケール変換を含む変換テスト"""
    config = {
        "image_x_mm_per_pixel": 2.0,
        "image_y_mm_per_pixel": 3.0,
    }
    transformer = CoordinateTransformer(np.eye(3), floormap_config=config)

    # スケール変換はpixel_to_mmで行われる
    mm = transformer.pixel_to_mm((10.0, 20.0))
    assert mm == pytest.approx((20.0, 60.0))


def test_transform_with_distortion_correction():
    """歪み補正を含む変換テスト（設定のみ、実際の補正は実装されていない可能性がある）"""
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)

    transformer = CoordinateTransformer(
        np.eye(3), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, use_distortion_correction=True
    )
    assert transformer.use_distortion_correction is True


def test_transform_with_distortion_correction_no_params():
    """歪み補正が有効だがパラメータがない場合のテスト"""
    transformer = CoordinateTransformer(np.eye(3), use_distortion_correction=True)
    # パラメータがない場合は自動的に無効化される
    assert transformer.use_distortion_correction is False


def test_transform_batch_large():
    """大量の点のバッチ変換テスト"""
    transformer = CoordinateTransformer(np.eye(3))
    points = [(float(i), float(i * 2)) for i in range(1000)]
    results = transformer.transform_batch(points)
    assert len(results) == 1000
    for i, (x, y) in enumerate(results):
        assert x == pytest.approx(float(i))
        assert y == pytest.approx(float(i * 2))
