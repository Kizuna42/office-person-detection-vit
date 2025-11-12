"""Unit tests for preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.preprocessing import (
    apply_blur,
    apply_clahe,
    apply_deskew,
    apply_invert,
    apply_morphology,
    apply_pipeline,
    apply_resize,
    apply_threshold,
    apply_unsharp_mask,
)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """テスト用のグレースケール画像"""

    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """テスト用のBGR画像"""

    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_apply_invert_enabled(sample_grayscale_image: np.ndarray):
    """画像を反転できる。"""

    result = apply_invert(sample_grayscale_image, enabled=True)

    assert result.shape == sample_grayscale_image.shape
    assert result.dtype == sample_grayscale_image.dtype
    # 反転チェック: 元の画像と結果の合計が255*sizeになる
    assert np.allclose(result + sample_grayscale_image, 255)


def test_apply_invert_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_invert(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_clahe_enabled(sample_grayscale_image: np.ndarray):
    """CLAHEを適用できる。"""

    result = apply_clahe(sample_grayscale_image, enabled=True)

    assert result.shape == sample_grayscale_image.shape
    assert result.dtype == sample_grayscale_image.dtype


def test_apply_clahe_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_clahe(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_clahe_with_params(sample_grayscale_image: np.ndarray):
    """パラメータを指定してCLAHEを適用できる。"""

    result = apply_clahe(
        sample_grayscale_image,
        clip_limit=3.0,
        tile_grid_size=(16, 16),
        enabled=True,
    )

    assert result.shape == sample_grayscale_image.shape


def test_apply_resize_enabled(sample_grayscale_image: np.ndarray):
    """画像をリサイズできる。"""

    result = apply_resize(sample_grayscale_image, fx=2.0, enabled=True)

    assert result.shape == (200, 200)


def test_apply_resize_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_resize(sample_grayscale_image, fx=2.0, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_resize_fx_fy(sample_grayscale_image: np.ndarray):
    """fx と fy を個別に指定できる。"""

    result = apply_resize(sample_grayscale_image, fx=2.0, fy=3.0, enabled=True)

    assert result.shape == (300, 200)


def test_apply_resize_no_change(sample_grayscale_image: np.ndarray):
    """fx=fy=1.0 の場合は元の画像が返される。"""

    result = apply_resize(sample_grayscale_image, fx=1.0, fy=1.0, enabled=True)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_threshold_otsu(sample_grayscale_image: np.ndarray):
    """Otsu二値化を適用できる。"""

    result = apply_threshold(sample_grayscale_image, method="otsu", enabled=True)

    assert result.shape == sample_grayscale_image.shape
    assert np.all((result == 0) | (result == 255))


def test_apply_threshold_adaptive(sample_grayscale_image: np.ndarray):
    """Adaptive二値化を適用できる。"""

    result = apply_threshold(
        sample_grayscale_image,
        method="adaptive",
        block_size=11,
        C=2,
        enabled=True,
    )

    assert result.shape == sample_grayscale_image.shape
    assert np.all((result == 0) | (result == 255))


def test_apply_threshold_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_threshold(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_threshold_invalid_method(sample_grayscale_image: np.ndarray):
    """不正なmethodの場合はOtsuが使用される。"""

    result = apply_threshold(sample_grayscale_image, method="invalid", enabled=True)

    assert result.shape == sample_grayscale_image.shape


def test_apply_blur_enabled(sample_grayscale_image: np.ndarray):
    """ブラーを適用できる。"""

    result = apply_blur(sample_grayscale_image, kernel_size=5, enabled=True)

    assert result.shape == sample_grayscale_image.shape


def test_apply_blur_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_blur(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_blur_even_kernel_size(sample_grayscale_image: np.ndarray):
    """偶数のカーネルサイズは奇数に調整される。"""

    result = apply_blur(sample_grayscale_image, kernel_size=4, enabled=True)

    assert result.shape == sample_grayscale_image.shape


def test_apply_unsharp_mask_enabled(sample_grayscale_image: np.ndarray):
    """アンシャープマスクを適用できる。"""

    result = apply_unsharp_mask(
        sample_grayscale_image,
        amount=1.5,
        radius=1.0,
        threshold=0,
        enabled=True,
    )

    assert result.shape == sample_grayscale_image.shape
    assert result.dtype == sample_grayscale_image.dtype


def test_apply_unsharp_mask_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_unsharp_mask(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_morphology_open(sample_grayscale_image: np.ndarray):
    """Opening処理を適用できる。"""

    # 二値化された画像が必要
    binary = apply_threshold(sample_grayscale_image, enabled=True)

    result = apply_morphology(
        binary,
        operation="open",
        kernel_size=3,
        iterations=1,
        enabled=True,
    )

    assert result.shape == binary.shape


def test_apply_morphology_close(sample_grayscale_image: np.ndarray):
    """Closing処理を適用できる。"""

    binary = apply_threshold(sample_grayscale_image, enabled=True)

    result = apply_morphology(
        binary,
        operation="close",
        kernel_size=3,
        iterations=1,
        enabled=True,
    )

    assert result.shape == binary.shape


def test_apply_morphology_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像が返される。"""

    result = apply_morphology(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)


def test_apply_morphology_invalid_operation(sample_grayscale_image: np.ndarray):
    """不正なoperationの場合はcloseが使用される。"""

    binary = apply_threshold(sample_grayscale_image, enabled=True)

    result = apply_morphology(binary, operation="invalid", enabled=True)

    assert result.shape == binary.shape


def test_apply_deskew_enabled(sample_grayscale_image: np.ndarray):
    """傾き補正を適用できる。"""

    binary = apply_threshold(sample_grayscale_image, enabled=True)

    result, angle = apply_deskew(binary, max_angle=5.0, enabled=True)

    assert result.shape == binary.shape
    assert isinstance(angle, float)


def test_apply_deskew_disabled(sample_grayscale_image: np.ndarray):
    """enabled=False の場合は元の画像と角度0が返される。"""

    result, angle = apply_deskew(sample_grayscale_image, enabled=False)

    assert np.array_equal(result, sample_grayscale_image)
    assert angle == 0.0


def test_apply_pipeline_bgr_to_grayscale(sample_bgr_image: np.ndarray):
    """BGR画像が自動的にグレースケールに変換される。"""

    params = {
        "threshold": {"enabled": True, "method": "otsu"},
    }

    result = apply_pipeline(sample_bgr_image, params)

    assert len(result.shape) == 2  # グレースケール
    assert result.shape[:2] == sample_bgr_image.shape[:2]


def test_apply_pipeline_full(sample_bgr_image: np.ndarray):
    """全前処理パイプラインを適用できる。"""

    params = {
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        "resize": {"enabled": True, "fx": 2.0},
        "threshold": {"enabled": True, "method": "otsu"},
        "invert_after_threshold": {"enabled": True},
        "morphology": {
            "enabled": True,
            "operation": "close",
            "kernel_size": 2,
            "iterations": 1,
        },
    }

    result = apply_pipeline(sample_bgr_image, params)

    assert len(result.shape) == 2  # グレースケール


def test_apply_pipeline_partial(sample_bgr_image: np.ndarray):
    """一部の前処理のみを適用できる。"""

    params = {
        "resize": {"enabled": True, "fx": 1.5},
        "threshold": {"enabled": False},
    }

    result = apply_pipeline(sample_bgr_image, params)

    assert result.shape[:2] == (
        int(sample_bgr_image.shape[0] * 1.5),
        int(sample_bgr_image.shape[1] * 1.5),
    )


def test_apply_pipeline_deskew(sample_bgr_image: np.ndarray):
    """Deskewを含むパイプラインを適用できる。"""

    params = {
        "threshold": {"enabled": True, "method": "otsu"},
        "deskew": {"enabled": True, "max_angle": 5.0},
    }

    result = apply_pipeline(sample_bgr_image, params)

    assert len(result.shape) == 2


def test_apply_pipeline_empty_params(sample_bgr_image: np.ndarray):
    """パラメータが空の場合はグレースケールのみ。"""

    result = apply_pipeline(sample_bgr_image, {})

    assert len(result.shape) == 2
    assert result.shape[:2] == sample_bgr_image.shape[:2]


def test_apply_pipeline_invert_order(sample_bgr_image: np.ndarray):
    """invert と invert_after_threshold の順序が正しい。"""

    params = {
        "invert": {"enabled": True},
        "threshold": {"enabled": True, "method": "otsu"},
        "invert_after_threshold": {"enabled": True},
    }

    result = apply_pipeline(sample_bgr_image, params)

    assert len(result.shape) == 2


def test_apply_threshold_auto_switch(sample_grayscale_image: np.ndarray):
    """自動切替機能のテスト"""
    import cv2  # noqa: F401

    # 低コントラスト画像を作成（標準偏差が小さい）
    low_contrast_image = np.full((100, 100), 128, dtype=np.uint8).astype(np.int16)
    low_contrast_image += np.random.randint(-10, 10, (100, 100), dtype=np.int16)
    low_contrast_image = np.clip(low_contrast_image, 0, 255).astype(np.uint8)

    # auto_switchが有効な場合、低コントラストではadaptive thresholdに切替
    result = apply_threshold(low_contrast_image, method="otsu", auto_switch=True)

    assert result.shape == low_contrast_image.shape
    assert result.dtype == np.uint8

    # 高コントラスト画像（標準偏差が大きい）ではOtsuを使用
    high_contrast_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result2 = apply_threshold(high_contrast_image, method="otsu", auto_switch=True)

    assert result2.shape == high_contrast_image.shape


def test_apply_deskew_with_lines(sample_grayscale_image: np.ndarray):
    """線検出がある場合のdeskew"""
    import cv2  # noqa: F401

    # 水平線を含む画像を作成
    image_with_lines = np.zeros((100, 100), dtype=np.uint8)
    # 水平線を描画
    image_with_lines[50, :] = 255

    result, angle = apply_deskew(image_with_lines, enabled=True)

    assert result.shape == image_with_lines.shape
    assert isinstance(angle, float)


def test_apply_deskew_no_lines(sample_grayscale_image: np.ndarray):
    """線検出がない場合のdeskew"""
    # 線が検出されない画像（均一な画像）
    uniform_image = np.full((100, 100), 128, dtype=np.uint8)

    result, angle = apply_deskew(uniform_image, enabled=True)

    assert result.shape == uniform_image.shape
    assert isinstance(angle, float)
    # 線が検出されない場合、角度は0に近い
    assert abs(angle) < 1.0


def test_apply_pipeline_threshold_auto_switch(sample_bgr_image: np.ndarray):
    """パイプライン内での自動切替"""
    import cv2

    # 低コントラスト画像を作成
    low_contrast_gray = np.full((100, 100), 128, dtype=np.uint8).astype(np.int16)
    low_contrast_gray += np.random.randint(-10, 10, (100, 100), dtype=np.int16)
    low_contrast_gray = np.clip(low_contrast_gray, 0, 255).astype(np.uint8)
    # BGR画像に変換
    low_contrast_bgr = cv2.cvtColor(low_contrast_gray, cv2.COLOR_GRAY2BGR)

    params = {
        "threshold": {"enabled": True, "method": "otsu", "auto_switch": True},
    }

    result = apply_pipeline(low_contrast_bgr, params)

    assert len(result.shape) == 2
    assert result.dtype == np.uint8


def test_apply_pipeline_deskew_with_angle(sample_bgr_image: np.ndarray):
    """傾きがある場合のdeskew"""
    import cv2

    # 傾きのある画像を作成（回転変換を使用）
    center = (50, 50)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)  # 5度回転
    rotated_image = cv2.warpAffine(sample_bgr_image, rotation_matrix, (100, 100), borderValue=(255, 255, 255))

    params = {
        "threshold": {"enabled": True, "method": "otsu"},
        "deskew": {"enabled": True, "max_angle": 10.0},
    }

    result = apply_pipeline(rotated_image, params)

    assert len(result.shape) == 2
    assert result.dtype == np.uint8
