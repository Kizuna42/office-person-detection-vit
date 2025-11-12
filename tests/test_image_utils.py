"""Test cases for image_utils module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from src.models import Detection
from src.utils.image_utils import save_detection_image

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_frame() -> np.ndarray:
    """テスト用のフレーム画像"""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
        ),
        Detection(
            bbox=(300.0, 400.0, 60.0, 120.0),
            confidence=0.92,
            class_id=1,
            class_name="person",
            camera_coords=(330.0, 520.0),
        ),
    ]


@pytest.fixture
def sample_logger() -> logging.Logger:
    """テスト用のロガー"""
    logger = logging.getLogger("test_image_utils")
    logger.setLevel(logging.DEBUG)
    return logger


def test_save_detection_image_success(
    sample_frame: np.ndarray,
    sample_detections: list[Detection],
    sample_logger: logging.Logger,
    tmp_path: Path,
):
    """検出画像の保存が成功する"""
    output_dir = tmp_path / "detections"
    timestamp = "2025/08/26 16:04:56"

    save_detection_image(sample_frame, sample_detections, timestamp, output_dir, sample_logger)

    # ファイルが作成されていることを確認
    expected_filename = "detection_2025_08_26_160456.jpg"
    expected_path = output_dir / expected_filename
    assert expected_path.exists(), f"ファイルが作成されていません: {expected_path}"

    # 画像が正しく読み込めることを確認
    saved_image = cv2.imread(str(expected_path))
    assert saved_image is not None, "保存された画像が読み込めません"
    assert saved_image.shape == sample_frame.shape, "画像サイズが一致しません"


def test_save_detection_image_timestamp_sanitization(
    sample_frame: np.ndarray,
    sample_detections: list[Detection],
    sample_logger: logging.Logger,
    tmp_path: Path,
):
    """タイムスタンプの特殊文字が正しく置換される"""
    # 様々な特殊文字を含むタイムスタンプをテスト
    test_cases = [
        ("2025/08/26 16:04:56", "detection_2025_08_26_160456.jpg"),
        ("2025/12/31 23:59:59", "detection_2025_12_31_235959.jpg"),
        ("2025/01/01 00:00:00", "detection_2025_01_01_000000.jpg"),
    ]

    for timestamp, expected_filename in test_cases:
        output_dir_test = tmp_path / f"test_{timestamp.replace('/', '_')}"
        save_detection_image(sample_frame, sample_detections, timestamp, output_dir_test, sample_logger)

        expected_path = output_dir_test / expected_filename
        assert expected_path.exists(), f"ファイルが作成されていません: {expected_path} (タイムスタンプ: {timestamp})"


def test_save_detection_image_empty_detections(
    sample_frame: np.ndarray,
    sample_logger: logging.Logger,
    tmp_path: Path,
):
    """検出結果が空の場合でも画像が保存される"""
    output_dir = tmp_path / "detections"
    timestamp = "2025/08/26 16:04:56"

    save_detection_image(sample_frame, [], timestamp, output_dir, sample_logger)

    expected_filename = "detection_2025_08_26_160456.jpg"
    expected_path = output_dir / expected_filename
    assert expected_path.exists(), "検出結果が空でもファイルが作成されるべきです"


def test_save_detection_image_creates_directory(
    sample_frame: np.ndarray,
    sample_detections: list[Detection],
    sample_logger: logging.Logger,
    tmp_path: Path,
):
    """存在しないディレクトリが自動作成される"""
    output_dir = tmp_path / "new" / "detections"
    timestamp = "2025/08/26 16:04:56"

    assert not output_dir.exists(), "ディレクトリはまだ存在しないはずです"

    save_detection_image(sample_frame, sample_detections, timestamp, output_dir, sample_logger)

    assert output_dir.exists(), "ディレクトリが自動作成されるべきです"


def test_save_detection_image_invalid_timestamp_characters(
    sample_frame: np.ndarray,
    sample_detections: list[Detection],
    sample_logger: logging.Logger,
    tmp_path: Path,
):
    """無効な文字を含むタイムスタンプでも正しく処理される"""
    output_dir = tmp_path / "detections"
    # 特殊文字を含むタイムスタンプ（実際には発生しないが、防御的プログラミング）
    timestamp = "2025/08/26 16:04:56<>|?*"

    save_detection_image(sample_frame, sample_detections, timestamp, output_dir, sample_logger)

    # ファイル名に無効な文字が含まれていないことを確認
    files = list(output_dir.glob("detection_*.jpg"))
    assert len(files) == 1, "1つのファイルが作成されるべきです"

    filename = files[0].name
    # 無効な文字が含まれていないことを確認
    invalid_chars = ["/", ":", "<", ">", "|", "?", "*"]
    for char in invalid_chars:
        assert char not in filename, f"ファイル名に無効な文字 '{char}' が含まれています: {filename}"
