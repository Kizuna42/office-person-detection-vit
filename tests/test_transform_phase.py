"""Unit tests for TransformPhase."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.transform_phase import TransformPhase


@pytest.fixture
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")
    config.set("homography.matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    config.set(
        "floormap",
        {
            "image_width": 1878,
            "image_height": 1369,
            "image_origin_x": 7,
            "image_origin_y": 9,
            "image_x_mm_per_pixel": 28.1926406926406,
            "image_y_mm_per_pixel": 28.241430700447,
        },
    )
    config.set(
        "zones",
        [
            {
                "id": "zone_a",
                "name": "Zone A",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "priority": 1,
            }
        ],
    )
    return config


@pytest.fixture
def sample_logger():
    """テスト用のロガー"""
    logger = logging.getLogger("test_transform_phase")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
        ),
        Detection(
            bbox=(200.0, 300.0, 60.0, 120.0),
            confidence=0.8,
            class_id=1,
            class_name="person",
            camera_coords=(230.0, 420.0),
        ),
    ]


@pytest.fixture
def sample_detection_results(
    sample_detections,
) -> list[tuple[int, str, list[Detection]]]:
    """テスト用の検出結果リスト"""
    return [
        (0, "2025/08/26 16:05:00", sample_detections),
        (1, "2025/08/26 16:10:00", sample_detections[:1]),
    ]


def test_initialize(sample_config, sample_logger):
    """初期化が正しく動作する"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    assert phase.coordinate_transformer is not None
    assert phase.zone_classifier is not None


def test_initialize_missing_homography(sample_config, sample_logger):
    """ホモグラフィ行列が設定されていない場合"""
    sample_config.set("homography.matrix", None)

    phase = TransformPhase(sample_config, sample_logger)

    with pytest.raises(ValueError, match="ホモグラフィ行列が設定されていません"):
        phase.initialize()


def test_initialize_empty_zones(sample_config, sample_logger):
    """ゾーン定義が空の場合"""
    sample_config.set("zones", [])

    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    assert phase.zone_classifier is not None


def test_execute_success(sample_config, sample_logger, sample_detection_results):
    """executeが正しく動作する"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_detection_results)

    assert len(results) == 2
    assert isinstance(results[0], FrameResult)
    assert results[0].frame_number == 0
    assert results[0].timestamp == "2025/08/26 16:05:00"
    assert len(results[0].detections) == 2

    # 座標変換が適用されていることを確認
    for detection in results[0].detections:
        assert detection.floor_coords is not None
        assert detection.floor_coords_mm is not None
        assert isinstance(detection.zone_ids, list)


def test_execute_without_initialize(
    sample_config, sample_logger, sample_detection_results
):
    """初期化前にexecuteを呼ぶとエラー"""
    phase = TransformPhase(sample_config, sample_logger)

    with pytest.raises(RuntimeError, match="変換器または分類器が初期化されていません"):
        phase.execute(sample_detection_results)


def test_execute_empty_detections(sample_config, sample_logger):
    """検出結果が空の場合"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    empty_results = [(0, "2025/08/26 16:05:00", [])]
    results = phase.execute(empty_results)

    assert len(results) == 1
    assert len(results[0].detections) == 0


def test_execute_coordinate_transform_error(
    sample_config, sample_logger, sample_detection_results
):
    """座標変換でエラーが発生した場合"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    # 無効な座標を持つ検出結果を作成
    invalid_detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(None, None),  # 無効な座標
    )
    invalid_results = [(0, "2025/08/26 16:05:00", [invalid_detection])]

    # エラーが発生しても処理は続行される
    results = phase.execute(invalid_results)

    assert len(results) == 1
    # エラーが発生した検出は座標がNoneになる
    assert results[0].detections[0].floor_coords is None
    assert results[0].detections[0].zone_ids == []


def test_export_results(
    sample_config, sample_logger, sample_detection_results, tmp_path
):
    """結果のエクスポートが正しく動作する"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_detection_results)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    phase.export_results(results, output_path)

    json_path = output_path / "coordinate_transformations.json"
    assert json_path.exists()

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["frame_number"] == 0
    assert "detections" in data[0]
    assert len(data[0]["detections"]) == 2


def test_export_results_empty(sample_config, sample_logger, tmp_path):
    """空の結果をエクスポートする場合"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    empty_results = []
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    phase.export_results(empty_results, output_path)

    json_path = output_path / "coordinate_transformations.json"
    assert json_path.exists()

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 0


def test_export_results_with_missing_coords(sample_config, sample_logger, tmp_path):
    """座標が欠損している検出結果をエクスポートする場合"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    # 座標がNoneの検出結果
    detection_without_coords = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
    )
    detection_without_coords.floor_coords = None
    detection_without_coords.floor_coords_mm = None

    results = [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=[detection_without_coords],
            zone_counts={},
        )
    ]

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    phase.export_results(results, output_path)

    json_path = output_path / "coordinate_transformations.json"
    assert json_path.exists()

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert "floor_coords" not in data[0]["detections"][0]
    assert "floor_coords_mm" not in data[0]["detections"][0]
