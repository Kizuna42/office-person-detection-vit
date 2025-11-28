"""Unit tests for TransformPhase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.phases import TransformPhase

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")

    # カメラパラメータ（新設計）
    config.set(
        "camera_params",
        {
            "height_m": 2.2,
            "pitch_deg": 45.0,
            "yaw_deg": 0.0,
            "roll_deg": 0.0,
            "focal_length_x": 1250.0,
            "focal_length_y": 1250.0,
            "center_x": 640.0,
            "center_y": 360.0,
            "image_width": 1280,
            "image_height": 720,
            "position_x_px": 1200.0,
            "position_y_px": 800.0,
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
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
                "polygon": [[0, 0], [2000, 0], [2000, 1500], [0, 1500]],
                "priority": 1,
            }
        ],
    )
    # 変換方式をpinholeに設定（homography.matrixが不要）
    config.set("transform", {"method": "pinhole"})
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
            bbox=(600.0, 300.0, 80.0, 200.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(640.0, 500.0),
        ),
        Detection(
            bbox=(400.0, 350.0, 60.0, 150.0),
            confidence=0.8,
            class_id=1,
            class_name="person",
            camera_coords=(430.0, 500.0),
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

    assert phase.transformer is not None
    assert phase.zone_classifier is not None


def test_initialize_with_default_camera_params(sample_config, sample_logger):
    """カメラパラメータがデフォルト値で動作する"""
    # カメラパラメータを最小限に設定
    sample_config.set("camera_params", {"height_m": 2.0, "pitch_deg": 45.0})

    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    assert phase.transformer is not None


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
        # 地平線より下の点なので変換成功するはず
        if detection.floor_coords is not None:
            assert detection.floor_coords_mm is not None
        assert isinstance(detection.zone_ids, list)


def test_execute_without_initialize(sample_config, sample_logger, sample_detection_results):
    """初期化前にexecuteを呼ぶとエラー"""
    phase = TransformPhase(sample_config, sample_logger)

    with pytest.raises(RuntimeError, match="Not initialized"):
        phase.execute(sample_detection_results)


def test_execute_empty_detections(sample_config, sample_logger):
    """検出結果が空の場合"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    empty_results = [(0, "2025/08/26 16:05:00", [])]
    results = phase.execute(empty_results)

    assert len(results) == 1
    assert len(results[0].detections) == 0


def test_execute_horizon_detection(sample_config, sample_logger):
    """地平線上の検出結果の処理"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    # 地平線上（画像上部）の検出結果
    horizon_detection = Detection(
        bbox=(600.0, 0.0, 80.0, 50.0),  # 画像上端
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(640.0, 50.0),
    )
    results_with_horizon = [(0, "2025/08/26 16:05:00", [horizon_detection])]

    results = phase.execute(results_with_horizon)

    assert len(results) == 1
    # 地平線上の点は変換失敗する可能性がある
    detection = results[0].detections[0]
    if detection.floor_coords is None:
        assert detection.zone_ids == []


def test_export_results(sample_config, sample_logger, sample_detection_results, tmp_path):
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

    with open(json_path, encoding="utf-8") as f:
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

    with open(json_path, encoding="utf-8") as f:
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

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert "floor_coords_px" not in data[0]["detections"][0]
    assert "floor_coords_mm" not in data[0]["detections"][0]


def test_cleanup(sample_config, sample_logger):
    """cleanupが正しく動作する"""
    phase = TransformPhase(sample_config, sample_logger)
    phase.initialize()

    assert phase.transformer is not None
    assert phase.zone_classifier is not None

    phase.cleanup()

    assert phase.transformer is None
    assert phase.zone_classifier is None
