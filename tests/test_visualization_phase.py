"""Unit tests for VisualizationPhase."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.aggregation import Aggregator
from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.visualization_phase import VisualizationPhase


@pytest.fixture()
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")
    config.set("output.debug_mode", False)
    config.set("output.save_floormap_images", True)
    config.set("floormap.image_path", str(tmp_path / "floormap.png"))
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
    config.set("camera", {})
    return config


@pytest.fixture()
def sample_logger():
    """テスト用のロガー"""
    logger = logging.getLogger("test_visualization_phase")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture()
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
            floor_coords=(50.0, 50.0),
            zone_ids=["zone_a"],
        ),
    ]


@pytest.fixture()
def sample_frame_results(sample_detections) -> list[FrameResult]:
    """テスト用のFrameResultリスト"""
    return [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=sample_detections,
            zone_counts={"zone_a": 1},
        ),
    ]


@pytest.fixture()
def sample_aggregator() -> Aggregator:
    """テスト用のAggregator"""
    aggregator = Aggregator()
    aggregator.aggregate_frame(
        "2025/08/26 16:05:00",
        [
            Detection(
                bbox=(100.0, 200.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125.0, 300.0),
                floor_coords=(50.0, 50.0),
                zone_ids=["zone_a"],
            ),
        ],
    )
    return aggregator


@patch("src.pipeline.visualization_phase.Visualizer")
def test_execute_success(
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    sample_frame_results,
    tmp_path,
):
    """executeが正しく動作する"""
    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = True
    mock_visualizer.plot_zone_statistics.return_value = True
    mock_visualizer_class.return_value = mock_visualizer

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    phase.execute(sample_aggregator, sample_frame_results, output_path)

    mock_visualizer.plot_time_series.assert_called_once()
    mock_visualizer.plot_zone_statistics.assert_called_once()


@patch("src.pipeline.visualization_phase.Visualizer")
@patch("src.pipeline.visualization_phase.FloormapVisualizer")
def test_execute_with_floormap(
    mock_floormap_visualizer_class,
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    sample_frame_results,
    tmp_path,
):
    """フロアマップ可視化が有効な場合"""
    import cv2
    import numpy as np

    # フロアマップ画像を作成
    floormap_path = tmp_path / "floormap.png"
    floormap_image = np.ones((1369, 1878, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(floormap_path), floormap_image)

    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = True
    mock_visualizer.plot_zone_statistics.return_value = True
    mock_visualizer_class.return_value = mock_visualizer

    mock_floormap_visualizer = MagicMock()
    mock_floormap_visualizer.visualize_frame.return_value = floormap_image
    mock_floormap_visualizer_class.return_value = mock_floormap_visualizer

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)
    (output_path / "floormaps").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    phase.execute(sample_aggregator, sample_frame_results, output_path)

    mock_floormap_visualizer_class.assert_called_once()
    assert mock_floormap_visualizer.visualize_frame.call_count == len(sample_frame_results)
    assert mock_floormap_visualizer.save_visualization.call_count == len(sample_frame_results)


@patch("src.pipeline.visualization_phase.Visualizer")
def test_execute_without_floormap(
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    sample_frame_results,
    tmp_path,
):
    """フロアマップ可視化が無効な場合"""
    sample_config.set("output.save_floormap_images", False)

    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = True
    mock_visualizer.plot_zone_statistics.return_value = True
    mock_visualizer_class.return_value = mock_visualizer

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    phase.execute(sample_aggregator, sample_frame_results, output_path)

    mock_visualizer.plot_time_series.assert_called_once()
    mock_visualizer.plot_zone_statistics.assert_called_once()


@patch("src.pipeline.visualization_phase.Visualizer")
@patch("src.pipeline.visualization_phase.FloormapVisualizer")
def test_execute_floormap_file_not_found(
    mock_floormap_visualizer_class,
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    sample_frame_results,
    tmp_path,
):
    """フロアマップ画像が見つからない場合"""
    sample_config.set("floormap.image_path", str(tmp_path / "nonexistent.png"))

    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = True
    mock_visualizer.plot_zone_statistics.return_value = True
    mock_visualizer_class.return_value = mock_visualizer

    mock_floormap_visualizer_class.side_effect = FileNotFoundError("Floormap not found")

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    # エラーが発生しても処理は続行される
    phase.execute(sample_aggregator, sample_frame_results, output_path)

    mock_visualizer.plot_time_series.assert_called_once()


@patch("src.pipeline.visualization_phase.Visualizer")
def test_execute_empty_results(
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    tmp_path,
):
    """空の結果でexecuteを呼ぶ"""
    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = True
    mock_visualizer.plot_zone_statistics.return_value = True
    mock_visualizer_class.return_value = mock_visualizer

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    phase.execute(sample_aggregator, [], output_path)

    mock_visualizer.plot_time_series.assert_called_once()
    mock_visualizer.plot_zone_statistics.assert_called_once()


@patch("src.pipeline.visualization_phase.Visualizer")
def test_execute_graph_generation_failure(
    mock_visualizer_class,
    sample_config,
    sample_logger,
    sample_aggregator,
    sample_frame_results,
    tmp_path,
):
    """グラフ生成に失敗した場合"""
    mock_visualizer = MagicMock()
    mock_visualizer.plot_time_series.return_value = False
    mock_visualizer.plot_zone_statistics.return_value = False
    mock_visualizer_class.return_value = mock_visualizer

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    phase = VisualizationPhase(sample_config, sample_logger)
    # エラーが発生しても処理は続行される
    phase.execute(sample_aggregator, sample_frame_results, output_path)

    mock_visualizer.plot_time_series.assert_called_once()
    mock_visualizer.plot_zone_statistics.assert_called_once()
