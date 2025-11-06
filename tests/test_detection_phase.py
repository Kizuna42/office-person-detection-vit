"""Unit tests for DetectionPhase."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.config import ConfigManager
from src.models import Detection
from src.pipeline.detection_phase import DetectionPhase


@pytest.fixture
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")
    config.set("detection.model_name", "facebook/detr-resnet-50")
    config.set("detection.confidence_threshold", 0.5)
    config.set("detection.device", "cpu")
    config.set("detection.batch_size", 2)
    config.set("output.save_detection_images", False)
    config.set("output.directory", str(tmp_path / "output"))
    return config


@pytest.fixture
def sample_logger():
    """テスト用のロガー"""
    logger = logging.getLogger("test_detection_phase")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_frames() -> list[tuple[int, str, np.ndarray]]:
    """テスト用のフレームリスト"""
    return [
        (0, "2025/08/26 16:05:00", np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)),
        (1, "2025/08/26 16:10:00", np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)),
        (2, "2025/08/26 16:15:00", np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)),
    ]


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


@patch("src.pipeline.detection_phase.ViTDetector")
def test_initialize(mock_detector_class, sample_config, sample_logger):
    """初期化が正しく動作する"""
    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    mock_detector_class.assert_called_once_with(
        "facebook/detr-resnet-50", 0.5, "cpu"
    )
    mock_detector.load_model.assert_called_once()
    assert phase.detector is mock_detector


@patch("src.pipeline.detection_phase.ViTDetector")
def test_execute_success(mock_detector_class, sample_config, sample_logger, sample_frames, sample_detections):
    """executeが正しく動作する"""
    mock_detector = MagicMock()
    # バッチサイズ2なので、最初の2フレームが1バッチ、最後の1フレームが1バッチ
    mock_detector.detect_batch.side_effect = [
        [sample_detections, sample_detections[:1]],  # 最初のバッチ（2フレーム）
        [sample_detections],  # 2番目のバッチ（1フレーム）
    ]
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_frames)

    assert len(results) == 3
    assert results[0][0] == 0  # frame_num
    assert results[0][1] == "2025/08/26 16:05:00"  # timestamp
    assert len(results[0][2]) == 2  # detections
    assert results[1][0] == 1
    assert len(results[1][2]) == 1
    assert results[2][0] == 2
    assert len(results[2][2]) == 2  # 最後のバッチはsample_detectionsが返される


@patch("src.pipeline.detection_phase.ViTDetector")
def test_execute_without_initialize(mock_detector_class, sample_config, sample_logger, sample_frames):
    """初期化前にexecuteを呼ぶとエラー"""
    phase = DetectionPhase(sample_config, sample_logger)

    with pytest.raises(RuntimeError, match="検出器が初期化されていません"):
        phase.execute(sample_frames)


@patch("src.pipeline.detection_phase.ViTDetector")
@patch("src.pipeline.detection_phase.save_detection_image")
def test_execute_with_image_saving(
    mock_save_image,
    mock_detector_class,
    sample_config,
    sample_logger,
    sample_frames,
    sample_detections,
    tmp_path,
):
    """検出画像の保存が有効な場合"""
    sample_config.set("output.save_detection_images", True)
    sample_config.set("output.directory", str(tmp_path / "output"))

    mock_detector = MagicMock()
    mock_detector.detect_batch.return_value = [sample_detections]
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_frames[:1])

    assert len(results) == 1
    # 検出がある場合は画像保存が呼ばれる
    mock_save_image.assert_called_once()


@patch("src.pipeline.detection_phase.ViTDetector")
def test_execute_batch_processing(
    mock_detector_class, sample_config, sample_logger, sample_frames, sample_detections
):
    """バッチ処理が正しく動作する"""
    sample_config.set("detection.batch_size", 2)

    mock_detector = MagicMock()
    # バッチサイズ2なので、2回呼ばれる（2フレーム + 1フレーム）
    mock_detector.detect_batch.side_effect = [
        [sample_detections, sample_detections],
        [sample_detections],
    ]
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_frames)

    assert mock_detector.detect_batch.call_count == 2
    assert len(results) == 3


@patch("src.pipeline.detection_phase.ViTDetector")
def test_execute_error_handling(
    mock_detector_class, sample_config, sample_logger, sample_frames
):
    """エラーハンドリングが正しく動作する"""
    mock_detector = MagicMock()
    mock_detector.detect_batch.side_effect = Exception("Detection error")
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute(sample_frames)

    # エラーが発生しても空の結果が返される
    assert len(results) == 3
    assert all(len(detections) == 0 for _, _, detections in results)


@patch("src.pipeline.detection_phase.ViTDetector")
@patch("src.pipeline.detection_phase.calculate_detection_statistics")
def test_log_statistics(
    mock_calc_stats,
    mock_detector_class,
    sample_config,
    sample_logger,
    sample_frames,
    sample_detections,
    tmp_path,
):
    """統計情報のログ出力が正しく動作する"""
    from types import SimpleNamespace

    mock_stats = SimpleNamespace(
        total_detections=5,
        avg_detections_per_frame=1.67,
        confidence_mean=0.85,
        confidence_min=0.8,
        confidence_max=0.9,
        confidence_std=0.05,
        confidence_median=0.85,
        frame_count=3,
    )
    mock_calc_stats.return_value = mock_stats

    mock_detector = MagicMock()
    mock_detector.detect_batch.return_value = [sample_detections]
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()
    results = phase.execute(sample_frames[:1])

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    phase.log_statistics(results, output_path)

    mock_calc_stats.assert_called_once_with(results)
    assert (output_path / "detection_statistics.json").exists()


@patch("src.pipeline.detection_phase.ViTDetector")
def test_execute_empty_frames(mock_detector_class, sample_config, sample_logger):
    """空のフレームリストでexecuteを呼ぶ"""
    mock_detector = MagicMock()
    mock_detector.detect_batch.return_value = []
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    phase.initialize()

    results = phase.execute([])

    assert len(results) == 0
    mock_detector.detect_batch.assert_not_called()


@patch("src.pipeline.detection_phase.ViTDetector")
def test_output_path_setting(mock_detector_class, sample_config, sample_logger, tmp_path):
    """output_pathが設定されている場合"""
    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector

    phase = DetectionPhase(sample_config, sample_logger)
    custom_output_path = tmp_path / "custom_output"
    phase.output_path = custom_output_path
    phase.initialize()

    assert phase.output_path == custom_output_path

