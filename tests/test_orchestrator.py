"""Test cases for PipelineOrchestrator."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")
    config.set("output.directory", str(tmp_path / "output"))
    config.set("output.use_session_management", False)
    config.set("video.input_path", "test_video.mov")
    config.set("video.frame_interval_minutes", 5)
    config.set("video.tolerance_seconds", 10.0)
    config.set("video.fps", 30.0)
    config.set("timestamp.extraction_mode", "manual_targets")
    config.set("timestamp.extraction", {})
    config.set("timestamp.sampling", {})
    config.set("timestamp.target", {})
    config.set("ocr", {"engines": ["tesseract"]})
    config.set("detection.model_name", "facebook/detr-resnet-50")
    config.set("detection.confidence_threshold", 0.5)
    config.set("detection.device", "cpu")
    config.set("detection.batch_size", 2)
    config.set("homography.matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    config.set("floormap", {})
    config.set("zones", [])
    return config


@pytest.fixture
def sample_logger() -> logging.Logger:
    """テスト用のロガー"""
    logger = logging.getLogger("test_orchestrator")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_extraction_results() -> list[dict]:
    """テスト用の抽出結果"""
    return [
        {
            "frame_idx": 0,
            "timestamp": datetime(2025, 8, 26, 16, 5, 0),
            "frame": np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            "confidence": 0.9,
        },
        {
            "frame_idx": 1,
            "timestamp": datetime(2025, 8, 26, 16, 10, 0),
            "frame": np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            "confidence": 0.85,
        },
    ]


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
    ]


def test_init(sample_config: ConfigManager, sample_logger: logging.Logger):
    """初期化が正しく動作する"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    assert orchestrator.config == sample_config
    assert orchestrator.logger == sample_logger
    assert orchestrator.output_manager is None
    assert orchestrator.session_dir is None
    assert orchestrator.output_path == Path(sample_config.get("output.directory"))


def test_setup_output_directories_without_session(
    sample_config: ConfigManager, sample_logger: logging.Logger, tmp_path: Path
):
    """セッション管理なしで出力ディレクトリをセットアップ"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    assert orchestrator.output_manager is None
    assert orchestrator.session_dir is None
    # 従来のディレクトリ構造が作成されていることを確認
    assert (orchestrator.output_path / "detections").exists()
    assert (orchestrator.output_path / "floormaps").exists()
    assert (orchestrator.output_path / "graphs").exists()


def test_setup_output_directories_with_session(
    sample_config: ConfigManager, sample_logger: logging.Logger, tmp_path: Path
):
    """セッション管理ありで出力ディレクトリをセットアップ"""
    sample_config.set("output.use_session_management", True)
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    args = Mock()
    args.config = "config.yaml"
    orchestrator.setup_output_directories(use_session_management=True, args=args)

    assert orchestrator.output_manager is not None
    assert orchestrator.session_dir is not None
    assert orchestrator.session_dir.exists()


def test_get_phase_output_dir_without_session(sample_config: ConfigManager, sample_logger: logging.Logger):
    """セッション管理なしでのフェーズ出力ディレクトリ取得"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    output_dir = orchestrator.get_phase_output_dir("phase1_extraction")
    assert output_dir == orchestrator.output_path


def test_get_phase_output_dir_with_session(sample_config: ConfigManager, sample_logger: logging.Logger, tmp_path: Path):
    """セッション管理ありでのフェーズ出力ディレクトリ取得"""
    sample_config.set("output.use_session_management", True)
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=True)

    output_dir = orchestrator.get_phase_output_dir("phase1_extraction")
    assert output_dir == orchestrator.session_dir / "phase1_extraction"


def test_parse_datetime_range(sample_config: ConfigManager, sample_logger: logging.Logger):
    """日時範囲のパースが正しく動作する"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    target_config = {
        "start_datetime": "2025-08-26 16:00:00",
        "end_datetime": "2025-08-26 18:00:00",
    }

    start, end = orchestrator._parse_datetime_range(target_config, "16:30", "17:30")

    assert start is not None
    assert end is not None
    assert start.hour == 16
    assert start.minute == 30
    assert end.hour == 17
    assert end.minute == 30


def test_parse_datetime_range_no_config(sample_config: ConfigManager, sample_logger: logging.Logger):
    """設定がない場合の日時範囲パース"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    start, end = orchestrator._parse_datetime_range({}, None, None)

    assert start is None
    assert end is None


@patch("src.pipeline.orchestrator.FrameExtractionPipeline")
def test_extract_frames(
    mock_pipeline_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_extraction_results: list[dict],
    tmp_path: Path,
):
    """フレーム抽出が正しく動作する"""
    mock_pipeline = Mock()
    mock_pipeline.run.return_value = sample_extraction_results
    # extractor.get_cache_stats()のMock設定
    mock_extractor = Mock()
    mock_extractor.get_cache_stats.return_value = {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_size": 0,
        "hit_rate": 0.0,
    }
    mock_pipeline.extractor = mock_extractor
    mock_pipeline_class.return_value = mock_pipeline

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    results = orchestrator.extract_frames("test_video.mov")

    assert results == sample_extraction_results
    mock_pipeline.run.assert_called_once()


@patch("src.pipeline.orchestrator.VideoProcessor")
def test_prepare_frames_for_detection(
    mock_video_processor_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_extraction_results: list[dict],
):
    """検出用フレーム準備が正しく動作する"""
    mock_video_processor = Mock()
    mock_video_processor.get_frame.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_video_processor_class.return_value = mock_video_processor

    # フレームがNoneの場合をテスト
    extraction_results_no_frame = [
        {
            "frame_idx": 0,
            "timestamp": datetime(2025, 8, 26, 16, 5, 0),
            "frame": None,
        }
    ]

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    frames = orchestrator.prepare_frames_for_detection(extraction_results_no_frame, "test_video.mov")

    assert len(frames) == 1
    assert frames[0][0] == 0
    mock_video_processor.open.assert_called_once()
    mock_video_processor.release.assert_called_once()


def test_prepare_frames_for_detection_with_frames(
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_extraction_results: list[dict],
):
    """フレームが既に存在する場合の準備"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    frames = orchestrator.prepare_frames_for_detection(sample_extraction_results, "test_video.mov")

    assert len(frames) == 2
    assert frames[0][0] == 0
    assert frames[1][0] == 1


@patch("src.pipeline.orchestrator.DetectionPhase")
def test_run_detection(
    mock_detection_phase_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
):
    """人物検出が正しく動作する"""
    mock_phase = Mock()
    mock_phase.execute.return_value = [(0, "2025/08/26 16:05:00", sample_detections)]
    mock_phase.detector = None
    mock_detection_phase_class.return_value = mock_phase

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    sample_frames = [(0, "2025/08/26 16:05:00", np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))]
    results, phase = orchestrator.run_detection(sample_frames)

    assert len(results) == 1
    assert phase == mock_phase
    mock_phase.initialize.assert_called_once()
    mock_phase.execute.assert_called_once()


@patch("src.pipeline.orchestrator.TransformPhase")
def test_run_transform(
    mock_transform_phase_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
):
    """座標変換とゾーン判定が正しく動作する"""
    mock_phase = Mock()
    frame_result = FrameResult(
        frame_number=0,
        timestamp="2025/08/26 16:05:00",
        detections=sample_detections,
        zone_counts={},
    )
    mock_phase.execute.return_value = [frame_result]
    mock_transform_phase_class.return_value = mock_phase

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    detection_results = [(0, "2025/08/26 16:05:00", sample_detections)]
    results, phase = orchestrator.run_transform(detection_results)

    assert len(results) == 1
    assert phase == mock_phase
    mock_phase.initialize.assert_called_once()
    mock_phase.execute.assert_called_once()


@patch("src.pipeline.orchestrator.AggregationPhase")
def test_run_aggregation(
    mock_aggregation_phase_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
):
    """集計処理が正しく動作する"""
    mock_phase = Mock()
    mock_aggregator = Mock()
    mock_aggregator.get_statistics.return_value = {"zone_a": {"average": 1.0}}
    mock_phase.execute.return_value = mock_aggregator
    mock_aggregation_phase_class.return_value = mock_phase

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    frame_results = [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=sample_detections,
            zone_counts={},
        )
    ]

    phase, aggregator = orchestrator.run_aggregation(frame_results)

    assert phase == mock_phase
    assert aggregator == mock_aggregator
    mock_phase.execute.assert_called_once()


@patch("src.pipeline.orchestrator.VisualizationPhase")
def test_run_visualization(
    mock_visualization_phase_class,
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
):
    """可視化が正しく動作する"""
    mock_phase = Mock()
    mock_visualization_phase_class.return_value = mock_phase

    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    mock_aggregator = Mock()
    frame_results = [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=sample_detections,
            zone_counts={},
        )
    ]

    orchestrator.run_visualization(mock_aggregator, frame_results)

    mock_phase.execute.assert_called_once()


def test_save_session_summary_without_session(
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
):
    """セッション管理なしでのサマリー保存（何もしない）"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=False)

    mock_aggregator = Mock()
    mock_aggregator.get_statistics.return_value = {"zone_a": {"average": 1.0}}

    # エラーが発生しないことを確認
    orchestrator.save_session_summary(
        [],
        [(0, "2025/08/26 16:05:00", sample_detections)],
        [],
        mock_aggregator,
    )


def test_save_session_summary_with_session(
    sample_config: ConfigManager,
    sample_logger: logging.Logger,
    sample_detections: list[Detection],
    tmp_path: Path,
):
    """セッション管理ありでのサマリー保存"""
    sample_config.set("output.use_session_management", True)
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)
    orchestrator.setup_output_directories(use_session_management=True)

    mock_aggregator = Mock()
    mock_aggregator.get_statistics.return_value = {"zone_a": {"average": 1.0}}

    extraction_results = [{"frame_idx": 0, "timestamp": datetime(2025, 8, 26, 16, 5, 0)}]
    frame_results = [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=sample_detections,
            zone_counts={},
        )
    ]

    orchestrator.save_session_summary(
        extraction_results,
        [(0, "2025/08/26 16:05:00", sample_detections)],
        frame_results,
        mock_aggregator,
    )

    # サマリーファイルが作成されていることを確認
    summary_path = orchestrator.session_dir / "summary.json"
    assert summary_path.exists()


def test_cleanup(sample_config: ConfigManager, sample_logger: logging.Logger):
    """クリーンアップが正しく動作する"""
    orchestrator = PipelineOrchestrator(sample_config, sample_logger)

    mock_detector_phase = Mock()
    mock_detector_phase.detector = None

    # エラーが発生しないことを確認
    orchestrator.cleanup(mock_detector_phase)
    orchestrator.cleanup()
