"""Unit tests for tracking phase module."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import ConfigManager
from src.models.data_models import Detection
from src.pipeline.phases.tracking import TrackingPhase


class TestTrackingPhase:
    """TrackingPhaseのテスト"""

    @pytest.fixture
    def config(self):
        """テスト用のConfigManagerを作成"""
        config_data = {
            "tracking": {
                "enabled": True,
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "appearance_weight": 0.7,
                "motion_weight": 0.3,
            },
            "detection": {
                "yolov8_model_path": "test_model.pt",
                "confidence_threshold": 0.5,
                "device": "cpu",
            },
            "output": {
                "save_tracking_images": True,
            },
        }
        # 一時的な設定ファイルを作成
        from pathlib import Path
        import tempfile

        import yaml

        # 最小限の必須設定を追加
        full_config = {
            "video": {"input_path": "test.mov"},
            "floormap": {
                "image_path": "test.png",
                "image_width": 100,
                "image_height": 50,
                "image_origin_x": 0,
                "image_origin_y": 0,
                "image_x_mm_per_pixel": 1.0,
                "image_y_mm_per_pixel": 1.0,
            },
            "homography": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "zones": [],
            "output": {"directory": "output"},
        }
        full_config.update(config_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(full_config, f)

            config = ConfigManager(str(config_path))
            return config

    @pytest.fixture
    def logger(self):
        """テスト用のロガーを作成"""
        import logging

        logger = logging.getLogger("test_tracking_phase")
        logger.setLevel(logging.DEBUG)
        return logger

    @pytest.fixture
    def tracking_phase(self, config, logger):
        """TrackingPhaseインスタンスを作成"""
        return TrackingPhase(config, logger)

    def test_init(self, tracking_phase):
        """初期化テスト"""
        assert tracking_phase.tracker is None
        assert tracking_phase.detector is None
        assert tracking_phase.tracks == []
        assert tracking_phase.tracked_results == []
        assert tracking_phase.sample_frames == []

    def test_initialize_tracking_disabled(self, config, logger):
        """追跡が無効な場合の初期化テスト"""
        config.config["tracking"]["enabled"] = False
        phase = TrackingPhase(config, logger)
        phase.initialize()
        assert phase.tracker is None
        assert phase.detector is None

    @patch("src.pipeline.phases.tracking.YOLOv8Detector")
    @patch("src.pipeline.phases.tracking.Tracker")
    def test_initialize_tracking_enabled(self, mock_tracker_class, mock_detector_class, config, logger):
        """追跡が有効な場合の初期化テスト"""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        phase = TrackingPhase(config, logger)
        phase.initialize()

        assert phase.tracker is not None
        assert phase.detector is not None
        mock_tracker_class.assert_called_once()
        mock_detector_class.assert_called_once()
        mock_detector.load_model.assert_called_once()

    def test_execute_tracking_disabled(self, tracking_phase, config):
        """追跡が無効な場合の実行テスト"""
        config.config["tracking"]["enabled"] = False
        tracking_phase.initialize()

        detection_results = [
            (
                1,
                "12:00:00",
                [
                    Detection(
                        bbox=(100, 100, 50, 100),
                        confidence=0.9,
                        class_id=1,
                        class_name="person",
                        camera_coords=(125, 200),
                    )
                ],
            ),
        ]
        sample_frames = [(1, "12:00:00", np.zeros((720, 1280, 3), dtype=np.uint8))]

        result = tracking_phase.execute(detection_results, sample_frames)
        assert result == detection_results

    def test_execute_tracking_not_initialized(self, tracking_phase, config):
        """初期化されていない場合の実行テスト"""
        config.config["tracking"]["enabled"] = True
        # initialize()を呼ばない

        detection_results = [
            (
                1,
                "12:00:00",
                [
                    Detection(
                        bbox=(100, 100, 50, 100),
                        confidence=0.9,
                        class_id=1,
                        class_name="person",
                        camera_coords=(125, 200),
                    )
                ],
            ),
        ]
        sample_frames = [(1, "12:00:00", np.zeros((720, 1280, 3), dtype=np.uint8))]

        with pytest.raises(RuntimeError, match="トラッカーまたは検出器が初期化されていません"):
            tracking_phase.execute(detection_results, sample_frames)

    @patch("src.pipeline.phases.tracking.YOLOv8Detector")
    @patch("src.pipeline.phases.tracking.Tracker")
    def test_execute_basic(self, mock_tracker_class, mock_detector_class, config, logger):
        """基本的な実行テスト"""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = [
            Detection(
                bbox=(100, 100, 50, 100),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125, 200),
                track_id=1,
            ),
        ]
        mock_tracker.get_confirmed_tracks.return_value = []
        mock_tracker_class.return_value = mock_tracker

        mock_detector = MagicMock()
        mock_detector.extract_features.return_value = [np.random.rand(256).astype(np.float32)]
        mock_detector_class.return_value = mock_detector

        phase = TrackingPhase(config, logger)
        phase.initialize()

        detection_results = [
            (
                1,
                "12:00:00",
                [
                    Detection(
                        bbox=(100, 100, 50, 100),
                        confidence=0.9,
                        class_id=1,
                        class_name="person",
                        camera_coords=(125, 200),
                    )
                ],
            ),
        ]
        sample_frames = [(1, "12:00:00", np.zeros((720, 1280, 3), dtype=np.uint8))]

        result = phase.execute(detection_results, sample_frames)
        assert len(result) == 1
        assert len(result[0][2]) == 1  # detections
        assert result[0][2][0].track_id == 1

    @patch("src.pipeline.phases.tracking.YOLOv8Detector")
    @patch("src.pipeline.phases.tracking.Tracker")
    def test_execute_missing_frame(self, mock_tracker_class, mock_detector_class, config, logger):
        """フレームが見つからない場合のテスト"""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = []
        mock_tracker.get_confirmed_tracks.return_value = []
        mock_tracker_class.return_value = mock_tracker

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        phase = TrackingPhase(config, logger)
        phase.initialize()

        detection_results = [
            (
                1,
                "12:00:00",
                [
                    Detection(
                        bbox=(100, 100, 50, 100),
                        confidence=0.9,
                        class_id=1,
                        class_name="person",
                        camera_coords=(125, 200),
                    )
                ],
            ),
        ]
        sample_frames = [(2, "12:00:05", np.zeros((720, 1280, 3), dtype=np.uint8))]  # frame_numが一致しない

        result = phase.execute(detection_results, sample_frames)
        assert len(result) == 1
        # フレームが見つからない場合は元の検出結果を返す

    @patch("src.pipeline.phases.tracking.YOLOv8Detector")
    @patch("src.pipeline.phases.tracking.Tracker")
    def test_execute_feature_extraction_failure(self, mock_tracker_class, mock_detector_class, config, logger):
        """特徴量抽出に失敗した場合のテスト"""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = []
        mock_tracker.get_confirmed_tracks.return_value = []
        mock_tracker_class.return_value = mock_tracker

        mock_detector = MagicMock()
        mock_detector.extract_features.side_effect = Exception("Feature extraction failed")
        mock_detector_class.return_value = mock_detector

        phase = TrackingPhase(config, logger)
        phase.initialize()

        detection_results = [
            (
                1,
                "12:00:00",
                [
                    Detection(
                        bbox=(100, 100, 50, 100),
                        confidence=0.9,
                        class_id=1,
                        class_name="person",
                        camera_coords=(125, 200),
                    )
                ],
            ),
        ]
        sample_frames = [(1, "12:00:00", np.zeros((720, 1280, 3), dtype=np.uint8))]

        # エラーが発生しても処理は続行される
        result = phase.execute(detection_results, sample_frames)
        assert len(result) == 1

    def test_export_results_tracking_disabled(self, tracking_phase, config):
        """追跡が無効な場合のエクスポートテスト"""
        config.config["tracking"]["enabled"] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            tracking_phase.export_results(output_path)
            # 何も出力されない

    def test_export_results_no_tracks(self, tracking_phase, config):
        """トラックがない場合のエクスポートテスト"""
        config.config["tracking"]["enabled"] = True
        tracking_phase.tracks = []
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            tracking_phase.export_results(output_path)
            # 警告が出力されるが、エラーは発生しない

    @patch("src.pipeline.phases.tracking.TrajectoryExporter")
    @patch("src.pipeline.phases.tracking.save_tracked_detection_image")
    def test_export_results_with_tracks(self, mock_save_image, mock_exporter_class, tracking_phase, config):
        """トラックがある場合のエクスポートテスト"""
        from src.tracking.kalman_filter import KalmanFilter
        from src.tracking.track import Track

        config.config["tracking"]["enabled"] = True
        config.config["output"]["save_tracking_images"] = True

        detection = Detection(
            bbox=(100, 100, 50, 100), confidence=0.9, class_id=1, class_name="person", camera_coords=(125, 200)
        )
        kf = KalmanFilter()
        kf.init(np.array([125.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        tracking_phase.tracks = [track]
        tracking_phase.tracked_results = [
            (1, "12:00:00", [detection]),
        ]
        tracking_phase.sample_frames = [
            (1, "12:00:00", np.zeros((720, 1280, 3), dtype=np.uint8)),
        ]

        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            tracking_phase.export_results(output_path)

            # TrajectoryExporterが呼ばれる
            mock_exporter.export_json.assert_called_once()
            mock_exporter.export_csv.assert_called_once()

    def test_get_tracks(self, tracking_phase):
        """トラック取得テスト"""
        from src.tracking.kalman_filter import KalmanFilter
        from src.tracking.track import Track

        detection = Detection(
            bbox=(100, 100, 50, 100), confidence=0.9, class_id=1, class_name="person", camera_coords=(125, 200)
        )
        kf = KalmanFilter()
        kf.init(np.array([125.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        tracking_phase.tracks = [track]
        result = tracking_phase.get_tracks()
        assert len(result) == 1
        assert result[0].track_id == 1
        # コピーが返されることを確認
        assert result is not tracking_phase.tracks

    def test_cleanup(self, tracking_phase):
        """クリーンアップテスト"""
        from src.detection import YOLOv8Detector

        mock_detector = MagicMock(spec=YOLOv8Detector)
        tracking_phase.detector = mock_detector

        tracking_phase.cleanup()
        # エラーが発生しないことを確認
