"""Unit tests for YOLOv8Detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.yolov8_detector import YOLOv8Detector
from src.models.data_models import Detection


class TestYOLOv8Detector:
    """YOLOv8Detectorのテスト"""

    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        """テスト用のフレーム"""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    @pytest.fixture
    def detector(self) -> YOLOv8Detector:
        """テスト用のdetectorインスタンス"""
        return YOLOv8Detector(
            model_path="test_model.pt",
            confidence_threshold=0.5,
            device="cpu",
            iou_threshold=0.45,
        )

    def test_init(self, detector: YOLOv8Detector):
        """初期化テスト"""
        assert detector.model_path == "test_model.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45
        assert detector.device == "cpu"
        assert detector.model is None
        assert detector.feature_extractor is not None

    def test_init_default_values(self):
        """デフォルト値での初期化テスト"""
        detector = YOLOv8Detector()
        assert detector.model_path == "runs/detect/person_ft/weights/best.pt"
        assert detector.confidence_threshold == 0.25
        assert detector.iou_threshold == 0.45

    @patch("src.detection.yolov8_detector.torch")
    def test_setup_device_cuda(self, mock_torch):
        """CUDAデバイス選択テスト"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        detector = YOLOv8Detector(device=None)
        assert detector.device == "cuda"

    @patch("src.detection.yolov8_detector.torch")
    def test_setup_device_mps(self, mock_torch):
        """MPSデバイス選択テスト"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        detector = YOLOv8Detector(device=None)
        assert detector.device == "mps"

    @patch("src.detection.yolov8_detector.torch")
    def test_setup_device_cpu_fallback(self, mock_torch):
        """CPUフォールバックテスト"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        detector = YOLOv8Detector(device=None)
        assert detector.device == "cpu"

    def test_setup_device_explicit(self):
        """明示的なデバイス指定テスト"""
        detector = YOLOv8Detector(device="cuda")
        assert detector.device == "cuda"

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_load_model_existing(self, mock_path, mock_yolo, detector: YOLOv8Detector):
        """既存モデルロードテスト"""
        mock_path.return_value.exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector.load_model()

        mock_yolo.assert_called_once()
        assert detector.model is not None

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_load_model_fallback_to_base(self, mock_path, mock_yolo, detector: YOLOv8Detector):
        """ベースモデルへのフォールバックテスト"""
        mock_path.return_value.exists.return_value = False
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector.load_model()

        mock_yolo.assert_called_once_with("yolov8x.pt")
        assert detector.model is not None

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_load_model_error(self, mock_path, mock_yolo, detector: YOLOv8Detector):
        """モデルロードエラーテスト"""
        mock_path.return_value.exists.return_value = True
        mock_yolo.side_effect = Exception("Model load error")

        with pytest.raises(RuntimeError, match="Failed to load YOLOv8 model"):
            detector.load_model()

    def test_detect_without_model(self, detector: YOLOv8Detector, sample_frame: np.ndarray):
        """モデル未ロードでの検出テスト"""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.detect(sample_frame)

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_detect_success(self, mock_path, mock_yolo, sample_frame: np.ndarray):
        """検出成功テスト"""
        mock_path.return_value.exists.return_value = True

        # モックの検出結果を設定
        mock_boxes = MagicMock()
        mock_boxes.xyxy = [np.array([100, 200, 150, 300])]
        mock_boxes.conf = [np.array([0.9])]
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = YOLOv8Detector(model_path="person_ft/best.pt", device="cpu")
        detector.load_model()
        detections = detector.detect(sample_frame)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].confidence == 0.9
        assert detections[0].class_name == "person"

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_detect_no_boxes(self, mock_path, mock_yolo, sample_frame: np.ndarray):
        """検出結果なしテスト"""
        mock_path.return_value.exists.return_value = True

        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = YOLOv8Detector(model_path="person_ft/best.pt", device="cpu")
        detector.load_model()
        detections = detector.detect(sample_frame)

        assert len(detections) == 0

    @patch("src.detection.yolov8_detector.YOLO")
    @patch("src.detection.yolov8_detector.Path")
    def test_detect_with_features(self, mock_path, mock_yolo, sample_frame: np.ndarray):
        """特徴量付き検出テスト"""
        mock_path.return_value.exists.return_value = True

        mock_boxes = MagicMock()
        mock_boxes.xyxy = [np.array([100, 200, 150, 300])]
        mock_boxes.conf = [np.array([0.9])]
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = YOLOv8Detector(model_path="person_ft/best.pt", device="cpu")
        detector.load_model()
        detections, features = detector.detect_with_features(sample_frame)

        assert len(detections) == 1
        assert len(features) == 1

    def test_extract_features_empty(self, detector: YOLOv8Detector, sample_frame: np.ndarray):
        """空の検出結果での特徴量抽出テスト"""
        features = detector.extract_features(sample_frame, [])
        assert len(features) == 0

    def test_extract_features_with_detections(self, detector: YOLOv8Detector, sample_frame: np.ndarray):
        """検出結果ありでの特徴量抽出テスト"""
        detections = [
            Detection(
                bbox=(100.0, 200.0, 50.0, 100.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
                camera_coords=(125.0, 300.0),
            ),
        ]
        features = detector.extract_features(sample_frame, detections)
        assert len(features) == 1

    def test_extract_features_invalid_bbox(self, detector: YOLOv8Detector, sample_frame: np.ndarray):
        """無効なbboxでの特徴量抽出テスト"""
        detections = [
            Detection(
                bbox=(0.0, 0.0, 0.0, 0.0),  # 無効なサイズ
                confidence=0.9,
                class_id=0,
                class_name="person",
                camera_coords=(0.0, 0.0),
            ),
        ]
        features = detector.extract_features(sample_frame, detections)
        assert len(features) == 1

    def test_get_foot_position(self, detector: YOLOv8Detector):
        """足元座標計算テスト"""
        bbox = (100.0, 200.0, 50.0, 100.0)
        foot_x, foot_y = detector._get_foot_position(bbox)

        assert foot_x == 125.0  # x + w/2
        assert foot_y == 300.0  # y + h

    def test_get_attention_map(self, detector: YOLOv8Detector, sample_frame: np.ndarray):
        """Attention Map取得テスト（YOLOv8では未サポート）"""
        result = detector.get_attention_map(sample_frame)
        assert result is None

    def test_postprocess(self, detector: YOLOv8Detector):
        """後処理テスト"""
        mock_boxes = MagicMock()
        mock_boxes.xyxy = [
            np.array([100, 200, 150, 300]),
            np.array([200, 100, 280, 250]),
        ]
        mock_boxes.conf = [np.array([0.9]), np.array([0.8])]
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        detections = detector._postprocess([mock_result], (720, 1280, 3))

        assert len(detections) == 2
        assert detections[0].confidence == 0.9
        assert detections[1].confidence == 0.8

    def test_postprocess_empty(self, detector: YOLOv8Detector):
        """空の後処理テスト"""
        mock_result = MagicMock()
        mock_result.boxes = None

        detections = detector._postprocess([mock_result], (720, 1280, 3))

        assert len(detections) == 0


class TestYOLOv8DetectorIntegration:
    """YOLOv8Detectorの統合テスト（モックなし）"""

    def test_detector_initialization_chain(self):
        """初期化チェーンテスト"""
        detector = YOLOv8Detector(
            model_path="nonexistent.pt",
            confidence_threshold=0.3,
            device="cpu",
            iou_threshold=0.5,
        )

        assert detector.model is None
        assert detector.confidence_threshold == 0.3
        assert detector.device == "cpu"
        assert detector.iou_threshold == 0.5
