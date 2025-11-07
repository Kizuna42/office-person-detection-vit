"""Test cases for feature extraction functionality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detection.vit_detector import ViTDetector
from src.models.data_models import Detection


@pytest.fixture()
def sample_frame() -> np.ndarray:
    """テスト用のフレーム画像"""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture()
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


def test_extract_features_not_loaded(sample_frame: np.ndarray, sample_detections: list[Detection]):
    """モデルがロードされていない場合のエラー"""
    detector = ViTDetector()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        detector.extract_features(sample_frame, sample_detections)


def test_extract_features_empty_detections(sample_frame: np.ndarray):
    """空の検出結果の場合"""
    detector = ViTDetector(device="cpu")
    detector.load_model()

    features = detector.extract_features(sample_frame, [])

    assert features == []


def test_detect_with_features_extraction(sample_frame: np.ndarray):
    """特徴量抽出付き検出のテスト"""
    detector = ViTDetector(device="cpu", confidence_threshold=0.0)
    detector.load_model()

    detections = detector.detect(sample_frame, extract_features=True)

    # 検出結果が返されることを確認
    assert isinstance(detections, list)
    # 特徴量が抽出されている場合、featuresフィールドが設定されている
    for detection in detections:
        assert hasattr(detection, "features")


def test_extract_detection_features_shape():
    """特徴量抽出の形状テスト"""
    detector = ViTDetector(device="cpu")
    detector.load_model()

    # モックエンコーダー特徴量を作成
    seq_len = 100  # パッチ数 + CLSトークン
    hidden_dim = 256
    encoder_features = torch.randn(seq_len, hidden_dim)

    bbox = (100.0, 200.0, 50.0, 100.0)
    image_size = (720, 1280)

    feature = detector._extract_detection_features(encoder_features, bbox, image_size)

    # 特徴量の形状と正規化を確認
    assert feature.shape == (hidden_dim,)
    assert np.isclose(np.linalg.norm(feature), 1.0, atol=1e-6)  # L2正規化確認


def test_extract_detection_features_normalization():
    """特徴量のL2正規化確認"""
    detector = ViTDetector(device="cpu")
    detector.load_model()

    seq_len = 100
    hidden_dim = 256
    encoder_features = torch.randn(seq_len, hidden_dim)

    bbox = (100.0, 200.0, 50.0, 100.0)
    image_size = (720, 1280)

    feature = detector._extract_detection_features(encoder_features, bbox, image_size)

    # L2正規化されていることを確認
    norm = np.linalg.norm(feature)
    assert np.isclose(norm, 1.0, atol=1e-6), f"Expected norm=1.0, got {norm}"


def test_detection_model_extension():
    """Detectionデータモデルの拡張確認"""
    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.85,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        track_id=1,
        features=np.random.randn(256).astype(np.float32),
        appearance_score=0.95,
    )

    assert detection.track_id == 1
    assert detection.features is not None
    assert detection.features.shape == (256,)
    assert detection.appearance_score == 0.95


def test_detection_model_defaults():
    """Detectionデータモデルのデフォルト値確認"""
    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.85,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
    )

    assert detection.track_id is None
    assert detection.features is None
    assert detection.appearance_score is None
