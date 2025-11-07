"""Test cases for similarity calculation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.data_models import Detection
from src.tracking.similarity import SimilarityCalculator


@pytest.fixture()
def similarity_calculator() -> SimilarityCalculator:
    """テスト用のSimilarityCalculator"""
    return SimilarityCalculator(appearance_weight=0.7, motion_weight=0.3, iou_threshold=0.3)


@pytest.fixture()
def sample_detections() -> tuple[Detection, Detection]:
    """テスト用の検出結果ペア"""
    det1 = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.85,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        features=np.random.randn(256).astype(np.float32),
    )
    # L2正規化
    det1.features = det1.features / np.linalg.norm(det1.features)

    det2 = Detection(
        bbox=(110.0, 210.0, 55.0, 105.0),
        confidence=0.90,
        class_id=1,
        class_name="person",
        camera_coords=(137.5, 315.0),
        features=np.random.randn(256).astype(np.float32),
    )
    # L2正規化
    det2.features = det2.features / np.linalg.norm(det2.features)

    return (det1, det2)


def test_similarity_calculator_init():
    """SimilarityCalculatorの初期化テスト"""
    calc = SimilarityCalculator(appearance_weight=0.7, motion_weight=0.3)
    assert calc.appearance_weight == 0.7
    assert calc.motion_weight == 0.3
    assert calc.iou_threshold == 0.3


def test_similarity_calculator_init_invalid_weights():
    """無効な重みでの初期化エラーテスト"""
    with pytest.raises(ValueError, match="appearance_weight.*motion_weight.*must equal 1.0"):
        SimilarityCalculator(appearance_weight=0.8, motion_weight=0.3)


def test_cosine_similarity_identical(similarity_calculator: SimilarityCalculator):
    """同一特徴量のコサイン類似度テスト"""
    feat = np.random.randn(256).astype(np.float32)
    feat = feat / np.linalg.norm(feat)

    similarity = similarity_calculator.cosine_similarity(feat, feat)

    assert np.isclose(similarity, 1.0, atol=1e-6)


def test_cosine_similarity_orthogonal(similarity_calculator: SimilarityCalculator):
    """直交特徴量のコサイン類似度テスト"""
    feat1 = np.array([1.0, 0.0], dtype=np.float32)
    feat2 = np.array([0.0, 1.0], dtype=np.float32)

    similarity = similarity_calculator.cosine_similarity(feat1, feat2)

    assert np.isclose(similarity, 0.0, atol=1e-6)


def test_cosine_distance(similarity_calculator: SimilarityCalculator):
    """コサイン距離のテスト"""
    feat1 = np.random.randn(256).astype(np.float32)
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = np.random.randn(256).astype(np.float32)
    feat2 = feat2 / np.linalg.norm(feat2)

    distance = similarity_calculator.cosine_distance(feat1, feat2)

    assert 0.0 <= distance <= 1.0


def test_cosine_similarity_none_features(similarity_calculator: SimilarityCalculator):
    """None特徴量のコサイン類似度テスト"""
    feat = np.random.randn(256).astype(np.float32)
    feat = feat / np.linalg.norm(feat)

    similarity = similarity_calculator.cosine_similarity(feat, None)

    assert similarity == 0.0


def test_iou_identical(similarity_calculator: SimilarityCalculator):
    """同一バウンディングボックスのIoUテスト"""
    bbox = (100.0, 200.0, 50.0, 100.0)

    iou_value = similarity_calculator.iou(bbox, bbox)

    assert np.isclose(iou_value, 1.0, atol=1e-6)


def test_iou_no_overlap(similarity_calculator: SimilarityCalculator):
    """重複なしバウンディングボックスのIoUテスト"""
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (200.0, 400.0, 50.0, 100.0)

    iou_value = similarity_calculator.iou(bbox1, bbox2)

    assert iou_value == 0.0


def test_iou_partial_overlap(similarity_calculator: SimilarityCalculator):
    """部分的重複バウンディングボックスのIoUテスト"""
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (120.0, 220.0, 50.0, 100.0)

    iou_value = similarity_calculator.iou(bbox1, bbox2)

    assert 0.0 < iou_value < 1.0


def test_iou_distance(similarity_calculator: SimilarityCalculator):
    """IoU距離のテスト"""
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (110.0, 210.0, 55.0, 105.0)

    distance = similarity_calculator.iou_distance(bbox1, bbox2)

    assert 0.0 <= distance <= 1.0


def test_iou_distance_below_threshold(similarity_calculator: SimilarityCalculator):
    """IoU閾値以下のIoU距離テスト"""
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (200.0, 400.0, 50.0, 100.0)  # 重複なし

    distance = similarity_calculator.iou_distance(bbox1, bbox2)

    assert distance == 1.0


def test_compute_similarity(similarity_calculator: SimilarityCalculator, sample_detections: tuple[Detection, Detection]):
    """統合類似度計算のテスト"""
    det1, det2 = sample_detections

    combined_dist, cosine_dist, iou_dist = similarity_calculator.compute_similarity(det1, det2)

    assert 0.0 <= combined_dist <= 1.0
    assert 0.0 <= cosine_dist <= 1.0
    assert 0.0 <= iou_dist <= 1.0
    assert np.isclose(
        combined_dist,
        similarity_calculator.appearance_weight * cosine_dist + similarity_calculator.motion_weight * iou_dist,
        atol=1e-6,
    )


def test_compute_similarity_no_features(
    similarity_calculator: SimilarityCalculator, sample_detections: tuple[Detection, Detection]
):
    """特徴量なしの統合類似度計算テスト"""
    det1, det2 = sample_detections
    det1.features = None

    combined_dist, cosine_dist, iou_dist = similarity_calculator.compute_similarity(det1, det2)

    assert cosine_dist == 1.0  # 特徴量がない場合は最大距離
    assert 0.0 <= combined_dist <= 1.0
    assert 0.0 <= iou_dist <= 1.0


def test_compute_similarity_matrix(similarity_calculator: SimilarityCalculator):
    """類似度行列計算のテスト"""
    detections1 = [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
            features=np.random.randn(256).astype(np.float32),
        )
        for _ in range(3)
    ]
    # L2正規化
    for det in detections1:
        det.features = det.features / np.linalg.norm(det.features)

    detections2 = [
        Detection(
            bbox=(110.0, 210.0, 55.0, 105.0),
            confidence=0.90,
            class_id=1,
            class_name="person",
            camera_coords=(137.5, 315.0),
            features=np.random.randn(256).astype(np.float32),
        )
        for _ in range(2)
    ]
    # L2正規化
    for det in detections2:
        det.features = det.features / np.linalg.norm(det.features)

    similarity_matrix = similarity_calculator.compute_similarity_matrix(detections1, detections2)

    assert similarity_matrix.shape == (3, 2)
    assert np.all((similarity_matrix >= 0.0) & (similarity_matrix <= 1.0))

