"""Unit tests for tracking module."""

import numpy as np
import pytest

from src.models.data_models import Detection
from src.tracking.hungarian import HungarianAlgorithm
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.similarity import SimilarityCalculator
from src.tracking.track import Track
from src.tracking.tracker import Tracker


class TestKalmanFilter:
    """KalmanFilterのテスト"""

    def test_init(self):
        """初期化テスト"""
        kf = KalmanFilter()
        assert kf.ndim == 4
        assert kf.x is None

    def test_init_with_measurement(self):
        """測定値での初期化テスト"""
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0])
        kf.init(measurement)
        assert kf.x is not None
        assert kf.x[0] == 100.0
        assert kf.x[1] == 200.0

    def test_predict(self):
        """予測テスト"""
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        predicted = kf.predict()
        assert predicted is not None
        assert len(predicted) == 4

    def test_update(self):
        """更新テスト"""
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        kf.predict()
        updated = kf.update(np.array([105.0, 205.0]))
        assert updated is not None
        assert len(updated) == 4

    def test_get_position(self):
        """位置取得テスト"""
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        position = kf.get_position()
        assert len(position) == 2
        assert position[0] == 100.0
        assert position[1] == 200.0


class TestSimilarityCalculator:
    """SimilarityCalculatorのテスト"""

    def test_init(self):
        """初期化テスト"""
        calc = SimilarityCalculator(appearance_weight=0.7, motion_weight=0.3)
        assert calc.appearance_weight == 0.7
        assert calc.motion_weight == 0.3

    def test_init_invalid_weights(self):
        """無効な重みでの初期化テスト"""
        with pytest.raises(ValueError, match=".*must equal.*"):
            SimilarityCalculator(appearance_weight=0.8, motion_weight=0.3)

    def test_cosine_similarity(self):
        """コサイン類似度テスト"""
        calc = SimilarityCalculator()
        feat1 = np.array([1.0, 0.0])
        feat2 = np.array([1.0, 0.0])
        sim = calc.cosine_similarity(feat1, feat2)
        assert sim == pytest.approx(1.0)

    def test_cosine_distance(self):
        """コサイン距離テスト"""
        calc = SimilarityCalculator()
        feat1 = np.array([1.0, 0.0])
        feat2 = np.array([0.0, 1.0])
        dist = calc.cosine_distance(feat1, feat2)
        assert dist == pytest.approx(1.0)

    def test_iou(self):
        """IoUテスト"""
        calc = SimilarityCalculator()
        bbox1 = (0.0, 0.0, 10.0, 10.0)
        bbox2 = (5.0, 5.0, 10.0, 10.0)
        iou = calc.iou(bbox1, bbox2)
        assert 0.0 <= iou <= 1.0

    def test_iou_distance(self):
        """IoU距離テスト"""
        calc = SimilarityCalculator()
        bbox1 = (0.0, 0.0, 10.0, 10.0)
        bbox2 = (0.0, 0.0, 10.0, 10.0)
        dist = calc.iou_distance(bbox1, bbox2)
        assert dist == pytest.approx(0.0)

    def test_compute_similarity(self):
        """統合類似度テスト"""
        calc = SimilarityCalculator()
        det1 = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        det2 = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        sim = calc.compute_similarity(det1, det2)
        assert 0.0 <= sim <= 1.0


class TestTrack:
    """Trackのテスト"""

    def test_init(self):
        """初期化テスト"""
        detection = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        kf = KalmanFilter()
        track = Track(track_id=1, detection=detection, kalman_filter=kf)
        assert track.track_id == 1
        assert track.age == 1
        assert track.hits == 1
        assert len(track.trajectory) == 1

    def test_predict(self):
        """予測テスト"""
        detection = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        kf = KalmanFilter()
        track = Track(track_id=1, detection=detection, kalman_filter=kf)
        predicted = track.predict()
        assert len(predicted) == 2

    def test_update(self):
        """更新テスト"""
        detection1 = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        detection2 = Detection(
            bbox=(5.0, 5.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(10.0, 15.0),
        )
        kf = KalmanFilter()
        track = Track(track_id=1, detection=detection1, kalman_filter=kf)
        track.update(detection2)
        assert track.hits == 2
        assert len(track.trajectory) == 2

    def test_is_confirmed(self):
        """確立確認テスト"""
        detection = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        kf = KalmanFilter()
        track = Track(track_id=1, detection=detection, kalman_filter=kf)
        assert not track.is_confirmed(min_hits=3)
        track.hits = 3
        assert track.is_confirmed(min_hits=3)


class TestTracker:
    """Trackerのテスト"""

    def test_init(self):
        """初期化テスト"""
        tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.next_id == 1

    def test_update_empty_detections(self):
        """空の検出結果での更新テスト"""
        tracker = Tracker()
        detections = []
        result = tracker.update(detections)
        assert len(result) == 0

    def test_update_single_detection(self):
        """単一検出結果での更新テスト"""
        tracker = Tracker(min_hits=1)  # min_hitsを1に設定して確立を容易に
        detection = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        # 特徴量を追加
        detection.features = np.random.rand(256).astype(np.float32)
        detection.features = detection.features / np.linalg.norm(detection.features)

        detections = [detection]
        result = tracker.update(detections)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_update_multiple_detections(self):
        """複数検出結果での更新テスト"""
        tracker = Tracker(min_hits=1)  # min_hitsを1に設定
        detections = [
            Detection(
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(5.0, 10.0),
            ),
            Detection(
                bbox=(20.0, 20.0, 10.0, 10.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(25.0, 30.0),
            ),
        ]
        # 特徴量を追加
        for det in detections:
            det.features = np.random.rand(256).astype(np.float32)
            det.features = det.features / np.linalg.norm(det.features)

        result = tracker.update(detections)
        assert len(result) == 2
        assert result[0].track_id != result[1].track_id

    def test_reset(self):
        """リセットテスト"""
        tracker = Tracker()
        detection = Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        )
        tracker.update([detection])
        assert len(tracker.tracks) > 0
        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1


class TestHungarianAlgorithm:
    """HungarianAlgorithmのテスト"""

    def test_init(self):
        """初期化テスト"""
        hungarian = HungarianAlgorithm()
        assert hungarian is not None

    def test_solve_empty_matrix(self):
        """空の行列でのテスト"""
        hungarian = HungarianAlgorithm()
        cost_matrix = np.array([]).reshape(0, 0)
        assignment, cost = hungarian.solve(cost_matrix)
        assert len(assignment) == 0
        assert cost == 0.0

    def test_solve_simple_matrix(self):
        """シンプルな行列でのテスト"""
        hungarian = HungarianAlgorithm()
        cost_matrix = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        assignment, cost = hungarian.solve(cost_matrix)
        assert len(assignment) == 2
        assert cost >= 0.0
