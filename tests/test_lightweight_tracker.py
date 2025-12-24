"""Tests for lightweight tracker module."""

import numpy as np
import pytest

from src.models.data_models import Detection
from src.tracking.lightweight_tracker import (
    LightweightTrack,
    LightweightTracker,
    OpticalFlowTracker,
)


class TestLightweightTrack:
    """LightweightTrackのテスト"""

    def test_init(self) -> None:
        """初期化テスト"""
        center = np.array([100.0, 200.0])
        track = LightweightTrack(
            track_id=1,
            bbox=(50.0, 150.0, 100.0, 100.0),
            center=center,
        )

        assert track.track_id == 1
        assert track.bbox == (50.0, 150.0, 100.0, 100.0)
        assert track.age == 0
        assert track.time_since_update == 0
        assert track.hits == 1

    def test_predict(self) -> None:
        """予測テスト"""
        center = np.array([100.0, 200.0])
        track = LightweightTrack(
            track_id=1,
            bbox=(50.0, 150.0, 100.0, 100.0),
            center=center,
        )

        predicted = track.predict()

        assert track.age == 1
        assert track.time_since_update == 1
        assert predicted.shape == (2,)

    def test_update(self) -> None:
        """更新テスト"""
        center = np.array([100.0, 200.0])
        track = LightweightTrack(
            track_id=1,
            bbox=(50.0, 150.0, 100.0, 100.0),
            center=center,
        )

        # 予測後に更新
        track.predict()
        new_bbox = (60.0, 160.0, 100.0, 100.0)
        new_center = np.array([110.0, 210.0])
        track.update(new_bbox, new_center)

        assert track.bbox == new_bbox
        assert track.time_since_update == 0
        assert track.hits == 2


class TestLightweightTracker:
    """LightweightTrackerのテスト"""

    def test_init(self) -> None:
        """初期化テスト"""
        tracker = LightweightTracker(
            max_age=30,
            iou_threshold=0.3,
            use_optical_flow=False,  # テストでは無効化
        )

        assert tracker.max_age == 30
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0

    def test_update_with_detections_empty(self) -> None:
        """空の検出結果での更新テスト"""
        tracker = LightweightTracker(use_optical_flow=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = tracker.update_with_detections(frame, [])

        assert len(result) == 0
        assert len(tracker.tracks) == 0

    def test_update_with_detections_single(self) -> None:
        """単一検出での更新テスト"""
        tracker = LightweightTracker(use_optical_flow=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )

        result = tracker.update_with_detections(frame, [det])

        assert len(result) == 1
        assert result[0].track_id == 1
        assert len(tracker.tracks) == 1

    def test_update_with_detections_multiple(self) -> None:
        """複数検出での更新テスト"""
        tracker = LightweightTracker(use_optical_flow=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = [
            Detection(
                bbox=(100.0, 100.0, 50.0, 100.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
                camera_coords=(125.0, 200.0),
            ),
            Detection(
                bbox=(300.0, 200.0, 50.0, 100.0),
                confidence=0.8,
                class_id=0,
                class_name="person",
                camera_coords=(325.0, 300.0),
            ),
        ]

        result = tracker.update_with_detections(frame, detections)

        assert len(result) == 2
        assert result[0].track_id is not None
        assert result[1].track_id is not None
        assert result[0].track_id != result[1].track_id

    def test_interpolate(self) -> None:
        """補間テスト（Kalman予測のみ）"""
        tracker = LightweightTracker(use_optical_flow=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        tracker.update_with_detections(frame, [det])

        # 補間
        interpolated = tracker.interpolate(frame)

        assert len(interpolated) == 1
        track_id, bbox = interpolated[0]
        assert track_id == 1
        assert len(bbox) == 4

    def test_track_continuity(self) -> None:
        """トラックの連続性テスト"""
        tracker = LightweightTracker(use_optical_flow=False, iou_threshold=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # フレーム1: 初期検出
        det1 = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        tracker.update_with_detections(frame, [det1])
        first_id = det1.track_id

        # フレーム2: 少し移動した検出
        det2 = Detection(
            bbox=(110.0, 105.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(135.0, 205.0),
        )
        tracker.update_with_detections(frame, [det2])

        # 同じIDが維持されるはず
        assert det2.track_id == first_id

    def test_compute_iou(self) -> None:
        """IoU計算テスト"""
        # 完全一致
        iou = LightweightTracker._compute_iou(
            (0.0, 0.0, 100.0, 100.0),
            (0.0, 0.0, 100.0, 100.0),
        )
        assert iou == pytest.approx(1.0)

        # 重複なし
        iou = LightweightTracker._compute_iou(
            (0.0, 0.0, 100.0, 100.0),
            (200.0, 200.0, 100.0, 100.0),
        )
        assert iou == pytest.approx(0.0)

        # 50%重複
        iou = LightweightTracker._compute_iou(
            (0.0, 0.0, 100.0, 100.0),
            (50.0, 0.0, 100.0, 100.0),
        )
        assert 0.3 < iou < 0.4  # 50*100 / (100*100 + 100*100 - 50*100) = 0.333...

    def test_reset(self) -> None:
        """リセットテスト"""
        tracker = LightweightTracker(use_optical_flow=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        tracker.update_with_detections(frame, [det])

        assert len(tracker.tracks) == 1

        tracker.reset()

        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1


class TestOpticalFlowTracker:
    """OpticalFlowTrackerのテスト"""

    def test_init(self) -> None:
        """初期化テスト"""
        tracker = OpticalFlowTracker()

        assert tracker.max_corners == 100
        assert tracker.prev_gray is None

    def test_initialize(self) -> None:
        """初期化（検出から）テスト"""
        tracker = OpticalFlowTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        det.track_id = 1

        tracker.initialize(frame, [det])

        assert tracker.prev_gray is not None
        assert len(tracker.prev_track_ids) == 1

    def test_track_empty(self) -> None:
        """空の追跡テスト"""
        tracker = OpticalFlowTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = tracker.track(frame)

        assert len(result) == 0

    def test_reset(self) -> None:
        """リセットテスト"""
        tracker = OpticalFlowTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        det.track_id = 1
        tracker.initialize(frame, [det])

        tracker.reset()

        assert tracker.prev_gray is None
        assert len(tracker.prev_track_ids) == 0
