"""Test cases for tracker module."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.data_models import Detection
from src.tracking.tracker import Track, TrackState, Tracker


@pytest.fixture()
def tracker() -> Tracker:
    """テスト用のTracker"""
    return Tracker(max_age=30, min_hits=3, iou_threshold=0.3)


@pytest.fixture()
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    detections = []
    for i in range(3):
        feat = np.random.randn(256).astype(np.float32)
        feat = feat / np.linalg.norm(feat)

        det = Detection(
            bbox=(100.0 + i * 10, 200.0 + i * 10, 50.0, 100.0),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(125.0 + i * 10, 300.0 + i * 10),
            floor_coords=(500.0 + i * 20, 600.0 + i * 20),
            features=feat,
        )
        detections.append(det)
    return detections


def test_track_creation():
    """Track作成のテスト"""
    track = Track(
        track_id=1,
        bbox=(100.0, 200.0, 50.0, 100.0),
        floor_coords=(500.0, 600.0),
        features=np.random.randn(256).astype(np.float32),
    )

    assert track.track_id == 1
    assert track.state == TrackState.TENTATIVE
    assert track.hits == 1
    assert track.time_since_update == 0
    assert len(track.trajectory) == 1


def test_track_update():
    """Track更新のテスト"""
    track = Track(
        track_id=1,
        bbox=(100.0, 200.0, 50.0, 100.0),
        floor_coords=(500.0, 600.0),
        features=np.random.randn(256).astype(np.float32),
    )

    new_detection = Detection(
        bbox=(110.0, 210.0, 55.0, 105.0),
        confidence=0.90,
        class_id=1,
        class_name="person",
        camera_coords=(137.5, 315.0),
        floor_coords=(520.0, 620.0),
        features=np.random.randn(256).astype(np.float32),
    )

    initial_hits = track.hits
    track.update(new_detection)

    assert track.hits == initial_hits + 1
    assert track.bbox == new_detection.bbox
    assert track.floor_coords == new_detection.floor_coords
    assert len(track.trajectory) == 2


def test_track_mark_missed():
    """Trackのマッチ失敗記録のテスト"""
    track = Track(
        track_id=1,
        bbox=(100.0, 200.0, 50.0, 100.0),
        floor_coords=(500.0, 600.0),
        features=None,
    )

    initial_time = track.time_since_update
    track.mark_missed()

    assert track.time_since_update == initial_time + 1


def test_tracker_initialization():
    """Tracker初期化のテスト"""
    tracker = Tracker(max_age=30, min_hits=3)

    assert tracker.max_age == 30
    assert tracker.min_hits == 3
    assert tracker.next_id == 1
    assert len(tracker.tracks) == 0


def test_tracker_update_first_frame(tracker: Tracker, sample_detections: list[Detection]):
    """最初のフレームでのTracker更新テスト"""
    tracked = tracker.update(sample_detections)

    # 最初のフレームでは全て新しいTrackが作成される
    assert len(tracker.tracks) == len(sample_detections)
    assert len(tracked) == len(sample_detections)

    # 各検出結果にtrack_idが割り当てられている
    for det in tracked:
        assert det.track_id is not None


def test_tracker_update_second_frame(tracker: Tracker, sample_detections: list[Detection]):
    """2フレーム目でのTracker更新テスト"""
    # 最初のフレーム
    tracker.update(sample_detections)

    # 2フレーム目（同じ検出結果）
    tracked = tracker.update(sample_detections)

    # Trackが更新されている
    assert len(tracker.tracks) == len(sample_detections)
    assert len(tracked) == len(sample_detections)

    # track_idが維持されている
    track_ids = {det.track_id for det in tracked}
    assert len(track_ids) == len(sample_detections)


def test_tracker_update_with_new_detections(tracker: Tracker, sample_detections: list[Detection]):
    """新しい検出結果が追加された場合のテスト"""
    # 最初のフレーム
    tracker.update(sample_detections)

    # 新しい検出結果を追加
    new_detection = Detection(
        bbox=(200.0, 300.0, 60.0, 120.0),
        confidence=0.90,
        class_id=1,
        class_name="person",
        camera_coords=(230.0, 420.0),
        floor_coords=(700.0, 800.0),
        features=np.random.randn(256).astype(np.float32),
    )
    new_detection.features = new_detection.features / np.linalg.norm(new_detection.features)

    updated_detections = sample_detections + [new_detection]
    tracked = tracker.update(updated_detections)

    # 新しいTrackが作成されている
    assert len(tracker.tracks) == len(updated_detections)
    assert len(tracked) == len(updated_detections)


def test_tracker_get_tracks(tracker: Tracker, sample_detections: list[Detection]):
    """Track取得のテスト"""
    tracker.update(sample_detections)

    tracks = tracker.get_tracks()

    assert len(tracks) == len(sample_detections)
    assert all(track.state != TrackState.DELETED for track in tracks)


def test_tracker_get_trajectories(tracker: Tracker, sample_detections: list[Detection]):
    """軌跡取得のテスト"""
    # 複数フレームで更新
    tracker.update(sample_detections)
    tracker.update(sample_detections)

    trajectories = tracker.get_trajectories()

    assert len(trajectories) == len(sample_detections)
    for track_id, trajectory in trajectories.items():
        assert len(trajectory) >= 1


def test_tracker_track_deletion(tracker: Tracker, sample_detections: list[Detection]):
    """Track削除のテスト"""
    tracker.update(sample_detections)

    initial_track_count = len(tracker.tracks)

    # 空の検出結果でmax_age回更新
    for _ in range(tracker.max_age + 1):
        tracker.update([])

    # Trackが削除されている
    assert len(tracker.tracks) < initial_track_count


def test_tracker_track_confirmation(tracker: Tracker, sample_detections: list[Detection]):
    """Track確定のテスト"""
    # min_hits回マッチング
    for _ in range(tracker.min_hits):
        tracker.update(sample_detections)

    # Trackが確定状態になっている
    tracks = tracker.get_tracks()
    assert all(track.state == TrackState.CONFIRMED for track in tracks)

