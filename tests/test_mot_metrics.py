"""Unit tests for MOT metrics evaluation module."""

import numpy as np

from src.evaluation.mot_metrics import MOTMetrics
from src.models.data_models import Detection
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.track import Track


class TestMOTMetrics:
    """MOTMetricsのテスト"""

    def test_init(self):
        """初期化テスト"""
        metrics = MOTMetrics()
        assert metrics is not None

    def test_calculate_mota_empty_ground_truth(self):
        """空のGround TruthでのMOTA計算テスト"""
        metrics = MOTMetrics()
        predicted_tracks = []
        result = metrics.calculate_mota([], predicted_tracks, 100)
        assert result == 0.0

    def test_calculate_mota_empty_predicted(self):
        """空の予測トラックでのMOTA計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]
        predicted_tracks = []
        result = metrics.calculate_mota(gt_tracks, predicted_tracks, 100)
        assert 0.0 <= result <= 1.0

    def test_calculate_mota_perfect_match(self):
        """完全一致の場合のMOTA計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        # 予測トラックを作成（Ground Truthに近い位置）
        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        result = metrics.calculate_mota(gt_tracks, [track], 100)
        assert 0.0 <= result <= 1.0

    def test_calculate_mota_no_matches(self):
        """マッチングがない場合のMOTA計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        # 予測トラックを作成（Ground Truthから遠い位置）
        detection = Detection(
            bbox=(1000.0, 2000.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(1000.0, 2000.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([1000.0, 2000.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        result = metrics.calculate_mota(gt_tracks, [track], 100)
        assert 0.0 <= result <= 1.0

    def test_calculate_idf1_both_empty(self):
        """両方が空の場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        result = metrics.calculate_idf1([], [])
        assert result == 1.0

    def test_calculate_idf1_ground_truth_empty(self):
        """Ground Truthが空の場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        result = metrics.calculate_idf1([], [track])
        assert result == 0.0

    def test_calculate_idf1_predicted_empty(self):
        """予測トラックが空の場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]
        result = metrics.calculate_idf1(gt_tracks, [])
        assert result == 0.0

    def test_calculate_idf1_normal_case(self):
        """通常の場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}, {"trajectory": [{"x": 200, "y": 300}]}]

        detection1 = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf1 = KalmanFilter()
        kf1.init(np.array([100.0, 200.0]))
        track1 = Track(track_id=1, detection=detection1, kalman_filter=kf1)

        detection2 = Detection(
            bbox=(200.0, 300.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(200.0, 300.0),
        )
        kf2 = KalmanFilter()
        kf2.init(np.array([200.0, 300.0]))
        track2 = Track(track_id=2, detection=detection2, kalman_filter=kf2)

        result = metrics.calculate_idf1(gt_tracks, [track1, track2])
        assert 0.0 <= result <= 1.0

    def test_calculate_tracking_metrics(self):
        """追跡メトリクス一括計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        result = metrics.calculate_tracking_metrics(gt_tracks, [track], 100)

        assert "MOTA" in result
        assert "IDF1" in result
        assert "ID_Switches" in result
        assert 0.0 <= result["MOTA"] <= 1.0
        assert 0.0 <= result["IDF1"] <= 1.0
        assert result["ID_Switches"] >= 0.0

    def test_calculate_tracking_metrics_empty(self):
        """空のデータでの追跡メトリクス計算テスト"""
        metrics = MOTMetrics()
        result = metrics.calculate_tracking_metrics([], [], 100)

        assert "MOTA" in result
        assert "IDF1" in result
        assert "ID_Switches" in result

    def test_count_matches_with_trajectory(self):
        """軌跡がある場合のマッチング数計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        matches = metrics._count_matches(gt_tracks, [track])
        assert matches >= 0

    def test_count_matches_empty_trajectory(self):
        """空の軌跡でのマッチング数計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": []}]

        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        matches = metrics._count_matches(gt_tracks, [track])
        assert matches == 0

    def test_count_matches_far_distance(self):
        """距離が遠い場合のマッチング数計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        # 距離が50ピクセル以上離れている
        detection = Detection(
            bbox=(1000.0, 2000.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(1000.0, 2000.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([1000.0, 2000.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        matches = metrics._count_matches(gt_tracks, [track])
        assert matches == 0

    def test_count_id_matches(self):
        """IDマッチング数計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}, {"trajectory": [{"x": 200, "y": 300}]}]

        detection1 = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf1 = KalmanFilter()
        kf1.init(np.array([100.0, 200.0]))
        track1 = Track(track_id=1, detection=detection1, kalman_filter=kf1)

        matches = metrics._count_id_matches(gt_tracks, [track1])
        assert matches == min(len(gt_tracks), 1)

    def test_count_id_switches(self):
        """IDスイッチ数計算テスト"""
        metrics = MOTMetrics()

        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        id_switches = metrics._count_id_switches([track])
        assert id_switches == 0  # 簡易実装では常に0を返す

    def test_calculate_mota_multiple_tracks(self):
        """複数トラックでのMOTA計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [
            {"trajectory": [{"x": 100, "y": 200}]},
            {"trajectory": [{"x": 200, "y": 300}]},
            {"trajectory": [{"x": 300, "y": 400}]},
        ]

        tracks = []
        for i, gt_track in enumerate(gt_tracks):
            traj = gt_track["trajectory"][0]
            detection = Detection(
                bbox=(traj["x"], traj["y"], 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(traj["x"], traj["y"]),
            )
            kf = KalmanFilter()
            kf.init(np.array([traj["x"], traj["y"]]))
            track = Track(track_id=i + 1, detection=detection, kalman_filter=kf)
            tracks.append(track)

        result = metrics.calculate_mota(gt_tracks, tracks, 100)
        assert 0.0 <= result <= 1.0

    def test_calculate_idf1_more_gt_than_pred(self):
        """Ground Truthが予測より多い場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [
            {"trajectory": [{"x": 100, "y": 200}]},
            {"trajectory": [{"x": 200, "y": 300}]},
            {"trajectory": [{"x": 300, "y": 400}]},
        ]

        detection = Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(100.0, 200.0),
        )
        kf = KalmanFilter()
        kf.init(np.array([100.0, 200.0]))
        track = Track(track_id=1, detection=detection, kalman_filter=kf)

        result = metrics.calculate_idf1(gt_tracks, [track])
        assert 0.0 <= result <= 1.0

    def test_calculate_idf1_more_pred_than_gt(self):
        """予測がGround Truthより多い場合のIDF1計算テスト"""
        metrics = MOTMetrics()
        gt_tracks = [{"trajectory": [{"x": 100, "y": 200}]}]

        tracks = []
        for i in range(3):
            detection = Detection(
                bbox=(100.0 + i * 10, 200.0 + i * 10, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(100.0 + i * 10, 200.0 + i * 10),
            )
            kf = KalmanFilter()
            kf.init(np.array([100.0 + i * 10, 200.0 + i * 10]))
            track = Track(track_id=i + 1, detection=detection, kalman_filter=kf)
            tracks.append(track)

        result = metrics.calculate_idf1(gt_tracks, tracks)
        assert 0.0 <= result <= 1.0
