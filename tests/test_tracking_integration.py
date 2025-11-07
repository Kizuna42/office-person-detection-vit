"""Integration tests for tracking pipeline."""

import pytest
import numpy as np

from src.detection.vit_detector import ViTDetector
from src.models.data_models import Detection
from src.tracking.tracker import Tracker
from src.transform.coordinate_transformer import CoordinateTransformer
from src.visualization.floormap_visualizer import FloormapVisualizer


class TestTrackingPipeline:
    """追跡パイプラインの統合テスト"""

    def test_tracking_with_detections(self):
        """検出結果を使った追跡テスト"""
        tracker = Tracker(max_age=30, min_hits=3)

        # フレーム1の検出結果
        detections_frame1 = [
            Detection(
                bbox=(100.0, 100.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125.0, 200.0),
            ),
        ]

        # 特徴量を追加（模擬）
        for det in detections_frame1:
            det.features = np.random.rand(256).astype(np.float32)
            det.features = det.features / np.linalg.norm(det.features)

        result1 = tracker.update(detections_frame1)
        assert len(result1) == 1
        assert result1[0].track_id == 1

        # フレーム2の検出結果（同じ人物が少し移動）
        detections_frame2 = [
            Detection(
                bbox=(105.0, 105.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(130.0, 205.0),
            ),
        ]

        # 特徴量を追加（同じ特徴量を使用）
        for det in detections_frame2:
            det.features = detections_frame1[0].features.copy()

        result2 = tracker.update(detections_frame2)
        assert len(result2) == 1
        assert result2[0].track_id == 1  # 同じIDが割り当てられる

    def test_tracking_multiple_objects(self):
        """複数オブジェクトの追跡テスト"""
        tracker = Tracker()

        # フレーム1: 2人の検出
        detections1 = [
            Detection(
                bbox=(100.0, 100.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125.0, 200.0),
            ),
            Detection(
                bbox=(200.0, 200.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(225.0, 300.0),
            ),
        ]

        for det in detections1:
            det.features = np.random.rand(256).astype(np.float32)
            det.features = det.features / np.linalg.norm(det.features)

        result1 = tracker.update(detections1)
        assert len(result1) == 2
        assert result1[0].track_id != result1[1].track_id

        # フレーム2: 同じ2人が移動
        detections2 = [
            Detection(
                bbox=(105.0, 105.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(130.0, 205.0),
            ),
            Detection(
                bbox=(205.0, 205.0, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(230.0, 305.0),
            ),
        ]

        for i, det in enumerate(detections2):
            det.features = detections1[i].features.copy()

        result2 = tracker.update(detections2)
        assert len(result2) == 2
        # IDが維持されているか確認
        assert result2[0].track_id in [result1[0].track_id, result1[1].track_id]
        assert result2[1].track_id in [result1[0].track_id, result1[1].track_id]

    def test_tracking_with_coordinate_transformation(self, tmp_path):
        """座標変換を含む追跡テスト"""
        # ホモグラフィ行列（単位行列）
        homography_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        transformer = CoordinateTransformer(
            homography_matrix,
            floormap_config={
                "image_width": 1000,
                "image_height": 1000,
                "image_origin_x": 0,
                "image_origin_y": 0,
            },
        )

        tracker = Tracker()

        # 検出結果を作成
        detection = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        detection.features = np.random.rand(256).astype(np.float32)
        detection.features = detection.features / np.linalg.norm(detection.features)

        # 座標変換
        floor_coords = transformer.transform(detection.camera_coords)
        detection.floor_coords = floor_coords

        # 追跡
        result = tracker.update([detection])
        assert len(result) == 1
        assert result[0].floor_coords is not None

    def test_tracking_visualization(self, tmp_path):
        """追跡可視化のテスト"""
        # フロアマップ画像を作成（ダミー）
        floormap = np.zeros((100, 100, 3), dtype=np.uint8)

        # フロアマップ画像を保存
        import cv2

        floormap_path = tmp_path / "floormap.png"
        cv2.imwrite(str(floormap_path), floormap)

        visualizer = FloormapVisualizer(
            floormap_path=str(floormap_path),
            floormap_config={"image_width": 100, "image_height": 100},
            zones=[],
        )

        # トラックデータを作成
        tracks = [
            {
                "track_id": 1,
                "trajectory": [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0)],
            },
            {
                "track_id": 2,
                "trajectory": [(50.0, 50.0), (55.0, 55.0)],
            },
        ]

        # 軌跡を描画
        image = floormap.copy()
        result_image = visualizer.draw_trajectories(image, tracks)
        assert result_image is not None
        assert result_image.shape == floormap.shape
