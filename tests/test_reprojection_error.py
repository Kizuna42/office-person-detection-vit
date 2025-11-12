"""Unit tests for reprojection error evaluation module."""

import numpy as np
import pytest

from src.calibration import ReprojectionErrorEvaluator


class TestReprojectionErrorEvaluator:
    """ReprojectionErrorEvaluatorのテスト"""

    def test_init(self):
        """初期化テスト"""
        evaluator = ReprojectionErrorEvaluator()
        assert evaluator.camera_matrix is None
        assert evaluator.dist_coeffs is None

    def test_init_with_params(self):
        """パラメータ付き初期化テスト"""
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        evaluator = ReprojectionErrorEvaluator(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        assert evaluator.camera_matrix is not None
        assert evaluator.dist_coeffs is not None

    def test_evaluate_homography_empty_points(self):
        """空の点リストでの評価テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        result = evaluator.evaluate_homography([], [], H)

        assert result["mean_error"] == 0.0
        assert result["max_error"] == 0.0
        assert result["min_error"] == 0.0
        assert result["std_error"] == 0.0
        assert result["errors"] == []

    def test_evaluate_homography_mismatched_points(self):
        """点の数が一致しない場合のテスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        src_points = [(0, 0), (1, 1)]
        dst_points = [(0, 0)]

        with pytest.raises(ValueError, match="点の数が一致しません"):
            evaluator.evaluate_homography(src_points, dst_points, H)

    def test_evaluate_homography_identity_matrix(self):
        """単位行列での評価テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        src_points = [(0, 0), (10, 20), (100, 200)]
        dst_points = [(0, 0), (10, 20), (100, 200)]

        result = evaluator.evaluate_homography(src_points, dst_points, H)

        assert result["mean_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["min_error"] == pytest.approx(0.0, abs=1e-6)
        assert len(result["errors"]) == 3

    def test_evaluate_homography_with_error(self):
        """誤差がある場合の評価テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        src_points = [(0, 0), (10, 20)]
        dst_points = [(1, 1), (11, 21)]  # 1ピクセルずつずれている

        result = evaluator.evaluate_homography(src_points, dst_points, H)

        assert result["mean_error"] > 0.0
        assert result["max_error"] > 0.0
        assert result["min_error"] > 0.0
        assert len(result["errors"]) == 2
        # 各点の誤差は約√2（1ピクセルずつx, y方向にずれているため）
        assert all(error > 0.0 for error in result["errors"])

    def test_evaluate_homography_custom_matrix(self):
        """カスタムホモグラフィ行列での評価テスト"""
        evaluator = ReprojectionErrorEvaluator()
        # 2倍スケールの変換行列
        H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)

        src_points = [(10, 20), (30, 40)]
        dst_points = [(20, 40), (60, 80)]  # 2倍に変換された座標

        result = evaluator.evaluate_homography(src_points, dst_points, H)

        assert result["mean_error"] == pytest.approx(0.0, abs=1e-6)
        assert len(result["errors"]) == 2

    def test_evaluate_camera_calibration_no_params(self):
        """パラメータなしでのカメラキャリブレーション評価テスト"""
        evaluator = ReprojectionErrorEvaluator()

        object_points = [np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)]
        image_points = [np.array([[100, 100], [200, 100]], dtype=np.float32)]

        with pytest.raises(ValueError, match="カメラ行列または歪み係数が設定されていません"):
            evaluator.evaluate_camera_calibration(object_points, image_points)

    def test_evaluate_camera_calibration_with_params(self):
        """パラメータ付きでのカメラキャリブレーション評価テスト"""
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        evaluator = ReprojectionErrorEvaluator(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

        # 簡単なテストケース
        object_points = [np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)]
        image_points = [np.array([[320, 240], [400, 240]], dtype=np.float32)]

        result = evaluator.evaluate_camera_calibration(object_points, image_points)

        assert "mean_error" in result
        assert "max_error" in result
        assert "min_error" in result
        assert "std_error" in result
        assert "per_image_errors" in result
        assert len(result["per_image_errors"]) == 1

    def test_evaluate_camera_calibration_empty_points(self):
        """空の点リストでのカメラキャリブレーション評価テスト"""
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        evaluator = ReprojectionErrorEvaluator(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

        result = evaluator.evaluate_camera_calibration([], [])

        assert result["mean_error"] == 0.0
        assert result["max_error"] == 0.0
        assert result["min_error"] == 0.0
        assert result["std_error"] == 0.0
        assert result["per_image_errors"] == []

    def test_create_error_map(self):
        """誤差マップ生成テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        src_points = [(10, 20), (30, 40)]
        dst_points = [(11, 21), (31, 41)]  # 1ピクセルずつずれている
        image_shape = (100, 200)

        error_map = evaluator.create_error_map(src_points, dst_points, H, image_shape)

        assert error_map.shape == image_shape
        assert error_map.dtype == np.float32
        assert np.max(error_map) > 0.0

    def test_create_error_map_empty_points(self):
        """空の点リストでの誤差マップ生成テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)
        image_shape = (100, 200)

        error_map = evaluator.create_error_map([], [], H, image_shape)

        assert error_map.shape == image_shape
        assert np.max(error_map) == 0.0

    def test_create_error_map_custom_shape(self):
        """カスタム形状での誤差マップ生成テスト"""
        evaluator = ReprojectionErrorEvaluator()
        H = np.eye(3, dtype=np.float64)

        src_points = [(50, 50)]
        dst_points = [(50, 50)]
        image_shape = (480, 640)

        error_map = evaluator.create_error_map(src_points, dst_points, H, image_shape)

        assert error_map.shape == image_shape
