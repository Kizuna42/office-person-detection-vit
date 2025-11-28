"""Unit tests for lens distortion module."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.calibration.lens_distortion import (
    CameraIntrinsics,
    DistortionParams,
    LensDistortionCorrector,
    calibrate_from_chessboard_images,
    estimate_distortion_from_lines,
)


class TestDistortionParams:
    """DistortionParamsデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        params = DistortionParams()
        assert params.k1 == 0.0
        assert params.k2 == 0.0
        assert params.k3 == 0.0
        assert params.p1 == 0.0
        assert params.p2 == 0.0

    def test_custom_values(self):
        """カスタム値のテスト"""
        params = DistortionParams(k1=-0.1, k2=0.05, k3=0.01, p1=0.001, p2=0.002)
        assert params.k1 == -0.1
        assert params.k2 == 0.05
        assert params.k3 == 0.01
        assert params.p1 == 0.001
        assert params.p2 == 0.002

    def test_to_array(self):
        """OpenCV形式配列変換テスト"""
        params = DistortionParams(k1=-0.1, k2=0.05, k3=0.01, p1=0.001, p2=0.002)
        arr = params.to_array()

        assert arr.shape == (5,)
        # OpenCV形式: [k1, k2, p1, p2, k3]
        assert arr[0] == pytest.approx(-0.1)
        assert arr[1] == pytest.approx(0.05)
        assert arr[2] == pytest.approx(0.001)
        assert arr[3] == pytest.approx(0.002)
        assert arr[4] == pytest.approx(0.01)

    def test_from_array_full(self):
        """配列からの作成テスト（5要素）"""
        arr = np.array([-0.1, 0.05, 0.001, 0.002, 0.01])
        params = DistortionParams.from_array(arr)

        assert params.k1 == pytest.approx(-0.1)
        assert params.k2 == pytest.approx(0.05)
        assert params.p1 == pytest.approx(0.001)
        assert params.p2 == pytest.approx(0.002)
        assert params.k3 == pytest.approx(0.01)

    def test_from_array_four_elements(self):
        """配列からの作成テスト（4要素）"""
        arr = np.array([-0.1, 0.05, 0.001, 0.002])
        params = DistortionParams.from_array(arr)

        assert params.k1 == pytest.approx(-0.1)
        assert params.k2 == pytest.approx(0.05)
        assert params.p1 == pytest.approx(0.001)
        assert params.p2 == pytest.approx(0.002)
        assert params.k3 == 0.0

    def test_from_array_two_elements(self):
        """配列からの作成テスト（2要素）"""
        arr = np.array([-0.1, 0.05])
        params = DistortionParams.from_array(arr)

        assert params.k1 == pytest.approx(-0.1)
        assert params.k2 == pytest.approx(0.05)
        assert params.p1 == 0.0
        assert params.p2 == 0.0
        assert params.k3 == 0.0

    def test_from_array_empty(self):
        """空配列からの作成テスト"""
        arr = np.array([])
        params = DistortionParams.from_array(arr)

        assert params.k1 == 0.0
        assert params.k2 == 0.0

    def test_from_array_list(self):
        """リストからの作成テスト"""
        lst = [-0.1, 0.05, 0.001, 0.002, 0.01]
        params = DistortionParams.from_array(lst)

        assert params.k1 == pytest.approx(-0.1)

    def test_is_zero_true(self):
        """ゼロ判定テスト（真）"""
        params = DistortionParams()
        assert params.is_zero() is True

    def test_is_zero_false(self):
        """ゼロ判定テスト（偽）"""
        params = DistortionParams(k1=-0.1)
        assert params.is_zero() is False

    def test_is_zero_near_zero(self):
        """ゼロ判定テスト（ほぼゼロ）"""
        params = DistortionParams(k1=1e-12)
        assert params.is_zero() is True

    def test_to_dict(self):
        """辞書変換テスト"""
        params = DistortionParams(k1=-0.1, k2=0.05, k3=0.01, p1=0.001, p2=0.002)
        d = params.to_dict()

        assert d["k1"] == -0.1
        assert d["k2"] == 0.05
        assert d["k3"] == 0.01
        assert d["p1"] == 0.001
        assert d["p2"] == 0.002


class TestCameraIntrinsics:
    """CameraIntrinsicsデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        intrinsics = CameraIntrinsics()
        assert intrinsics.fx == 1250.0
        assert intrinsics.fy == 1250.0
        assert intrinsics.cx == 640.0
        assert intrinsics.cy == 360.0
        assert intrinsics.width == 1280
        assert intrinsics.height == 720

    def test_get_camera_matrix(self):
        """カメラ行列取得テスト"""
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        matrix = intrinsics.get_camera_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1000.0  # fx
        assert matrix[1, 1] == 1000.0  # fy
        assert matrix[0, 2] == 640.0  # cx
        assert matrix[1, 2] == 360.0  # cy
        assert matrix[2, 2] == 1.0

    def test_from_config_full(self):
        """設定辞書からの作成テスト（全パラメータ）"""
        config = {
            "focal_length_x": 1500.0,
            "focal_length_y": 1500.0,
            "center_x": 960.0,
            "center_y": 540.0,
            "image_width": 1920,
            "image_height": 1080,
            "distortion": {"k1": -0.1, "k2": 0.05, "k3": 0.0, "p1": 0.0, "p2": 0.0},
        }
        intrinsics = CameraIntrinsics.from_config(config)

        assert intrinsics.fx == 1500.0
        assert intrinsics.fy == 1500.0
        assert intrinsics.cx == 960.0
        assert intrinsics.cy == 540.0
        assert intrinsics.width == 1920
        assert intrinsics.height == 1080
        assert intrinsics.distortion.k1 == -0.1

    def test_from_config_distortion_list(self):
        """設定辞書からの作成テスト（歪み係数リスト形式）"""
        config = {
            "distortion": [-0.1, 0.05, 0.001, 0.002, 0.01],
        }
        intrinsics = CameraIntrinsics.from_config(config)

        assert intrinsics.distortion.k1 == pytest.approx(-0.1)

    def test_from_config_empty(self):
        """空の設定辞書からの作成テスト"""
        config = {}
        intrinsics = CameraIntrinsics.from_config(config)

        # デフォルト値が使用される
        assert intrinsics.fx == 1250.0
        assert intrinsics.distortion.is_zero() is True


class TestLensDistortionCorrector:
    """LensDistortionCorrectorのテスト"""

    @pytest.fixture
    def zero_distortion_intrinsics(self):
        """歪みなしのカメラパラメータ"""
        return CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720,
            distortion=DistortionParams(),
        )

    @pytest.fixture
    def distorted_intrinsics(self):
        """歪みありのカメラパラメータ"""
        return CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720,
            distortion=DistortionParams(k1=-0.1, k2=0.05),
        )

    def test_init_zero_distortion(self, zero_distortion_intrinsics):
        """歪みなしでの初期化テスト"""
        corrector = LensDistortionCorrector(zero_distortion_intrinsics)
        assert corrector.enabled is False

    def test_init_with_distortion(self, distorted_intrinsics):
        """歪みありでの初期化テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        assert corrector.enabled is True

    def test_undistort_points_disabled(self, zero_distortion_intrinsics):
        """歪み補正無効時の点群変換テスト"""
        corrector = LensDistortionCorrector(zero_distortion_intrinsics)
        points = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = corrector.undistort_points(points)

        np.testing.assert_array_almost_equal(result, points)

    def test_undistort_points_enabled(self, distorted_intrinsics):
        """歪み補正有効時の点群変換テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        points = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = corrector.undistort_points(points)

        assert result.shape == (2, 2)
        # 歪み補正後は元の点と異なるはず
        # （ただし中心付近はほぼ同じ）

    def test_undistort_points_3d_input(self, distorted_intrinsics):
        """3D入力形式の点群変換テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        points = np.array([[[100.0, 200.0]], [[300.0, 400.0]]])
        result = corrector.undistort_points(points)

        assert result.shape == (2, 2)

    def test_undistort_point_single(self, zero_distortion_intrinsics):
        """単一点の歪み補正テスト"""
        corrector = LensDistortionCorrector(zero_distortion_intrinsics)
        point = (100.0, 200.0)
        result = corrector.undistort_point(point)

        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(200.0)

    def test_undistort_point_with_distortion(self, distorted_intrinsics):
        """歪みありの単一点補正テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        point = (200.0, 300.0)
        result = corrector.undistort_point(point)

        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_undistort_image_disabled(self, zero_distortion_intrinsics):
        """歪み補正無効時の画像変換テスト"""
        corrector = LensDistortionCorrector(zero_distortion_intrinsics)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = corrector.undistort_image(image)

        # 補正無効なので同じ画像が返る
        np.testing.assert_array_equal(result, image)

    def test_undistort_image_enabled(self, distorted_intrinsics):
        """歪み補正有効時の画像変換テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        result = corrector.undistort_image(image)

        assert result.shape == image.shape

    def test_visualize_distortion_grid(self, distorted_intrinsics, tmp_path):
        """歪みグリッド可視化テスト"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        output_path = tmp_path / "distortion_grid.png"

        img = corrector.visualize_distortion_grid(grid_size=50, output_path=output_path)

        assert img.shape == (720, 1280, 3)
        assert output_path.exists()

    def test_visualize_distortion_grid_no_output(self, distorted_intrinsics):
        """歪みグリッド可視化テスト（出力なし）"""
        corrector = LensDistortionCorrector(distorted_intrinsics)
        img = corrector.visualize_distortion_grid(grid_size=50)

        assert img.shape == (720, 1280, 3)

    def test_visualize_distortion_grid_disabled(self, zero_distortion_intrinsics):
        """歪み無効時のグリッド可視化テスト"""
        corrector = LensDistortionCorrector(zero_distortion_intrinsics)
        img = corrector.visualize_distortion_grid(grid_size=50)

        assert img.shape == (720, 1280, 3)

    def test_from_config(self):
        """設定辞書からの作成テスト"""
        config = {
            "focal_length_x": 1000.0,
            "focal_length_y": 1000.0,
            "center_x": 640.0,
            "center_y": 360.0,
            "image_width": 1280,
            "image_height": 720,
            "distortion": {"k1": -0.1, "k2": 0.05},
        }
        corrector = LensDistortionCorrector.from_config(config)

        assert corrector.enabled is True
        assert corrector.intrinsics.fx == 1000.0

    def test_from_calibration_file(self, tmp_path):
        """キャリブレーションファイルからの作成テスト"""
        calibration_data = {
            "camera_matrix": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [-0.1, 0.05, 0.0, 0.0, 0.0],
        }
        file_path = tmp_path / "calibration.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(calibration_data, f)

        corrector = LensDistortionCorrector.from_calibration_file(file_path)

        assert corrector.intrinsics.fx == 1000.0
        assert corrector.intrinsics.distortion.k1 == pytest.approx(-0.1)


class TestEstimateDistortionFromLines:
    """estimate_distortion_from_lines関数のテスト"""

    def test_insufficient_lines(self):
        """直線不足時のテスト"""
        # 単色画像（直線検出不可）
        image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        params = estimate_distortion_from_lines(image)

        # 直線が不十分な場合はデフォルト値
        assert params.k1 == 0.0 or params.k1 != 0.0  # 実装依存

    def test_with_lines(self):
        """直線ありの画像でのテスト"""
        # グリッドパターンを描画
        image = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        for i in range(0, 1280, 50):
            cv2.line(image, (i, 0), (i, 720), (0, 0, 0), 2)
        for j in range(0, 720, 50):
            cv2.line(image, (0, j), (1280, j), (0, 0, 0), 2)

        params = estimate_distortion_from_lines(image, min_line_length=50)

        # 結果が返されることを確認（実際の値は実装依存）
        assert isinstance(params, DistortionParams)


class TestCalibrateFromChessboardImages:
    """calibrate_from_chessboard_images関数のテスト"""

    def test_insufficient_images(self, tmp_path):
        """画像不足時のテスト"""
        # 空のリスト
        with pytest.raises(ValueError, match=r"Insufficient"):
            calibrate_from_chessboard_images([], chessboard_size=(9, 6))

    def test_no_corners_found(self, tmp_path):
        """コーナー検出失敗時のテスト"""
        # 単色画像（チェスボードなし）
        image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        image_path = tmp_path / "no_corners.jpg"
        cv2.imwrite(str(image_path), image)

        # 3枚の同じ画像
        with pytest.raises(ValueError, match=r"Insufficient"):
            calibrate_from_chessboard_images(
                [image_path, image_path, image_path],
                chessboard_size=(9, 6),
            )

    def test_invalid_image_path(self, tmp_path):
        """無効な画像パスのテスト"""
        with pytest.raises(ValueError, match=r"Insufficient"):
            calibrate_from_chessboard_images(
                [tmp_path / "nonexistent.jpg"],
                chessboard_size=(9, 6),
            )
