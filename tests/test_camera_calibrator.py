"""Unit tests for camera calibration module."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from src.calibration.camera_calibrator import CameraCalibrator


class TestCameraCalibrator:
    """CameraCalibratorのテスト"""

    def test_init(self):
        """初期化テスト"""
        calibrator = CameraCalibrator(chessboard_size=(9, 6))
        assert calibrator.chessboard_size == (9, 6)
        assert calibrator.camera_matrix is None
        assert calibrator.dist_coeffs is None
        assert calibrator.calibrated is False

    def test_init_default_size(self):
        """デフォルトサイズでの初期化テスト"""
        calibrator = CameraCalibrator()
        assert calibrator.chessboard_size == (9, 6)

    def test_get_camera_matrix_before_calibration(self):
        """キャリブレーション前のカメラ行列取得テスト"""
        calibrator = CameraCalibrator()
        assert calibrator.get_camera_matrix() is None

    def test_get_distortion_coefficients_before_calibration(self):
        """キャリブレーション前の歪み係数取得テスト"""
        calibrator = CameraCalibrator()
        assert calibrator.get_distortion_coefficients() is None

    def test_undistort_image_before_calibration(self):
        """キャリブレーション前の歪み補正テスト"""
        calibrator = CameraCalibrator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="Camera not calibrated"):
            calibrator.undistort_image(image)

    def test_calibrate_from_images_insufficient_images(self):
        """画像数不足のテスト"""
        calibrator = CameraCalibrator()

        # 空のリスト
        with pytest.raises(ValueError, match="Insufficient images"):
            calibrator.calibrate_from_images([])

        # 2枚のみ（3枚以上必要）
        with tempfile.TemporaryDirectory() as tmpdir:
            # ダミー画像を作成（チェスボードではない）
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)

            path1 = Path(tmpdir) / "img1.jpg"
            path2 = Path(tmpdir) / "img2.jpg"
            cv2.imwrite(str(path1), img1)
            cv2.imwrite(str(path2), img2)

            with pytest.raises(ValueError, match="Insufficient images"):
                calibrator.calibrate_from_images([str(path1), str(path2)])

    def test_calibrate_from_images_invalid_path(self):
        """無効なパスのテスト"""
        calibrator = CameraCalibrator()

        with pytest.raises(ValueError, match="Insufficient images"):
            calibrator.calibrate_from_images(["nonexistent.jpg"])

    def test_undistort_image_after_calibration(self):
        """キャリブレーション後の歪み補正テスト"""
        calibrator = CameraCalibrator(chessboard_size=(9, 6))

        # ダミーのカメラ行列と歪み係数を設定
        calibrator.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        calibrator.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        calibrator.calibrated = True

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        undistorted = calibrator.undistort_image(image)

        assert undistorted is not None
        assert undistorted.shape == image.shape

    def test_get_camera_matrix_after_calibration(self):
        """キャリブレーション後のカメラ行列取得テスト"""
        calibrator = CameraCalibrator()
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        calibrator.camera_matrix = camera_matrix
        calibrator.calibrated = True

        result = calibrator.get_camera_matrix()
        assert result is not None
        assert np.array_equal(result, camera_matrix)

    def test_get_distortion_coefficients_after_calibration(self):
        """キャリブレーション後の歪み係数取得テスト"""
        calibrator = CameraCalibrator()
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        calibrator.dist_coeffs = dist_coeffs
        calibrator.calibrated = True

        result = calibrator.get_distortion_coefficients()
        assert result is not None
        assert np.array_equal(result, dist_coeffs)
