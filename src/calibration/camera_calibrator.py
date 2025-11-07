"""Camera calibration module for estimating camera intrinsics and distortion coefficients."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """カメラキャリブレーションクラス

    チェスボード画像からカメラ内部パラメータと歪み係数を推定します。
    """

    def __init__(self, chessboard_size: tuple[int, int] = (9, 6)):
        """CameraCalibratorを初期化

        Args:
            chessboard_size: チェスボードの内部コーナー数 (width, height)
        """
        self.chessboard_size = chessboard_size
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.calibrated = False

        logger.info(f"CameraCalibrator initialized with chessboard size: {chessboard_size}")

    def calibrate_from_images(self, image_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        """複数のチェスボード画像からキャリブレーションを実行

        Args:
            image_paths: チェスボード画像のパスリスト

        Returns:
            (カメラ行列, 歪み係数)

        Raises:
            ValueError: 十分な画像が提供されない場合
        """
        objpoints = []  # 3D点（実世界座標）
        imgpoints = []  # 2D点（画像座標）

        # チェスボードの3D座標を生成
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.chessboard_size[0], 0 : self.chessboard_size[1]].T.reshape(-1, 2)

        # 各画像からコーナーを検出
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                logger.debug(f"Found corners in: {img_path}")
            else:
                logger.warning(f"No corners found in: {img_path}")

        if len(objpoints) < 3:
            raise ValueError(f"Insufficient images with detected corners: {len(objpoints)} < 3")

        # キャリブレーション実行
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibrated = True

        logger.info(f"Calibration completed with {len(objpoints)} images")
        logger.info(f"Camera matrix:\n{camera_matrix}")
        logger.info(f"Distortion coefficients: {dist_coeffs}")

        return camera_matrix, dist_coeffs

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """画像の歪みを補正

        Args:
            image: 入力画像

        Returns:
            歪み補正された画像

        Raises:
            RuntimeError: キャリブレーションが完了していない場合
        """
        if not self.calibrated:
            raise RuntimeError("Camera not calibrated. Call calibrate_from_images() first.")

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return dst

    def get_camera_matrix(self) -> np.ndarray | None:
        """カメラ行列を取得

        Returns:
            カメラ行列（キャリブレーション未完了時はNone）
        """
        return self.camera_matrix

    def get_distortion_coefficients(self) -> np.ndarray | None:
        """歪み係数を取得

        Returns:
            歪み係数（キャリブレーション未完了時はNone）
        """
        return self.dist_coeffs
