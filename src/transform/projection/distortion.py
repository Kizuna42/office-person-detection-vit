"""歪み補正モジュール。

レンズ歪みの補正と逆変換を提供します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.transform.projection.pinhole_model import CameraIntrinsics


class DistortionCorrector:
    """歪み補正クラス。

    OpenCV の歪み補正機能をラップし、単一点およびバッチ処理を提供。
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """初期化。

        Args:
            intrinsics: カメラ内部パラメータ
        """
        self._intrinsics = intrinsics
        self._K = intrinsics.K
        self._dist_coeffs = intrinsics.dist_coeffs
        self._has_distortion = intrinsics.has_distortion()

    def undistort_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """単一点の歪みを補正。

        Args:
            point: 歪んだ画像座標 (u, v)

        Returns:
            補正後の画像座標 (u', v')
        """
        if not self._has_distortion:
            return point

        src = np.array([[[point[0], point[1]]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, self._K, self._dist_coeffs, P=self._K)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """複数点の歪みを補正（バッチ処理）。

        Args:
            points: 歪んだ画像座標 (N, 2)

        Returns:
            補正後の画像座標 (N, 2)
        """
        if not self._has_distortion:
            return points.copy()

        if points.ndim == 1:
            points = points.reshape(1, 2)

        # OpenCV の入力形式に変換 (N, 1, 2)
        src = points.reshape(-1, 1, 2).astype(np.float64)
        dst = cv2.undistortPoints(src, self._K, self._dist_coeffs, P=self._K)
        return dst.reshape(-1, 2)

    def undistort_to_normalized(self, point: tuple[float, float]) -> tuple[float, float]:
        """歪み補正して正規化カメラ座標を返す。

        正規化座標は焦点距離 1 の仮想カメラでの座標。
        (x_n, y_n) where ray = (x_n, y_n, 1)

        Args:
            point: 歪んだ画像座標 (u, v)

        Returns:
            正規化カメラ座標 (x_n, y_n)
        """
        src = np.array([[[point[0], point[1]]]], dtype=np.float64)
        # P=None で正規化座標を返す
        dst = cv2.undistortPoints(src, self._K, self._dist_coeffs, P=None)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))

    def undistort_to_normalized_batch(self, points: np.ndarray) -> np.ndarray:
        """複数点を歪み補正して正規化カメラ座標を返す。

        Args:
            points: 歪んだ画像座標 (N, 2)

        Returns:
            正規化カメラ座標 (N, 2)
        """
        if points.ndim == 1:
            points = points.reshape(1, 2)

        src = points.reshape(-1, 1, 2).astype(np.float64)
        dst = cv2.undistortPoints(src, self._K, self._dist_coeffs, P=None)
        return dst.reshape(-1, 2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """画像全体の歪みを補正。

        Args:
            image: 入力画像 (H, W, C) or (H, W)

        Returns:
            補正後の画像
        """
        if not self._has_distortion:
            return image.copy()

        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self._K, self._dist_coeffs, (w, h), 1, (w, h))
        return cv2.undistort(image, self._K, self._dist_coeffs, None, new_K)
