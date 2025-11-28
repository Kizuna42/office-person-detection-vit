"""レイキャストモジュール。

画像座標から床面座標への射影変換を提供します。
ピンホールカメラモデルに基づき、画像上の点から3Dレイを生成し、
床面（Z=0平面）との交点を計算します。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.transform.projection.distortion import DistortionCorrector

if TYPE_CHECKING:
    from src.transform.projection.pinhole_model import (
        CameraExtrinsics,
        CameraIntrinsics,
    )

logger = logging.getLogger(__name__)


class RayCaster:
    """レイキャストクラス。

    画像座標を床面座標（World XY平面）に投影する。
    """

    # 数値計算の許容誤差
    EPSILON = 1e-9

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ):
        """初期化。

        Args:
            intrinsics: カメラ内部パラメータ
            extrinsics: カメラ外部パラメータ
        """
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics

        # 歪み補正器
        self._distortion = DistortionCorrector(intrinsics)

        # パフォーマンス最適化のために逆行列をキャッシュ
        self._K_inv = intrinsics.K_inv
        self._R_inv = extrinsics.R_inv
        self._camera_center = extrinsics.camera_position_world.copy()

        logger.debug(f"RayCaster initialized: camera at {self._camera_center}, fx={intrinsics.fx}, fy={intrinsics.fy}")

    @property
    def camera_center(self) -> np.ndarray:
        """カメラ中心の World 座標を返す。

        Returns:
            カメラ位置 (3,) [meters]
        """
        return self._camera_center.copy()

    def image_to_floor(
        self,
        pixel: tuple[float, float],
        floor_z: float = 0.0,
    ) -> tuple[float, float] | None:
        """画像座標を床面座標に投影（単一点）。

        Args:
            pixel: 画像座標 (u, v)
            floor_z: 床面の高さ [meters]（デフォルト: 0.0）

        Returns:
            床面座標 (X, Y) [meters]、または None（交差しない場合）
        """
        u, v = pixel

        # 1. 歪み補正して正規化カメラ座標を取得
        x_n, y_n = self._distortion.undistort_to_normalized((u, v))

        # 2. カメラ座標系でのレイ方向
        ray_camera = np.array([x_n, y_n, 1.0], dtype=np.float64)

        # 3. World座標系でのレイ方向
        ray_world = self._R_inv @ ray_camera

        # 4. 床面との交差判定
        # Ray: P = C + s * ray_world
        # Plane: Z = floor_z
        # C_z + s * ray_world_z = floor_z
        # s = (floor_z - C_z) / ray_world_z

        ray_z = ray_world[2]
        if abs(ray_z) < self.EPSILON:
            # レイが床面と平行
            return None

        s = (floor_z - self._camera_center[2]) / ray_z

        if s < 0:
            # 交点がカメラの後方
            return None

        # 交点座標
        intersection = self._camera_center + s * ray_world

        return (float(intersection[0]), float(intersection[1]))

    def batch_image_to_floor(
        self,
        pixels: np.ndarray,
        floor_z: float = 0.0,
    ) -> np.ndarray:
        """画像座標を床面座標に投影（バッチ処理）。

        Args:
            pixels: 画像座標 (N, 2)
            floor_z: 床面の高さ [meters]

        Returns:
            床面座標 (N, 2) [meters]
            無効な点は NaN で埋められる
        """
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)

        n_points = pixels.shape[0]
        result = np.full((n_points, 2), np.nan, dtype=np.float64)

        # 1. 歪み補正して正規化座標を取得
        normalized = self._distortion.undistort_to_normalized_batch(pixels)

        # 2. カメラ座標系でのレイ方向 (N, 3)
        rays_camera = np.column_stack([normalized, np.ones(n_points)])

        # 3. World座標系でのレイ方向 (N, 3)
        rays_world = (self._R_inv @ rays_camera.T).T

        # 4. 床面との交差判定（ベクトル化）
        ray_z = rays_world[:, 2]

        # 平行でないレイを特定
        valid_mask = np.abs(ray_z) > self.EPSILON

        # s パラメータを計算
        s = np.zeros(n_points)
        s[valid_mask] = (floor_z - self._camera_center[2]) / ray_z[valid_mask]

        # s > 0 のレイのみ有効
        valid_mask &= s > 0

        # 交点を計算
        for i in range(n_points):
            if valid_mask[i]:
                intersection = self._camera_center + s[i] * rays_world[i]
                result[i, 0] = intersection[0]
                result[i, 1] = intersection[1]

        return result

    def floor_to_image(
        self,
        world_point: tuple[float, float, float],
    ) -> tuple[float, float] | None:
        """World座標を画像座標に投影。

        Args:
            world_point: World座標 (X, Y, Z) [meters]

        Returns:
            画像座標 (u, v)、または None（カメラの後方の場合）
        """
        P_world = np.array(world_point, dtype=np.float64)

        # Camera座標系へ変換
        P_camera = self._extrinsics.R @ P_world + self._extrinsics.t

        if P_camera[2] <= 0:
            # カメラの後方
            return None

        # 画像座標へ投影
        K = self._intrinsics.K
        p = K @ P_camera
        u = p[0] / p[2]
        v = p[1] / p[2]

        return (float(u), float(v))

    def get_foot_point(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """バウンディングボックスから足元座標を計算。

        Args:
            bbox: (x, y, width, height)

        Returns:
            足元の画像座標 (u, v)
        """
        x, y, w, h = bbox
        return (x + w / 2.0, y + h)

    def transform_detection(
        self,
        bbox: tuple[float, float, float, float],
        floor_z: float = 0.0,
    ) -> tuple[float, float] | None:
        """検出結果を床面座標に変換。

        Args:
            bbox: (x, y, width, height)
            floor_z: 床面の高さ [meters]

        Returns:
            床面座標 (X, Y) [meters]、または None
        """
        foot_point = self.get_foot_point(bbox)
        return self.image_to_floor(foot_point, floor_z)

    def is_above_horizon(self, pixel: tuple[float, float]) -> bool:
        """画像座標が地平線より上か判定。

        Args:
            pixel: 画像座標 (u, v)

        Returns:
            True if 地平線より上（床面と交差しない）
        """
        return self.image_to_floor(pixel) is None

    def get_horizon_v(self, u: float | None = None) -> float | None:
        """指定した u 座標での地平線の v 座標を計算。

        Args:
            u: 画像 X 座標（None の場合は画像中心）

        Returns:
            地平線の v 座標、または None（地平線が画像外の場合）
        """
        if u is None:
            u = self._intrinsics.cx

        # 二分探索で地平線を見つける
        v_min, v_max = 0.0, float(self._intrinsics.image_height)

        for _ in range(50):  # 精度 ~0.001 pixel
            v_mid = (v_min + v_max) / 2
            if self.image_to_floor((u, v_mid)) is None:
                v_min = v_mid
            else:
                v_max = v_mid

        return v_max if v_max < self._intrinsics.image_height else None
