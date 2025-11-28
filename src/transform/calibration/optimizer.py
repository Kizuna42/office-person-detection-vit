"""キャリブレーション最適化モジュール。

対応点データからカメラ外部パラメータを推定し、
Levenberg-Marquardt 法で残差を最小化します。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from src.transform.calibration.correspondence import (
        CorrespondenceData,
    )
    from src.transform.projection.pinhole_model import (
        CameraExtrinsics,
        CameraIntrinsics,
    )

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """キャリブレーション結果。

    Attributes:
        extrinsics: 推定された外部パラメータ
        reprojection_error: 再投影誤差 RMSE [pixels]
        inlier_ratio: インライアー比率 (0-1)
        optimization_converged: 最適化が収束したか
        iterations: 反復回数
        residuals: 残差ベクトル
    """

    extrinsics: CameraExtrinsics
    reprojection_error: float
    inlier_ratio: float
    optimization_converged: bool
    iterations: int = 0
    residuals: np.ndarray | None = None


class CorrespondenceCalibrator:
    """対応点からのキャリブレーションクラス。

    線分-点対応データを使用してカメラ外部パラメータを推定。
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        floormap_scale_x_mm_per_px: float = 28.1926406926406,
        floormap_scale_y_mm_per_px: float = 28.241430700447,
    ):
        """初期化。

        Args:
            intrinsics: カメラ内部パラメータ
            floormap_scale_x_mm_per_px: フロアマップX軸スケール [mm/pixel]
            floormap_scale_y_mm_per_px: フロアマップY軸スケール [mm/pixel]
        """
        self._intrinsics = intrinsics
        self._K = intrinsics.K
        self._dist_coeffs = intrinsics.dist_coeffs
        self._scale_x = floormap_scale_x_mm_per_px / 1000.0  # m/pixel
        self._scale_y = floormap_scale_y_mm_per_px / 1000.0  # m/pixel

    def calibrate_from_correspondences(
        self,
        correspondence_data: CorrespondenceData,
        initial_guess: CameraExtrinsics | None = None,
        camera_position_px: tuple[float, float] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> CalibrationResult:
        """対応点データからキャリブレーションを実行。

        Args:
            correspondence_data: 対応点データ
            initial_guess: 初期推定値（None の場合は自動推定）
            camera_position_px: カメラ位置（フロアマップ座標）
            max_iterations: 最大反復回数
            tolerance: 収束許容誤差

        Returns:
            CalibrationResult インスタンス
        """
        # 足元点の対応を取得
        foot_correspondences = correspondence_data.get_foot_points()

        if len(foot_correspondences) < 4:
            raise ValueError(f"At least 4 correspondences required, got {len(foot_correspondences)}")

        return self.calibrate_from_point_pairs(
            point_pairs=foot_correspondences,
            initial_guess=initial_guess,
            camera_position_px=camera_position_px,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def calibrate_from_point_pairs(
        self,
        point_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
        initial_guess: CameraExtrinsics | None = None,
        camera_position_px: tuple[float, float] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> CalibrationResult:
        """点-点対応からキャリブレーションを実行。

        Args:
            point_pairs: [(image_point, floormap_point), ...] のリスト
            initial_guess: 初期推定値
            camera_position_px: カメラ位置（フロアマップ座標）
            max_iterations: 最大反復回数
            tolerance: 収束許容誤差

        Returns:
            CalibrationResult インスタンス
        """

        # 画像点とフロアマップ点を分離
        image_points = np.array([p[0] for p in point_pairs], dtype=np.float64)
        floormap_points = np.array([p[1] for p in point_pairs], dtype=np.float64)

        # フロアマップ座標をWorld座標（メートル）に変換
        if camera_position_px is None:
            # カメラ位置が不明な場合、フロアマップ点の重心を使用
            camera_position_px = (
                float(np.mean(floormap_points[:, 0])),
                float(np.mean(floormap_points[:, 1])),
            )

        world_points = self._floormap_to_world(floormap_points, camera_position_px)

        # 初期推定
        if initial_guess is not None:
            # 明示的な初期値が渡された場合はそれを使用
            initial_params = self._extrinsics_to_params(initial_guess)
            logger.info(
                f"Using provided initial guess: height={initial_params[0]:.2f}m, "
                f"pitch={initial_params[1]:.1f}deg, yaw={initial_params[2]:.1f}deg"
            )
        else:
            # solvePnP で推定を試みる
            initial_params = self._estimate_initial_params(image_points, world_points, camera_position_px)

        # 初期残差を確認
        initial_residuals = self._residual_function(initial_params, image_points, world_points)
        if not np.all(np.isfinite(initial_residuals)):
            logger.warning("Initial residuals not finite, trying fallback parameters")
            # フォールバック：デフォルト値を使用
            initial_params = np.array([2.2, 45.0, 0.0, 0.0, 0.0, 0.0])
            initial_residuals = self._residual_function(initial_params, image_points, world_points)
            if not np.all(np.isfinite(initial_residuals)):
                raise ValueError(
                    "Residuals are not finite even with default parameters. "
                    "Check correspondence points and camera position."
                )

        initial_rmse = np.sqrt(np.mean(initial_residuals**2))
        logger.info(f"Initial RMSE: {initial_rmse:.2f} pixels")

        # Trust Region Reflective 最適化（境界付き）
        # 境界を設定して物理的に意味のある範囲に制限
        bounds = (
            [0.5, -90.0, -180.0, -90.0, -50.0, -50.0],  # 下限
            [10.0, 90.0, 180.0, 90.0, 50.0, 50.0],  # 上限
        )

        result = least_squares(
            fun=self._residual_function,
            x0=initial_params,
            args=(image_points, world_points),
            method="trf",  # Trust Region Reflective（境界対応）
            bounds=bounds,
            max_nfev=max_iterations,
            ftol=tolerance,
            xtol=tolerance,
        )

        # 結果を CameraExtrinsics に変換
        optimized_extrinsics = self._params_to_extrinsics(result.x)

        # 再投影誤差を計算
        residuals = result.fun
        reprojection_error = float(np.sqrt(np.mean(residuals**2)))

        # インライアー比率を計算（誤差 < 10 pixels をインライアーとする）
        point_errors = np.sqrt(residuals[::2] ** 2 + residuals[1::2] ** 2)
        inlier_ratio = float(np.mean(point_errors < 10.0))

        logger.info(
            f"Calibration completed: RMSE={reprojection_error:.2f}px, "
            f"inlier_ratio={inlier_ratio:.2%}, "
            f"converged={result.success}"
        )

        return CalibrationResult(
            extrinsics=optimized_extrinsics,
            reprojection_error=reprojection_error,
            inlier_ratio=inlier_ratio,
            optimization_converged=result.success,
            iterations=result.nfev,
            residuals=residuals,
        )

    def _floormap_to_world(
        self,
        floormap_points: np.ndarray,
        camera_position_px: tuple[float, float],
    ) -> np.ndarray:
        """フロアマップ座標をWorld座標に変換。

        Args:
            floormap_points: フロアマップ座標 (N, 2) [pixels]
            camera_position_px: カメラ位置 [pixels]

        Returns:
            World座標 (N, 3) [meters], Z=0
        """
        # 1次元配列の場合は2次元に変換
        floormap_points = np.atleast_2d(floormap_points)

        # 空の配列の場合は空の結果を返す
        if floormap_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)

        n = floormap_points.shape[0]
        world_points = np.zeros((n, 3), dtype=np.float64)

        # カメラ位置を原点としてWorld座標に変換
        world_points[:, 0] = (floormap_points[:, 0] - camera_position_px[0]) * self._scale_x
        world_points[:, 1] = (floormap_points[:, 1] - camera_position_px[1]) * self._scale_y
        world_points[:, 2] = 0.0  # 床面

        return world_points

    def _estimate_initial_params(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        _camera_position_px: tuple[float, float],
    ) -> np.ndarray:
        """初期パラメータを推定。

        OpenCV の solvePnP を使用して初期推定を行う。

        Args:
            image_points: 画像座標 (N, 2)
            world_points: World座標 (N, 3)
            camera_position_px: カメラ位置

        Returns:
            パラメータベクトル [height, pitch, yaw, roll, cam_x, cam_y]
        """
        # solvePnP で初期推定
        success, rvec, tvec = cv2.solvePnP(
            world_points.reshape(-1, 1, 3).astype(np.float32),
            image_points.reshape(-1, 1, 2).astype(np.float32),
            self._K.astype(np.float32),
            self._dist_coeffs.astype(np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.warning("solvePnP failed, using default initial values")
            return np.array([2.2, 45.0, 0.0, 0.0, 0.0, 0.0])

        # rvec, tvec からパラメータを抽出
        R, _ = cv2.Rodrigues(rvec)

        # カメラ位置を計算
        camera_pos = -R.T @ tvec.flatten()
        height = float(camera_pos[2])

        # 回転行列から角度を抽出
        pitch, yaw, roll = self._rotation_to_angles(R)

        return np.array([height, np.degrees(pitch), np.degrees(yaw), np.degrees(roll), 0.0, 0.0])

    def _residual_function(
        self,
        params: np.ndarray,
        image_points: np.ndarray,
        world_points: np.ndarray,
    ) -> np.ndarray:
        """残差関数（最適化用）。

        Args:
            params: [height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y]
            image_points: 画像座標 (N, 2)
            world_points: World座標 (N, 3)

        Returns:
            残差ベクトル (2N,)
        """
        try:
            extrinsics = self._params_to_extrinsics(params)

            # World座標を画像座標に投影
            projected = self._project_points(world_points, extrinsics)

            # NaN チェック: 無効な点にはペナルティを課す
            nan_mask = np.isnan(projected).any(axis=1)
            if np.any(nan_mask):
                # 無効な点には大きな残差を設定
                projected[nan_mask] = image_points[nan_mask] + 1000.0

            # 残差を計算
            residuals = (projected - image_points).flatten()

            # 最終的なNaNチェック
            if not np.all(np.isfinite(residuals)):
                residuals = np.array(np.where(np.isfinite(residuals), residuals, 1e4), dtype=np.float64)

            return cast("np.ndarray", residuals)

        except Exception as e:
            logger.warning(f"Error in residual function: {e}")
            return np.full(image_points.size, 1e4, dtype=np.float64)

    def _project_points(
        self,
        world_points: np.ndarray,
        extrinsics: CameraExtrinsics,
    ) -> np.ndarray:
        """World座標を画像座標に投影。

        Args:
            world_points: World座標 (N, 3)
            extrinsics: 外部パラメータ

        Returns:
            画像座標 (N, 2)
        """
        n = world_points.shape[0]
        projected = np.zeros((n, 2), dtype=np.float64)

        for i in range(n):
            P_world = world_points[i]
            P_camera = extrinsics.R @ P_world + extrinsics.t

            if P_camera[2] <= 0:
                # カメラの後方
                projected[i] = [np.nan, np.nan]
                continue

            # 画像座標へ投影
            p = self._K @ P_camera
            projected[i, 0] = p[0] / p[2]
            projected[i, 1] = p[1] / p[2]

        return projected

    def _params_to_extrinsics(self, params: np.ndarray) -> CameraExtrinsics:
        """パラメータベクトルを CameraExtrinsics に変換。

        Args:
            params: [height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y]

        Returns:
            CameraExtrinsics インスタンス
        """
        from src.transform.projection.pinhole_model import CameraExtrinsics

        height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y = params

        return CameraExtrinsics.from_pose(
            camera_height_m=float(height),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            roll_deg=float(roll_deg),
            camera_x_m=float(cam_x),
            camera_y_m=float(cam_y),
        )

    def _extrinsics_to_params(self, extrinsics: CameraExtrinsics) -> np.ndarray:
        """CameraExtrinsics をパラメータベクトルに変換。

        Args:
            extrinsics: 外部パラメータ

        Returns:
            パラメータベクトル [height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y]
        """
        camera_pos = extrinsics.camera_position_world
        pose = extrinsics.to_pose_params()

        return np.array(
            [
                camera_pos[2],  # height
                pose["pitch_deg"],
                pose["yaw_deg"],
                pose["roll_deg"],
                camera_pos[0],  # cam_x
                camera_pos[1],  # cam_y
            ]
        )

    def _rotation_to_angles(self, R: np.ndarray) -> tuple[float, float, float]:
        """回転行列から角度を抽出。

        Args:
            R: 回転行列 (3, 3)

        Returns:
            (pitch, yaw, roll) in radians
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        return (pitch, yaw, roll)

    def compute_reprojection_error(
        self,
        point_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
        extrinsics: CameraExtrinsics,
        camera_position_px: tuple[float, float],
    ) -> float:
        """再投影誤差を計算。

        Args:
            point_pairs: [(image_point, floormap_point), ...] のリスト
            extrinsics: 外部パラメータ
            camera_position_px: カメラ位置

        Returns:
            RMSE [pixels]
        """
        if len(point_pairs) == 0:
            return float("inf")

        image_points = np.array([p[0] for p in point_pairs], dtype=np.float64)
        floormap_points = np.array([p[1] for p in point_pairs], dtype=np.float64)

        # 2次元配列であることを保証
        if image_points.ndim == 1:
            image_points = image_points.reshape(1, -1)
        if floormap_points.ndim == 1:
            floormap_points = floormap_points.reshape(1, -1)

        world_points = self._floormap_to_world(floormap_points, camera_position_px)

        projected = self._project_points(world_points, extrinsics)
        errors = np.linalg.norm(projected - image_points, axis=1)

        return float(np.sqrt(np.mean(errors**2)))


class InteractiveCalibrator:
    """インタラクティブキャリブレーションクラス。

    パラメータを手動で調整し、リアルタイムで誤差を確認できる。
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        initial_extrinsics: CameraExtrinsics,
        floormap_scale_x_mm_per_px: float = 28.1926406926406,
        floormap_scale_y_mm_per_px: float = 28.241430700447,
    ):
        """初期化。

        Args:
            intrinsics: カメラ内部パラメータ
            initial_extrinsics: 初期外部パラメータ
            floormap_scale_x_mm_per_px: フロアマップX軸スケール
            floormap_scale_y_mm_per_px: フロアマップY軸スケール
        """
        self._intrinsics = intrinsics
        self._current_extrinsics = initial_extrinsics
        self._calibrator = CorrespondenceCalibrator(intrinsics, floormap_scale_x_mm_per_px, floormap_scale_y_mm_per_px)

        # 現在のパラメータを保持
        pose = initial_extrinsics.to_pose_params()
        self._height = float(initial_extrinsics.camera_position_world[2])
        self._pitch_deg = float(pose["pitch_deg"])
        self._yaw_deg = float(pose["yaw_deg"])
        self._roll_deg = float(pose["roll_deg"])
        self._cam_x = float(initial_extrinsics.camera_position_world[0])
        self._cam_y = float(initial_extrinsics.camera_position_world[1])

    @property
    def current_extrinsics(self) -> CameraExtrinsics:
        """現在の外部パラメータを返す。"""
        return self._current_extrinsics

    def adjust_parameter(self, param_name: str, delta: float) -> CameraExtrinsics:
        """パラメータを調整。

        Args:
            param_name: パラメータ名 ("height", "pitch", "yaw", "roll", "cam_x", "cam_y")
            delta: 増減量

        Returns:
            更新後の CameraExtrinsics
        """
        from src.transform.projection.pinhole_model import CameraExtrinsics

        if param_name == "height":
            self._height += delta
        elif param_name == "pitch":
            self._pitch_deg += delta
        elif param_name == "yaw":
            self._yaw_deg += delta
        elif param_name == "roll":
            self._roll_deg += delta
        elif param_name == "cam_x":
            self._cam_x += delta
        elif param_name == "cam_y":
            self._cam_y += delta
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        self._current_extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=self._height,
            pitch_deg=self._pitch_deg,
            yaw_deg=self._yaw_deg,
            roll_deg=self._roll_deg,
            camera_x_m=self._cam_x,
            camera_y_m=self._cam_y,
        )

        return self._current_extrinsics

    def get_current_error(
        self,
        point_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
        camera_position_px: tuple[float, float],
    ) -> float:
        """現在設定での再投影誤差を計算。

        Args:
            point_pairs: [(image_point, floormap_point), ...] のリスト
            camera_position_px: カメラ位置

        Returns:
            RMSE [pixels]
        """
        return self._calibrator.compute_reprojection_error(point_pairs, self._current_extrinsics, camera_position_px)

    def get_current_params(self) -> dict:
        """現在のパラメータを返す。

        Returns:
            パラメータ辞書
        """
        return {
            "height_m": self._height,
            "pitch_deg": self._pitch_deg,
            "yaw_deg": self._yaw_deg,
            "roll_deg": self._roll_deg,
            "camera_x_m": self._cam_x,
            "camera_y_m": self._cam_y,
        }

    def set_params(
        self,
        height_m: float | None = None,
        pitch_deg: float | None = None,
        yaw_deg: float | None = None,
        roll_deg: float | None = None,
        camera_x_m: float | None = None,
        camera_y_m: float | None = None,
    ) -> CameraExtrinsics:
        """パラメータを直接設定。

        Args:
            height_m: カメラ高さ [m]
            pitch_deg: 俯角 [deg]
            yaw_deg: 方位角 [deg]
            roll_deg: 回転角 [deg]
            camera_x_m: カメラX位置 [m]
            camera_y_m: カメラY位置 [m]

        Returns:
            更新後の CameraExtrinsics
        """
        from src.transform.projection.pinhole_model import CameraExtrinsics

        if height_m is not None:
            self._height = height_m
        if pitch_deg is not None:
            self._pitch_deg = pitch_deg
        if yaw_deg is not None:
            self._yaw_deg = yaw_deg
        if roll_deg is not None:
            self._roll_deg = roll_deg
        if camera_x_m is not None:
            self._cam_x = camera_x_m
        if camera_y_m is not None:
            self._cam_y = camera_y_m

        self._current_extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=self._height,
            pitch_deg=self._pitch_deg,
            yaw_deg=self._yaw_deg,
            roll_deg=self._roll_deg,
            camera_x_m=self._cam_x,
            camera_y_m=self._cam_y,
        )

        return self._current_extrinsics
