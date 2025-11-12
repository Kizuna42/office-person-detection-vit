"""Reprojection error evaluation module for coordinate transformation accuracy."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ReprojectionErrorEvaluator:
    """再投影誤差評価クラス

    座標変換の精度を評価するために、再投影誤差を計算します。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ):
        """ReprojectionErrorEvaluatorを初期化

        Args:
            camera_matrix: カメラ内部パラメータ行列（3x3、オプション）
            dist_coeffs: 歪み係数（オプション）
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        logger.info("ReprojectionErrorEvaluator initialized")

    def evaluate_homography(
        self,
        src_points: list[tuple[float, float]],
        dst_points: list[tuple[float, float]],
        homography_matrix: np.ndarray,
    ) -> dict[str, float | list[float]]:
        """ホモグラフィ変換の再投影誤差を評価

        Args:
            src_points: 変換元の点のリスト（カメラ座標）
            dst_points: 変換先の点のリスト（フロアマップ座標）
            homography_matrix: ホモグラフィ変換行列（3x3）

        Returns:
            評価結果の辞書:
                - mean_error: 平均再投影誤差（ピクセル）
                - max_error: 最大再投影誤差（ピクセル）
                - min_error: 最小再投影誤差（ピクセル）
                - std_error: 標準偏差（ピクセル）
                - errors: 各点の誤差リスト
        """
        if len(src_points) != len(dst_points):
            raise ValueError(f"点の数が一致しません: {len(src_points)} vs {len(dst_points)}")

        if len(src_points) == 0:
            return {
                "mean_error": 0.0,
                "max_error": 0.0,
                "min_error": 0.0,
                "std_error": 0.0,
                "errors": [],
            }

        errors: list[float] = []
        H = np.array(homography_matrix, dtype=np.float64)

        for src_pt, dst_pt in zip(src_points, dst_points, strict=False):
            # 変換元の点を同次座標に変換
            src_homogeneous = np.array([src_pt[0], src_pt[1], 1.0])

            # ホモグラフィ変換を適用
            transformed = H @ src_homogeneous
            w = transformed[2]

            if abs(w) < 1e-10:
                logger.warning(f"変換後のw成分が0に近い値です: {w}")
                continue

            # 正規化
            transformed_x = transformed[0] / w
            transformed_y = transformed[1] / w

            # 再投影誤差を計算（ユークリッド距離）
            error = np.sqrt((transformed_x - dst_pt[0]) ** 2 + (transformed_y - dst_pt[1]) ** 2)
            errors.append(float(error))

        if not errors:
            return {
                "mean_error": 0.0,
                "max_error": 0.0,
                "min_error": 0.0,
                "std_error": 0.0,
                "errors": [],
            }

        errors_array = np.array(errors)

        result: dict[str, float | list[float]] = {
            "mean_error": float(np.mean(errors_array)),
            "max_error": float(np.max(errors_array)),
            "min_error": float(np.min(errors_array)),
            "std_error": float(np.std(errors_array)),
            "errors": errors,
        }

        logger.info(
            f"再投影誤差評価完了: 平均={result['mean_error']:.2f}px, "
            f"最大={result['max_error']:.2f}px, 標準偏差={result['std_error']:.2f}px"
        )

        return result

    def evaluate_camera_calibration(
        self,
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> dict[str, float | list[list[float]]]:
        """カメラキャリブレーションの再投影誤差を評価

        Args:
            object_points: 3Dオブジェクト点のリスト
            image_points: 2D画像点のリスト
            camera_matrix: カメラ内部パラメータ行列（指定されない場合はself.camera_matrixを使用）
            dist_coeffs: 歪み係数（指定されない場合はself.dist_coeffsを使用）

        Returns:
            評価結果の辞書:
                - mean_error: 平均再投影誤差（ピクセル）
                - max_error: 最大再投影誤差（ピクセル）
                - min_error: 最小再投影誤差（ピクセル）
                - std_error: 標準偏差（ピクセル）
                - per_image_errors: 画像ごとの誤差リスト
        """
        cam_matrix = camera_matrix if camera_matrix is not None else self.camera_matrix
        dist_coeff = dist_coeffs if dist_coeffs is not None else self.dist_coeffs

        if cam_matrix is None or dist_coeff is None:
            raise ValueError("カメラ行列または歪み係数が設定されていません。")

        all_errors = []
        per_image_errors = []

        for obj_pts, img_pts in zip(object_points, image_points, strict=False):
            # 再投影
            rvecs = np.zeros((3, 1))
            tvecs = np.zeros((3, 1))
            projected_points, _ = cv2.projectPoints(obj_pts, rvecs, tvecs, cam_matrix, dist_coeff)

            # 誤差を計算
            errors = []
            for i in range(len(img_pts)):
                error = np.sqrt(
                    (projected_points[i, 0, 0] - img_pts[i, 0]) ** 2 + (projected_points[i, 0, 1] - img_pts[i, 1]) ** 2
                )
                errors.append(float(error))
                all_errors.append(float(error))

            per_image_errors.append(errors)

        if not all_errors:
            return {
                "mean_error": 0.0,
                "max_error": 0.0,
                "min_error": 0.0,
                "std_error": 0.0,
                "per_image_errors": [],
            }

        errors_array = np.array(all_errors)

        result: dict[str, float | list[list[float]]] = {
            "mean_error": float(np.mean(errors_array)),
            "max_error": float(np.max(errors_array)),
            "min_error": float(np.min(errors_array)),
            "std_error": float(np.std(errors_array)),
            "per_image_errors": per_image_errors,
        }

        logger.info(
            f"カメラキャリブレーション再投影誤差評価完了: "
            f"平均={result['mean_error']:.2f}px, 最大={result['max_error']:.2f}px"
        )

        return result

    def create_error_map(
        self,
        src_points: list[tuple[float, float]],
        dst_points: list[tuple[float, float]],
        homography_matrix: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """再投影誤差マップを生成

        Args:
            src_points: 変換元の点のリスト
            dst_points: 変換先の点のリスト
            homography_matrix: ホモグラフィ変換行列
            image_shape: 画像形状 (height, width)

        Returns:
            誤差マップ（画像と同じ形状）
        """
        error_map = np.zeros(image_shape, dtype=np.float32)

        H = np.array(homography_matrix, dtype=np.float64)

        for src_pt, dst_pt in zip(src_points, dst_points, strict=False):
            # 変換元の点を同次座標に変換
            src_homogeneous = np.array([src_pt[0], src_pt[1], 1.0])

            # ホモグラフィ変換を適用
            transformed = H @ src_homogeneous
            w = transformed[2]

            if abs(w) < 1e-10:
                continue

            # 正規化
            transformed_x = transformed[0] / w
            transformed_y = transformed[1] / w

            # 誤差を計算
            error = np.sqrt((transformed_x - dst_pt[0]) ** 2 + (transformed_y - dst_pt[1]) ** 2)

            # 誤差マップに記録（最近傍のピクセルに記録）
            x = int(np.clip(transformed_x, 0, image_shape[1] - 1))
            y = int(np.clip(transformed_y, 0, image_shape[0] - 1))
            error_map[y, x] = max(error_map[y, x], error)

        return error_map
