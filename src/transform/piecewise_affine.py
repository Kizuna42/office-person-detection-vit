"""Piecewise Affine (PWA) 変換モジュール

Delaunay三角形分割を使用した高精度座標変換を提供します。
単一ホモグラフィの限界を克服し、非線形な変形にも対応できます。
オプションでレンズ歪み補正を適用できます。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial import Delaunay

if TYPE_CHECKING:
    from src.calibration.lens_distortion import LensDistortionCorrector
    from src.transform.floormap_config import FloorMapConfig

logger = logging.getLogger(__name__)


@dataclass
class PWATransformResult:
    """PWA変換結果

    Attributes:
        floor_coords_px: フロアマップ座標 (px, py) [pixels]
        floor_coords_mm: フロアマップ座標 (x, y) [mm]
        is_valid: 変換が有効か
        error_reason: 無効時のエラー理由
        is_within_bounds: フロアマップ範囲内か
        triangle_index: 使用した三角形のインデックス（-1は外挿）
        is_extrapolated: 外挿されたか
    """

    floor_coords_px: tuple[float, float] | None = None
    floor_coords_mm: tuple[float, float] | None = None
    is_valid: bool = False
    error_reason: str | None = None
    is_within_bounds: bool = False
    triangle_index: int = -1
    is_extrapolated: bool = False


class PiecewiseAffineTransformer:
    """Piecewise Affine変換器

    Delaunay三角形分割を使用して、対応点間の局所的なアフィン変換を実行します。
    三角形外の点は最近傍三角形のアフィン行列で外挿します。
    オプションでレンズ歪み補正を適用できます。
    """

    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        floormap_config: FloorMapConfig | None = None,
        distortion_corrector: LensDistortionCorrector | None = None,
    ):
        """初期化

        Args:
            src_points: 変換元の点 (N, 2)
            dst_points: 変換先の点 (N, 2)
            floormap_config: フロアマップ設定（オプション）
            distortion_corrector: レンズ歪み補正器（オプション）

        Raises:
            ValueError: 点の数が不足している場合
        """
        self.src_points = np.array(src_points, dtype=np.float64)
        self.dst_points = np.array(dst_points, dtype=np.float64)
        self.floormap_config = floormap_config
        self.distortion_corrector = distortion_corrector

        if len(self.src_points) < 3:
            raise ValueError("最低3点の対応点が必要です")

        if len(self.src_points) != len(self.dst_points):
            raise ValueError("src_points と dst_points の数が一致しません")

        # Delaunay三角形分割
        self.delaunay = Delaunay(self.src_points)

        # 各三角形のアフィン行列を事前計算
        self.affine_matrices = self._compute_affine_matrices()

        # 外挿用の最近傍三角形マッピング
        self._fallback_triangle_indices: dict[int, int] = {}

        logger.info(
            f"PiecewiseAffineTransformer initialized: "
            f"{len(src_points)} points, {len(self.delaunay.simplices)} triangles"
        )

    def _compute_affine_matrices(self) -> list[np.ndarray]:
        """各三角形のアフィン変換行列を計算

        Returns:
            アフィン行列のリスト
        """
        matrices = []

        for simplex in self.delaunay.simplices:
            # 三角形の頂点
            src_tri = self.src_points[simplex]
            dst_tri = self.dst_points[simplex]

            # アフィン変換行列を計算
            # [dst] = A * [src]
            # A = dst * pinv(src) in augmented form
            src_aug = np.vstack([src_tri.T, np.ones(3)])
            dst_aug = np.vstack([dst_tri.T, np.ones(3)])

            # 最小二乗法でアフィン行列を求める
            A, _, _, _ = np.linalg.lstsq(src_aug.T, dst_aug.T, rcond=None)
            matrices.append(A.T)

        return matrices

    def _find_triangle(self, point: np.ndarray) -> int:
        """点を含む三角形を見つける

        Args:
            point: 座標 (x, y)

        Returns:
            三角形のインデックス（-1は三角形外）
        """
        return int(self.delaunay.find_simplex(point))

    def _find_nearest_triangle(self, point: np.ndarray) -> int:
        """最近傍の三角形を見つける

        Args:
            point: 座標 (x, y)

        Returns:
            最近傍三角形のインデックス
        """
        # 各三角形の重心との距離を計算
        centroids = np.mean(
            self.src_points[self.delaunay.simplices],
            axis=1,
        )
        distances = np.linalg.norm(centroids - point, axis=1)
        return int(np.argmin(distances))

    def transform_pixel(self, image_point: tuple[float, float]) -> PWATransformResult:
        """1点を変換

        Args:
            image_point: 画像座標 (x, y)

        Returns:
            PWATransformResult
        """
        # レンズ歪み補正を適用（有効な場合）
        if self.distortion_corrector is not None:
            corrected = self.distortion_corrector.undistort_point(image_point)
            point = np.array([corrected[0], corrected[1]])
        else:
            point = np.array([image_point[0], image_point[1]])

        # 三角形を見つける
        tri_idx = self._find_triangle(point)
        is_extrapolated = tri_idx < 0

        if is_extrapolated:
            # 外挿: 最近傍三角形を使用
            tri_idx = self._find_nearest_triangle(point)

        # アフィン変換を適用
        A = self.affine_matrices[tri_idx]
        point_aug = np.array([point[0], point[1], 1.0])
        transformed = A @ point_aug
        floor_x, floor_y = float(transformed[0]), float(transformed[1])

        # 境界チェック
        is_within = True
        if self.floormap_config:
            is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

        # mm座標を計算
        floor_mm = None
        if self.floormap_config:
            floor_mm = (
                floor_x * self.floormap_config.scale_x_mm_per_px,
                floor_y * self.floormap_config.scale_y_mm_per_px,
            )

        return PWATransformResult(
            floor_coords_px=(floor_x, floor_y),
            floor_coords_mm=floor_mm,
            is_valid=True,
            is_within_bounds=is_within,
            triangle_index=tri_idx,
            is_extrapolated=is_extrapolated,
        )

    def transform_detection(
        self,
        bbox: tuple[float, float, float, float],
    ) -> PWATransformResult:
        """検出結果（BBox）をフロアマップ座標に変換

        Args:
            bbox: (x, y, width, height)

        Returns:
            PWATransformResult
        """
        # 足元点（中心下端）を使用
        x, y, w, h = bbox
        foot_point = (x + w / 2, y + h)
        return self.transform_pixel(foot_point)

    def transform_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[PWATransformResult]:
        """バッチ変換

        Args:
            bboxes: バウンディングボックス [(x, y, w, h), ...]

        Returns:
            PWATransformResult のリスト
        """
        return [self.transform_detection(bbox) for bbox in bboxes]

    def evaluate_training_error(self) -> dict:
        """訓練データでの誤差を評価

        Returns:
            評価結果
        """
        errors = []
        for src, dst in zip(self.src_points, self.dst_points, strict=False):
            result = self.transform_pixel((src[0], src[1]))
            if result.is_valid and result.floor_coords_px:
                error = np.sqrt((result.floor_coords_px[0] - dst[0]) ** 2 + (result.floor_coords_px[1] - dst[1]) ** 2)
                errors.append(error)

        if not errors:
            return {"rmse": 0.0, "max_error": 0.0, "mean_error": 0.0}

        errors_array = np.array(errors)
        return {
            "rmse": float(np.sqrt(np.mean(errors_array**2))),
            "max_error": float(np.max(errors_array)),
            "mean_error": float(np.mean(errors_array)),
            "min_error": float(np.min(errors_array)),
            "std_error": float(np.std(errors_array)),
            "num_points": len(errors),
        }

    def get_info(self) -> dict:
        """変換器の情報を返す

        Returns:
            設定情報の辞書
        """
        info = {
            "method": "piecewise_affine",
            "num_points": len(self.src_points),
            "num_triangles": len(self.delaunay.simplices),
            "training_error": self.evaluate_training_error(),
            "distortion_correction_enabled": self.distortion_corrector is not None,
        }
        if self.distortion_corrector is not None:
            info["distortion_params"] = self.distortion_corrector.intrinsics.distortion.to_dict()
        return info

    def save(self, path: Path | str) -> None:
        """モデルを保存

        Args:
            path: 保存先パス
        """
        data = {
            "src_points": self.src_points,
            "dst_points": self.dst_points,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"PWA model saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path | str,
        floormap_config: FloorMapConfig | None = None,
        distortion_corrector: LensDistortionCorrector | None = None,
    ) -> PiecewiseAffineTransformer:
        """モデルを読み込み

        Args:
            path: モデルファイルのパス
            floormap_config: フロアマップ設定
            distortion_corrector: レンズ歪み補正器（オプション）

        Returns:
            PiecewiseAffineTransformer インスタンス
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            src_points=data["src_points"],
            dst_points=data["dst_points"],
            floormap_config=floormap_config,
            distortion_corrector=distortion_corrector,
        )

    @classmethod
    def from_correspondence_file(
        cls,
        file_path: Path | str,
        floormap_config: FloorMapConfig | None = None,
        distortion_corrector: LensDistortionCorrector | None = None,
    ) -> PiecewiseAffineTransformer:
        """対応点ファイルから作成

        Args:
            file_path: 対応点JSONファイルのパス
            floormap_config: フロアマップ設定
            distortion_corrector: レンズ歪み補正器（オプション）

        Returns:
            PiecewiseAffineTransformer インスタンス
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("point_correspondences", [])
        src_points = np.array([p["src_point"] for p in points])
        dst_points = np.array([p["dst_point"] for p in points])

        return cls(src_points, dst_points, floormap_config, distortion_corrector)

    def visualize_triangulation(
        self,
        image: np.ndarray | Path | str | None = None,
        image_size: tuple[int, int] = (1280, 720),
        output_path: Path | str | None = None,
    ) -> np.ndarray:
        """三角形分割を可視化

        Args:
            image: 背景画像（オプション）
            image_size: 画像サイズ (width, height)
            output_path: 出力パス

        Returns:
            可視化画像
        """
        # 背景画像
        if image is not None:
            if isinstance(image, str | Path):
                img = cv2.imread(str(image))
                if img is None:
                    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
            else:
                img = image.copy()
        else:
            img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

        # 三角形を描画
        for simplex in self.delaunay.simplices:
            pts = self.src_points[simplex].astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 1)

        # 対応点を描画
        for _i, pt in enumerate(self.src_points):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(img, (x, y), 5, (0, 0, 0), 1)

        # 情報表示
        cv2.putText(img, f"Points: {len(self.src_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(
            img, f"Triangles: {len(self.delaunay.simplices)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        if output_path:
            cv2.imwrite(str(output_path), img)

        return img


class ThinPlateSplineTransformer:
    """Thin-Plate Spline (TPS) 変換器

    滑らかな非線形変換を提供します。
    PWAよりも滑らかな補間が可能ですが、計算コストが高くなります。
    オプションでレンズ歪み補正を適用できます。
    """

    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        floormap_config: FloorMapConfig | None = None,
        regularization: float = 0.0,
        distortion_corrector: LensDistortionCorrector | None = None,
    ):
        """初期化

        Args:
            src_points: 変換元の点 (N, 2)
            dst_points: 変換先の点 (N, 2)
            floormap_config: フロアマップ設定
            regularization: 正則化パラメータ（0.0で厳密補間）
            distortion_corrector: レンズ歪み補正器（オプション）
        """
        self.src_points = np.array(src_points, dtype=np.float64)
        self.dst_points = np.array(dst_points, dtype=np.float64)
        self.floormap_config = floormap_config
        self.regularization = regularization
        self.distortion_corrector = distortion_corrector

        if len(self.src_points) < 3:
            raise ValueError("最低3点の対応点が必要です")

        # TPS係数を計算
        self.weights_x, self.weights_y, self.affine_x, self.affine_y = self._compute_tps_coefficients()

        logger.info(f"ThinPlateSplineTransformer initialized with {len(src_points)} points")

    def _radial_basis(self, r: np.ndarray) -> np.ndarray:
        """放射基底関数 U(r) = r^2 * log(r)"""
        # r=0でのlog(0)を避ける
        mask = r > 0
        result = np.zeros_like(r)
        result[mask] = r[mask] ** 2 * np.log(r[mask])
        return result

    def _compute_tps_coefficients(self):
        """TPS係数を計算"""
        n = len(self.src_points)

        # カーネル行列 K
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(self.src_points[i] - self.src_points[j])
                    K[i, j] = self._radial_basis(np.array([r]))[0]

        # 行列 P = [1, x, y]
        P = np.hstack([np.ones((n, 1)), self.src_points])

        # 連立方程式を構築
        # [K + λI  P ] [w]   [v]
        # [P^T     0 ] [a] = [0]
        K_reg = K + self.regularization * np.eye(n)

        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K_reg
        L[:n, n:] = P
        L[n:, :n] = P.T

        # 右辺
        v_x = np.zeros(n + 3)
        v_y = np.zeros(n + 3)
        v_x[:n] = self.dst_points[:, 0]
        v_y[:n] = self.dst_points[:, 1]

        # 解く
        coef_x = np.linalg.solve(L, v_x)
        coef_y = np.linalg.solve(L, v_y)

        weights_x = coef_x[:n]
        weights_y = coef_y[:n]
        affine_x = coef_x[n:]
        affine_y = coef_y[n:]

        return weights_x, weights_y, affine_x, affine_y

    def transform_pixel(self, image_point: tuple[float, float]) -> PWATransformResult:
        """1点を変換"""
        # レンズ歪み補正を適用（有効な場合）
        if self.distortion_corrector is not None:
            corrected = self.distortion_corrector.undistort_point(image_point)
            point = np.array([corrected[0], corrected[1]])
        else:
            point = np.array([image_point[0], image_point[1]])

        # 放射基底の寄与
        rbf_x = 0.0
        rbf_y = 0.0
        for i, src_pt in enumerate(self.src_points):
            r = np.linalg.norm(point - src_pt)
            u = self._radial_basis(np.array([r]))[0]
            rbf_x += self.weights_x[i] * u
            rbf_y += self.weights_y[i] * u

        # アフィン部分
        floor_x = self.affine_x[0] + self.affine_x[1] * point[0] + self.affine_x[2] * point[1] + rbf_x
        floor_y = self.affine_y[0] + self.affine_y[1] * point[0] + self.affine_y[2] * point[1] + rbf_y

        # 境界チェック
        is_within = True
        if self.floormap_config:
            is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

        # mm座標
        floor_mm = None
        if self.floormap_config:
            floor_mm = (
                floor_x * self.floormap_config.scale_x_mm_per_px,
                floor_y * self.floormap_config.scale_y_mm_per_px,
            )

        return PWATransformResult(
            floor_coords_px=(float(floor_x), float(floor_y)),
            floor_coords_mm=floor_mm,
            is_valid=True,
            is_within_bounds=is_within,
        )

    def transform_detection(
        self,
        bbox: tuple[float, float, float, float],
    ) -> PWATransformResult:
        """検出結果を変換"""
        x, y, w, h = bbox
        foot_point = (x + w / 2, y + h)
        return self.transform_pixel(foot_point)

    def transform_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[PWATransformResult]:
        """バッチ変換"""
        return [self.transform_detection(bbox) for bbox in bboxes]

    def evaluate_training_error(self) -> dict:
        """訓練データでの誤差を評価"""
        errors = []
        for src, dst in zip(self.src_points, self.dst_points, strict=False):
            result = self.transform_pixel((src[0], src[1]))
            if result.is_valid and result.floor_coords_px:
                error = np.sqrt((result.floor_coords_px[0] - dst[0]) ** 2 + (result.floor_coords_px[1] - dst[1]) ** 2)
                errors.append(error)

        errors_array = np.array(errors)
        return {
            "rmse": float(np.sqrt(np.mean(errors_array**2))),
            "max_error": float(np.max(errors_array)),
            "mean_error": float(np.mean(errors_array)),
            "num_points": len(errors),
        }

    def get_info(self) -> dict:
        """変換器の情報"""
        info = {
            "method": "thin_plate_spline",
            "num_points": len(self.src_points),
            "regularization": self.regularization,
            "training_error": self.evaluate_training_error(),
            "distortion_correction_enabled": self.distortion_corrector is not None,
        }
        if self.distortion_corrector is not None:
            info["distortion_params"] = self.distortion_corrector.intrinsics.distortion.to_dict()
        return info

    @classmethod
    def from_correspondence_file(
        cls,
        file_path: Path | str,
        floormap_config: FloorMapConfig | None = None,
        regularization: float = 0.0,
        distortion_corrector: LensDistortionCorrector | None = None,
    ) -> ThinPlateSplineTransformer:
        """対応点ファイルから作成"""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("point_correspondences", [])
        src_points = np.array([p["src_point"] for p in points])
        dst_points = np.array([p["dst_point"] for p in points])

        return cls(src_points, dst_points, floormap_config, regularization, distortion_corrector)
