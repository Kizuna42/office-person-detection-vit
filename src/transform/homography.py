"""ホモグラフィ変換モジュール。

2D画像座標からフロアマップ座標への直接変換を提供します。
これはメインパイプラインで使用される座標変換方式です。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.transform.floormap_config import FloorMapConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """座標変換結果。

    Attributes:
        floor_coords_px: フロアマップ座標 (px, py) [pixels]、無効時は None
        floor_coords_mm: フロアマップ座標 (x, y) [mm]、無効時は None
        is_valid: 変換が有効か
        error_reason: 無効時のエラー理由
        is_within_bounds: フロアマップ範囲内か
    """

    floor_coords_px: tuple[float, float] | None = None
    floor_coords_mm: tuple[float, float] | None = None
    is_valid: bool = False
    error_reason: str | None = None
    is_within_bounds: bool = False


class HomographyTransformer:
    """ホモグラフィ変換器。

    2D画像座標からフロアマップ座標への直接変換を実行します。
    バウンディングボックスの足元点（中心下端）を変換対象とします。
    """

    def __init__(
        self,
        homography_matrix: np.ndarray,
        floormap_config: FloorMapConfig,
    ):
        """初期化。

        Args:
            homography_matrix: 3x3ホモグラフィ行列
            floormap_config: フロアマップ設定

        Raises:
            ValueError: ホモグラフィ行列が不正な形式の場合
        """
        self.H = self._validate_matrix(homography_matrix)
        self.floormap_config = floormap_config

        logger.debug(f"HomographyTransformer initialized with matrix:\n{self.H}")

    def _validate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """ホモグラフィ行列を検証。

        Args:
            matrix: 入力行列

        Returns:
            検証済みの3x3行列

        Raises:
            ValueError: 行列が不正な形式の場合
        """
        H = np.array(matrix, dtype=np.float64)

        if H.shape != (3, 3):
            raise ValueError(f"ホモグラフィ行列は3x3である必要があります: {H.shape}")

        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            raise ValueError(f"ホモグラフィ行列が特異行列です（行列式={det}）")

        cond = np.linalg.cond(H)
        if cond > 1e12:
            logger.warning(f"ホモグラフィ行列の条件数が大きい: {cond}")

        return H

    def _get_foot_point(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """バウンディングボックスから足元点を計算。

        Args:
            bbox: (x, y, width, height)

        Returns:
            足元点 (x, y) - 中心下端
        """
        x, y, w, h = bbox
        return (x + w / 2, y + h)

    def transform_pixel(self, image_point: tuple[float, float]) -> TransformResult:
        """1点を変換。

        Args:
            image_point: 画像座標 (x, y)

        Returns:
            TransformResult
        """
        pt = np.array([[image_point[0], image_point[1], 1.0]])
        transformed = (self.H @ pt.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        floor_x, floor_y = float(transformed[0, 0]), float(transformed[0, 1])

        # 境界チェック
        is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

        # mm座標を計算
        floor_mm = (
            floor_x * self.floormap_config.scale_x_mm_per_px,
            floor_y * self.floormap_config.scale_y_mm_per_px,
        )

        return TransformResult(
            is_valid=True,
            floor_coords_px=(floor_x, floor_y),
            floor_coords_mm=floor_mm,
            is_within_bounds=is_within,
        )

    def transform_detection(
        self,
        bbox: tuple[float, float, float, float],
    ) -> TransformResult:
        """検出結果（BBox）をフロアマップ座標に変換。

        Args:
            bbox: (x, y, width, height)

        Returns:
            TransformResult
        """
        foot_point = self._get_foot_point(bbox)
        return self.transform_pixel(foot_point)

    def transform_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[TransformResult]:
        """バッチ変換（足元点を使用）。

        Args:
            bboxes: バウンディングボックス [(x, y, w, h), ...]

        Returns:
            TransformResult のリスト
        """
        if not bboxes:
            return []

        # 足元点を計算
        foot_points = np.array(
            [[x + w / 2, y + h] for x, y, w, h in bboxes],
            dtype=np.float64,
        )

        # ホモグラフィ変換
        ones = np.ones((len(foot_points), 1))
        pts_h = np.hstack([foot_points, ones])
        transformed = (self.H @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        results = []
        for i in range(len(bboxes)):
            floor_x, floor_y = float(transformed[i, 0]), float(transformed[i, 1])

            is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

            floor_mm = (
                floor_x * self.floormap_config.scale_x_mm_per_px,
                floor_y * self.floormap_config.scale_y_mm_per_px,
            )

            results.append(
                TransformResult(
                    is_valid=True,
                    floor_coords_px=(floor_x, floor_y),
                    floor_coords_mm=floor_mm,
                    is_within_bounds=is_within,
                )
            )

        return results

    def get_info(self) -> dict:
        """変換器の情報を返す（デバッグ用）。

        Returns:
            設定情報の辞書
        """
        return {
            "method": "homography",
            "matrix": self.H.tolist(),
            "floormap_size": (self.floormap_config.width_px, self.floormap_config.height_px),
            "scale_mm_per_px": (
                self.floormap_config.scale_x_mm_per_px,
                self.floormap_config.scale_y_mm_per_px,
            ),
        }
