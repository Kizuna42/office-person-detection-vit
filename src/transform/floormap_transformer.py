"""フロアマップ変換モジュール。

World座標系（メートル）からフロアマップ座標系（ピクセル）への変換を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

try:
    from typing import Self
except ImportError:
    from typing import Self

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FloorMapConfig:
    """フロアマップ設定。

    Attributes:
        width_px: フロアマップ画像幅 [pixel]
        height_px: フロアマップ画像高さ [pixel]
        origin_x_px: 座標系原点 X オフセット [pixel]
        origin_y_px: 座標系原点 Y オフセット [pixel]
        scale_x_mm_per_px: X軸スケール [mm/pixel]
        scale_y_mm_per_px: Y軸スケール [mm/pixel]
    """

    width_px: int = 1878
    height_px: int = 1369
    origin_x_px: float = 7.0
    origin_y_px: float = 9.0
    scale_x_mm_per_px: float = 28.1926406926406
    scale_y_mm_per_px: float = 28.241430700447

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """設定辞書から FloorMapConfig を作成。

        Args:
            config: floormap セクションの設定辞書

        Returns:
            FloorMapConfig インスタンス
        """
        return cls(
            width_px=int(config.get("image_width", 1878)),
            height_px=int(config.get("image_height", 1369)),
            origin_x_px=float(config.get("image_origin_x", 7.0)),
            origin_y_px=float(config.get("image_origin_y", 9.0)),
            scale_x_mm_per_px=float(config.get("image_x_mm_per_pixel", 28.1926406926406)),
            scale_y_mm_per_px=float(config.get("image_y_mm_per_pixel", 28.241430700447)),
        )

    @property
    def scale_x_m_per_px(self) -> float:
        """X軸スケール [m/pixel]"""
        return self.scale_x_mm_per_px / 1000.0

    @property
    def scale_y_m_per_px(self) -> float:
        """Y軸スケール [m/pixel]"""
        return self.scale_y_mm_per_px / 1000.0

    @property
    def scale_x_px_per_m(self) -> float:
        """X軸スケール [pixel/m]"""
        return 1000.0 / self.scale_x_mm_per_px

    @property
    def scale_y_px_per_m(self) -> float:
        """Y軸スケール [pixel/m]"""
        return 1000.0 / self.scale_y_mm_per_px


class FloorMapTransformer:
    """フロアマップ変換クラス。

    World座標系（メートル）からフロアマップ座標系（ピクセル）への変換を実行。
    カメラ位置をフロアマップ上の原点として使用。
    """

    def __init__(
        self,
        config: FloorMapConfig,
        camera_position_px: tuple[float, float],
    ):
        """初期化。

        Args:
            config: フロアマップ設定
            camera_position_px: カメラ位置（フロアマップ座標系）[pixel]
        """
        self._config = config
        self._camera_x_px = float(camera_position_px[0])
        self._camera_y_px = float(camera_position_px[1])

        # スケール係数（mm → pixel）
        self._scale_x = 1000.0 / config.scale_x_mm_per_px
        self._scale_y = 1000.0 / config.scale_y_mm_per_px

        logger.debug(
            f"FloorMapTransformer initialized: "
            f"camera_pos=({self._camera_x_px:.1f}, {self._camera_y_px:.1f}), "
            f"scale=({self._scale_x:.2f}, {self._scale_y:.2f}) px/m"
        )

    @property
    def config(self) -> FloorMapConfig:
        """フロアマップ設定を返す。"""
        return self._config

    @property
    def camera_position_px(self) -> tuple[float, float]:
        """カメラ位置（フロアマップ座標系）を返す。"""
        return (self._camera_x_px, self._camera_y_px)

    def world_to_floormap(
        self,
        world_point: tuple[float, float],
    ) -> tuple[float, float]:
        """World座標をフロアマップ座標に変換。

        World座標系:
            - 原点: カメラ直下の床面
            - X: 右方向 [meters]
            - Y: 前方（カメラが向いている方向）[meters]

        フロアマップ座標系:
            - 原点: 左上
            - X: 右方向 [pixels]
            - Y: 下方向 [pixels]

        変換式:
            px = camera_x_px + world_x * scale_x_px_per_m
            py = camera_y_px + world_y * scale_y_px_per_m

        Args:
            world_point: World座標 (X, Y) [meters]

        Returns:
            フロアマップ座標 (px, py) [pixels]
        """
        world_x, world_y = world_point

        # World座標をフロアマップ座標に変換
        # X: World X → Floormap X（同方向）
        # Y: World Y → Floormap Y（World Y+ は前方、Floormap Y+ は下方向）
        px = self._camera_x_px + world_x * self._scale_x
        py = self._camera_y_px + world_y * self._scale_y

        return (px, py)

    def world_to_floormap_batch(
        self,
        world_points: np.ndarray,
    ) -> np.ndarray:
        """World座標をフロアマップ座標に変換（バッチ処理）。

        Args:
            world_points: World座標 (N, 2) [meters]

        Returns:
            フロアマップ座標 (N, 2) [pixels]
            NaN を含む入力はそのまま NaN として出力
        """
        if world_points.ndim == 1:
            world_points = world_points.reshape(1, 2)

        result = np.full_like(world_points, np.nan, dtype=np.float64)

        # NaN でない点のみ変換
        valid_mask = ~np.isnan(world_points).any(axis=1)

        if np.any(valid_mask):
            result[valid_mask, 0] = self._camera_x_px + world_points[valid_mask, 0] * self._scale_x
            result[valid_mask, 1] = self._camera_y_px + world_points[valid_mask, 1] * self._scale_y

        return result

    def floormap_to_world(
        self,
        floormap_point: tuple[float, float],
    ) -> tuple[float, float]:
        """フロアマップ座標をWorld座標に変換（逆変換）。

        Args:
            floormap_point: フロアマップ座標 (px, py) [pixels]

        Returns:
            World座標 (X, Y) [meters]
        """
        px, py = floormap_point

        world_x = (px - self._camera_x_px) / self._scale_x
        world_y = (py - self._camera_y_px) / self._scale_y

        return (world_x, world_y)

    def is_within_bounds(
        self,
        floormap_point: tuple[float, float],
    ) -> bool:
        """フロアマップ座標が画像範囲内か判定。

        Args:
            floormap_point: フロアマップ座標 (px, py) [pixels]

        Returns:
            True if 範囲内
        """
        px, py = floormap_point
        return 0 <= px < self._config.width_px and 0 <= py < self._config.height_px

    def is_within_bounds_batch(
        self,
        floormap_points: np.ndarray,
    ) -> np.ndarray:
        """フロアマップ座標が画像範囲内か判定（バッチ処理）。

        Args:
            floormap_points: フロアマップ座標 (N, 2) [pixels]

        Returns:
            (N,) bool配列
        """
        if floormap_points.ndim == 1:
            floormap_points = floormap_points.reshape(1, 2)

        valid_x = (floormap_points[:, 0] >= 0) & (floormap_points[:, 0] < self._config.width_px)
        valid_y = (floormap_points[:, 1] >= 0) & (floormap_points[:, 1] < self._config.height_px)

        return valid_x & valid_y

    def pixel_to_mm(
        self,
        floormap_point: tuple[float, float],
    ) -> tuple[float, float]:
        """フロアマップ座標をmm座標に変換。

        Args:
            floormap_point: フロアマップ座標 (px, py) [pixels]

        Returns:
            (x_mm, y_mm)
        """
        px, py = floormap_point
        x_mm = px * self._config.scale_x_mm_per_px
        y_mm = py * self._config.scale_y_mm_per_px
        return (x_mm, y_mm)

    def mm_to_pixel(
        self,
        mm_point: tuple[float, float],
    ) -> tuple[float, float]:
        """mm座標をフロアマップ座標に変換。

        Args:
            mm_point: (x_mm, y_mm)

        Returns:
            フロアマップ座標 (px, py) [pixels]
        """
        x_mm, y_mm = mm_point
        px = x_mm / self._config.scale_x_mm_per_px
        py = y_mm / self._config.scale_y_mm_per_px
        return (px, py)

    def get_info(self) -> dict:
        """変換器の情報を返す（デバッグ用）。

        Returns:
            設定情報の辞書
        """
        return {
            "floormap_size": (self._config.width_px, self._config.height_px),
            "camera_position_px": (self._camera_x_px, self._camera_y_px),
            "scale_px_per_m": (self._scale_x, self._scale_y),
            "scale_mm_per_px": (
                self._config.scale_x_mm_per_px,
                self._config.scale_y_mm_per_px,
            ),
        }
