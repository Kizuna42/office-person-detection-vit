"""フロアマップ設定モジュール。

フロアマップの座標系とスケール情報を管理するデータクラスを提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
else:
    from typing import Self


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
