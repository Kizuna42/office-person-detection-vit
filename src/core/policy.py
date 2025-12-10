"""出力やI/O挙動を束ねるポリシー定義。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OutputPolicy:
    """画像・動画などの出力可否と間引き設定をまとめる。"""

    save_detection_images: bool = True
    save_tracking_images: bool = True
    save_floormap_images: bool = True
    save_side_by_side_video: bool = True
    detection_image_stride: int = 1
    tracking_image_stride: int = 1
