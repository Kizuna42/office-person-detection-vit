"""共通ユーティリティ関数"""

import cv2
import numpy as np


def generate_track_color(track_id: int) -> tuple[int, int, int]:
    """トラックIDから色を生成（HSV色空間、黄金角を使用）

    Args:
        track_id: トラックID

    Returns:
        BGR色のタプル (B, G, R)
    """
    hue = (track_id * 137) % 180
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)


def clip_coordinates(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """座標を画像範囲内にクリップ

    Args:
        x: X座標
        y: Y座標
        width: 画像幅
        height: 画像高さ

    Returns:
        クリップされた座標 (x, y)
    """
    clipped_x = max(0, min(int(x), width - 1))
    clipped_y = max(0, min(int(y), height - 1))
    return clipped_x, clipped_y


def is_within_bounds(x: float, y: float, width: int, height: int) -> bool:
    """座標が画像範囲内かチェック

    Args:
        x: X座標
        y: Y座標
        width: 画像幅
        height: 画像高さ

    Returns:
        範囲内の場合True
    """
    return 0 <= x < width and 0 <= y < height


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """2点間の距離を計算

    Args:
        x1, y1: 点1の座標
        x2, y2: 点2の座標

    Returns:
        距離
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
