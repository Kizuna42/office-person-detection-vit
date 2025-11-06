"""前処理パイプラインモジュール

OCR精度向上のための画像前処理をパラメタ化して提供。
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_invert(image: np.ndarray, enabled: bool = True) -> np.ndarray:
    """画像を反転（白文字→黒背景）

    Args:
        image: 入力画像（グレースケール）
        enabled: 反転を有効化するか

    Returns:
        反転後の画像
    """
    if not enabled:
        return image
    return cv2.bitwise_not(image)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    enabled: bool = True,
) -> np.ndarray:
    """CLAHE（Contrast Limited Adaptive Histogram Equalization）を適用

    Args:
        image: 入力画像（グレースケール）
        clip_limit: コントラスト制限値
        tile_grid_size: タイルサイズ (width, height)
        enabled: CLAHEを有効化するか

    Returns:
        コントラスト強調後の画像
    """
    if not enabled:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_resize(
    image: np.ndarray,
    fx: float = 1.0,
    fy: Optional[float] = None,
    interpolation: int = cv2.INTER_CUBIC,
    enabled: bool = True,
) -> np.ndarray:
    """画像をリサイズ

    Args:
        image: 入力画像
        fx: X方向の倍率
        fy: Y方向の倍率（Noneの場合はfxと同じ）
        interpolation: 補間方法
        enabled: リサイズを有効化するか

    Returns:
        リサイズ後の画像
    """
    if not enabled:
        return image

    if fy is None:
        fy = fx

    if fx == 1.0 and fy == 1.0:
        return image

    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def apply_threshold(
    image: np.ndarray,
    method: str = "otsu",
    block_size: int = 11,
    C: int = 2,
    enabled: bool = True,
    auto_switch: bool = False,
) -> np.ndarray:
    """二値化を適用

    Args:
        image: 入力画像（グレースケール）
        method: 二値化方法 ("otsu" または "adaptive")
        block_size: adaptive threshold用のブロックサイズ（奇数）
        C: adaptive threshold用の定数
        enabled: 二値化を有効化するか
        auto_switch: 低コントラスト時に自動的にadaptive thresholdに切替

    Returns:
        二値化後の画像
    """
    if not enabled:
        return image

    # 自動切替: コントラストが低い場合はadaptive threshold
    if auto_switch and method == "otsu":
        std_dev = np.std(image)
        if std_dev < 30.0:  # 低コントラスト判定
            method = "adaptive"
            logger.debug(f"低コントラスト検出 (std={std_dev:.1f})。Adaptive thresholdに切替")

    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1  # 奇数に調整
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )
        return binary
    else:
        logger.warning(f"未知の二値化方法: {method}。Otsuを使用します。")
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def apply_blur(
    image: np.ndarray, kernel_size: int = 3, sigma: float = 0.0, enabled: bool = True
) -> np.ndarray:
    """Gaussianブラーを適用

    Args:
        image: 入力画像
        kernel_size: カーネルサイズ（奇数）
        sigma: 標準偏差（0の場合は自動計算）
        enabled: ブラーを有効化するか

    Returns:
        ブラー後の画像
    """
    if not enabled:
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1  # 奇数に調整

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_unsharp_mask(
    image: np.ndarray,
    amount: float = 1.5,
    radius: float = 1.0,
    threshold: int = 0,
    enabled: bool = True,
) -> np.ndarray:
    """アンシャープマスク（シャープ化）を適用

    Args:
        image: 入力画像（グレースケール）
        amount: シャープ化の強度
        radius: ガウシアンブラーの半径
        threshold: 閾値（この値以下の差は無視）
        enabled: シャープ化を有効化するか

    Returns:
        シャープ化後の画像
    """
    if not enabled:
        return image

    # ガウシアンブラー
    blurred = cv2.GaussianBlur(image, (0, 0), radius)

    # 差分を計算
    diff = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))

    # 閾値以下の差分を無視
    if threshold > 0:
        diff[np.abs(diff) < threshold] = 0

    # シャープ化
    sharpened = image.astype(np.float32) + amount * diff
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def apply_morphology(
    image: np.ndarray,
    operation: str = "close",
    kernel_size: int = 3,
    iterations: int = 1,
    enabled: bool = True,
) -> np.ndarray:
    """モルフォロジー変換を適用

    Args:
        image: 入力画像（二値画像）
        operation: 操作 ("open" または "close")
        kernel_size: カーネルサイズ
        iterations: 反復回数
        enabled: モルフォロジー変換を有効化するか

    Returns:
        変換後の画像
    """
    if not enabled:
        return image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        logger.warning(f"未知のモルフォロジー操作: {operation}。closeを使用します。")
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_deskew(
    image: np.ndarray, max_angle: float = 5.0, enabled: bool = True
) -> Tuple[np.ndarray, float]:
    """傾き補正（deskew）を適用

    Args:
        image: 入力画像（二値画像）
        max_angle: 最大補正角度（度）
        enabled: 傾き補正を有効化するか

    Returns:
        (補正後の画像, 検出された角度) のタプル
    """
    if not enabled:
        return image, 0.0

    # エッジ検出
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Hough線変換で傾きを検出
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) == 0:
        return image, 0.0

    # 角度を計算
    angles = []
    for line in lines[:20]:  # 最初の20本のみ使用
        if (
            isinstance(line, (list, tuple))
            and len(line) >= 2
            and not isinstance(line[0], (list, np.ndarray))
        ):
            rho, theta = line[0], line[1]
        else:
            rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if -max_angle <= angle <= max_angle:
            angles.append(angle)

    if not angles:
        return image, 0.0

    # 中央値を角度として使用
    angle = np.median(angles)

    # 回転
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle


def apply_pipeline(image: np.ndarray, params: Dict) -> np.ndarray:
    """前処理パイプラインを適用

    パラメータ例:
    {
        "invert": {"enabled": True},
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        "resize": {"enabled": True, "fx": 2.0},
        "blur": {"enabled": False},
        "unsharp": {"enabled": False},
        "threshold": {"enabled": True, "method": "otsu"},
        "morphology": {"enabled": True, "operation": "close", "kernel_size": 2, "iterations": 1},
        "deskew": {"enabled": False}
    }

    Args:
        image: 入力画像（BGRまたはグレースケール）
        params: 前処理パラメータの辞書

    Returns:
        前処理後の画像（グレースケール）
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    result = gray

    # invert（最初に適用する場合もあるが、通常は後）
    if "invert" in params and params["invert"].get("enabled", False):
        result = apply_invert(result, **params["invert"])

    # CLAHE
    if "clahe" in params and params["clahe"].get("enabled", False):
        clahe_params = params["clahe"].copy()
        if "tile_grid_size" in clahe_params and isinstance(
            clahe_params["tile_grid_size"], list
        ):
            clahe_params["tile_grid_size"] = tuple(clahe_params["tile_grid_size"])
        result = apply_clahe(result, **clahe_params)

    # blur（ノイズ除去）
    if "blur" in params and params["blur"].get("enabled", False):
        result = apply_blur(result, **params["blur"])

    # unsharp（シャープ化）
    if "unsharp" in params and params["unsharp"].get("enabled", False):
        result = apply_unsharp_mask(result, **params["unsharp"])

    # resize
    if "resize" in params and params["resize"].get("enabled", False):
        result = apply_resize(result, **params["resize"])

    # threshold（二値化）
    if "threshold" in params and params["threshold"].get("enabled", False):
        result = apply_threshold(result, **params["threshold"])

    # invert（二値化後に適用する場合）
    if "invert_after_threshold" in params and params["invert_after_threshold"].get(
        "enabled", False
    ):
        if np.mean(result) < 127:
            result = apply_invert(result, enabled=True)

    # morphology
    if "morphology" in params and params["morphology"].get("enabled", False):
        result = apply_morphology(result, **params["morphology"])

    # deskew（最後に適用）
    if "deskew" in params and params["deskew"].get("enabled", False):
        result, angle = apply_deskew(result, **params["deskew"])
        if abs(angle) > 0.1:
            logger.debug(f"傾き補正適用: {angle:.2f}度")

    return result
