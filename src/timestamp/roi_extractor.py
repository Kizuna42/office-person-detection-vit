"""ROI (Region of Interest) extraction for timestamp detection."""

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TimestampROIExtractor:
    """画像の右上領域からタイムスタンプ領域を抽出

    フレーム画像からタイムスタンプが表示されている領域を切り出し、
    OCR精度向上のための前処理を実行します。
    """

    def __init__(self, roi_config: Dict[str, float] = None):
        """TimestampROIExtractorを初期化

        Args:
            roi_config: ROI設定辞書。以下のキーを持つ:
                - x_ratio: 右端からの開始位置比率（0.0-1.0）
                - y_ratio: 上端からの開始位置比率（0.0-1.0）
                - width_ratio: 幅の比率（0.0-1.0）
                - height_ratio: 高さの比率（0.0-1.0）
        """
        # デフォルト設定（画像を見て調整）
        self.roi_config = roi_config or {
            "x_ratio": 0.65,  # 右から35%の位置から
            "y_ratio": 0.0,  # 上端から
            "width_ratio": 0.35,  # 幅35%
            "height_ratio": 0.08,  # 高さ8%
        }

    def extract_roi(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """フレームからタイムスタンプ領域を切り出し

        Args:
            frame: 入力フレーム画像（BGR形式）

        Returns:
            (ROI画像, (x, y, width, height)) のタプル
        """
        h, w = frame.shape[:2]

        x = int(w * self.roi_config["x_ratio"])
        y = int(h * self.roi_config["y_ratio"])
        roi_w = int(w * self.roi_config["width_ratio"])
        roi_h = int(h * self.roi_config["height_ratio"])

        # 境界チェック
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)

        roi = frame[y : y + roi_h, x : x + roi_w]
        return roi, (x, y, roi_w, roi_h)

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """OCR精度向上のための前処理（最適化版）

        最適化テストの結果、グレースケールのみ（二値化なし）が
        最も高い精度を示すため、その手法を採用しています。

        Args:
            roi: ROI画像（BGR形式）

        Returns:
            前処理済み画像（グレースケール、CLAHE適用済み）
        """
        # ROI画像を拡大（OCR精度向上のため、最小サイズを確保）
        # 最適化テストの結果、300pxが最適
        h, w = roi.shape[:2]
        min_size = 300  # 最適化された最小サイズ
        scale = max(min_size / w, min_size / h, 1.0)
        if scale > 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # グレースケール化
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # ガウシアンブラーでノイズを軽減（軽度）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # コントラスト強調（CLAHE）- 最適化されたパラメータ
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # 最適化テストの結果、二値化せずにグレースケールのまま返す
        # これにより、Tesseractが内部で最適な二値化を実行できる
        return enhanced
