"""Feature extraction module for object tracking."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """特徴量抽出クラス

    DETRモデルから特徴量を抽出し、正規化を行います。
    """

    def __init__(self):
        """FeatureExtractorを初期化"""
        logger.info("FeatureExtractor initialized")

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """特徴量をL2正規化

        Args:
            features: 特徴量配列 (num_samples, feature_dim)

        Returns:
            L2正規化された特徴量配列
        """
        if features.size == 0:
            return features

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized: np.ndarray = features / (norms + 1e-8)

        return normalized

    def extract_roi_features(
        self,
        encoder_features: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """ROI領域の特徴量を抽出（簡易実装）

        注意: この実装は簡易版です。本番環境ではROI AlignまたはROI Poolingを使用することを推奨します。

        Args:
            encoder_features: エンコーダー特徴量 (H, W, feature_dim)
            bboxes: バウンディングボックスリスト (x, y, width, height)
            image_shape: 画像形状 (height, width)

        Returns:
            ROI特徴量配列 (num_bboxes, feature_dim)
        """
        if encoder_features.ndim != 3:
            raise ValueError(f"Expected 3D encoder features, got {encoder_features.ndim}D")

        h, w, feature_dim = encoder_features.shape
        img_h, img_w = image_shape

        roi_features = []

        for bbox in bboxes:
            x, y, width, height = bbox

            # 画像座標を特徴量マップ座標に変換
            x_min = int((x / img_w) * w)
            y_min = int((y / img_h) * h)
            x_max = int(((x + width) / img_w) * w)
            y_max = int(((y + height) / img_h) * h)

            # 範囲チェック
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            # ROI領域の特徴量を平均プーリング
            roi_feat = encoder_features[y_min:y_max, x_min:x_max, :].mean(axis=(0, 1))
            roi_features.append(roi_feat)

        if not roi_features:
            return np.array([]).reshape(0, feature_dim)

        features_array = np.array(roi_features)
        return self.normalize_features(features_array)

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """クロップ画像のバッチから特徴量を抽出

        Args:
            crops: クロップ画像のリスト (各要素は BGR format numpy array)

        Returns:
            特徴量配列 (num_crops, feature_dim)
            簡易実装ではカラーヒストグラム特徴を使用
        """
        if not crops:
            return np.array([]).reshape(0, 256)

        features = []
        for crop in crops:
            if crop is None or crop.size == 0:
                features.append(np.zeros(256))
                continue

            # 簡易実装: カラーヒストグラムを特徴量として使用
            # 本番環境ではRe-IDモデル（OSNet等）を使用することを推奨
            try:
                # 各チャンネルのヒストグラムを計算
                hist_b = np.histogram(crop[:, :, 0], bins=64, range=(0, 256))[0]
                hist_g = np.histogram(crop[:, :, 1], bins=64, range=(0, 256))[0]
                hist_r = np.histogram(crop[:, :, 2], bins=64, range=(0, 256))[0]

                # 各チャンネルの統計量を追加
                stats = np.array(
                    [
                        crop[:, :, 0].mean(),
                        crop[:, :, 0].std(),
                        crop[:, :, 1].mean(),
                        crop[:, :, 1].std(),
                        crop[:, :, 2].mean(),
                        crop[:, :, 2].std(),
                    ]
                )

                # ヒストグラム + 統計量を連結 (64*3 + 6 = 198, パディングで256に)
                feature = np.concatenate([hist_b, hist_g, hist_r, stats])
                feature = np.pad(feature, (0, 256 - len(feature)))[:256]
                features.append(feature.astype(np.float32))
            except Exception:
                features.append(np.zeros(256, dtype=np.float32))

        features_array = np.array(features)
        return self.normalize_features(features_array)
