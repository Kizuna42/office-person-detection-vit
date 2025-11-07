"""Similarity calculation module for object tracking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.data_models import Detection

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """類似度計算クラス

    外観特徴量と位置情報を用いて、検出結果間の類似度を計算します。
    """

    def __init__(self, appearance_weight: float = 0.7, motion_weight: float = 0.3):
        """SimilarityCalculatorを初期化

        Args:
            appearance_weight: 外観特徴量の重み（0.0-1.0）
            motion_weight: 位置情報の重み（0.0-1.0）
                合計が1.0になる必要がある
        """
        if abs(appearance_weight + motion_weight - 1.0) > 1e-6:
            raise ValueError(
                f"appearance_weight ({appearance_weight}) + motion_weight ({motion_weight}) must equal 1.0"
            )

        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight

        logger.info(
            f"SimilarityCalculator initialized: appearance_weight={appearance_weight}, "
            f"motion_weight={motion_weight}"
        )

    def cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """コサイン類似度を計算

        Args:
            feat1: 特徴量1 (feature_dim,)
            feat2: 特徴量2 (feature_dim,)

        Returns:
            コサイン類似度 (0.0-1.0)
        """
        if feat1.shape != feat2.shape:
            raise ValueError(f"Feature shape mismatch: {feat1.shape} vs {feat2.shape}")

        # L2正規化されていることを前提
        dot_product = np.dot(feat1, feat2)
        cosine_sim = float(np.clip(dot_product, -1.0, 1.0))

        return cosine_sim

    def cosine_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """コサイン距離を計算

        Args:
            feat1: 特徴量1 (feature_dim,)
            feat2: 特徴量2 (feature_dim,)

        Returns:
            コサイン距離 (0.0-2.0, 0が最も類似)
        """
        cosine_sim = self.cosine_similarity(feat1, feat2)
        return 1.0 - cosine_sim

    def iou(self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]) -> float:
        """IoU (Intersection over Union) を計算

        Args:
            bbox1: バウンディングボックス1 (x, y, width, height)
            bbox2: バウンディングボックス2 (x, y, width, height)

        Returns:
            IoU (0.0-1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 座標を (x_min, y_min, x_max, y_max) 形式に変換
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # 交差領域を計算
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 各バウンディングボックスの面積
        area1 = w1 * h1
        area2 = w2 * h2

        # 和集合の面積
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        iou_value = inter_area / union_area
        return float(np.clip(iou_value, 0.0, 1.0))

    def iou_distance(self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]) -> float:
        """IoU距離を計算

        Args:
            bbox1: バウンディングボックス1 (x, y, width, height)
            bbox2: バウンディングボックス2 (x, y, width, height)

        Returns:
            IoU距離 (0.0-1.0, 0が最も類似)
        """
        iou_value = self.iou(bbox1, bbox2)
        return 1.0 - iou_value

    def compute_similarity(
        self,
        det1: Detection,
        det2: Detection,
        use_appearance: bool = True,
        use_motion: bool = True,
    ) -> float:
        """統合類似度を計算

        Args:
            det1: 検出結果1
            det2: 検出結果2
            use_appearance: 外観特徴量を使用するか
            use_motion: 位置情報を使用するか

        Returns:
            統合類似度スコア (0.0-1.0, 1.0が最も類似)
        """
        similarity_score = 0.0
        total_weight = 0.0

        # 外観特徴量による類似度
        if use_appearance and det1.features is not None and det2.features is not None:
            cosine_sim = self.cosine_similarity(det1.features, det2.features)
            similarity_score += self.appearance_weight * cosine_sim
            total_weight += self.appearance_weight
        elif use_appearance:
            logger.warning("Features not available, skipping appearance similarity")

        # 位置情報による類似度
        if use_motion:
            iou_value = self.iou(det1.bbox, det2.bbox)
            similarity_score += self.motion_weight * iou_value
            total_weight += self.motion_weight

        # 重みの正規化
        similarity_score = similarity_score / total_weight if total_weight > 0 else 0.0

        return float(np.clip(similarity_score, 0.0, 1.0))

    def compute_distance(
        self,
        det1: Detection,
        det2: Detection,
        use_appearance: bool = True,
        use_motion: bool = True,
    ) -> float:
        """統合距離を計算

        Args:
            det1: 検出結果1
            det2: 検出結果2
            use_appearance: 外観特徴量を使用するか
            use_motion: 位置情報を使用するか

        Returns:
            統合距離 (0.0-1.0, 0が最も類似)
        """
        similarity = self.compute_similarity(det1, det2, use_appearance, use_motion)
        return 1.0 - similarity

    def compute_similarity_matrix(self, detections1: list[Detection], detections2: list[Detection]) -> np.ndarray:
        """検出結果間の類似度行列を計算

        Args:
            detections1: 検出結果リスト1
            detections2: 検出結果リスト2

        Returns:
            類似度行列 (len(detections1), len(detections2))
        """
        matrix = np.zeros((len(detections1), len(detections2)), dtype=np.float32)

        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                matrix[i, j] = self.compute_similarity(det1, det2)

        return matrix

    def compute_distance_matrix(self, detections1: list[Detection], detections2: list[Detection]) -> np.ndarray:
        """検出結果間の距離行列を計算

        Args:
            detections1: 検出結果リスト1
            detections2: 検出結果リスト2

        Returns:
            距離行列 (len(detections1), len(detections2))
        """
        similarity_matrix = self.compute_similarity_matrix(detections1, detections2)
        return 1.0 - similarity_matrix
