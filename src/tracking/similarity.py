"""Similarity calculation module for object tracking.

This module provides similarity metrics for matching detections across frames,
combining appearance features and spatial information.
"""

import logging

import numpy as np

from src.models.data_models import Detection

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """類似度計算クラス

    外観特徴量と位置情報を組み合わせて、検出結果間の類似度を計算します。
    """

    def __init__(
        self,
        appearance_weight: float = 0.7,
        motion_weight: float = 0.3,
        iou_threshold: float = 0.3,
    ):
        """SimilarityCalculatorを初期化

        Args:
            appearance_weight: 外観特徴量の重み（0.0-1.0）
            motion_weight: 位置情報の重み（0.0-1.0、appearance_weight + motion_weight = 1.0）
            iou_threshold: IoU距離の閾値（この値以下のIoUは距離1.0として扱う）
        """
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError(f"appearance_weight must be between 0.0 and 1.0, got {appearance_weight}")
        if not 0.0 <= motion_weight <= 1.0:
            raise ValueError(f"motion_weight must be between 0.0 and 1.0, got {motion_weight}")
        if abs(appearance_weight + motion_weight - 1.0) > 1e-6:
            raise ValueError(
                f"appearance_weight + motion_weight must equal 1.0, " f"got {appearance_weight + motion_weight}"
            )

        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.iou_threshold = iou_threshold

        logger.info(
            f"SimilarityCalculator initialized: "
            f"appearance_weight={appearance_weight}, motion_weight={motion_weight}, "
            f"iou_threshold={iou_threshold}"
        )

    def cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """コサイン類似度を計算

        Args:
            feat1: 特徴量ベクトル1（L2正規化済み）
            feat2: 特徴量ベクトル2（L2正規化済み）

        Returns:
            コサイン類似度（0.0-1.0、1.0が最も類似）
        """
        if feat1 is None or feat2 is None:
            logger.warning("One or both features are None. Returning 0.0 similarity.")
            return 0.0

        if feat1.shape != feat2.shape:
            raise ValueError(f"Feature shapes must match: {feat1.shape} vs {feat2.shape}")

        # L2正規化済みなので、内積がコサイン類似度になる
        similarity = np.dot(feat1, feat2)

        # 範囲を0.0-1.0にクリップ（数値誤差対策）
        similarity = max(0.0, min(1.0, similarity))

        return float(similarity)

    def cosine_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """コサイン距離を計算

        Args:
            feat1: 特徴量ベクトル1（L2正規化済み）
            feat2: 特徴量ベクトル2（L2正規化済み）

        Returns:
            コサイン距離（0.0-1.0、0.0が最も類似）
        """
        similarity = self.cosine_similarity(feat1, feat2)
        return 1.0 - similarity

    def iou(self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]) -> float:
        """IoU（Intersection over Union）を計算

        Args:
            bbox1: バウンディングボックス1 (x, y, width, height)
            bbox2: バウンディングボックス2 (x, y, width, height)

        Returns:
            IoU値（0.0-1.0、1.0が完全一致）
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # バウンディングボックスを(x_min, y_min, x_max, y_max)形式に変換
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1

        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # 交差領域を計算
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 交差領域がない場合
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        # 交差面積と和集合面積を計算
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def iou_distance(self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]) -> float:
        """IoU距離を計算

        Args:
            bbox1: バウンディングボックス1 (x, y, width, height)
            bbox2: バウンディングボックス2 (x, y, width, height)

        Returns:
            IoU距離（0.0-1.0、0.0が完全一致）
        """
        iou_value = self.iou(bbox1, bbox2)

        # IoUが閾値以下の場合は距離1.0として扱う
        if iou_value < self.iou_threshold:
            return 1.0

        return 1.0 - iou_value

    def compute_similarity(self, detection1: Detection, detection2: Detection) -> tuple[float, float, float]:
        """統合類似度を計算

        外観特徴量と位置情報を組み合わせて、2つの検出結果間の統合距離を計算します。

        Args:
            detection1: 検出結果1
            detection2: 検出結果2

        Returns:
            (統合距離, コサイン距離, IoU距離)のタプル
            - 統合距離: 0.0-1.0（0.0が最も類似）
            - コサイン距離: 0.0-1.0
            - IoU距離: 0.0-1.0
        """
        # コサイン距離を計算
        cosine_dist = 0.0
        if detection1.features is not None and detection2.features is not None:
            cosine_dist = self.cosine_distance(detection1.features, detection2.features)
        else:
            # 特徴量がない場合は最大距離として扱う
            cosine_dist = 1.0
            logger.debug("Features not available for one or both detections. Using max distance.")

        # IoU距離を計算
        iou_dist = self.iou_distance(detection1.bbox, detection2.bbox)

        # 統合距離を計算
        combined_distance = self.appearance_weight * cosine_dist + self.motion_weight * iou_dist

        return (combined_distance, cosine_dist, iou_dist)

    def compute_similarity_matrix(self, detections1: list[Detection], detections2: list[Detection]) -> np.ndarray:
        """類似度行列を計算

        2つの検出結果リスト間の全ペアの類似度を計算します。

        Args:
            detections1: 検出結果リスト1
            detections2: 検出結果リスト2

        Returns:
            類似度行列 (len(detections1), len(detections2))
            各要素は統合距離（0.0-1.0、0.0が最も類似）
        """
        n1 = len(detections1)
        n2 = len(detections2)

        similarity_matrix = np.zeros((n1, n2), dtype=np.float32)

        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                combined_dist, _, _ = self.compute_similarity(det1, det2)
                similarity_matrix[i, j] = combined_dist

        return similarity_matrix
