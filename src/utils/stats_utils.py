"""Statistics calculation utilities."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.models import Detection


@dataclass
class DetectionStatistics:
    """検出統計情報データクラス"""
    total_detections: int
    avg_detections_per_frame: float
    frame_count: int
    confidence_mean: float
    confidence_min: float
    confidence_max: float
    confidence_std: float
    confidence_median: float


def calculate_detection_statistics(
    detection_results: List[Tuple[int, str, List[Detection]]]
) -> DetectionStatistics:
    """検出結果から統計情報を計算する
    
    Args:
        detection_results: 検出結果のリスト [(frame_num, timestamp, detections), ...]
        
    Returns:
        DetectionStatistics: 統計情報
    """
    total_detections = sum(len(dets) for _, _, dets in detection_results)
    avg_detections = total_detections / len(detection_results) if detection_results else 0.0
    
    # 信頼度スコアの統計を計算
    all_confidences = [
        detection.confidence
        for _, _, detections in detection_results
        for detection in detections
    ]
    
    if all_confidences:
        confidence_mean = float(np.mean(all_confidences))
        confidence_min = float(np.min(all_confidences))
        confidence_max = float(np.max(all_confidences))
        confidence_std = float(np.std(all_confidences))
        confidence_median = float(np.median(all_confidences))
    else:
        confidence_mean = 0.0
        confidence_min = 0.0
        confidence_max = 0.0
        confidence_std = 0.0
        confidence_median = 0.0
    
    return DetectionStatistics(
        total_detections=total_detections,
        avg_detections_per_frame=float(avg_detections),
        frame_count=len(detection_results),
        confidence_mean=confidence_mean,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        confidence_std=confidence_std,
        confidence_median=confidence_median
    )

