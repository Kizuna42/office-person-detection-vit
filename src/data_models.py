"""Data models for the office person detection system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Detection:
    """検出結果データクラス

    Attributes:
        bbox: バウンディングボックス座標 (x, y, width, height)
        confidence: 信頼度スコア (0.0-1.0)
        class_id: クラスID
        class_name: クラス名
        camera_coords: カメラ座標系でのバウンディングボックス足元座標 (x, y)
        floor_coords: フロアマップ座標系での変換後座標 (x, y) ピクセル単位
        floor_coords_mm: フロアマップ座標系での変換後座標 (x, y) mm単位
        zone_ids: 所属するゾーンIDのリスト
    """

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    camera_coords: Tuple[float, float]
    floor_coords: Optional[Tuple[float, float]] = None
    floor_coords_mm: Optional[Tuple[float, float]] = None
    zone_ids: List[str] = field(default_factory=list)


@dataclass
class FrameResult:
    """フレーム処理結果データクラス

    Attributes:
        frame_number: フレーム番号
        timestamp: タイムスタンプ (HH:MM形式)
        detections: 検出結果のリスト
        zone_counts: ゾーン別人数カウント {zone_id: count}
    """

    frame_number: int
    timestamp: str
    detections: List[Detection]
    zone_counts: Dict[str, int]


@dataclass
class AggregationResult:
    """集計結果データクラス

    Attributes:
        timestamp: タイムスタンプ (HH:MM形式)
        zone_id: ゾーンID
        count: 人数カウント
    """

    timestamp: str
    zone_id: str
    count: int


@dataclass
class EvaluationMetrics:
    """評価指標データクラス

    Attributes:
        precision: 精度 (Precision)
        recall: 再現率 (Recall)
        f1_score: F1スコア
        true_positives: 真陽性の数
        false_positives: 偽陽性の数
        false_negatives: 偽陰性の数
        confidence_threshold: 使用した信頼度閾値
    """

    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    confidence_threshold: float
