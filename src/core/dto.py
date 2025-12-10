"""パイプライン共通で扱うDTO定義。

現行の `src/models` で利用されるフィールドを踏襲しつつ、
フェーズ間データ受け渡しを型で明示するための薄いデータコンテナ。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(slots=True)
class FrameDTO:
    """動画フレームのDTO。

    Attributes:
        frame_number: フレーム番号
        timestamp: 文字列表現のタイムスタンプ
        image: 画像 (H, W, C) ndarray
        metadata: 任意の付帯情報
    """

    frame_number: int
    timestamp: str
    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionDTO:
    """検出結果のDTO。

    bboxは (x, y, w, h) 形式で保持する。
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None
    track_id: int | None = None
    camera_coords: tuple[float, float] | None = None
    floor_coords_px: tuple[float, float] | None = None
    floor_coords_mm: tuple[float, float] | None = None
    zone_ids: list[str] = field(default_factory=list)
    features: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameResultDTO:
    """1フレームの後処理結果DTO。"""

    frame_number: int
    timestamp: str
    detections: list[DetectionDTO]
    zone_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AggregationResultDTO:
    """集計済みタイムスロットの結果DTO。"""

    timestamp: str
    zone_counts: dict[str, int]


# ユーティリティ型
FrameBatch = Sequence[FrameDTO] | Iterable[FrameDTO]
DetectionBatch = Sequence[DetectionDTO] | Iterable[DetectionDTO]
FrameResultBatch = Sequence[FrameResultDTO] | Iterable[FrameResultDTO]
