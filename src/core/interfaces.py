"""ポートインターフェース定義。

各フェーズはここで定義されるProtocolに依存し、具体実装は adapters 層へ分離する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.core.dto import (
        AggregationResultDTO,
        DetectionBatch,
        DetectionDTO,
        FrameBatch,
        FrameDTO,
        FrameResultBatch,
    )


class FrameSourcePort(Protocol):
    """フレーム取得ポート。"""

    def get_frames(self) -> Iterable[FrameDTO]:
        """ストリーミング/ジェネレータでフレームを供給する。"""


class DetectorPort(Protocol):
    """人物検出ポート。"""

    def detect(self, frames: Sequence[FrameDTO]) -> Sequence[Sequence[DetectionDTO]]:
        """バッチ単位で検出を返す。"""


class TrackerPort(Protocol):
    """追跡ポート。"""

    def track(self, detections: DetectionBatch, frames: FrameBatch | None = None) -> Sequence[DetectionDTO]:
        """追跡IDを付与した検出結果を返す。"""


class TransformerPort(Protocol):
    """座標変換・ゾーン判定ポート。"""

    def transform(self, detections: DetectionBatch) -> Sequence[DetectionDTO]:
        """床面座標/ゾーン情報を付与した検出結果を返す。"""


class AggregatorPort(Protocol):
    """集計ポート。"""

    def aggregate(self, frames: FrameResultBatch) -> AggregationResultDTO:
        """時系列集計を返す。"""


class VisualizerPort(Protocol):
    """可視化ポート。"""

    def render(
        self,
        aggregation: AggregationResultDTO,
        frames: FrameResultBatch,
    ) -> None:
        """図表や画像/動画を生成する。"""
