"""テスト向けの軽量な Fake 実装群。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.dto import AggregationResultDTO, DetectionDTO, FrameDTO, FrameResultDTO
from src.core.interfaces import AggregatorPort, DetectorPort, TrackerPort, TransformerPort, VisualizerPort
from src.core.policy import OutputPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class FakeDetectorPort(DetectorPort):
    def detect(self, frames: Sequence[FrameDTO]) -> Sequence[Sequence[DetectionDTO]]:
        return [[] for _ in frames]


class FakeTrackerPort(TrackerPort):
    def track(
        self, detections: Iterable[DetectionDTO], frames: Iterable[FrameDTO] | None = None
    ) -> Sequence[DetectionDTO]:
        _ = frames  # 未使用引数
        return list(detections)


class FakeTransformerPort(TransformerPort):
    def transform(self, detections: Iterable[DetectionDTO]) -> Sequence[DetectionDTO]:
        return list(detections)


class FakeAggregatorPort(AggregatorPort):
    def aggregate(self, frames: Iterable[FrameResultDTO]) -> AggregationResultDTO:
        # 単一の疑似集計結果を返す
        first = next(iter(frames), None)
        ts = first.timestamp if first else "0"
        return AggregationResultDTO(timestamp=ts, zone_counts={})


class FakeVisualizerPort(VisualizerPort):
    def __init__(self, policy: OutputPolicy | None = None):
        self.policy = policy or OutputPolicy()

    def render(self, aggregation: AggregationResultDTO, frames: Iterable[FrameResultDTO]) -> None:
        _ = (aggregation, frames)  # 未使用引数
        # 実際の描画は行わない
        return
