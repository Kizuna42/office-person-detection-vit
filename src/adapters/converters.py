"""既存 data_models と DTO 間の薄い変換ヘルパー。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.dto import AggregationResultDTO, DetectionDTO, FrameDTO, FrameResultDTO
from src.models.data_models import AggregationResult, Detection, FrameResult

if TYPE_CHECKING:
    from collections.abc import Iterable


def detection_to_dto(det: Detection) -> DetectionDTO:
    return DetectionDTO(
        bbox=det.bbox,
        confidence=det.confidence,
        class_id=det.class_id,
        class_name=det.class_name,
        track_id=det.track_id,
        camera_coords=det.camera_coords,
        floor_coords_px=det.floor_coords,
        floor_coords_mm=det.floor_coords_mm,
        zone_ids=list(det.zone_ids) if det.zone_ids else [],
        features=det.features,
        extra={"appearance_score": det.appearance_score} if det.appearance_score is not None else {},
    )


def dto_to_detection(dto: DetectionDTO) -> Detection:
    return Detection(
        bbox=dto.bbox,
        confidence=dto.confidence,
        class_id=dto.class_id or 0,
        class_name=dto.class_name or "",
        camera_coords=dto.camera_coords or (0.0, 0.0),
        floor_coords=dto.floor_coords_px,
        floor_coords_mm=dto.floor_coords_mm,
        zone_ids=list(dto.zone_ids),
        track_id=dto.track_id,
        features=dto.features,
        appearance_score=dto.extra.get("appearance_score") if dto.extra else None,
    )


def frame_result_to_dto(fr: FrameResult) -> FrameResultDTO:
    return FrameResultDTO(
        frame_number=fr.frame_number,
        timestamp=fr.timestamp,
        detections=[detection_to_dto(d) for d in fr.detections],
        zone_counts=dict(fr.zone_counts),
    )


def dto_to_frame_result(dto: FrameResultDTO) -> FrameResult:
    return FrameResult(
        frame_number=dto.frame_number,
        timestamp=dto.timestamp,
        detections=[dto_to_detection(d) for d in dto.detections],
        zone_counts=dict(dto.zone_counts),
    )


def aggregation_to_dto(results: Iterable[AggregationResult]) -> AggregationResultDTO:
    zone_counts: dict[str, int] = {}
    timestamp = ""
    for r in results:
        timestamp = r.timestamp
        zone_counts[r.zone_id] = r.count
    return AggregationResultDTO(timestamp=timestamp, zone_counts=zone_counts)


def dto_to_aggregation(dto: AggregationResultDTO) -> list[AggregationResult]:
    return [AggregationResult(timestamp=dto.timestamp, zone_id=k, count=v) for k, v in dto.zone_counts.items()]


def frame_to_dto(frame_number: int, timestamp: str, image) -> FrameDTO:
    """フレーム番号と画像から DTO を生成する簡易ヘルパー。"""
    return FrameDTO(frame_number=frame_number, timestamp=timestamp, image=image)
