"""Detection module for person detection using YOLOv8."""

from src.detection.preprocessing import (
    apply_blur,
    apply_clahe,
    apply_deskew,
    apply_invert,
    apply_morphology,
    apply_pipeline,
    apply_resize,
    apply_threshold,
    apply_unsharp_mask,
)
from src.detection.yolov8_detector import YOLOv8Detector

__all__ = [
    "YOLOv8Detector",
    "apply_blur",
    "apply_clahe",
    "apply_deskew",
    "apply_invert",
    "apply_morphology",
    "apply_pipeline",
    "apply_resize",
    "apply_threshold",
    "apply_unsharp_mask",
]
