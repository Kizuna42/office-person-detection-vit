"""Data models for the person detection system."""

from src.models.data_models import (AggregationResult, Detection,
                                    EvaluationMetrics, FrameResult)

__all__ = [
    "Detection",
    "FrameResult",
    "AggregationResult",
    "EvaluationMetrics",
]
