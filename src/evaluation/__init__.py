"""Evaluation module for accuracy assessment."""

from src.evaluation.detection_benchmark import (
    DetectionBenchmark,
    DetectionDiagnostics,
    DetectionMetrics,
)
from src.evaluation.evaluation_module import EvaluationModule, run_evaluation
from src.evaluation.mot_metrics import MOTMetrics
from src.evaluation.tracking_benchmark import (
    IDSwitchEvent,
    TrackingBenchmark,
    TrackingDiagnostics,
    TrackingMetrics,
)
from src.evaluation.transform_evaluator import EvaluationMetrics, TransformEvaluator

__all__ = [
    "DetectionBenchmark",
    "DetectionDiagnostics",
    "DetectionMetrics",
    "EvaluationMetrics",
    "EvaluationModule",
    "IDSwitchEvent",
    "MOTMetrics",
    "TrackingBenchmark",
    "TrackingDiagnostics",
    "TrackingMetrics",
    "TransformEvaluator",
    "run_evaluation",
]
