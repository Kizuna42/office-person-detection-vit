"""Utility modules for the office person detection system."""

from src.utils.logging_utils import setup_logging
from src.utils.stats_utils import calculate_detection_statistics, DetectionStatistics

__all__ = [
    "setup_logging",
    "calculate_detection_statistics",
    "DetectionStatistics",
]

