"""Office Person Detection System

Vision Transformer-based person detection and zone counting system.
"""

__version__ = "0.1.0"

# Configuration
from src.config import ConfigManager

# Data models
from src.models import (
    AggregationResult,
    Detection,
    EvaluationMetrics,
    FrameResult,
)

# Video processing
from src.video import FrameSampler, VideoProcessor

# Timestamp extraction
from src.timestamp import TimestampExtractor

__all__ = [
    "ConfigManager",
    "Detection",
    "FrameResult",
    "AggregationResult",
    "EvaluationMetrics",
    "VideoProcessor",
    "TimestampExtractor",
    "FrameSampler",
]
