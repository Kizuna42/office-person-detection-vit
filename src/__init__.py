"""Office Person Detection System

Vision Transformer-based person detection and zone counting system.
"""

__version__ = "0.1.0"

from src.config_manager import ConfigManager
from src.data_models import (AggregationResult, Detection, EvaluationMetrics,
                             FrameResult)
from src.frame_sampler import FrameSampler
from src.timestamp_extractor import TimestampExtractor
from src.video_processor import VideoProcessor

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
