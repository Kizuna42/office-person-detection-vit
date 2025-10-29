"""Office Person Detection System

Vision Transformer-based person detection and zone counting system.
"""

__version__ = "0.1.0"

from src.config_manager import ConfigManager
from src.data_models import Detection, FrameResult, AggregationResult, EvaluationMetrics
from src.video_processor import VideoProcessor
from src.timestamp_extractor import TimestampExtractor
from src.frame_sampler import FrameSampler

__all__ = [
    'ConfigManager',
    'Detection',
    'FrameResult',
    'AggregationResult',
    'EvaluationMetrics',
    'VideoProcessor',
    'TimestampExtractor',
    'FrameSampler',
]
