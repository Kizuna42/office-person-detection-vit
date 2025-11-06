"""Video processing module for frame extraction and sampling."""

from src.video.frame_sampler import AdaptiveSampler, CoarseSampler, FineSampler
from src.video.video_processor import VideoProcessor

__all__ = [
    "VideoProcessor",
    "CoarseSampler",
    "FineSampler",
    "AdaptiveSampler",
]
