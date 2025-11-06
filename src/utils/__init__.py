"""Utility modules for the office person detection system."""

from src.utils.image_utils import (create_timestamp_overlay,
                                   save_detection_image)
from src.utils.logging_utils import setup_logging
from src.utils.memory_utils import cleanup_resources
from src.utils.output_utils import setup_output_directories
from src.utils.stats_utils import (DetectionStatistics,
                                   calculate_detection_statistics)

__all__ = [
    "setup_logging",
    "setup_output_directories",
    "calculate_detection_statistics",
    "DetectionStatistics",
    "cleanup_resources",
    "save_detection_image",
    "create_timestamp_overlay",
]
