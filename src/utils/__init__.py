"""Utility modules for the office person detection system."""

from src.utils.image_utils import (create_timestamp_overlay,
                                   save_detection_image)
from src.utils.logging_utils import setup_logging
from src.utils.memory_utils import cleanup_resources
from src.utils.output_manager import OutputManager
from src.utils.output_utils import setup_output_directories
from src.utils.stats_utils import (DetectionStatistics,
                                   calculate_detection_statistics)
from src.utils.text_metrics import (calculate_cer, calculate_token_metrics,
                                    calculate_wer)
from src.utils.torch_utils import get_device, setup_mps_compatibility

__all__ = [
    "setup_logging",
    "setup_output_directories",
    "OutputManager",
    "calculate_detection_statistics",
    "DetectionStatistics",
    "cleanup_resources",
    "save_detection_image",
    "create_timestamp_overlay",
    "calculate_cer",
    "calculate_wer",
    "calculate_token_metrics",
    "get_device",
    "setup_mps_compatibility",
]
