"""Camera calibration module for automatic and manual parameter estimation.

This module provides tools for calibrating camera extrinsic parameters
using correspondence points and optimization techniques.
"""

from src.transform.calibration.correspondence import (
    CorrespondenceData,
    LinePointCorrespondence,
    PointCorrespondence,
    load_correspondence_file,
    save_correspondence_file,
)
from src.transform.calibration.optimizer import (
    CalibrationResult,
    CorrespondenceCalibrator,
    InteractiveCalibrator,
)

__all__ = [
    "CalibrationResult",
    "CorrespondenceCalibrator",
    "CorrespondenceData",
    "InteractiveCalibrator",
    "LinePointCorrespondence",
    "PointCorrespondence",
    "load_correspondence_file",
    "save_correspondence_file",
]
