"""Coordinate transformation module for camera-to-floormap mapping.

This module provides high-precision coordinate transformation using:
- Pinhole camera model with distortion correction
- Ray casting for image-to-floor projection
- Floormap coordinate mapping
- Automatic and interactive calibration
"""

from src.transform.coordinate_transformer import CoordinateTransformer
from src.transform.floormap_transformer import FloorMapConfig, FloorMapTransformer
from src.transform.projection import (
    CameraExtrinsics,
    CameraIntrinsics,
    DistortionCorrector,
    RayCaster,
)
from src.transform.unified_transformer import (
    TransformPipelineBuilder,
    TransformResult,
    UnifiedTransformer,
)

__all__ = [
    "CameraExtrinsics",
    # Projection
    "CameraIntrinsics",
    # Legacy
    "CoordinateTransformer",
    "DistortionCorrector",
    # Floormap
    "FloorMapConfig",
    "FloorMapTransformer",
    "RayCaster",
    "TransformPipelineBuilder",
    "TransformResult",
    # Unified
    "UnifiedTransformer",
]
