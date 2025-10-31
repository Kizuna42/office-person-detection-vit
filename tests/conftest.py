"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.data_models import Detection


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the directory containing static test fixtures."""

    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Return a dummy frame (720x1280 BGR)."""

    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> Detection:
    """Return a detection instance located in zone_a."""

    return Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        floor_coords=(150.0, 350.0),
        zone_ids=["zone_a"],
    )


@pytest.fixture
def ground_truth_path(fixtures_dir: Path) -> Path:
    """Return the path to the sample COCO ground truth file."""

    return fixtures_dir / "sample_ground_truth.json"
