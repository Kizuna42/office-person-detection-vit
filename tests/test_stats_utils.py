"""Test cases for stats_utils."""

from __future__ import annotations

import pytest

from src.models import Detection
from src.utils.stats_utils import DetectionStatistics, calculate_detection_statistics


@pytest.fixture()
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
        ),
        Detection(
            bbox=(300.0, 400.0, 60.0, 120.0),
            confidence=0.92,
            class_id=1,
            class_name="person",
            camera_coords=(330.0, 520.0),
        ),
        Detection(
            bbox=(500.0, 600.0, 70.0, 130.0),
            confidence=0.78,
            class_id=1,
            class_name="person",
            camera_coords=(535.0, 730.0),
        ),
    ]


def test_calculate_detection_statistics(sample_detections: list[Detection]):
    """統計情報計算が正しく動作する"""
    detection_results = [
        (0, "2025/08/26 16:05:00", sample_detections),
        (1, "2025/08/26 16:10:00", sample_detections[:2]),
    ]

    stats = calculate_detection_statistics(detection_results)

    assert isinstance(stats, DetectionStatistics)
    assert stats.total_detections == 5
    assert stats.frame_count == 2
    assert stats.avg_detections_per_frame == 2.5
    assert 0.0 <= stats.confidence_mean <= 1.0
    assert 0.0 <= stats.confidence_min <= stats.confidence_max <= 1.0
    assert stats.confidence_std >= 0.0
    assert 0.0 <= stats.confidence_median <= 1.0


def test_calculate_detection_statistics_empty():
    """空の検出結果での統計情報計算"""
    detection_results: list[tuple[int, str, list[Detection]]] = []

    stats = calculate_detection_statistics(detection_results)

    assert stats.total_detections == 0
    assert stats.frame_count == 0
    assert stats.avg_detections_per_frame == 0.0
    assert stats.confidence_mean == 0.0
    assert stats.confidence_min == 0.0
    assert stats.confidence_max == 0.0
    assert stats.confidence_std == 0.0
    assert stats.confidence_median == 0.0


def test_calculate_detection_statistics_no_detections():
    """検出結果が空のフレームでの統計情報計算"""
    detection_results = [
        (0, "2025/08/26 16:05:00", []),
        (1, "2025/08/26 16:10:00", []),
    ]

    stats = calculate_detection_statistics(detection_results)

    assert stats.total_detections == 0
    assert stats.frame_count == 2
    assert stats.avg_detections_per_frame == 0.0
    assert stats.confidence_mean == 0.0


def test_calculate_detection_statistics_single_detection():
    """単一検出結果での統計情報計算"""
    detection_results = [
        (
            0,
            "2025/08/26 16:05:00",
            [
                Detection(
                    bbox=(100.0, 200.0, 50.0, 100.0),
                    confidence=0.85,
                    class_id=1,
                    class_name="person",
                    camera_coords=(125.0, 300.0),
                )
            ],
        )
    ]

    stats = calculate_detection_statistics(detection_results)

    assert stats.total_detections == 1
    assert stats.frame_count == 1
    assert stats.avg_detections_per_frame == 1.0
    assert stats.confidence_mean == 0.85
    assert stats.confidence_min == 0.85
    assert stats.confidence_max == 0.85
    assert stats.confidence_std == 0.0
    assert stats.confidence_median == 0.85
