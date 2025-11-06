"""Unit tests for data models."""

from __future__ import annotations

from src.models import (AggregationResult, Detection, EvaluationMetrics,
                        FrameResult)


def test_detection_creation():
    """Detectionデータクラスを正しく作成できる。"""

    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
    )

    assert detection.bbox == (100.0, 200.0, 50.0, 100.0)
    assert detection.confidence == 0.9
    assert detection.class_id == 1
    assert detection.class_name == "person"
    assert detection.camera_coords == (125.0, 300.0)
    assert detection.floor_coords is None
    assert detection.zone_ids == []


def test_detection_with_floor_coords():
    """floor_coordsを指定してDetectionを作成できる。"""

    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        floor_coords=(150.0, 350.0),
    )

    assert detection.floor_coords == (150.0, 350.0)


def test_detection_with_zone_ids():
    """zone_idsを指定してDetectionを作成できる。"""

    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        zone_ids=["zone_a", "zone_b"],
    )

    assert detection.zone_ids == ["zone_a", "zone_b"]


def test_detection_with_floor_coords_mm():
    """floor_coords_mmを指定してDetectionを作成できる。"""

    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
        floor_coords_mm=(1500.0, 3500.0),
    )

    assert detection.floor_coords_mm == (1500.0, 3500.0)


def test_frame_result_creation():
    """FrameResultデータクラスを正しく作成できる。"""

    detections = [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
            zone_ids=["zone_a"],
        ),
    ]

    frame_result = FrameResult(
        frame_number=0,
        timestamp="12:00",
        detections=detections,
        zone_counts={"zone_a": 1, "zone_b": 0},
    )

    assert frame_result.frame_number == 0
    assert frame_result.timestamp == "12:00"
    assert len(frame_result.detections) == 1
    assert frame_result.zone_counts == {"zone_a": 1, "zone_b": 0}


def test_frame_result_empty_detections():
    """検出結果が空でもFrameResultを作成できる。"""

    frame_result = FrameResult(
        frame_number=0,
        timestamp="12:00",
        detections=[],
        zone_counts={"zone_a": 0},
    )

    assert len(frame_result.detections) == 0
    assert frame_result.zone_counts == {"zone_a": 0}


def test_aggregation_result_creation():
    """AggregationResultデータクラスを正しく作成できる。"""

    result = AggregationResult(
        timestamp="12:00",
        zone_id="zone_a",
        count=5,
    )

    assert result.timestamp == "12:00"
    assert result.zone_id == "zone_a"
    assert result.count == 5


def test_aggregation_result_zero_count():
    """countが0でもAggregationResultを作成できる。"""

    result = AggregationResult(
        timestamp="12:00",
        zone_id="zone_a",
        count=0,
    )

    assert result.count == 0


def test_evaluation_metrics_creation():
    """EvaluationMetricsデータクラスを正しく作成できる。"""

    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.90,
        f1_score=0.875,
        true_positives=85,
        false_positives=15,
        false_negatives=10,
        confidence_threshold=0.5,
    )

    assert metrics.precision == 0.85
    assert metrics.recall == 0.90
    assert metrics.f1_score == 0.875
    assert metrics.true_positives == 85
    assert metrics.false_positives == 15
    assert metrics.false_negatives == 10
    assert metrics.confidence_threshold == 0.5


def test_evaluation_metrics_zero_values():
    """全ての値が0でもEvaluationMetricsを作成できる。"""

    metrics = EvaluationMetrics(
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        confidence_threshold=0.5,
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_evaluation_metrics_high_values():
    """高い値でもEvaluationMetricsを作成できる。"""

    metrics = EvaluationMetrics(
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        true_positives=100,
        false_positives=0,
        false_negatives=0,
        confidence_threshold=0.9,
    )

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0


def test_detection_mutable_fields():
    """Detectionのフィールドを変更できる。"""

    detection = Detection(
        bbox=(100.0, 200.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 300.0),
    )

    detection.floor_coords = (150.0, 350.0)
    detection.zone_ids.append("zone_a")

    assert detection.floor_coords == (150.0, 350.0)
    assert detection.zone_ids == ["zone_a"]


def test_frame_result_mutable_fields():
    """FrameResultのフィールドを変更できる。"""

    frame_result = FrameResult(
        frame_number=0,
        timestamp="12:00",
        detections=[],
        zone_counts={},
    )

    frame_result.zone_counts["zone_a"] = 5

    assert frame_result.zone_counts["zone_a"] == 5
