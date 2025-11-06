"""Unit tests for EvaluationModule."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation import EvaluationModule
from src.models import Detection, EvaluationMetrics


def _make_detection(bbox, confidence=0.9) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_id=1,
        class_name="person",
        camera_coords=(bbox[0], bbox[1]),
    )


def test_load_ground_truth_file_not_found(tmp_path: Path):
    """Ground Truth ファイルが存在しない場合は FileNotFoundError。"""

    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        EvaluationModule(str(missing_path))


def test_calculate_iou_overlap(ground_truth_path: Path):
    """IoU 計算が期待値になる。"""

    module = EvaluationModule(str(ground_truth_path))
    iou = module.calculate_iou((0, 0, 2, 2), (1, 1, 2, 2))
    assert iou == pytest.approx(1 / 7)


def test_calculate_metrics_handles_zero(ground_truth_path: Path):
    """TP, FP, FN がゼロのときもゼロ除算せずに計算できる。"""

    module = EvaluationModule(str(ground_truth_path))
    metrics = module.calculate_metrics(0, 0, 0)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_evaluate_metrics(ground_truth_path: Path):
    """検出結果との比較で Precision / Recall / F1 を計算する。"""

    module = EvaluationModule(str(ground_truth_path))
    detections = {
        "frame1.jpg": [
            _make_detection((100, 200, 50, 100), confidence=0.9),  # True Positive
            _make_detection((0, 0, 10, 10), confidence=0.8),  # False Positive
        ]
    }

    metrics = module.evaluate(detections)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1_score == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_export_report_csv_and_json(tmp_path: Path, ground_truth_path: Path):
    """評価結果を CSV / JSON で出力できる。"""

    module = EvaluationModule(str(ground_truth_path))
    metrics = EvaluationMetrics(
        precision=0.8,
        recall=0.7,
        f1_score=0.75,
        true_positives=8,
        false_positives=2,
        false_negatives=3,
        confidence_threshold=0.5,
    )

    csv_path = tmp_path / "report.csv"
    json_path = tmp_path / "report.json"

    module.export_report(metrics, str(csv_path), format="csv")
    module.export_report(metrics, str(json_path), format="json")

    assert csv_path.read_text(encoding="utf-8").startswith("Metric,Value")
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")
