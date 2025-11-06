"""Unit tests for EvaluationModule."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluation_module import EvaluationModule, run_evaluation
from src.models import Detection, EvaluationMetrics


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the directory containing static test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def ground_truth_path(fixtures_dir: Path) -> Path:
    """Return the path to the sample COCO ground truth file."""
    return fixtures_dir / "sample_ground_truth.json"


@pytest.fixture
def sample_detections() -> list[Detection]:
    """サンプル検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
        ),
        Detection(
            bbox=(200.0, 300.0, 60.0, 120.0),
            confidence=0.8,
            class_id=1,
            class_name="person",
            camera_coords=(230.0, 420.0),
        ),
    ]


def test_load_ground_truth_file_not_found(tmp_path: Path):
    """Ground Truthファイルが見つからない場合"""
    with pytest.raises(FileNotFoundError):
        EvaluationModule(str(tmp_path / "nonexistent.json"))


def test_calculate_iou_overlap(ground_truth_path: Path):
    """IoU 計算が期待値になる。"""
    module = EvaluationModule(str(ground_truth_path))

    # 重複しているボックス
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (110.0, 210.0, 50.0, 100.0)

    iou = module.calculate_iou(bbox1, bbox2)

    assert 0.0 <= iou <= 1.0
    assert iou > 0.0  # 重複しているので0より大きい


def test_calculate_metrics_handles_zero(ground_truth_path: Path):
    """TP, FP, FN がゼロのときもゼロ除算せずに計算できる。"""
    module = EvaluationModule(str(ground_truth_path))

    metrics = module.calculate_metrics(0, 0, 0)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_evaluate_metrics(ground_truth_path: Path, sample_detections):
    """検出結果との比較で Precision / Recall / F1 を計算する。"""
    module = EvaluationModule(str(ground_truth_path))

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": sample_detections,
    }

    metrics = module.evaluate(detections_dict)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.true_positives >= 0
    assert metrics.false_positives >= 0
    assert metrics.false_negatives >= 0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0


def test_export_report_csv_and_json(tmp_path: Path, ground_truth_path: Path):
    """CSVとJSONの両方でレポートをエクスポートできる。"""
    module = EvaluationModule(str(ground_truth_path))

    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.90,
        f1_score=0.875,
        true_positives=85,
        false_positives=15,
        false_negatives=10,
        confidence_threshold=0.5,
    )

    csv_path = tmp_path / "report.csv"
    json_path = tmp_path / "report.json"

    module.export_report(metrics, str(csv_path), format="csv")
    module.export_report(metrics, str(json_path), format="json")

    assert csv_path.exists()
    assert json_path.exists()


def test_calculate_iou_edge_cases(ground_truth_path: Path):
    """IoU計算のエッジケーステスト"""
    module = EvaluationModule(str(ground_truth_path))

    # 重複なし
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (20.0, 20.0, 10.0, 10.0)
    iou = module.calculate_iou(bbox1, bbox2)
    assert iou == 0.0

    # 完全包含
    bbox1 = (0.0, 0.0, 100.0, 100.0)
    bbox2 = (10.0, 10.0, 20.0, 20.0)
    iou = module.calculate_iou(bbox1, bbox2)
    assert 0.0 < iou < 1.0

    # 完全一致
    bbox1 = (100.0, 200.0, 50.0, 100.0)
    bbox2 = (100.0, 200.0, 50.0, 100.0)
    iou = module.calculate_iou(bbox1, bbox2)
    assert iou == 1.0

    # ゼロ面積
    bbox1 = (0.0, 0.0, 0.0, 0.0)
    bbox2 = (0.0, 0.0, 10.0, 10.0)
    iou = module.calculate_iou(bbox1, bbox2)
    assert iou == 0.0


def test_evaluate_with_multiple_images(ground_truth_path: Path, sample_detections):
    """複数画像の評価テスト"""
    module = EvaluationModule(str(ground_truth_path))

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": sample_detections,
        "frame_000001_20250826 16h10m00m.jpg": sample_detections[:1],
        "frame_000002_20250826 16h15m00m.jpg": [],
    }

    metrics = module.evaluate(detections_dict)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.true_positives >= 0
    assert metrics.false_positives >= 0
    assert metrics.false_negatives >= 0


def test_evaluate_with_no_detections(ground_truth_path: Path):
    """検出結果が空の場合のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": [],
    }

    metrics = module.evaluate(detections_dict)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives > 0  # Ground Truthは存在する


def test_evaluate_with_all_false_positives(ground_truth_path: Path):
    """すべてFalse Positiveの場合のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    # IoUが閾値未満の検出結果
    false_positive_detections = [
        Detection(
            bbox=(0.0, 0.0, 10.0, 10.0),  # Ground Truthとは重複しない
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5.0, 10.0),
        ),
    ]

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": false_positive_detections,
    }

    metrics = module.evaluate(detections_dict)

    assert metrics.true_positives == 0
    assert metrics.false_positives > 0
    assert metrics.false_negatives > 0


def test_evaluate_with_all_false_negatives(ground_truth_path: Path):
    """すべてFalse Negativeの場合のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": [],  # 検出結果なし
    }

    metrics = module.evaluate(detections_dict)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives > 0  # Ground Truthは存在する


def test_calculate_metrics_edge_cases(ground_truth_path: Path):
    """メトリクス計算のエッジケーステスト"""
    module = EvaluationModule(str(ground_truth_path))

    # TP=0, FP>0, FN=0
    metrics = module.calculate_metrics(0, 10, 0)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0

    # TP>0, FP=0, FN=0 (完璧な検出)
    metrics = module.calculate_metrics(10, 0, 0)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0

    # TP>0, FP>0, FN>0 (一般的なケース)
    metrics = module.calculate_metrics(8, 2, 2)
    assert abs(metrics.precision - 0.8) < 0.0001
    assert abs(metrics.recall - 0.8) < 0.0001
    assert abs(metrics.f1_score - 0.8) < 0.0001


def test_export_report_invalid_format(ground_truth_path: Path, tmp_path: Path):
    """無効な出力形式のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.90,
        f1_score=0.875,
        true_positives=85,
        false_positives=15,
        false_negatives=10,
        confidence_threshold=0.5,
    )

    output_path = tmp_path / "report.txt"

    with pytest.raises(ValueError, match="不正な出力形式"):
        module.export_report(metrics, str(output_path), format="txt")


def test_run_evaluation_helper(ground_truth_path: Path, tmp_path: Path):
    """ヘルパー関数のテスト"""
    from src.config import ConfigManager

    logger = logging.getLogger("test")
    config = ConfigManager("nonexistent_config.yaml")
    config.set("evaluation.ground_truth_path", str(ground_truth_path))
    config.set("evaluation.iou_threshold", 0.5)
    config.set("output.directory", str(tmp_path / "output"))

    detection_results = [
        (
            0,
            "2025/08/26 16:05:00",
            [
                Detection(
                    bbox=(100.0, 200.0, 50.0, 100.0),
                    confidence=0.9,
                    class_id=1,
                    class_name="person",
                    camera_coords=(125.0, 300.0),
                )
            ],
        )
    ]

    run_evaluation(detection_results, config, logger)

    # レポートファイルが生成されることを確認
    output_dir = tmp_path / "output"
    assert (output_dir / "evaluation_report.csv").exists()
    assert (output_dir / "evaluation_report.json").exists()


def test_get_annotations_by_image(ground_truth_path: Path):
    """アノテーション取得のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    annotations = module._get_annotations_by_image(1)

    assert len(annotations) == 2  # image_id=1には2つのアノテーション
    assert all(ann["image_id"] == 1 for ann in annotations)

    # 存在しない画像ID
    annotations = module._get_annotations_by_image(999)
    assert len(annotations) == 0


def test_get_image_by_filename(ground_truth_path: Path):
    """画像情報取得のテスト"""
    module = EvaluationModule(str(ground_truth_path))

    image_info = module._get_image_by_filename("frame_000000_20250826 16h05m00m.jpg")
    assert image_info is not None
    assert image_info["id"] == 1
    assert image_info["file_name"] == "frame_000000_20250826 16h05m00m.jpg"

    # 存在しないファイル名
    image_info = module._get_image_by_filename("nonexistent.jpg")
    assert image_info is None


def test_evaluate_with_missing_image(ground_truth_path: Path, sample_detections):
    """Ground Truthに存在しない画像の評価テスト"""
    module = EvaluationModule(str(ground_truth_path))

    detections_dict = {
        "nonexistent_frame.jpg": sample_detections,
    }

    metrics = module.evaluate(detections_dict)

    # 画像が見つからない場合は警告が出力されるが、処理は続行
    assert isinstance(metrics, EvaluationMetrics)


def test_evaluate_with_non_person_category(ground_truth_path: Path):
    """人物以外のカテゴリのテスト"""
    module = EvaluationModule(str(ground_truth_path))

    # category_id=1（人物以外）のアノテーションは無視される
    detections_dict = {
        "frame_000002_20250826 16h15m00m.jpg": [],  # category_id=1のアノテーションがある
    }

    metrics = module.evaluate(detections_dict)

    # 人物以外のカテゴリは評価対象外
    assert isinstance(metrics, EvaluationMetrics)


def test_calculate_iou_partial_overlap(ground_truth_path: Path):
    """部分的重複のIoU計算テスト"""
    module = EvaluationModule(str(ground_truth_path))

    # 部分的重複
    bbox1 = (0.0, 0.0, 100.0, 100.0)
    bbox2 = (50.0, 50.0, 100.0, 100.0)

    iou = module.calculate_iou(bbox1, bbox2)

    assert 0.0 < iou < 1.0
    # 交差面積: 50x50 = 2500
    # 和集合面積: 100x100 + 100x100 - 2500 = 17500
    # IoU = 2500 / 17500 ≈ 0.143
    assert iou > 0.1


def test_export_csv_format(ground_truth_path: Path, tmp_path: Path):
    """CSV出力フォーマットのテスト"""
    module = EvaluationModule(str(ground_truth_path))

    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.90,
        f1_score=0.875,
        true_positives=85,
        false_positives=15,
        false_negatives=10,
        confidence_threshold=0.5,
    )

    csv_path = tmp_path / "report.csv"
    module._export_csv(metrics, str(csv_path))

    assert csv_path.exists()

    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) > 0
    assert rows[0] == ["Metric", "Value"]
    assert any(row[0] == "Precision" for row in rows)
    assert any(row[0] == "Recall" for row in rows)
    assert any(row[0] == "F1-score" for row in rows)


def test_export_json_format(ground_truth_path: Path, tmp_path: Path):
    """JSON出力フォーマットのテスト"""
    module = EvaluationModule(str(ground_truth_path))

    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.90,
        f1_score=0.875,
        true_positives=85,
        false_positives=15,
        false_negatives=10,
        confidence_threshold=0.5,
    )

    json_path = tmp_path / "report.json"
    module._export_json(metrics, str(json_path))

    assert json_path.exists()

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert data["precision"] == 0.85
    assert data["iou_threshold"] == 0.5


def test_iou_threshold_parameter(ground_truth_path: Path, sample_detections):
    """IoU閾値パラメータのテスト"""
    # 低い閾値
    module_low = EvaluationModule(str(ground_truth_path), iou_threshold=0.3)

    detections_dict = {
        "frame_000000_20250826 16h05m00m.jpg": sample_detections,
    }

    metrics_low = module_low.evaluate(detections_dict)

    # 高い閾値
    module_high = EvaluationModule(str(ground_truth_path), iou_threshold=0.9)

    metrics_high = module_high.evaluate(detections_dict)

    # 低い閾値の方がTrue Positiveが多い可能性が高い
    assert isinstance(metrics_low, EvaluationMetrics)
    assert isinstance(metrics_high, EvaluationMetrics)

