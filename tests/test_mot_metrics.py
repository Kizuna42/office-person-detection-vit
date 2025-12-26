"""Unit tests for MOT metrics evaluation module (motmetrics)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.mot_metrics import MOTMetrics


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"FrameId": 1, "Id": 1, "X": 10.0, "Y": 20.0, "Width": 30.0, "Height": 40.0, "Confidence": 1.0},
            {"FrameId": 2, "Id": 1, "X": 15.0, "Y": 25.0, "Width": 30.0, "Height": 40.0, "Confidence": 1.0},
        ]
    )


def test_coco_to_mot_dataframe(tmp_path: Path):
    coco = {
        "images": [{"id": 0, "file_name": "frame_000001.jpg"}],
        "annotations": [{"id": 11, "image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40]}],
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(coco), encoding="utf-8")

    metrics = MOTMetrics()
    df = metrics.coco_to_mot_dataframe(gt_path)

    assert len(df) == 1
    assert df.iloc[0]["FrameId"] == 1
    assert df.iloc[0]["Id"] == 11


def test_evaluate_perfect_match():
    metrics = MOTMetrics()
    gt_df = _base_df()
    pred_df = _base_df()

    result = metrics.evaluate_from_dataframes(gt_df, pred_df)

    assert result["FP"] == 0.0
    assert result["FN"] == 0.0
    assert result["IDSW"] == 0.0
    assert result["MOTA"] == pytest.approx(1.0)
    assert result["IDF1"] == pytest.approx(1.0)


def test_evaluate_with_misses():
    metrics = MOTMetrics()
    gt_df = _base_df()
    empty_pred = pd.DataFrame(columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"])

    result = metrics.evaluate_from_dataframes(gt_df, empty_pred)

    assert result["FP"] == 0.0
    assert result["FN"] == len(gt_df)
    assert result["MOTA"] <= 1.0


def test_evaluate_from_files(tmp_path: Path):
    metrics = MOTMetrics()

    # _base_df()と一致するデータ: FrameId=1,2, Id=1（同一トラック）
    coco = {
        "images": [
            {"id": 0, "file_name": "frame_000001.jpg"},
            {"id": 1, "file_name": "frame_000002.jpg"},
        ],
        "annotations": [
            # annotation id=1 → pred_dfのId=1と一致
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [15, 25, 30, 40]},
        ],
    }

    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.csv"
    gt_path.write_text(json.dumps(coco), encoding="utf-8")

    pred_df = _base_df()
    pred_df.to_csv(pred_path, index=False)

    result = metrics.evaluate_from_files(gt_path, pred_path)

    assert result["MOTA"] == pytest.approx(1.0)
    assert result["IDF1"] == pytest.approx(1.0)
