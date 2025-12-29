"""TrackingBenchmark のユニットテスト"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.evaluation.tracking_benchmark import (
    IDSwitchEvent,
    TrackingBenchmark,
    TrackingDiagnostics,
    TrackingMetrics,
)


class TestTrackingMetrics:
    """TrackingMetrics のテスト"""

    def test_to_dict(self) -> None:
        """to_dict が正しい辞書を返すこと"""
        metrics = TrackingMetrics(
            mota=0.8,
            idf1=0.75,
            idsw=5,
            fp=10,
            fn=15,
            gt_count=100,
            num_frames=50,
            num_transitions=49,
            idsw_per_transition=0.102,
        )
        result = metrics.to_dict()

        assert result["mota"] == 0.8
        assert result["idf1"] == 0.75
        assert result["idsw"] == 5
        assert result["idsw_per_transition"] == 0.102

    def test_summary(self) -> None:
        """summary が文字列を返すこと"""
        metrics = TrackingMetrics(mota=0.8, idf1=0.75, idsw=5)
        summary = metrics.summary()

        assert "MOTA" in summary
        assert "IDF1" in summary
        assert "IDSW" in summary


class TestTrackingDiagnostics:
    """TrackingDiagnostics のテスト"""

    def test_to_dict(self) -> None:
        """to_dict が正しい形式を返すこと"""
        diag = TrackingDiagnostics(
            id_switches=[
                IDSwitchEvent(
                    frame_idx=10,
                    timestamp="2024-01-15T10:00:00",
                    old_track_id=1,
                    new_track_id=2,
                    gt_id=1,
                    bbox=(100, 200, 50, 100),
                    reason="low_similarity",
                )
            ],
            false_positives=[{"frame_idx": 5, "pred_id": 3}],
        )
        result = diag.to_dict()

        assert len(result["id_switches"]) == 1
        assert result["id_switches"][0]["old_track_id"] == 1
        assert result["summary"]["total_id_switches"] == 1
        assert result["summary"]["total_false_positives"] == 1

    def test_export_jsonl(self) -> None:
        """JSONL形式でエクスポートできること"""
        diag = TrackingDiagnostics(
            id_switches=[
                IDSwitchEvent(
                    frame_idx=10,
                    timestamp=None,
                    old_track_id=1,
                    new_track_id=2,
                    gt_id=1,
                    bbox=(100, 200, 50, 100),
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = diag.export_jsonl(output_dir)

            assert "id_switches" in paths
            assert paths["id_switches"].exists()

            # JSONLを読み込んで検証
            with paths["id_switches"].open() as f:
                lines = f.readlines()
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["frame_idx"] == 10
            assert data["old_track_id"] == 1


class TestTrackingBenchmark:
    """TrackingBenchmark のテスト"""

    @pytest.fixture
    def sample_gt_df(self) -> pd.DataFrame:
        """サンプルGT DataFrame"""
        return pd.DataFrame(
            {
                "FrameId": [1, 1, 2, 2, 3, 3],
                "Id": [1, 2, 1, 2, 1, 2],
                "X": [100, 200, 110, 210, 120, 220],
                "Y": [100, 100, 100, 100, 100, 100],
                "Width": [50, 50, 50, 50, 50, 50],
                "Height": [100, 100, 100, 100, 100, 100],
                "Confidence": [1.0] * 6,
            }
        )

    @pytest.fixture
    def sample_pred_df(self) -> pd.DataFrame:
        """サンプル予測 DataFrame (完全一致)"""
        return pd.DataFrame(
            {
                "FrameId": [1, 1, 2, 2, 3, 3],
                "Id": [1, 2, 1, 2, 1, 2],
                "X": [100, 200, 110, 210, 120, 220],
                "Y": [100, 100, 100, 100, 100, 100],
                "Width": [50, 50, 50, 50, 50, 50],
                "Height": [100, 100, 100, 100, 100, 100],
                "Confidence": [0.9] * 6,
            }
        )

    def test_evaluate_perfect_match(self, sample_gt_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> None:
        """完全一致時のメトリクス"""
        benchmark = TrackingBenchmark(iou_threshold=0.5)
        metrics = benchmark.evaluate(sample_gt_df, sample_pred_df)

        assert metrics.mota == pytest.approx(1.0)
        assert metrics.idf1 == pytest.approx(1.0)
        assert metrics.idsw == 0
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.num_frames == 3
        assert metrics.num_transitions == 2

    def test_evaluate_with_id_switch(self, sample_gt_df: pd.DataFrame) -> None:
        """ID Switchがある場合"""
        # 予測: フレーム2でID1とID2が入れ替わる
        pred_df = pd.DataFrame(
            {
                "FrameId": [1, 1, 2, 2, 3, 3],
                "Id": [1, 2, 2, 1, 2, 1],  # ID swapped at frame 2
                "X": [100, 200, 110, 210, 120, 220],
                "Y": [100, 100, 100, 100, 100, 100],
                "Width": [50, 50, 50, 50, 50, 50],
                "Height": [100, 100, 100, 100, 100, 100],
                "Confidence": [0.9] * 6,
            }
        )
        benchmark = TrackingBenchmark(iou_threshold=0.5)
        metrics = benchmark.evaluate(sample_gt_df, pred_df)

        # ID Switchが発生しているはず
        assert metrics.idsw > 0

    def test_evaluate_with_false_positives(self, sample_gt_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> None:
        """False Positiveがある場合"""
        # 予測に余分な検出を追加
        extra = pd.DataFrame(
            {
                "FrameId": [1],
                "Id": [99],
                "X": [500],
                "Y": [500],
                "Width": [50],
                "Height": [100],
                "Confidence": [0.9],
            }
        )
        pred_with_fp = pd.concat([sample_pred_df, extra], ignore_index=True)

        benchmark = TrackingBenchmark(iou_threshold=0.5)
        metrics = benchmark.evaluate(sample_gt_df, pred_with_fp)

        assert metrics.fp >= 1

    def test_evaluate_sparse_mode(self, sample_gt_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> None:
        """疎サンプリングモード"""
        benchmark = TrackingBenchmark(sparse_mode=True)
        metrics = benchmark.evaluate(sample_gt_df, sample_pred_df)

        assert metrics.idsw_per_transition == 0.0
        assert metrics.num_transitions == 2

    def test_diagnostics_collection(self, sample_gt_df: pd.DataFrame) -> None:
        """診断情報の収集"""
        # 一部検出が欠けている予測
        pred_df = pd.DataFrame(
            {
                "FrameId": [1, 2, 3],
                "Id": [1, 1, 1],
                "X": [100, 110, 120],
                "Y": [100, 100, 100],
                "Width": [50, 50, 50],
                "Height": [100, 100, 100],
                "Confidence": [0.9] * 3,
            }
        )
        benchmark = TrackingBenchmark(output_diagnostics=True)
        benchmark.evaluate(sample_gt_df, pred_df)

        diag = benchmark.get_diagnostics()
        # ID=2が全フレームで見逃されている
        assert len(diag.missed_detections) == 3

    def test_export_results(self, sample_gt_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> None:
        """結果のエクスポート"""
        benchmark = TrackingBenchmark()
        metrics = benchmark.evaluate(sample_gt_df, sample_pred_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = benchmark.export_results(metrics, output_dir)

            assert "metrics" in paths
            assert paths["metrics"].exists()

            # メトリクスファイルの内容を検証
            with paths["metrics"].open() as f:
                data = json.load(f)
            assert "metrics" in data
            assert data["metrics"]["mota"] == pytest.approx(1.0)

    def test_generate_report(self, sample_gt_df: pd.DataFrame, sample_pred_df: pd.DataFrame) -> None:
        """Markdownレポート生成"""
        benchmark = TrackingBenchmark(sparse_mode=True)
        metrics = benchmark.evaluate(sample_gt_df, sample_pred_df)

        report = benchmark.generate_report(metrics)

        assert "トラッキング評価レポート" in report
        assert "MOTA" in report
        assert "疎サンプリング" in report


class TestGoldGTFormat:
    """Gold GT形式のテスト"""

    def test_load_gold_gt(self) -> None:
        """Gold GT形式を読み込めること"""
        gold_gt = {
            "version": "1.0",
            "frames": [
                {
                    "frame_idx": 0,
                    "timestamp": "2024-01-15T10:00:00",
                    "annotations": [
                        {"person_id": 1, "bbox": [100, 100, 50, 100]},
                        {"person_id": 2, "bbox": [200, 100, 50, 100]},
                    ],
                },
                {
                    "frame_idx": 1,
                    "annotations": [
                        {"person_id": 1, "bbox": [110, 100, 50, 100]},
                    ],
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gold_gt, f)
            gt_path = Path(f.name)

        try:
            benchmark = TrackingBenchmark()
            df = benchmark._load_gold_gt(gt_path)

            assert len(df) == 3
            assert df["FrameId"].tolist() == [1, 1, 2]  # 1-indexed
            assert set(df["Id"].tolist()) == {1, 2}
        finally:
            gt_path.unlink()
