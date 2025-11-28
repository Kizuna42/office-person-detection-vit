"""Unit tests for transform evaluator module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.transform_evaluator import EvaluationMetrics, TransformEvaluator


class TestEvaluationMetrics:
    """EvaluationMetricsデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        metrics = EvaluationMetrics()
        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0
        assert metrics.max_error == 0.0
        assert metrics.min_error == 0.0
        assert metrics.std_error == 0.0
        assert metrics.percentile_90 == 0.0
        assert metrics.percentile_95 == 0.0
        assert metrics.num_points == 0
        assert metrics.num_valid == 0
        assert metrics.valid_ratio == 0.0
        assert metrics.per_point_errors == []

    def test_to_dict(self):
        """辞書変換テスト"""
        metrics = EvaluationMetrics(
            rmse=10.5,
            mae=8.2,
            max_error=25.0,
            min_error=2.0,
            std_error=5.5,
            percentile_90=18.0,
            percentile_95=22.0,
            num_points=100,
            num_valid=95,
            valid_ratio=0.95,
        )
        d = metrics.to_dict()

        assert d["rmse"] == 10.5
        assert d["mae"] == 8.2
        assert d["max_error"] == 25.0
        assert d["min_error"] == 2.0
        assert d["std_error"] == 5.5
        assert d["percentile_90"] == 18.0
        assert d["percentile_95"] == 22.0
        assert d["num_points"] == 100
        assert d["num_valid"] == 95
        assert d["valid_ratio"] == 0.95

    def test_meets_target_pass(self):
        """目標達成テスト（合格）"""
        metrics = EvaluationMetrics(
            rmse=15.0,
            max_error=40.0,
        )
        assert metrics.meets_target(rmse_target=20.0, max_error_target=50.0) is True

    def test_meets_target_fail_rmse(self):
        """目標達成テスト（RMSE不合格）"""
        metrics = EvaluationMetrics(
            rmse=25.0,
            max_error=40.0,
        )
        assert metrics.meets_target(rmse_target=20.0, max_error_target=50.0) is False

    def test_meets_target_fail_max_error(self):
        """目標達成テスト（最大誤差不合格）"""
        metrics = EvaluationMetrics(
            rmse=15.0,
            max_error=60.0,
        )
        assert metrics.meets_target(rmse_target=20.0, max_error_target=50.0) is False

    def test_meets_target_default_thresholds(self):
        """目標達成テスト（デフォルト閾値）"""
        metrics = EvaluationMetrics(rmse=15.0, max_error=40.0)
        assert metrics.meets_target() is True

        metrics_fail = EvaluationMetrics(rmse=25.0, max_error=60.0)
        assert metrics_fail.meets_target() is False

    def test_summary_ok(self):
        """サマリー文字列テスト（OK）"""
        metrics = EvaluationMetrics(
            rmse=15.0,
            mae=12.0,
            max_error=40.0,
            num_points=100,
            num_valid=95,
        )
        summary = metrics.summary()

        assert "RMSE: 15.00px" in summary
        assert "MAE: 12.00px" in summary
        assert "Max: 40.00px" in summary
        assert "Valid: 95/100" in summary
        assert "OK" in summary

    def test_summary_ng(self):
        """サマリー文字列テスト（NG）"""
        metrics = EvaluationMetrics(
            rmse=25.0,
            mae=20.0,
            max_error=60.0,
            num_points=100,
            num_valid=90,
        )
        summary = metrics.summary()

        assert "NG" in summary


class TestTransformEvaluator:
    """TransformEvaluatorのテスト"""

    @pytest.fixture
    def sample_points(self):
        """サンプル対応点"""
        return [
            {"src_point": [100, 200], "dst_point": [150, 250]},
            {"src_point": [200, 300], "dst_point": [250, 350]},
            {"src_point": [300, 400], "dst_point": [350, 450]},
        ]

    @pytest.fixture
    def correspondence_file(self, sample_points, tmp_path):
        """一時対応点ファイル"""
        file_path = tmp_path / "correspondence_points.json"
        data = {"point_correspondences": sample_points}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return file_path

    def test_init_with_points(self, sample_points):
        """対応点リストでの初期化テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)
        assert len(evaluator.points) == 3

    def test_init_with_file(self, correspondence_file):
        """対応点ファイルでの初期化テスト"""
        evaluator = TransformEvaluator(correspondence_file=correspondence_file)
        assert len(evaluator.points) == 3

    def test_init_empty(self):
        """空の初期化テスト"""
        evaluator = TransformEvaluator()
        assert len(evaluator.points) == 0

    def test_load_correspondence_file(self, correspondence_file):
        """対応点ファイル読み込みテスト"""
        evaluator = TransformEvaluator()
        evaluator.load_correspondence_file(correspondence_file)
        assert len(evaluator.points) == 3

    def test_load_correspondence_file_not_found(self):
        """存在しないファイルの読み込みテスト"""
        evaluator = TransformEvaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.load_correspondence_file("/nonexistent/path.json")

    def test_evaluate_empty_points(self):
        """空の対応点での評価テスト"""
        evaluator = TransformEvaluator()
        mock_transformer = MagicMock()
        metrics = evaluator.evaluate(mock_transformer)

        assert metrics.num_points == 0
        assert metrics.num_valid == 0

    def test_evaluate_with_mock_transformer(self, sample_points):
        """モック変換器での評価テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        # 完璧な変換をシミュレート（誤差0）
        mock_transformer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = True

        def transform_side_effect(point):
            result = MagicMock()
            result.is_valid = True
            # 入力と同じ座標を返す（テスト用）
            # sample_pointsのdst_pointと同じ値を返す
            idx = next(i for i, p in enumerate(sample_points) if p["src_point"] == list(point))
            result.floor_coords_px = tuple(sample_points[idx]["dst_point"])
            return result

        mock_transformer.transform_pixel.side_effect = transform_side_effect

        metrics = evaluator.evaluate(mock_transformer)

        assert metrics.num_points == 3
        assert metrics.num_valid == 3
        assert metrics.rmse == pytest.approx(0.0, abs=0.01)

    def test_evaluate_with_errors(self, sample_points):
        """誤差ありの評価テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        mock_transformer = MagicMock()

        def transform_side_effect(point):
            result = MagicMock()
            result.is_valid = True
            # 10pxのオフセットを追加
            idx = next(i for i, p in enumerate(sample_points) if p["src_point"] == list(point))
            dst = sample_points[idx]["dst_point"]
            result.floor_coords_px = (dst[0] + 10.0, dst[1])
            return result

        mock_transformer.transform_pixel.side_effect = transform_side_effect

        metrics = evaluator.evaluate(mock_transformer)

        assert metrics.num_points == 3
        assert metrics.num_valid == 3
        assert metrics.rmse == pytest.approx(10.0, abs=0.1)
        assert metrics.mae == pytest.approx(10.0, abs=0.1)
        assert metrics.max_error == pytest.approx(10.0, abs=0.1)

    def test_evaluate_with_invalid_transforms(self, sample_points):
        """無効な変換結果のテスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        mock_transformer = MagicMock()
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.floor_coords_px = None
        mock_transformer.transform_pixel.return_value = mock_result

        metrics = evaluator.evaluate(mock_transformer)

        assert metrics.num_points == 3
        assert metrics.num_valid == 0
        assert metrics.valid_ratio == 0.0

    def test_evaluate_homography_perfect(self, sample_points):
        """完璧なホモグラフィ評価テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        # サンプル点に完全に合う変換行列（平行移動50,50）
        H = np.array(
            [
                [1.0, 0.0, 50.0],
                [0.0, 1.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        metrics = evaluator.evaluate_homography(H)

        assert metrics.num_points == 3
        assert metrics.num_valid == 3
        assert metrics.rmse == pytest.approx(0.0, abs=0.01)

    def test_evaluate_homography_with_error(self, sample_points):
        """誤差ありのホモグラフィ評価テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        # 平行移動 (60, 50) - X方向に10pxの誤差
        H = np.array(
            [
                [1.0, 0.0, 60.0],
                [0.0, 1.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        metrics = evaluator.evaluate_homography(H)

        assert metrics.num_points == 3
        assert metrics.num_valid == 3
        assert metrics.rmse == pytest.approx(10.0, abs=0.1)

    def test_evaluate_homography_empty_points(self):
        """空の対応点でのホモグラフィ評価テスト"""
        evaluator = TransformEvaluator()
        H = np.eye(3)
        metrics = evaluator.evaluate_homography(H)

        assert metrics.num_points == 0

    def test_evaluate_homography_singular_case(self, sample_points):
        """ホモグラフィ特異ケースのテスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        # w=0 になるケースを作成（実際にはほぼ起こらない）
        H = np.array(
            [
                [1.0, 0.0, 50.0],
                [0.0, 1.0, 50.0],
                [0.0, 0.0, 0.0],  # これだと常にw=0
            ],
            dtype=np.float64,
        )

        metrics = evaluator.evaluate_homography(H)
        # 全点が無効になるはず
        assert metrics.num_valid == 0

    def test_generate_report(self, sample_points):
        """レポート生成テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)
        metrics = EvaluationMetrics(
            rmse=15.0,
            mae=12.0,
            max_error=25.0,
            min_error=5.0,
            std_error=6.0,
            percentile_90=20.0,
            percentile_95=23.0,
            num_points=3,
            num_valid=3,
            valid_ratio=1.0,
            per_point_errors=[
                {
                    "index": 0,
                    "src_point": [100, 200],
                    "dst_expected": [150, 250],
                    "dst_actual": [160, 250],
                    "error": 10.0,
                    "error_vector": [10.0, 0.0],
                    "is_valid": True,
                },
            ],
        )

        report = evaluator.generate_report(metrics)

        assert "座標変換精度評価レポート" in report
        assert "RMSE" in report
        assert "15.00" in report
        assert "目標達成" in report

    def test_generate_report_to_file(self, sample_points, tmp_path):
        """レポートファイル出力テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)
        metrics = EvaluationMetrics(
            rmse=15.0,
            mae=12.0,
            max_error=25.0,
            num_points=3,
            num_valid=3,
            valid_ratio=1.0,
        )

        output_path = tmp_path / "report.md"
        report = evaluator.generate_report(metrics, output_path=output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert content == report

    def test_cross_validate_insufficient_points(self):
        """不十分な点数でのクロスバリデーションテスト"""
        points = [{"src_point": [100, 200], "dst_point": [150, 250]}]
        evaluator = TransformEvaluator(correspondence_points=points)

        result = evaluator.cross_validate(lambda pts: MagicMock(), k_folds=5)
        assert result == {}

    def test_per_point_errors_tracking(self, sample_points):
        """点ごとの誤差追跡テスト"""
        evaluator = TransformEvaluator(correspondence_points=sample_points)

        mock_transformer = MagicMock()

        def transform_side_effect(point):
            result = MagicMock()
            result.is_valid = True
            idx = next(i for i, p in enumerate(sample_points) if p["src_point"] == list(point))
            dst = sample_points[idx]["dst_point"]
            # 各点に異なるオフセット
            result.floor_coords_px = (dst[0] + (idx + 1) * 5.0, dst[1])
            return result

        mock_transformer.transform_pixel.side_effect = transform_side_effect

        metrics = evaluator.evaluate(mock_transformer)

        assert len(metrics.per_point_errors) == 3
        assert all(e["is_valid"] for e in metrics.per_point_errors)
        # 誤差は5, 10, 15 なのでmin=5, max=15
        assert metrics.min_error == pytest.approx(5.0, abs=0.1)
        assert metrics.max_error == pytest.approx(15.0, abs=0.1)
