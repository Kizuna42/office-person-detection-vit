"""Unit tests for performance monitor module."""

import logging
import time
from unittest.mock import MagicMock

import pytest

from src.utils.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    """PerformanceMonitorのテスト"""

    def test_init(self):
        """初期化テスト"""
        monitor = PerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.start_times == {}

    def test_measure_context_manager(self):
        """コンテキストマネージャーとしての計測テスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)  # 10ms待機

        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["count"] == 1
        assert monitor.metrics["test_operation"]["total_time"] > 0
        assert monitor.metrics["test_operation"]["min_time"] > 0
        assert monitor.metrics["test_operation"]["max_time"] > 0

    def test_measure_multiple_operations(self):
        """複数の操作を計測するテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("operation1"):
            time.sleep(0.01)

        with monitor.measure("operation2"):
            time.sleep(0.01)

        assert "operation1" in monitor.metrics
        assert "operation2" in monitor.metrics
        assert monitor.metrics["operation1"]["count"] == 1
        assert monitor.metrics["operation2"]["count"] == 1

    def test_measure_same_operation_multiple_times(self):
        """同じ操作を複数回計測するテスト"""
        monitor = PerformanceMonitor()

        for _ in range(3):
            with monitor.measure("test_operation"):
                time.sleep(0.01)

        assert monitor.metrics["test_operation"]["count"] == 3
        assert monitor.metrics["test_operation"]["total_time"] > 0
        assert monitor.metrics["test_operation"]["min_time"] > 0
        assert monitor.metrics["test_operation"]["max_time"] > 0

    def test_measure_exception_handling(self):
        """例外が発生した場合でも計測が完了するテスト"""
        monitor = PerformanceMonitor()

        def _raise_test_error():
            time.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"), monitor.measure("test_operation"):
            _raise_test_error()

        # 例外が発生しても計測は完了する
        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["count"] == 1

    def test_get_metrics_specific_operation(self):
        """特定の操作のメトリクスを取得するテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)

        metrics = monitor.get_metrics("test_operation")
        assert "total_time" in metrics
        assert "count" in metrics
        assert "min_time" in metrics
        assert "max_time" in metrics

    def test_get_metrics_nonexistent_operation(self):
        """存在しない操作のメトリクスを取得するテスト"""
        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics("nonexistent")
        assert metrics == {}

    def test_get_metrics_all(self):
        """全メトリクスを取得するテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("operation1"):
            time.sleep(0.01)

        with monitor.measure("operation2"):
            time.sleep(0.01)

        all_metrics = monitor.get_metrics()
        assert "operation1" in all_metrics
        assert "operation2" in all_metrics
        assert len(all_metrics) == 2

    def test_get_summary(self):
        """サマリー統計を取得するテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)

        summary = monitor.get_summary()
        assert "test_operation" in summary
        assert "total_time" in summary["test_operation"]
        assert "count" in summary["test_operation"]
        assert "avg_time" in summary["test_operation"]
        assert "min_time" in summary["test_operation"]
        assert "max_time" in summary["test_operation"]

    def test_get_summary_empty(self):
        """メトリクスがない場合のサマリーテスト"""
        monitor = PerformanceMonitor()
        summary = monitor.get_summary()
        assert summary == {}

    def test_get_summary_avg_time_calculation(self):
        """平均時間の計算テスト"""
        monitor = PerformanceMonitor()

        for _ in range(3):
            with monitor.measure("test_operation"):
                time.sleep(0.01)

        summary = monitor.get_summary()
        avg_time = summary["test_operation"]["avg_time"]
        total_time = summary["test_operation"]["total_time"]
        count = summary["test_operation"]["count"]

        assert avg_time == pytest.approx(total_time / count, rel=1e-2)

    def test_log_summary(self):
        """サマリーをログ出力するテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)

        # ロガーをモック
        mock_logger = MagicMock(spec=logging.Logger)
        monitor.log_summary(mock_logger)

        # ログが呼ばれていることを確認
        assert mock_logger.info.called

    def test_log_summary_default_logger(self):
        """デフォルトロガーでのサマリー出力テスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)

        # デフォルトロガーでエラーが発生しないことを確認
        monitor.log_summary()

    def test_log_summary_empty(self):
        """メトリクスがない場合のログ出力テスト"""
        monitor = PerformanceMonitor()

        mock_logger = MagicMock(spec=logging.Logger)
        monitor.log_summary(mock_logger)

        # メトリクスがない場合のメッセージが出力される
        assert mock_logger.info.called

    def test_reset(self):
        """メトリクスをリセットするテスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)

        assert len(monitor.metrics) > 0

        monitor.reset()

        assert monitor.metrics == {}
        assert monitor.start_times == {}

    def test_measure_min_max_time(self):
        """最小・最大時間の計測テスト"""
        monitor = PerformanceMonitor()

        # 異なる時間の操作を計測
        with monitor.measure("test_operation"):
            time.sleep(0.01)

        with monitor.measure("test_operation"):
            time.sleep(0.02)

        with monitor.measure("test_operation"):
            time.sleep(0.015)

        assert monitor.metrics["test_operation"]["min_time"] < monitor.metrics["test_operation"]["max_time"]
        assert monitor.metrics["test_operation"]["count"] == 3

    def test_measure_very_short_operation(self):
        """非常に短い操作の計測テスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            pass  # 何もしない

        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["count"] == 1
        assert monitor.metrics["test_operation"]["total_time"] >= 0

    def test_measure_nested_operations(self):
        """ネストされた操作の計測テスト"""
        monitor = PerformanceMonitor()

        with monitor.measure("outer_operation"), monitor.measure("inner_operation"):
            time.sleep(0.01)

        assert "outer_operation" in monitor.metrics
        assert "inner_operation" in monitor.metrics
        # 外側の操作の方が時間が長いはず
        assert monitor.metrics["outer_operation"]["total_time"] >= monitor.metrics["inner_operation"]["total_time"]

    def test_get_summary_multiple_operations(self):
        """複数の操作のサマリー取得テスト"""
        monitor = PerformanceMonitor()

        for op_name in ["op1", "op2", "op3"]:
            with monitor.measure(op_name):
                time.sleep(0.01)

        summary = monitor.get_summary()
        assert len(summary) == 3
        assert all("avg_time" in summary[op] for op in ["op1", "op2", "op3"])
