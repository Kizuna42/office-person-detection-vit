"""Performance monitoring utilities for pipeline phases."""

from contextlib import contextmanager
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """パフォーマンスモニタリングクラス

    各フェーズの処理時間、メモリ使用量などを計測します。
    """

    def __init__(self):
        """PerformanceMonitorを初期化"""
        self.metrics: dict[str, dict] = {}
        self.start_times: dict[str, float] = {}

    @contextmanager
    def measure(self, operation_name: str):
        """処理時間を計測するコンテキストマネージャー

        Args:
            operation_name: 操作名（例: "ocr_extraction", "detection_batch"）

        Example:
            with monitor.measure("ocr_extraction"):
                result = extract_timestamp(frame)
        """
        start_time = time.time()
        self.start_times[operation_name] = start_time
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if operation_name not in self.metrics:
                self.metrics[operation_name] = {
                    "total_time": 0.0,
                    "count": 0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                }
            self.metrics[operation_name]["total_time"] += elapsed
            self.metrics[operation_name]["count"] += 1
            self.metrics[operation_name]["min_time"] = min(self.metrics[operation_name]["min_time"], elapsed)
            self.metrics[operation_name]["max_time"] = max(self.metrics[operation_name]["max_time"], elapsed)
            del self.start_times[operation_name]

    def get_metrics(self, operation_name: Optional[str] = None) -> dict:
        """メトリクスを取得

        Args:
            operation_name: 操作名（Noneの場合は全メトリクスを返す）

        Returns:
            メトリクスの辞書
        """
        if operation_name:
            return self.metrics.get(operation_name, {})
        return self.metrics.copy()

    def get_summary(self) -> dict:
        """サマリー統計を取得

        Returns:
            サマリー統計の辞書
        """
        summary = {}
        for op_name, metrics in self.metrics.items():
            if metrics["count"] > 0:
                summary[op_name] = {
                    "total_time": metrics["total_time"],
                    "count": metrics["count"],
                    "avg_time": metrics["total_time"] / metrics["count"],
                    "min_time": metrics["min_time"],
                    "max_time": metrics["max_time"],
                }
        return summary

    def log_summary(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """サマリーをログ出力

        Args:
            logger_instance: ロガーインスタンス（Noneの場合はデフォルトロガー）
        """
        log = logger_instance or logger
        summary = self.get_summary()

        if not summary:
            log.info("パフォーマンスメトリクスがありません")
            return

        log.info("=" * 80)
        log.info("パフォーマンスサマリー:")
        for op_name, stats in summary.items():
            log.info(f"  {op_name}:")
            log.info(f"    実行回数: {stats['count']}")
            log.info(f"    総処理時間: {stats['total_time']:.3f}秒")
            log.info(f"    平均処理時間: {stats['avg_time']:.3f}秒")
            log.info(f"    最小処理時間: {stats['min_time']:.3f}秒")
            log.info(f"    最大処理時間: {stats['max_time']:.3f}秒")
        log.info("=" * 80)

    def reset(self) -> None:
        """メトリクスをリセット"""
        self.metrics.clear()
        self.start_times.clear()
