"""性能計測の薄いサービスラッパー。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils import PerformanceMonitor

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class PerformanceService:
    """PerformanceMonitor をカプセル化し、依存注入しやすくする。"""

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def measure(self, name: str) -> AbstractContextManager:
        return self.monitor.measure(name)

    def summary(self) -> dict:
        return self.monitor.get_summary()

    def get_metrics(self, name: str) -> dict | None:
        return self.monitor.get_metrics(name)
