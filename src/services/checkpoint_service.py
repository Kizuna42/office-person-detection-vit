"""チェックポイント保存/読み込みを担当するサービス。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils import CheckpointManager

if TYPE_CHECKING:
    from pathlib import Path


class CheckpointService:
    """フェーズ進捗の保存と復元を行う薄いラッパー。"""

    def __init__(self, base_dir: Path):
        self.manager = CheckpointManager(base_dir)

    def save(self, phase_name: str, data: dict[str, Any] | None = None) -> None:
        self.manager.save_checkpoint(phase_name, data or {})

    def summary(self) -> dict:
        return self.manager.get_summary()

    def last_completed(self) -> str | None:
        return self.manager.get_last_completed_phase()
