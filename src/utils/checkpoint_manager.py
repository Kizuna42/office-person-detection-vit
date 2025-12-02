"""Checkpoint management for pipeline recovery."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """パイプラインのチェックポイント管理

    長時間処理の途中で中断した場合に、中断したフェーズから再開できるようにする。

    Attributes:
        checkpoint_dir: チェックポイントファイルの保存ディレクトリ
    """

    CHECKPOINT_FILE = "pipeline_checkpoint.json"

    def __init__(self, session_dir: Path):
        """CheckpointManagerを初期化

        Args:
            session_dir: セッションディレクトリ
        """
        self.session_dir = session_dir
        self.checkpoint_path = session_dir / self.CHECKPOINT_FILE

    def save_checkpoint(
        self,
        phase: str,
        data: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        """フェーズ完了時にチェックポイントを保存

        Args:
            phase: フェーズ名（例: "phase1_extraction"）
            data: 保存するデータ（オプション）
            status: ステータス（"completed", "in_progress", "failed"）
        """
        checkpoint = self._load_or_create_checkpoint()

        checkpoint["phases"][phase] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        }
        checkpoint["last_updated"] = datetime.now().isoformat()
        checkpoint["last_phase"] = phase

        self._save_checkpoint(checkpoint)
        logger.info(f"チェックポイント保存: {phase} ({status})")

    def load_checkpoint(self, phase: str) -> dict[str, Any] | None:
        """指定フェーズのチェックポイントデータを読み込み

        Args:
            phase: フェーズ名

        Returns:
            チェックポイントデータ。存在しない場合はNone
        """
        checkpoint = self._load_or_create_checkpoint()

        phase_data = checkpoint.get("phases", {}).get(phase)
        if phase_data and phase_data.get("status") == "completed":
            logger.info(f"チェックポイント読み込み: {phase}")
            return cast("dict[str, Any]", phase_data.get("data"))

        return None

    def is_phase_completed(self, phase: str) -> bool:
        """指定フェーズが完了しているか確認

        Args:
            phase: フェーズ名

        Returns:
            完了している場合はTrue
        """
        checkpoint = self._load_or_create_checkpoint()
        phase_data = checkpoint.get("phases", {}).get(phase)
        return phase_data is not None and phase_data.get("status") == "completed"

    def get_last_completed_phase(self) -> str | None:
        """最後に完了したフェーズ名を取得

        Returns:
            最後に完了したフェーズ名。完了したフェーズがない場合はNone
        """
        checkpoint = self._load_or_create_checkpoint()
        phases = checkpoint.get("phases", {})

        # フェーズの順序を定義（番号プレフィックス形式）
        phase_order = [
            "01_extraction",
            "02_detection",
            "03_tracking",
            "04_transform",
            "05_aggregation",
            "06_visualization",
        ]

        last_completed = None
        for phase in phase_order:
            if phases.get(phase, {}).get("status") == "completed":
                last_completed = phase

        return last_completed

    def get_resumable_phase(self) -> str | None:
        """再開可能なフェーズを取得

        Returns:
            再開すべきフェーズ名。最初から開始する場合はNone
        """
        last_completed = self.get_last_completed_phase()
        if last_completed is None:
            return None

        # フェーズの順序を定義（番号プレフィックス形式）
        phase_order = [
            "01_extraction",
            "02_detection",
            "03_tracking",
            "04_transform",
            "05_aggregation",
            "06_visualization",
        ]

        try:
            last_index = phase_order.index(last_completed)
            if last_index + 1 < len(phase_order):
                return phase_order[last_index + 1]
        except ValueError:
            pass

        return None

    def clear_checkpoint(self) -> None:
        """チェックポイントをクリア"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("チェックポイントをクリアしました")

    def get_summary(self) -> dict[str, Any]:
        """チェックポイントのサマリーを取得

        Returns:
            チェックポイントサマリー
        """
        checkpoint = self._load_or_create_checkpoint()
        return {
            "session_dir": str(self.session_dir),
            "last_updated": checkpoint.get("last_updated"),
            "last_phase": checkpoint.get("last_phase"),
            "completed_phases": [
                phase for phase, data in checkpoint.get("phases", {}).items() if data.get("status") == "completed"
            ],
            "resumable_from": self.get_resumable_phase(),
        }

    def _load_or_create_checkpoint(self) -> dict[str, Any]:
        """チェックポイントファイルを読み込みまたは作成

        Returns:
            チェックポイントデータ
        """
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, encoding="utf-8") as f:
                    return cast("dict[str, Any]", json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"チェックポイント読み込みエラー: {e}")

        # 全フェーズをpendingステータスで初期化
        initial_phases: dict[str, dict[str, Any]] = {
            "01_extraction": {"status": "pending", "timestamp": None, "data": {}},
            "02_detection": {"status": "pending", "timestamp": None, "data": {}},
            "03_tracking": {"status": "pending", "timestamp": None, "data": {}},
            "04_transform": {"status": "pending", "timestamp": None, "data": {}},
            "05_aggregation": {"status": "pending", "timestamp": None, "data": {}},
            "06_visualization": {"status": "pending", "timestamp": None, "data": {}},
        }

        return {
            "session_dir": str(self.session_dir),
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "last_phase": None,
            "phases": initial_phases,
        }

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """チェックポイントをファイルに保存

        Args:
            checkpoint: チェックポイントデータ
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
