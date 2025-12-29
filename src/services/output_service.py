"""出力ディレクトリ管理とメタデータ保存を担当するサービス。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils import OutputManager, setup_output_directories

if TYPE_CHECKING:
    from src.config import ConfigManager


class OutputService:
    """出力パスとセッション管理の窓口。"""

    def __init__(self, logger, base_output: str | Path = "output", use_phase_subdir: bool = False):
        self.logger = logger
        self.base_output = Path(base_output)
        self.output_manager: OutputManager | None = None
        self.session_dir: Path | None = None
        self.use_phase_subdir = use_phase_subdir

    def setup(self, use_session_management: bool, config: ConfigManager, args: Any | None = None) -> Path:
        """出力ディレクトリを初期化し、セッションを必要に応じて作成する。"""
        if use_session_management:
            self.output_manager = OutputManager(self.base_output)
            self.session_dir = self.output_manager.create_session()
            self.logger.info(f"セッション管理を有効化しました: {self.session_dir.name}")

            config_dict = config.config if hasattr(config, "config") else {}
            args_dict = vars(args) if args else {}
            self.output_manager.save_metadata(self.session_dir, config_dict, args_dict)
            self.output_manager.update_latest_link(self.session_dir)
            self.base_output = self.session_dir
        else:
            self.logger.info("セッション管理は無効です（従来の出力構造を使用）")
            setup_output_directories(self.base_output)

        self.logger.info(f"出力ディレクトリ: {self.base_output.absolute()}")
        return self.base_output

    def get_phase_dir(self, phase_name: str) -> Path:
        """フェーズ名に基づく出力ディレクトリを返す。"""
        root = self.session_dir if self.session_dir else self.base_output
        if self.use_phase_subdir:
            return root / "phases" / phase_name
        return root / phase_name

    def save_summary(self, summary: dict) -> Path | None:
        """セッションサマリーを保存（セッション管理時のみ）。"""
        if not self.output_manager or not self.session_dir:
            return None
        self.output_manager.save_summary(self.session_dir, summary)
        self.output_manager.update_latest_link(self.session_dir)
        return self.session_dir / "summary.json"
