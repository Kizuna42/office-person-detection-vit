"""Base class for pipeline phases."""

from abc import ABC, abstractmethod
import logging

from src.config import ConfigManager


class BasePhase(ABC):
    """パイプラインフェーズの基底クラス

    全てのPhaseクラスが共通して持つ機能を提供します。
    """

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        self.config = config
        self.logger = logger

    def log_phase_start(self, phase_name: str) -> None:
        """フェーズ開始のログを出力

        Args:
            phase_name: フェーズ名（例: "フェーズ2: ViT人物検出"）
        """
        self.logger.info("=" * 80)
        self.logger.info(phase_name)
        self.logger.info("=" * 80)

    @abstractmethod
    def execute(self, *args, **kwargs):
        """フェーズの実行処理（サブクラスで実装）

        Raises:
            NotImplementedError: サブクラスで実装されていない場合
        """
        raise NotImplementedError("Subclass must implement execute method")

    def cleanup(self) -> None:
        """リソースのクリーンアップ（オプション）

        サブクラスで必要に応じてオーバーライドします。
        """
