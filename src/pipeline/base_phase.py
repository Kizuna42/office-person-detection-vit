"""Base class for pipeline phases."""

import logging
from abc import ABC, abstractmethod

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
        pass
