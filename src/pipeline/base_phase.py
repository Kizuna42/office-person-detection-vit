"""Base class for pipeline phases."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import FrameSampler, VideoProcessor


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

    def _setup_frame_sampling_components(
        self, video_processor: Optional[VideoProcessor] = None
    ) -> Tuple[VideoProcessor, TimestampExtractor, FrameSampler]:
        """フレームサンプリングに必要なコンポーネントを初期化

        Args:
            video_processor: 既存のVideoProcessorインスタンス（オプション）

        Returns:
            (VideoProcessor, TimestampExtractor, FrameSampler)のタプル
        """
        # 動画処理の初期化
        if video_processor is None:
            video_path = self.config.get("video.input_path")
            self.logger.info(f"動画ファイル: {video_path}")
            video_processor = VideoProcessor(video_path)
            video_processor.open()

        # タイムスタンプ抽出器の初期化
        confidence_threshold = self.config.get(
            "timestamp.extraction.confidence_threshold", 0.3
        )
        timestamp_extractor = TimestampExtractor(
            confidence_threshold=confidence_threshold
        )
        output_dir = Path(self.config.get("output.directory", "output"))

        if self.config.get("output.debug_mode", False):
            debug_dir = output_dir / "debug" / "timestamps"
            timestamp_extractor.enable_debug(debug_dir)

        # フレームサンプラーの初期化
        interval_minutes = self.config.get("video.frame_interval_minutes", 5)
        tolerance_seconds = self.config.get("video.tolerance_seconds", 10)
        frame_sampler = FrameSampler(interval_minutes, tolerance_seconds)

        return video_processor, timestamp_extractor, frame_sampler
