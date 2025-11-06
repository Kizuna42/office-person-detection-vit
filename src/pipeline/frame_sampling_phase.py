"""Frame sampling phase of the pipeline."""

from typing import List, Optional, Tuple

import numpy as np

from src.pipeline.base_phase import BasePhase
from src.video import VideoProcessor


class FrameSamplingPhase(BasePhase):
    """フレームサンプリングフェーズ"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.video_processor: Optional[VideoProcessor] = None

    def execute(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> List[Tuple[int, str, np.ndarray]]:
        """フレームサンプリングを実行

        Args:
            start_time: 開始時刻（HH:MM形式、オプション）
            end_time: 終了時刻（HH:MM形式、オプション）

        Returns:
            サンプルフレームのリスト [(frame_num, timestamp, frame), ...]
        """
        self.logger.info("=" * 80)
        self.logger.info("フェーズ1: フレームサンプリング")
        self.logger.info("=" * 80)

        # 共通の初期化処理を使用
        (
            self.video_processor,
            timestamp_extractor,
            frame_sampler,
        ) = self._setup_frame_sampling_components()

        # フレームサンプリング実行
        self.logger.info("フレームサンプリングを開始します...")
        sample_frames = frame_sampler.extract_sample_frames(
            self.video_processor,
            timestamp_extractor,
            start_time=start_time,
            end_time=end_time,
        )

        self.logger.info(f"サンプルフレーム数: {len(sample_frames)}個")

        if not sample_frames:
            self.logger.error("サンプルフレームが抽出できませんでした")
            raise ValueError("サンプルフレームが抽出できませんでした")

        return sample_frames

    def cleanup(self) -> None:
        """リソースをクリーンアップ"""
        if self.video_processor is not None:
            try:
                self.video_processor.release()
            except Exception as e:
                self.logger.error(f"リソース解放中にエラーが発生しました: {e}")
            finally:
                self.video_processor = None
