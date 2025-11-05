"""Frame sampling phase of the pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import FrameSampler, VideoProcessor


class FrameSamplingPhase:
    """フレームサンプリングフェーズ"""
    
    def __init__(
        self,
        config: ConfigManager,
        logger: logging.Logger
    ):
        """初期化
        
        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        self.config = config
        self.logger = logger
        self.video_processor: Optional[VideoProcessor] = None
    
    def execute(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
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
        
        # 動画処理の初期化
        video_path = self.config.get('video.input_path')
        self.logger.info(f"動画ファイル: {video_path}")
        
        self.video_processor = VideoProcessor(video_path)
        self.video_processor.open()
        
        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        output_dir = Path(self.config.get('output.directory', 'output'))
        
        if self.config.get('output.debug_mode', False):
            debug_dir = output_dir / 'debug' / 'timestamps'
            timestamp_extractor.enable_debug(debug_dir)
        
        # フレームサンプラーの初期化
        interval_minutes = self.config.get('video.frame_interval_minutes', 5)
        tolerance_seconds = self.config.get('video.tolerance_seconds', 10)
        frame_sampler = FrameSampler(interval_minutes, tolerance_seconds)
        
        # フレームサンプリング実行
        self.logger.info("フレームサンプリングを開始します...")
        sample_frames = frame_sampler.extract_sample_frames(
            self.video_processor,
            timestamp_extractor,
            start_time=start_time,
            end_time=end_time
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

