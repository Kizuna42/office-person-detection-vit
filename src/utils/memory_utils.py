"""Memory management utilities."""

import gc
import logging
from typing import Optional

import torch

from src.detection import ViTDetector
from src.video import VideoProcessor


def cleanup_resources(
    video_processor: Optional[VideoProcessor] = None,
    detector: Optional[ViTDetector] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """リソースをクリーンアップする

    Args:
        video_processor: VideoProcessorインスタンス（オプション）
        detector: ViTDetectorインスタンス（オプション）
        logger: ロガー（オプション）
    """
    # 動画処理のクリーンアップ
    if video_processor is not None:
        try:
            video_processor.release()
        except Exception as e:
            if logger:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")

    # GPUメモリのクリーンアップ
    if detector is not None:
        try:
            if detector.device in ["mps", "cuda"]:
                if detector.device == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif detector.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            if logger:
                logger.error(f"メモリ解放中にエラーが発生しました: {e}")

    # ガベージコレクションを実行
    gc.collect()
    if logger:
        logger.debug("メモリクリーンアップを実行しました")
