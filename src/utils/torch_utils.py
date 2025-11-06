"""PyTorch関連のユーティリティ関数"""

import logging
import warnings
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def setup_mps_compatibility() -> None:
    """MPSデバイス使用時の互換性設定を適用

    MPSデバイスではpin_memoryがサポートされていないため、
    DataLoaderの警告を抑制します。
    """
    if not torch.backends.mps.is_available():
        return

    # MPSデバイス使用時のpin_memory警告を抑制
    # この警告は外部ライブラリ（PaddleOCR、EasyOCRなど）が
    # 内部的にDataLoaderを使用する際に発生します
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*argument is set as true but not supported on MPS.*",
        category=UserWarning,
    )
    # より広範囲なパターンも追加（念のため）
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*MPS.*",
        category=UserWarning,
    )

    logger.debug("MPS互換性設定を適用しました（pin_memory警告を抑制）")


def get_device(device: Optional[str] = None) -> str:
    """使用するデバイスを取得

    Args:
        device: 指定されたデバイス名 (None の場合は自動検出)

    Returns:
        使用するデバイス名 ("mps", "cuda", "cpu")
    """
    if device is not None:
        # ユーザー指定のデバイスを使用
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS is not available. Falling back to CPU.")
            return "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Falling back to CPU.")
            return "cpu"
        return device

    # 自動検出
    if torch.backends.mps.is_available():
        logger.info("MPS device detected and will be used for acceleration.")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA device detected and will be used for acceleration.")
        return "cuda"
    else:
        logger.info("No GPU acceleration available. Using CPU.")
        return "cpu"

