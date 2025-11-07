"""PyTorch関連のユーティリティ関数"""

import logging
from typing import Optional
import warnings

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
