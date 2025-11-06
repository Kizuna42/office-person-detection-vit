"""Temporal validation for timestamp sequences."""

import logging
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalValidator:
    """タイムスタンプの時系列整合性を検証

    連続するフレームのタイムスタンプが時系列的に妥当かを検証し、
    フレーム間の時間差が期待値と一致しているかをチェックします。
    """

    def __init__(self, fps: float = 30.0):
        """TemporalValidatorを初期化

        Args:
            fps: 動画のフレームレート（デフォルト: 30.0）
        """
        self.fps = fps
        self.last_timestamp: Optional[datetime] = None
        self.last_frame_idx: Optional[int] = None

    def validate(self, timestamp: datetime, frame_idx: int) -> Tuple[bool, float, str]:
        """タイムスタンプが時系列的に妥当かを検証

        Args:
            timestamp: 検証するタイムスタンプ
            frame_idx: フレーム番号

        Returns:
            (妥当性フラグ, 信頼度, 理由メッセージ) のタプル
        """
        if self.last_timestamp is None:
            # 初回は常に受け入れ
            self.last_timestamp = timestamp
            self.last_frame_idx = frame_idx
            return True, 1.0, "Initial timestamp"

        # フレーム差から期待される時間差を計算
        frame_diff = frame_idx - self.last_frame_idx
        expected_seconds = frame_diff / self.fps

        # 実際の時間差
        actual_diff = (timestamp - self.last_timestamp).total_seconds()

        # 許容範囲チェック（±20%）
        tolerance = expected_seconds * 0.2
        lower_bound = expected_seconds - tolerance
        upper_bound = expected_seconds + tolerance

        if lower_bound <= actual_diff <= upper_bound:
            confidence = 1.0 - abs(actual_diff - expected_seconds) / max(expected_seconds, 1.0)
            confidence = max(0.0, min(1.0, confidence))  # 0.0-1.0にクランプ
            self.last_timestamp = timestamp
            self.last_frame_idx = frame_idx
            return True, confidence, f"Valid: expected={expected_seconds:.1f}s, actual={actual_diff:.1f}s"
        else:
            return False, 0.0, f"Invalid: expected={expected_seconds:.1f}s, actual={actual_diff:.1f}s"

    def reset(self) -> None:
        """状態をリセット"""
        self.last_timestamp = None
        self.last_frame_idx = None
        logger.debug("TemporalValidator reset")

