"""Improved temporal validator with adaptive tolerance and outlier recovery."""

from collections import deque
from datetime import datetime, timedelta
import logging

import numpy as np

logger = logging.getLogger(__name__)


class TemporalValidatorV2:
    """改善された時系列検証器

    適応的許容範囲と異常値リカバリーを実装
    """

    def __init__(
        self,
        fps: float = 30.0,
        base_tolerance_seconds: float = 10.0,
        history_size: int = 10,
        z_score_threshold: float = 2.0,
    ):
        """TemporalValidatorV2を初期化

        Args:
            fps: 動画のフレームレート
            base_tolerance_seconds: ベース許容範囲（秒）
            history_size: 履歴サイズ（過去N個のフレーム間隔を保持）
            z_score_threshold: Z-score閾値（外れ値検出用）
        """
        self.fps = fps
        self.base_tolerance = base_tolerance_seconds
        self.history_size = history_size
        self.z_score_threshold = z_score_threshold

        self.last_timestamp: datetime | None = None
        self.last_frame_idx: int | None = None
        self.interval_history: deque = deque(maxlen=history_size)

    def validate(self, timestamp: datetime, frame_idx: int) -> tuple[bool, float, str]:
        """タイムスタンプの時系列整合性を検証（改善版）

        Args:
            timestamp: 検証するタイムスタンプ
            frame_idx: フレーム番号

        Returns:
            (有効性, 信頼度, 理由) のタプル
        """
        # 初回フレームは常に有効
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.last_frame_idx = frame_idx
            return True, 1.0, "First frame"

        # フレーム差を計算
        assert self.last_frame_idx is not None  # 型チェック用
        frame_diff = frame_idx - self.last_frame_idx
        if frame_diff <= 0:
            return False, 0.0, f"Invalid frame_diff: {frame_diff}"

        # 時間差を計算
        time_diff = (timestamp - self.last_timestamp).total_seconds()

        # 期待される時間差（フレーム差から計算）
        expected_seconds = frame_diff / self.fps

        # 適応的許容範囲を計算
        adaptive_tolerance = self._calculate_adaptive_tolerance()

        # 外れ値検出（Z-score法）
        is_outlier, z_score = self._detect_outlier(time_diff, expected_seconds)

        if is_outlier:
            # 異常値リカバリー: 前後フレームからの線形補間
            recovered_timestamp = self._recover_timestamp(frame_idx, expected_seconds)
            if recovered_timestamp:
                logger.warning(f"Outlier detected (z-score={z_score:.2f}), recovered: {recovered_timestamp}")
                # リカバリー後のタイムスタンプで再検証
                time_diff = (recovered_timestamp - self.last_timestamp).total_seconds()
                timestamp = recovered_timestamp

        lower_bound = expected_seconds - adaptive_tolerance
        upper_bound = expected_seconds + adaptive_tolerance

        if lower_bound <= time_diff <= upper_bound:
            # 履歴を更新
            self.interval_history.append(time_diff)

            confidence = 1.0 - abs(time_diff - expected_seconds) / max(adaptive_tolerance, 1.0)
            confidence = max(0.0, min(1.0, confidence))

            self.last_timestamp = timestamp
            self.last_frame_idx = frame_idx

            return (
                True,
                confidence,
                f"Valid: expected={expected_seconds:.1f}s, actual={time_diff:.1f}s, "
                f"tolerance={adaptive_tolerance:.1f}s",
            )
        return (
            False,
            0.0,
            f"Invalid: expected={expected_seconds:.1f}s, actual={time_diff:.1f}s, tolerance={adaptive_tolerance:.1f}s",
        )

    def _calculate_adaptive_tolerance(self) -> float:
        """適応的許容範囲を計算

        過去N個のフレーム間隔から動的に許容範囲を計算

        Returns:
            適応的許容範囲（秒）
        """
        if len(self.interval_history) < 3:
            # 履歴が少ない場合はベース許容範囲を使用
            return self.base_tolerance

        # 過去の間隔の標準偏差を計算
        intervals = list(self.interval_history)
        std_interval = np.std(intervals)

        # 適応的許容範囲 = ベース許容範囲 + 標準偏差の倍数
        adaptive_tolerance: float = self.base_tolerance + (std_interval * 1.5)

        # 最小値と最大値を設定
        adaptive_tolerance = max(
            self.base_tolerance * 0.5,
            min(adaptive_tolerance, self.base_tolerance * 3.0),
        )

        return float(adaptive_tolerance)

    def _detect_outlier(self, time_diff: float, _expected_seconds: float) -> tuple[bool, float]:
        """外れ値検出（Z-score法）

        Args:
            time_diff: 実際の時間差
            expected_seconds: 期待される時間差

        Returns:
            (外れ値かどうか, Z-score) のタプル
        """
        if len(self.interval_history) < 3:
            return False, 0.0

        # 過去の間隔の統計を計算
        intervals = list(self.interval_history)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if std_interval == 0:
            return False, 0.0

        # Z-scoreを計算
        z_score = abs(time_diff - mean_interval) / std_interval

        is_outlier = z_score > self.z_score_threshold

        return is_outlier, z_score

    def _recover_timestamp(self, _frame_idx: int, expected_seconds: float) -> datetime | None:
        """異常値のリカバリー（前後フレームからの線形補間）

        Args:
            frame_idx: 現在のフレーム番号
            expected_seconds: 期待される時間差

        Returns:
            リカバリー後のタイムスタンプ（Noneの場合はリカバリー不可）
        """
        if self.last_timestamp is None:
            return None

        # 線形補間: 前のタイムスタンプ + 期待される時間差
        recovered = self.last_timestamp + timedelta(seconds=expected_seconds)

        return recovered

    def reset(self) -> None:
        """状態をリセット"""
        self.last_timestamp = None
        self.last_frame_idx = None
        self.interval_history.clear()
        logger.debug("TemporalValidatorV2 reset")
