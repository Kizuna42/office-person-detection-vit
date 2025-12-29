"""Track class for object tracking.

Supports OC-SORT ORU (Observation-Centric Re-Update) for occlusion recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.data_models import Detection
    from src.tracking.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """トラック（追跡対象）クラス

    各オブジェクトの追跡状態を保持します。
    OC-SORT ORU 対応: 遮蔽復帰時にKalman状態を後方修正。
    """

    track_id: int
    detection: Detection
    kalman_filter: KalmanFilter
    age: int = 1  # トラックの年齢（フレーム数）
    hits: int = 1  # マッチング成功回数
    time_since_update: int = 0  # 最後の更新からの経過フレーム数
    features_history: list[np.ndarray] = field(default_factory=list)  # 特徴量履歴
    trajectory: list[tuple[float, float]] = field(default_factory=list)  # 軌跡（位置の履歴）
    _last_observation: np.ndarray | None = field(default=None, repr=False)  # ORU用: 最後の観測位置

    def __post_init__(self):
        """初期化後の処理"""
        # 初期位置をKalman Filterに設定
        position = np.array(self.detection.camera_coords, dtype=np.float32)
        self.kalman_filter.init(position)

        # 初期位置を軌跡に追加
        self.trajectory.append(self.detection.camera_coords)

        # ORU用: 最後の観測位置を保存
        self._last_observation = position.copy()

        # 特徴量があれば履歴に追加
        if self.detection.features is not None:
            self.features_history.append(self.detection.features)

        logger.debug(f"Track {self.track_id} initialized at {self.detection.camera_coords}")

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """次の位置を予測

        Args:
            dt: 時間ステップ（秒またはフレーム数）

        Returns:
            予測位置 [x, y]
        """
        state = self.kalman_filter.predict(dt)
        self.time_since_update += 1
        return state[:2]

    def update(self, detection: Detection, apply_oru: bool = True, oru_threshold: int = 3) -> None:
        """検出結果でトラックを更新

        Args:
            detection: 新しい検出結果
            apply_oru: OC-SORT ORU を適用するか
            oru_threshold: ORU を適用する最小遮蔽フレーム数
        """
        new_position = np.array(detection.camera_coords, dtype=np.float32)

        # === OC-SORT ORU: 遮蔽復帰時のKalman再更新 ===
        if apply_oru and self.time_since_update >= oru_threshold and self._last_observation is not None:
            self._apply_oru(new_position, self.time_since_update)

        # 位置を更新
        self.kalman_filter.update(new_position)

        # ORU用: 最後の観測位置を保存
        self._last_observation = new_position.copy()

        # 検出結果を更新
        self.detection = detection

        # カウンタを更新
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

        # 軌跡に追加
        self.trajectory.append(detection.camera_coords)

        # 特徴量履歴に追加（最新N個のみ保持）
        if detection.features is not None:
            self.features_history.append(detection.features)
            max_history = 10  # 最大履歴数
            if len(self.features_history) > max_history:
                self.features_history.pop(0)

        logger.debug(f"Track {self.track_id} updated, hits={self.hits}, age={self.age}")

    def _apply_oru(self, new_position: np.ndarray, missing_frames: int) -> None:
        """OC-SORT ORU (Observation-Centric Re-Update) を適用

        遮蔽期間中の仮想軌跡を線形補間で生成し、Kalman状態を後方修正。
        これにより、遮蔽中に蓄積したエラーを軽減。

        Args:
            new_position: 復帰時の新しい観測位置
            missing_frames: 遮蔽フレーム数
        """
        if self._last_observation is None:
            return

        # 線形補間で仮想軌跡を生成
        virtual_observations = self._interpolate_path(self._last_observation, new_position, missing_frames)

        # 仮想観測でKalmanを再更新（エラー蓄積防止）
        for obs in virtual_observations:
            # 各仮想観測でpredict→updateを実行
            self.kalman_filter.predict(dt=1.0)
            self.kalman_filter.update(obs)

        logger.debug(f"Track {self.track_id} ORU applied: {missing_frames} frames interpolated")

    def _interpolate_path(self, start: np.ndarray, end: np.ndarray, num_steps: int) -> list[np.ndarray]:
        """2点間を線形補間して仮想軌跡を生成

        Args:
            start: 開始位置
            end: 終了位置
            num_steps: 補間ステップ数

        Returns:
            補間された位置のリスト（開始点を含まない）
        """
        if num_steps <= 0:
            return []

        path = []
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)  # 0 < t < 1
            interpolated = start + t * (end - start)
            path.append(interpolated.astype(np.float32))

        return path

    def get_state(self) -> dict:
        """トラックの状態を取得

        Returns:
            状態辞書
        """
        position = self.kalman_filter.get_position()
        state = self.kalman_filter.get_state()

        return {
            "track_id": self.track_id,
            "position": position.tolist(),
            "velocity": state[2:].tolist(),
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "trajectory_length": len(self.trajectory),
        }

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """トラックが確立されているか確認

        Args:
            min_hits: 確立に必要な最小マッチング回数

        Returns:
            確立されている場合True
        """
        return self.hits >= min_hits

    def is_tentative(self) -> bool:
        """トラックが仮の状態か確認

        Returns:
            仮の状態の場合True
        """
        return not self.is_confirmed()

    def get_smoothed_feature(self, alpha: float = 0.9) -> np.ndarray | None:
        """特徴量履歴からEMA平滑化した特徴を取得する.

        Args:
            alpha: EMAの減衰係数（0-1）。大きいほど新しい特徴量を重視。

        Returns:
            L2正規化されたEMA特徴量、またはNone。
        """
        if not self.features_history:
            if self.detection.features is not None:
                return self.detection.features
            return None

        # EMA (Exponential Moving Average): 新しい特徴量を重視
        ema = self.features_history[0].copy()
        for feat in self.features_history[1:]:
            ema = alpha * feat + (1 - alpha) * ema

        # L2正規化（コサイン類似度計算用）
        norm = np.linalg.norm(ema)
        if norm > 1e-6:
            ema = ema / norm

        return ema
