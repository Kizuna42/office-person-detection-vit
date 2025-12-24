"""Track class for object tracking."""

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
    """

    track_id: int
    detection: Detection
    kalman_filter: KalmanFilter
    age: int = 1  # トラックの年齢（フレーム数）
    hits: int = 1  # マッチング成功回数
    time_since_update: int = 0  # 最後の更新からの経過フレーム数
    features_history: list[np.ndarray] = field(default_factory=list)  # 特徴量履歴
    trajectory: list[tuple[float, float]] = field(default_factory=list)  # 軌跡（位置の履歴）

    def __post_init__(self):
        """初期化後の処理"""
        # 初期位置をKalman Filterに設定
        position = np.array(self.detection.camera_coords, dtype=np.float32)
        self.kalman_filter.init(position)

        # 初期位置を軌跡に追加
        self.trajectory.append(self.detection.camera_coords)

        # 特徴量があれば履歴に追加
        if self.detection.features is not None:
            self.features_history.append(self.detection.features)

        logger.debug(f"Track {self.track_id} initialized at {self.detection.camera_coords}")

    def predict(self) -> np.ndarray:
        """次の位置を予測

        Returns:
            予測位置 [x, y]
        """
        state = self.kalman_filter.predict()
        self.time_since_update += 1
        return state[:2]

    def update(self, detection: Detection) -> None:
        """検出結果でトラックを更新

        Args:
            detection: 新しい検出結果
        """
        # 位置を更新
        position = np.array(detection.camera_coords, dtype=np.float32)
        self.kalman_filter.update(position)

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

    def get_smoothed_feature(self) -> np.ndarray | None:
        """特徴量履歴から平滑化した特徴を取得する."""
        if self.features_history:
            # 過去特徴量の単純平均を使用（ノイズ緩和）
            return np.mean(self.features_history, axis=0)
        if self.detection.features is not None:
            return self.detection.features
        return None
