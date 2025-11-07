"""Kalman Filter implementation for object tracking."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class KalmanFilter:
    """簡易Kalman Filter実装

    2D位置と速度を追跡するためのKalman Filterです。
    状態ベクトル: [x, y, vx, vy] (位置と速度)
    """

    def __init__(self):
        """KalmanFilterを初期化"""
        # 状態ベクトルの次元: [x, y, vx, vy]
        self.ndim = 4

        # 状態遷移行列 (F)
        # 等速直線運動を仮定
        dt = 1.0  # 時間ステップ（1フレーム）
        self.F = np.array(
            [
                [1, 0, dt, 0],  # x' = x + vx*dt
                [0, 1, 0, dt],  # y' = y + vy*dt
                [0, 0, 1, 0],  # vx' = vx
                [0, 0, 0, 1],  # vy' = vy
            ],
            dtype=np.float32,
        )

        # 観測行列 (H) - 位置のみ観測
        self.H = np.array(
            [
                [1, 0, 0, 0],  # xのみ観測
                [0, 1, 0, 0],  # yのみ観測
            ],
            dtype=np.float32,
        )

        # プロセスノイズ共分散行列 (Q)
        q = 0.1  # プロセスノイズの強度
        self.Q = np.eye(self.ndim, dtype=np.float32) * q

        # 観測ノイズ共分散行列 (R)
        r = 1.0  # 観測ノイズの強度
        self.R = np.eye(2, dtype=np.float32) * r

        # 状態共分散行列 (P)
        self.P = np.eye(self.ndim, dtype=np.float32) * 1000.0

        # 初期状態
        self.x: np.ndarray | None = None

        logger.debug("KalmanFilter initialized")

    def init(self, measurement: np.ndarray) -> None:
        """フィルタを初期化

        Args:
            measurement: 初期観測値 [x, y]
        """
        if measurement.shape != (2,):
            raise ValueError(f"Expected measurement shape (2,), got {measurement.shape}")

        # 初期状態: [x, y, 0, 0] (速度は0と仮定)
        self.x = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)

        # 初期共分散
        self.P = np.eye(self.ndim, dtype=np.float32) * 1000.0

        logger.debug(f"KalmanFilter initialized with measurement: {measurement}")

    def predict(self) -> np.ndarray:
        """次の状態を予測

        Returns:
            予測された状態ベクトル [x, y, vx, vy]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        # 状態予測: x' = F * x
        self.x = self.F @ self.x

        # 共分散予測: P' = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """観測値で状態を更新

        Args:
            measurement: 観測値 [x, y]

        Returns:
            更新された状態ベクトル [x, y, vx, vy]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        if measurement.shape != (2,):
            raise ValueError(f"Expected measurement shape (2,), got {measurement.shape}")

        # 観測残差: y = z - H * x
        y = measurement - self.H @ self.x

        # 残差共分散: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状態更新: x = x + K * y
        self.x = self.x + K @ y

        # 共分散更新: P = (I - K * H) * P
        identity = np.eye(self.ndim, dtype=np.float32)
        self.P = (identity - K @ self.H) @ self.P

        return self.x.copy()

    def get_state(self) -> np.ndarray:
        """現在の状態を取得

        Returns:
            状態ベクトル [x, y, vx, vy]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        return self.x.copy()

    def get_position(self) -> np.ndarray:
        """現在の位置を取得

        Returns:
            位置ベクトル [x, y]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        return self.x[:2].copy()
