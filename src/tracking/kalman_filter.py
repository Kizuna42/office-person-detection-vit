"""Kalman Filter implementation for object tracking.

Supports adaptive time step (dt) for variable frame rate environments.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Adaptive dt 対応 Kalman Filter実装

    2D位置と速度を追跡するためのKalman Filterです。
    状態ベクトル: [x, y, vx, vy] (位置と速度)
    可変フレームレート（タイムラプス等）に対応するため、predict時にdtを指定可能。
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """KalmanFilterを初期化

        Args:
            process_noise: プロセスノイズの基準強度（dtでスケール）
            measurement_noise: 観測ノイズの強度
        """
        # 状態ベクトルの次元: [x, y, vx, vy]
        self.ndim = 4

        # プロセスノイズの基準強度
        self._process_noise_base = process_noise

        # 状態遷移行列 (F) - predict時に動的更新
        self.F = np.eye(self.ndim, dtype=np.float32)

        # プロセスノイズ共分散行列 (Q) - predict時に動的更新
        self.Q = np.eye(self.ndim, dtype=np.float32) * process_noise

        # 観測行列 (H) - 位置のみ観測
        self.H = np.array(
            [
                [1, 0, 0, 0],  # xのみ観測
                [0, 1, 0, 0],  # yのみ観測
            ],
            dtype=np.float32,
        )

        # 観測ノイズ共分散行列 (R)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # 状態共分散行列 (P)
        self.P = np.eye(self.ndim, dtype=np.float32) * 1000.0

        # 初期状態
        self.x: np.ndarray | None = None

        # 最後のdt（デバッグ用）
        self._last_dt: float = 1.0

        logger.debug("KalmanFilter initialized (adaptive dt enabled)")

    def _update_transition_matrix(self, dt: float) -> None:
        """時間ステップに応じて遷移行列Fを更新

        Args:
            dt: 時間ステップ（秒またはフレーム数）
        """
        # 等速直線運動モデル: x' = x + vx*dt
        self.F = np.array(
            [
                [1, 0, dt, 0],  # x' = x + vx*dt
                [0, 1, 0, dt],  # y' = y + vy*dt
                [0, 0, 1, 0],  # vx' = vx
                [0, 0, 0, 1],  # vy' = vy
            ],
            dtype=np.float32,
        )
        self._last_dt = dt

    def _compute_process_noise(self, dt: float) -> np.ndarray:
        """時間ステップに応じてプロセスノイズQを計算

        離散化されたWhite Noise Acceleration モデルを使用。
        長いdtでは不確実性が増大する。

        Args:
            dt: 時間ステップ

        Returns:
            プロセスノイズ共分散行列 Q
        """
        q = self._process_noise_base

        # White Noise Acceleration モデル
        # Q = [dt^4/4  0      dt^3/2  0     ]
        #     [0       dt^4/4 0       dt^3/2]
        #     [dt^3/2  0      dt^2    0     ]
        #     [0       dt^3/2 0       dt^2  ]
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = (
            np.array(
                [
                    [dt4 / 4, 0, dt3 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2],
                    [dt3 / 2, 0, dt2, 0],
                    [0, dt3 / 2, 0, dt2],
                ],
                dtype=np.float32,
            )
            * q
        )

        return Q

    def init(self, measurement: np.ndarray) -> None:
        """フィルタを初期化

        Args:
            measurement: 初期観測値 [x, y]
        """
        if measurement.shape != (2,):
            raise ValueError(f"Expected measurement shape (2,), got {measurement.shape}")

        # 初期状態: [x, y, 0, 0] (速度は0と仮定)
        self.x = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)

        # 初期共分散（速度は不確実性が高い）
        self.P = np.diag([100.0, 100.0, 1000.0, 1000.0]).astype(np.float32)

        logger.debug(f"KalmanFilter initialized with measurement: {measurement}")

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """次の状態を予測（可変dt対応）

        Args:
            dt: 時間ステップ（秒またはフレーム数）。デフォルト1.0。
                5分間隔のタイムラプスでは dt=300.0 等を指定。

        Returns:
            予測された状態ベクトル [x, y, vx, vy]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        # dtに応じて遷移行列とプロセスノイズを更新
        self._update_transition_matrix(dt)
        self.Q = self._compute_process_noise(dt)

        # 状態予測: x' = F * x
        self.x = self.F @ self.x

        # 共分散予測: P' = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        assert self.x is not None  # 型チェック用
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

        assert self.x is not None  # 型チェック用
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

    def get_velocity(self) -> np.ndarray:
        """現在の速度を取得

        Returns:
            速度ベクトル [vx, vy]
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call init() first.")

        return self.x[2:].copy()
