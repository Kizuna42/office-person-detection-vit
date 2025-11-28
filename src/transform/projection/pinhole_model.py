"""ピンホールカメラモデルの定義。

このモジュールはカメラの内部パラメータ（焦点距離、主点、歪み係数）と
外部パラメータ（回転、並進、カメラ位置）を定義します。

座標系の定義:
    - Image (Pixel): (u, v) ∈ ℝ², 左上原点、右下正
    - Camera (3D): (x, y, z) ∈ ℝ³, カメラ中心原点、X右/Y下/Z前方
    - World (3D): (X, Y, Z) ∈ ℝ³, 床面原点、X右/Y前方/Z上
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from typing import Self
except ImportError:
    from typing import Self

import numpy as np


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ。

    Attributes:
        fx: 焦点距離 X [pixel]
        fy: 焦点距離 Y [pixel]
        cx: 主点 X [pixel]
        cy: 主点 Y [pixel]
        image_width: 画像幅 [pixel]
        image_height: 画像高さ [pixel]
        dist_coeffs: 歪み係数 [k1, k2, p1, p2, k3, ...]
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int = 1280
    image_height: int = 720
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    def __post_init__(self) -> None:
        """歪み係数を numpy 配列に変換"""
        if not isinstance(self.dist_coeffs, np.ndarray):
            self.dist_coeffs = np.array(self.dist_coeffs, dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 カメラ行列を返す。

        Returns:
            カメラ行列 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )

    @property
    def K_inv(self) -> np.ndarray:
        """カメラ行列の逆行列を返す（キャッシュ用）。

        Returns:
            K の逆行列
        """
        return np.linalg.inv(self.K)

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """設定辞書から CameraIntrinsics を作成。

        Args:
            config: camera_params セクションの設定辞書

        Returns:
            CameraIntrinsics インスタンス
        """
        dist_coeffs = config.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        return cls(
            fx=float(config.get("focal_length_x", 1000.0)),
            fy=float(config.get("focal_length_y", 1000.0)),
            cx=float(config.get("center_x", 640.0)),
            cy=float(config.get("center_y", 360.0)),
            image_width=int(config.get("image_width", 1280)),
            image_height=int(config.get("image_height", 720)),
            dist_coeffs=np.array(dist_coeffs, dtype=np.float64),
        )

    def has_distortion(self) -> bool:
        """歪み係数が設定されているか確認。

        Returns:
            True if 歪み係数が非ゼロ
        """
        return bool(np.any(np.abs(self.dist_coeffs) > 1e-10))


@dataclass
class CameraExtrinsics:
    """カメラ外部パラメータ。

    World座標系からCamera座標系への変換を定義。
    P_camera = R @ P_world + t

    Attributes:
        R: 回転行列 (3x3, World → Camera)
        t: 並進ベクトル (3,)
        camera_position_world: カメラ位置 World座標系 (3,) [meters]
        _pose_params: 元のポーズパラメータ（from_poseで作成時のみ）
    """

    R: np.ndarray
    t: np.ndarray
    camera_position_world: np.ndarray
    _pose_params: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """配列を numpy 配列に変換し、形状を検証"""
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64).flatten()
        self.camera_position_world = np.asarray(self.camera_position_world, dtype=np.float64).flatten()

        if self.R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")
        if self.camera_position_world.shape != (3,):
            raise ValueError(f"camera_position_world must be (3,), got {self.camera_position_world.shape}")

    @property
    def R_inv(self) -> np.ndarray:
        """回転行列の逆行列（転置）を返す。

        Returns:
            R の転置（逆行列）
        """
        return self.R.T

    @classmethod
    def from_pose(
        cls,
        camera_height_m: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float = 0.0,
        camera_x_m: float = 0.0,
        camera_y_m: float = 0.0,
    ) -> Self:
        """直感的なパラメータから外部パラメータを構築。

        カメラが World 座標系の (camera_x_m, camera_y_m, camera_height_m) にあり、
        指定された角度で回転していると仮定。

        座標系の規約:
            - World: X=右, Y=前方, Z=上
            - Camera: X=右, Y=下, Z=前方（光軸）
            - pitch: X軸周り回転（正=下向き）
            - yaw: Z軸周り回転（正=右向き）
            - roll: Z軸（光軸）周り回転

        Args:
            camera_height_m: カメラの高さ [meters]
            pitch_deg: 俯角 [degrees]（0=水平、正=下向き）
            yaw_deg: 方位角 [degrees]（0=Y+方向、正=右向き）
            roll_deg: 回転角 [degrees]
            camera_x_m: カメラX位置 [meters]
            camera_y_m: カメラY位置 [meters]

        Returns:
            CameraExtrinsics インスタンス
        """
        # ラジアンに変換
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        roll = np.radians(roll_deg)

        # 基底変換行列: World座標系からCamera座標系への標準変換
        # Camera初期状態: Y+方向（World前方）を見ている
        # World: X=右, Y=前方, Z=上
        # Camera: X=右, Y=下, Z=前方
        # => Cam_X = World_X, Cam_Y = -World_Z, Cam_Z = World_Y
        R_base = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )

        # Yaw: World Z軸周りの回転（カメラの向きを変える）
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array(
            [
                [c_yaw, -s_yaw, 0],
                [s_yaw, c_yaw, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Pitch: Camera X軸周りの回転（下を向く）
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        R_pitch = np.array(
            [
                [1, 0, 0],
                [0, c_pitch, -s_pitch],
                [0, s_pitch, c_pitch],
            ],
            dtype=np.float64,
        )

        # Roll: Camera Z軸周りの回転
        c_roll, s_roll = np.cos(roll), np.sin(roll)
        R_roll = np.array(
            [
                [c_roll, -s_roll, 0],
                [s_roll, c_roll, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # 合成: R = R_roll @ R_pitch @ R_base @ R_yaw
        # 順序: World座標系でYaw → 基底変換 → Camera座標系でPitch → Roll
        R = R_roll @ R_pitch @ R_base @ R_yaw

        # カメラ位置
        C = np.array([camera_x_m, camera_y_m, camera_height_m], dtype=np.float64)

        # 並進ベクトル: t = -R @ C
        t = -R @ C

        # 元のパラメータを保存
        pose_params = {
            "height_m": camera_height_m,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "roll_deg": roll_deg,
            "camera_x_m": camera_x_m,
            "camera_y_m": camera_y_m,
        }

        return cls(R=R, t=t, camera_position_world=C, _pose_params=pose_params)

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """設定辞書から CameraExtrinsics を作成。

        Args:
            config: camera_params セクションの設定辞書

        Returns:
            CameraExtrinsics インスタンス
        """
        return cls.from_pose(
            camera_height_m=float(config.get("height_m", 2.2)),
            pitch_deg=float(config.get("pitch_deg", 45.0)),
            yaw_deg=float(config.get("yaw_deg", 0.0)),
            roll_deg=float(config.get("roll_deg", 0.0)),
            camera_x_m=float(config.get("camera_x_m", 0.0)),
            camera_y_m=float(config.get("camera_y_m", 0.0)),
        )

    def to_pose_params(self) -> dict:
        """外部パラメータを直感的なパラメータに変換。

        from_pose()で作成された場合は元のパラメータを返す。
        そうでない場合は回転行列から推定（近似）。

        Returns:
            pitch_deg, yaw_deg, roll_deg, camera_position の辞書
        """
        # from_pose() で作成された場合は元のパラメータを返す
        if self._pose_params:
            return {
                "pitch_deg": self._pose_params["pitch_deg"],
                "yaw_deg": self._pose_params["yaw_deg"],
                "roll_deg": self._pose_params["roll_deg"],
                "camera_position_world": self.camera_position_world.tolist(),
            }

        # 回転行列から角度を抽出（Euler angles）
        # これは近似的な逆変換
        sy = np.sqrt(self.R[0, 0] ** 2 + self.R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(self.R[2, 1], self.R[2, 2])
            yaw = np.arctan2(-self.R[2, 0], sy)
            roll = np.arctan2(self.R[1, 0], self.R[0, 0])
        else:
            pitch = np.arctan2(-self.R[1, 2], self.R[1, 1])
            yaw = np.arctan2(-self.R[2, 0], sy)
            roll = 0

        return {
            "pitch_deg": np.degrees(pitch),
            "yaw_deg": np.degrees(yaw),
            "roll_deg": np.degrees(roll),
            "camera_position_world": self.camera_position_world.tolist(),
        }
