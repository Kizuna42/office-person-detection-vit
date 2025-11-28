"""統合座標変換モジュール。

Phase 3 で使用する統合変換クラスを提供します。
画像座標からフロアマップ座標への変換をワンストップで実行。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np

from src.transform.floormap_transformer import FloorMapConfig, FloorMapTransformer
from src.transform.projection import CameraExtrinsics, CameraIntrinsics, RayCaster

if TYPE_CHECKING:
    from src.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """座標変換結果。

    Attributes:
        floor_coords_px: フロアマップ座標 (px, py) [pixels]、無効時は None
        floor_coords_mm: フロアマップ座標 (x, y) [mm]、無効時は None
        world_coords_m: World座標 (X, Y) [meters]、無効時は None
        is_valid: 変換が有効か
        error_reason: 無効時のエラー理由
        is_within_bounds: フロアマップ範囲内か
    """

    floor_coords_px: tuple[float, float] | None = None
    floor_coords_mm: tuple[float, float] | None = None
    world_coords_m: tuple[float, float] | None = None
    is_valid: bool = False
    error_reason: str | None = None
    is_within_bounds: bool = False


class UnifiedTransformer:
    """統合座標変換クラス。

    画像座標からフロアマップ座標への変換を一貫したインターフェースで提供。
    内部で RayCaster と FloorMapTransformer を使用。
    """

    def __init__(
        self,
        ray_caster: RayCaster,
        floormap_transformer: FloorMapTransformer,
    ):
        """初期化。

        Args:
            ray_caster: レイキャストモジュール
            floormap_transformer: フロアマップ変換モジュール
        """
        self._ray_caster = ray_caster
        self._floormap_transformer = floormap_transformer

        logger.debug("UnifiedTransformer initialized")

    @classmethod
    def from_config(cls, config: ConfigManager) -> UnifiedTransformer:
        """設定から UnifiedTransformer を作成。

        Args:
            config: ConfigManager インスタンス

        Returns:
            UnifiedTransformer インスタンス
        """
        camera_params = config.get("camera_params", {})
        floormap_config = config.get("floormap", {})

        # カメラ内部パラメータ
        intrinsics = CameraIntrinsics.from_config(camera_params)

        # カメラ外部パラメータ
        extrinsics = CameraExtrinsics.from_config(camera_params)

        # RayCaster
        ray_caster = RayCaster(intrinsics, extrinsics)

        # フロアマップ設定
        fm_config = FloorMapConfig.from_config(floormap_config)

        # カメラ位置（フロアマップ座標系）
        camera_position_px = (
            float(camera_params.get("position_x_px", 1200.0)),
            float(camera_params.get("position_y_px", 800.0)),
        )

        floormap_transformer = FloorMapTransformer(fm_config, camera_position_px)

        return cls(ray_caster, floormap_transformer)

    @property
    def ray_caster(self) -> RayCaster:
        """RayCaster を返す。"""
        return self._ray_caster

    @property
    def floormap_transformer(self) -> FloorMapTransformer:
        """FloorMapTransformer を返す。"""
        return self._floormap_transformer

    def transform_pixel(
        self,
        pixel: tuple[float, float],
        floor_z: float = 0.0,
    ) -> TransformResult:
        """画像座標をフロアマップ座標に変換。

        Args:
            pixel: 画像座標 (u, v)
            floor_z: 床面の高さ [meters]

        Returns:
            TransformResult
        """
        # 1. 画像座標 → World座標（床面）
        world_point = self._ray_caster.image_to_floor(pixel, floor_z)

        if world_point is None:
            return TransformResult(
                is_valid=False,
                error_reason="Point is above horizon or behind camera",
            )

        # 2. World座標 → フロアマップ座標
        floor_px = self._floormap_transformer.world_to_floormap(world_point)

        # 3. mm座標を計算
        floor_mm = self._floormap_transformer.pixel_to_mm(floor_px)

        # 4. 範囲チェック
        is_within = self._floormap_transformer.is_within_bounds(floor_px)

        return TransformResult(
            floor_coords_px=floor_px,
            floor_coords_mm=floor_mm,
            world_coords_m=world_point,
            is_valid=True,
            is_within_bounds=is_within,
        )

    def transform_detection(
        self,
        bbox: tuple[float, float, float, float],
        floor_z: float = 0.0,
    ) -> TransformResult:
        """検出結果（BBox）をフロアマップ座標に変換。

        BBox の足元点（中心下端）を変換対象とする。

        Args:
            bbox: (x, y, width, height)
            floor_z: 床面の高さ [meters]

        Returns:
            TransformResult
        """
        foot_point = self._ray_caster.get_foot_point(bbox)
        return self.transform_pixel(foot_point, floor_z)

    def transform_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
        floor_z: float = 0.0,
    ) -> list[TransformResult]:
        """複数の検出結果をバッチ変換。

        Args:
            bboxes: BBox のリスト
            floor_z: 床面の高さ [meters]

        Returns:
            TransformResult のリスト
        """
        if not bboxes:
            return []

        # 足元点を抽出
        foot_points = np.array(
            [self._ray_caster.get_foot_point(bbox) for bbox in bboxes],
            dtype=np.float64,
        )

        # バッチでWorld座標に変換
        world_points = self._ray_caster.batch_image_to_floor(foot_points, floor_z)

        # バッチでフロアマップ座標に変換
        floor_points = self._floormap_transformer.world_to_floormap_batch(world_points)

        # 範囲チェック
        within_bounds = self._floormap_transformer.is_within_bounds_batch(floor_points)

        # 結果を構築
        results = []
        for i in range(len(bboxes)):
            if np.isnan(world_points[i]).any():
                results.append(
                    TransformResult(
                        is_valid=False,
                        error_reason="Point is above horizon or behind camera",
                    )
                )
            else:
                floor_px = (float(floor_points[i, 0]), float(floor_points[i, 1]))
                floor_mm = self._floormap_transformer.pixel_to_mm(floor_px)
                world_m = (float(world_points[i, 0]), float(world_points[i, 1]))

                results.append(
                    TransformResult(
                        floor_coords_px=floor_px,
                        floor_coords_mm=floor_mm,
                        world_coords_m=world_m,
                        is_valid=True,
                        is_within_bounds=bool(within_bounds[i]),
                    )
                )

        return results

    def get_camera_info(self) -> dict:
        """カメラ情報を返す（デバッグ用）。

        Returns:
            カメラ情報の辞書
        """
        return {
            "camera_center_world": self._ray_caster.camera_center.tolist(),
            "camera_position_floormap_px": self._floormap_transformer.camera_position_px,
            "floormap_info": self._floormap_transformer.get_info(),
        }


class TransformPipelineBuilder:
    """変換パイプラインビルダー。

    柔軟な設定で UnifiedTransformer を構築する。
    """

    def __init__(self):
        """初期化。"""
        self._intrinsics: CameraIntrinsics | None = None
        self._extrinsics: CameraExtrinsics | None = None
        self._floormap_config: FloorMapConfig | None = None
        self._camera_position_px: tuple[float, float] | None = None

    def with_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_width: int = 1280,
        image_height: int = 720,
        dist_coeffs: list[float] | None = None,
    ) -> TransformPipelineBuilder:
        """カメラ内部パラメータを設定。

        Args:
            fx: 焦点距離 X [pixel]
            fy: 焦点距離 Y [pixel]
            cx: 主点 X [pixel]
            cy: 主点 Y [pixel]
            image_width: 画像幅 [pixel]
            image_height: 画像高さ [pixel]
            dist_coeffs: 歪み係数

        Returns:
            self
        """
        self._intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_width=image_width,
            image_height=image_height,
            dist_coeffs=np.array(dist_coeffs or [0.0] * 5, dtype=np.float64),
        )
        return self

    def with_extrinsics(
        self,
        height_m: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float = 0.0,
        camera_x_m: float = 0.0,
        camera_y_m: float = 0.0,
    ) -> TransformPipelineBuilder:
        """カメラ外部パラメータを設定。

        Args:
            height_m: カメラ高さ [meters]
            pitch_deg: 俯角 [degrees]
            yaw_deg: 方位角 [degrees]
            roll_deg: 回転角 [degrees]
            camera_x_m: カメラX位置 [meters]
            camera_y_m: カメラY位置 [meters]

        Returns:
            self
        """
        self._extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=height_m,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            camera_x_m=camera_x_m,
            camera_y_m=camera_y_m,
        )
        return self

    def with_floormap(
        self,
        width_px: int = 1878,
        height_px: int = 1369,
        origin_x_px: float = 7.0,
        origin_y_px: float = 9.0,
        scale_x_mm_per_px: float = 28.1926406926406,
        scale_y_mm_per_px: float = 28.241430700447,
    ) -> TransformPipelineBuilder:
        """フロアマップ設定を行う。

        Args:
            width_px: フロアマップ幅 [pixel]
            height_px: フロアマップ高さ [pixel]
            origin_x_px: 原点Xオフセット [pixel]
            origin_y_px: 原点Yオフセット [pixel]
            scale_x_mm_per_px: X軸スケール [mm/pixel]
            scale_y_mm_per_px: Y軸スケール [mm/pixel]

        Returns:
            self
        """
        self._floormap_config = FloorMapConfig(
            width_px=width_px,
            height_px=height_px,
            origin_x_px=origin_x_px,
            origin_y_px=origin_y_px,
            scale_x_mm_per_px=scale_x_mm_per_px,
            scale_y_mm_per_px=scale_y_mm_per_px,
        )
        return self

    def with_camera_position(
        self,
        x_px: float,
        y_px: float,
    ) -> TransformPipelineBuilder:
        """フロアマップ上のカメラ位置を設定。

        Args:
            x_px: カメラX位置 [pixel]
            y_px: カメラY位置 [pixel]

        Returns:
            self
        """
        self._camera_position_px = (x_px, y_px)
        return self

    def build(self) -> UnifiedTransformer:
        """UnifiedTransformer を構築。

        Returns:
            UnifiedTransformer インスタンス

        Raises:
            ValueError: 必須パラメータが設定されていない場合
        """
        if self._intrinsics is None:
            raise ValueError("Camera intrinsics not set")
        if self._extrinsics is None:
            raise ValueError("Camera extrinsics not set")
        if self._floormap_config is None:
            self._floormap_config = FloorMapConfig()
        if self._camera_position_px is None:
            raise ValueError("Camera position on floormap not set")

        ray_caster = RayCaster(self._intrinsics, self._extrinsics)
        floormap_transformer = FloorMapTransformer(self._floormap_config, self._camera_position_px)

        return UnifiedTransformer(ray_caster, floormap_transformer)
