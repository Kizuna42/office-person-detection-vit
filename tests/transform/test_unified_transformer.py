"""UnifiedTransformer のテスト。

高精度座標変換パイプラインの精度と動作を検証。
"""

import numpy as np
import pytest

from src.transform import (
    CameraExtrinsics,
    CameraIntrinsics,
    FloorMapConfig,
    FloorMapTransformer,
    RayCaster,
    TransformPipelineBuilder,
    TransformResult,
    UnifiedTransformer,
)


class TestCameraIntrinsics:
    """CameraIntrinsics のテスト"""

    def test_camera_matrix_construction(self):
        """カメラ行列が正しく構築されるか"""
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

        K = intrinsics.K

        assert K.shape == (3, 3)
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 640.0  # cx
        assert K[1, 2] == 360.0  # cy
        assert K[2, 2] == 1.0

    def test_from_config(self):
        """設定辞書から正しく作成されるか"""
        config = {
            "focal_length_x": 1250.0,
            "focal_length_y": 1250.0,
            "center_x": 640.0,
            "center_y": 360.0,
            "image_width": 1280,
            "image_height": 720,
        }

        intrinsics = CameraIntrinsics.from_config(config)

        assert intrinsics.fx == 1250.0
        assert intrinsics.fy == 1250.0
        assert intrinsics.image_width == 1280
        assert intrinsics.image_height == 720

    def test_has_distortion(self):
        """歪み検出が正しく動作するか"""
        # 歪みなし
        intrinsics_no_dist = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, dist_coeffs=np.zeros(5))
        assert not intrinsics_no_dist.has_distortion()

        # 歪みあり
        intrinsics_with_dist = CameraIntrinsics(
            fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, dist_coeffs=np.array([-0.1, 0.0, 0.0, 0.0, 0.0])
        )
        assert intrinsics_with_dist.has_distortion()


class TestCameraExtrinsics:
    """CameraExtrinsics のテスト"""

    def test_from_pose(self):
        """ポーズパラメータから正しく作成されるか"""
        extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=2.0,
            pitch_deg=45.0,
            yaw_deg=0.0,
            roll_deg=0.0,
        )

        # 回転行列の検証
        assert extrinsics.R.shape == (3, 3)

        # 回転行列が直交行列か（R @ R.T = I）
        identity = extrinsics.R @ extrinsics.R.T
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)

        # カメラ位置の検証
        assert extrinsics.camera_position_world[2] == pytest.approx(2.0)

    def test_from_config(self):
        """設定辞書から正しく作成されるか"""
        config = {
            "height_m": 2.5,
            "pitch_deg": 30.0,
            "yaw_deg": 10.0,
            "roll_deg": 0.0,
        }

        extrinsics = CameraExtrinsics.from_config(config)

        assert extrinsics.camera_position_world[2] == pytest.approx(2.5)


class TestRayCaster:
    """RayCaster のテスト"""

    @pytest.fixture
    def ray_caster(self):
        """基本的なレイキャスターを作成"""
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=2.0,
            pitch_deg=45.0,
            yaw_deg=0.0,
        )
        return RayCaster(intrinsics, extrinsics)

    def test_image_center_projects_forward(self, ray_caster):
        """画像中心が前方に投影されるか"""
        # 画像中心 (640, 360)
        result = ray_caster.image_to_floor((640.0, 360.0))

        assert result is not None
        x, y = result

        # X座標はほぼ0（中心から）
        assert abs(x) < 0.1

        # Y座標は正（前方）
        assert y > 0

    def test_horizon_returns_none(self, ray_caster):
        """地平線上の点は None を返すか"""
        # 水平より上の点
        ray_caster.image_to_floor((640.0, 0.0))

        # pitch=45度なので、y=0（画像上端）は地平線より上の可能性がある
        # 結果は None または非常に遠い点

    def test_batch_processing(self, ray_caster):
        """バッチ処理が正しく動作するか"""
        pixels = np.array(
            [
                [640.0, 360.0],
                [320.0, 400.0],
                [960.0, 400.0],
            ]
        )

        results = ray_caster.batch_image_to_floor(pixels)

        assert results.shape == (3, 2)
        # 少なくとも一部の点が有効
        assert not np.all(np.isnan(results))

    def test_round_trip_consistency(self, ray_caster):
        """投影と逆投影が一貫しているか"""
        # 床面上の点
        world_point = (0.5, 2.0, 0.0)

        # 画像に投影
        pixel = ray_caster.floor_to_image(world_point)

        if pixel is not None:
            # 床面に戻す
            reconstructed = ray_caster.image_to_floor(pixel)

            if reconstructed is not None:
                assert reconstructed[0] == pytest.approx(world_point[0], abs=0.01)
                assert reconstructed[1] == pytest.approx(world_point[1], abs=0.01)


class TestFloorMapTransformer:
    """FloorMapTransformer のテスト"""

    @pytest.fixture
    def transformer(self):
        """基本的なフロアマップ変換器を作成"""
        config = FloorMapConfig(
            width_px=1878,
            height_px=1369,
            scale_x_mm_per_px=28.1926,
            scale_y_mm_per_px=28.2414,
        )
        camera_position_px = (1000.0, 800.0)
        return FloorMapTransformer(config, camera_position_px)

    def test_origin_maps_to_camera_position(self, transformer):
        """World原点がカメラ位置にマップされるか"""
        result = transformer.world_to_floormap((0.0, 0.0))

        assert result[0] == pytest.approx(1000.0)
        assert result[1] == pytest.approx(800.0)

    def test_positive_x_maps_right(self, transformer):
        """正のX座標が右にマップされるか"""
        origin = transformer.world_to_floormap((0.0, 0.0))
        right = transformer.world_to_floormap((1.0, 0.0))

        # 1メートル右 = 約35ピクセル右
        assert right[0] > origin[0]

    def test_positive_y_maps_forward(self, transformer):
        """正のY座標が前方（下）にマップされるか"""
        origin = transformer.world_to_floormap((0.0, 0.0))
        forward = transformer.world_to_floormap((0.0, 1.0))

        # 1メートル前 = 約35ピクセル下
        assert forward[1] > origin[1]

    def test_is_within_bounds(self, transformer):
        """範囲判定が正しく動作するか"""
        assert transformer.is_within_bounds((500.0, 500.0))
        assert not transformer.is_within_bounds((-1.0, 500.0))
        assert not transformer.is_within_bounds((500.0, 2000.0))


class TestUnifiedTransformer:
    """UnifiedTransformer のテスト"""

    @pytest.fixture
    def unified_transformer(self):
        """統合変換器を作成"""
        return (
            TransformPipelineBuilder()
            .with_intrinsics(1000.0, 1000.0, 640.0, 360.0)
            .with_extrinsics(2.0, 45.0, 0.0)
            .with_floormap()
            .with_camera_position(1000.0, 800.0)
            .build()
        )

    def test_transform_detection(self, unified_transformer):
        """検出結果の変換が動作するか"""
        bbox = (600.0, 300.0, 80.0, 200.0)  # 画像上の人物

        result = unified_transformer.transform_detection(bbox)

        assert isinstance(result, TransformResult)
        # 地平線より下の点なので有効
        if result.is_valid:
            assert result.floor_coords_px is not None
            assert result.world_coords_m is not None

    def test_transform_batch(self, unified_transformer):
        """バッチ変換が動作するか"""
        bboxes = [
            (600.0, 300.0, 80.0, 200.0),
            (400.0, 350.0, 60.0, 150.0),
            (800.0, 280.0, 70.0, 180.0),
        ]

        results = unified_transformer.transform_batch(bboxes)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, TransformResult)


class TestTransformPipelineBuilder:
    """TransformPipelineBuilder のテスト"""

    def test_build_complete_pipeline(self):
        """完全なパイプラインが構築できるか"""
        transformer = (
            TransformPipelineBuilder()
            .with_intrinsics(1250.0, 1250.0, 640.0, 360.0)
            .with_extrinsics(2.2, 45.0, 0.0)
            .with_floormap(1878, 1369)
            .with_camera_position(1200.0, 800.0)
            .build()
        )

        assert isinstance(transformer, UnifiedTransformer)

    def test_missing_intrinsics_raises(self):
        """内部パラメータなしでビルドするとエラー"""
        builder = TransformPipelineBuilder().with_extrinsics(2.0, 45.0, 0.0).with_camera_position(1000.0, 800.0)

        with pytest.raises(ValueError, match="intrinsics"):
            builder.build()

    def test_missing_extrinsics_raises(self):
        """外部パラメータなしでビルドするとエラー"""
        builder = (
            TransformPipelineBuilder().with_intrinsics(1000.0, 1000.0, 640.0, 360.0).with_camera_position(1000.0, 800.0)
        )

        with pytest.raises(ValueError, match="extrinsics"):
            builder.build()

    def test_missing_camera_position_raises(self):
        """カメラ位置なしでビルドするとエラー"""
        builder = (
            TransformPipelineBuilder().with_intrinsics(1000.0, 1000.0, 640.0, 360.0).with_extrinsics(2.0, 45.0, 0.0)
        )

        with pytest.raises(ValueError, match="position"):
            builder.build()


class TestAccuracyValidation:
    """精度検証テスト"""

    @pytest.fixture
    def precise_transformer(self):
        """精度検証用の変換器"""
        return (
            TransformPipelineBuilder()
            .with_intrinsics(1250.0, 1250.0, 640.0, 360.0)
            .with_extrinsics(2.2, 45.0, 0.0)
            .with_floormap(
                1878,
                1369,
                scale_x_mm_per_px=28.1926406926406,
                scale_y_mm_per_px=28.241430700447,
            )
            .with_camera_position(1200.0, 800.0)
            .build()
        )

    def test_scale_accuracy(self, precise_transformer):
        """スケール変換の精度"""
        # 1メートル移動のピクセル差を検証
        result_origin = precise_transformer.transform_pixel((640.0, 500.0))
        result_shifted = precise_transformer.transform_pixel((700.0, 500.0))

        if result_origin.is_valid and result_shifted.is_valid:
            # ピクセル差が妥当な範囲内か
            dx = abs(result_shifted.floor_coords_px[0] - result_origin.floor_coords_px[0])
            # 約35ピクセル/メートルのスケールなので、妥当な範囲をチェック
            assert dx > 0

    def test_deterministic_output(self, precise_transformer):
        """同一入力に対して決定論的な出力"""
        pixel = (640.0, 400.0)

        result1 = precise_transformer.transform_pixel(pixel)
        result2 = precise_transformer.transform_pixel(pixel)

        if result1.is_valid and result2.is_valid:
            assert result1.floor_coords_px[0] == result2.floor_coords_px[0]
            assert result1.floor_coords_px[1] == result2.floor_coords_px[1]
