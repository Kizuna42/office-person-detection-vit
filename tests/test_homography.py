"""Unit tests for homography transform module."""

import numpy as np
import pytest

from src.transform.floormap_config import FloorMapConfig
from src.transform.homography import HomographyTransformer, TransformResult


class TestTransformResult:
    """TransformResultデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        result = TransformResult()
        assert result.floor_coords_px is None
        assert result.floor_coords_mm is None
        assert result.is_valid is False
        assert result.error_reason is None
        assert result.is_within_bounds is False

    def test_valid_result(self):
        """有効な結果のテスト"""
        result = TransformResult(
            floor_coords_px=(100.0, 200.0),
            floor_coords_mm=(2819.26, 5648.29),
            is_valid=True,
            is_within_bounds=True,
        )
        assert result.floor_coords_px == (100.0, 200.0)
        assert result.floor_coords_mm == (2819.26, 5648.29)
        assert result.is_valid is True
        assert result.is_within_bounds is True

    def test_invalid_result_with_reason(self):
        """無効な結果（理由付き）のテスト"""
        result = TransformResult(
            is_valid=False,
            error_reason="Point outside image bounds",
        )
        assert result.is_valid is False
        assert result.error_reason == "Point outside image bounds"


class TestHomographyTransformer:
    """HomographyTransformerのテスト"""

    @pytest.fixture
    def floormap_config(self):
        """テスト用フロアマップ設定"""
        return FloorMapConfig(
            width_px=1878,
            height_px=1369,
            origin_x_px=7.0,
            origin_y_px=9.0,
            scale_x_mm_per_px=28.1926406926406,
            scale_y_mm_per_px=28.241430700447,
        )

    @pytest.fixture
    def identity_homography(self):
        """単位行列（恒等変換）"""
        return np.eye(3, dtype=np.float64)

    @pytest.fixture
    def sample_homography(self):
        """サンプルホモグラフィ行列（スケーリングと平行移動）"""
        # 2倍スケーリング + (100, 50)平行移動
        H = np.array(
            [
                [2.0, 0.0, 100.0],
                [0.0, 2.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return H

    def test_init_valid_matrix(self, identity_homography, floormap_config):
        """有効な行列での初期化テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        assert transformer.H is not None
        assert transformer.floormap_config == floormap_config

    def test_init_invalid_matrix_shape(self, floormap_config):
        """無効な行列形状での初期化テスト"""
        invalid_matrix = np.eye(2)
        with pytest.raises(ValueError, match=r"3x3"):
            HomographyTransformer(invalid_matrix, floormap_config)

    def test_init_singular_matrix(self, floormap_config):
        """特異行列での初期化テスト"""
        singular_matrix = np.zeros((3, 3))
        with pytest.raises(ValueError, match=r"特異行列"):
            HomographyTransformer(singular_matrix, floormap_config)

    def test_transform_pixel_identity(self, identity_homography, floormap_config):
        """恒等変換での点変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((100.0, 200.0))

        assert result.is_valid is True
        assert result.floor_coords_px is not None
        assert result.floor_coords_px[0] == pytest.approx(100.0)
        assert result.floor_coords_px[1] == pytest.approx(200.0)

    def test_transform_pixel_with_scaling(self, sample_homography, floormap_config):
        """スケーリング変換での点変換テスト"""
        transformer = HomographyTransformer(sample_homography, floormap_config)
        result = transformer.transform_pixel((50.0, 100.0))

        assert result.is_valid is True
        assert result.floor_coords_px is not None
        # 2 * 50 + 100 = 200, 2 * 100 + 50 = 250
        assert result.floor_coords_px[0] == pytest.approx(200.0)
        assert result.floor_coords_px[1] == pytest.approx(250.0)

    def test_transform_pixel_mm_conversion(self, identity_homography, floormap_config):
        """mm座標変換のテスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((100.0, 200.0))

        assert result.floor_coords_mm is not None
        expected_x_mm = 100.0 * floormap_config.scale_x_mm_per_px
        expected_y_mm = 200.0 * floormap_config.scale_y_mm_per_px
        assert result.floor_coords_mm[0] == pytest.approx(expected_x_mm)
        assert result.floor_coords_mm[1] == pytest.approx(expected_y_mm)

    def test_transform_pixel_within_bounds(self, identity_homography, floormap_config):
        """境界内の点変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((500.0, 500.0))

        assert result.is_within_bounds is True

    def test_transform_pixel_outside_bounds(self, identity_homography, floormap_config):
        """境界外の点変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((2000.0, 2000.0))

        assert result.is_valid is True  # 変換自体は有効
        assert result.is_within_bounds is False  # 境界外

    def test_transform_detection(self, identity_homography, floormap_config):
        """検出結果（BBox）の変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        bbox = (100.0, 200.0, 50.0, 100.0)  # x, y, w, h
        result = transformer.transform_detection(bbox)

        assert result.is_valid is True
        # 足元点: (100 + 50/2, 200 + 100) = (125, 300)
        assert result.floor_coords_px is not None
        assert result.floor_coords_px[0] == pytest.approx(125.0)
        assert result.floor_coords_px[1] == pytest.approx(300.0)

    def test_transform_batch_empty(self, identity_homography, floormap_config):
        """空のバッチ変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        results = transformer.transform_batch([])

        assert len(results) == 0

    def test_transform_batch_single(self, identity_homography, floormap_config):
        """単一要素のバッチ変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        bboxes = [(100.0, 200.0, 50.0, 100.0)]
        results = transformer.transform_batch(bboxes)

        assert len(results) == 1
        assert results[0].is_valid is True
        # 足元点: (125, 300)
        assert results[0].floor_coords_px is not None
        assert results[0].floor_coords_px[0] == pytest.approx(125.0)
        assert results[0].floor_coords_px[1] == pytest.approx(300.0)

    def test_transform_batch_multiple(self, identity_homography, floormap_config):
        """複数要素のバッチ変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        bboxes = [
            (100.0, 200.0, 50.0, 100.0),  # 足元点: (125, 300)
            (200.0, 300.0, 60.0, 120.0),  # 足元点: (230, 420)
            (300.0, 400.0, 40.0, 80.0),  # 足元点: (320, 480)
        ]
        results = transformer.transform_batch(bboxes)

        assert len(results) == 3
        assert all(r.is_valid for r in results)

        expected_foot_points = [(125.0, 300.0), (230.0, 420.0), (320.0, 480.0)]
        for result, expected in zip(results, expected_foot_points, strict=False):
            assert result.floor_coords_px is not None
            assert result.floor_coords_px[0] == pytest.approx(expected[0])
            assert result.floor_coords_px[1] == pytest.approx(expected[1])

    def test_get_info(self, sample_homography, floormap_config):
        """情報取得テスト"""
        transformer = HomographyTransformer(sample_homography, floormap_config)
        info = transformer.get_info()

        assert info["method"] == "homography"
        assert "matrix" in info
        assert info["floormap_size"] == (1878, 1369)
        assert "scale_mm_per_px" in info

    def test_perspective_transform(self, floormap_config):
        """透視変換（実際のホモグラフィ）のテスト"""
        # 4点対応から計算されたホモグラフィ行列（例）
        H = np.array(
            [
                [1.5, 0.2, 50.0],
                [0.1, 1.8, 30.0],
                [0.0001, 0.0002, 1.0],
            ],
            dtype=np.float64,
        )
        transformer = HomographyTransformer(H, floormap_config)
        result = transformer.transform_pixel((100.0, 100.0))

        assert result.is_valid is True
        assert result.floor_coords_px is not None
        # 透視変換の結果を確認（正確な値は行列依存）
        assert isinstance(result.floor_coords_px[0], float)
        assert isinstance(result.floor_coords_px[1], float)

    def test_transform_origin(self, identity_homography, floormap_config):
        """原点の変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((0.0, 0.0))

        assert result.is_valid is True
        assert result.floor_coords_px is not None
        assert result.floor_coords_px[0] == pytest.approx(0.0)
        assert result.floor_coords_px[1] == pytest.approx(0.0)
        assert result.is_within_bounds is True

    def test_transform_negative_coords(self, identity_homography, floormap_config):
        """負の座標の変換テスト"""
        transformer = HomographyTransformer(identity_homography, floormap_config)
        result = transformer.transform_pixel((-10.0, -10.0))

        assert result.is_valid is True
        assert result.floor_coords_px is not None
        assert result.is_within_bounds is False  # 負の座標は境界外
