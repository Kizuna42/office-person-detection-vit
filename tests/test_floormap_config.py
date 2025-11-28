"""Unit tests for floormap config module."""

import pytest

from src.transform.floormap_config import FloorMapConfig


class TestFloorMapConfig:
    """FloorMapConfigデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = FloorMapConfig()

        assert config.width_px == 1878
        assert config.height_px == 1369
        assert config.origin_x_px == 7.0
        assert config.origin_y_px == 9.0
        assert config.scale_x_mm_per_px == pytest.approx(28.1926406926406)
        assert config.scale_y_mm_per_px == pytest.approx(28.241430700447)

    def test_custom_values(self):
        """カスタム値のテスト"""
        config = FloorMapConfig(
            width_px=1920,
            height_px=1080,
            origin_x_px=0.0,
            origin_y_px=0.0,
            scale_x_mm_per_px=25.0,
            scale_y_mm_per_px=25.0,
        )

        assert config.width_px == 1920
        assert config.height_px == 1080
        assert config.origin_x_px == 0.0
        assert config.origin_y_px == 0.0
        assert config.scale_x_mm_per_px == 25.0
        assert config.scale_y_mm_per_px == 25.0

    def test_from_config_full(self):
        """設定辞書からの作成テスト（全パラメータ）"""
        config_dict = {
            "image_width": 1920,
            "image_height": 1080,
            "image_origin_x": 10.0,
            "image_origin_y": 15.0,
            "image_x_mm_per_pixel": 30.0,
            "image_y_mm_per_pixel": 30.0,
        }
        config = FloorMapConfig.from_config(config_dict)

        assert config.width_px == 1920
        assert config.height_px == 1080
        assert config.origin_x_px == 10.0
        assert config.origin_y_px == 15.0
        assert config.scale_x_mm_per_px == 30.0
        assert config.scale_y_mm_per_px == 30.0

    def test_from_config_partial(self):
        """設定辞書からの作成テスト（一部パラメータ）"""
        config_dict = {
            "image_width": 2000,
            "image_height": 1500,
        }
        config = FloorMapConfig.from_config(config_dict)

        assert config.width_px == 2000
        assert config.height_px == 1500
        # デフォルト値が使用される
        assert config.origin_x_px == 7.0
        assert config.origin_y_px == 9.0

    def test_from_config_empty(self):
        """空の設定辞書からの作成テスト"""
        config = FloorMapConfig.from_config({})

        # すべてデフォルト値
        assert config.width_px == 1878
        assert config.height_px == 1369

    def test_scale_x_m_per_px(self):
        """X軸スケール（m/pixel）プロパティテスト"""
        config = FloorMapConfig(scale_x_mm_per_px=28.0)

        assert config.scale_x_m_per_px == pytest.approx(0.028)

    def test_scale_y_m_per_px(self):
        """Y軸スケール（m/pixel）プロパティテスト"""
        config = FloorMapConfig(scale_y_mm_per_px=28.0)

        assert config.scale_y_m_per_px == pytest.approx(0.028)

    def test_scale_x_px_per_m(self):
        """X軸スケール（pixel/m）プロパティテスト"""
        config = FloorMapConfig(scale_x_mm_per_px=25.0)

        # 1000 / 25 = 40
        assert config.scale_x_px_per_m == pytest.approx(40.0)

    def test_scale_y_px_per_m(self):
        """Y軸スケール（pixel/m）プロパティテスト"""
        config = FloorMapConfig(scale_y_mm_per_px=20.0)

        # 1000 / 20 = 50
        assert config.scale_y_px_per_m == pytest.approx(50.0)

    def test_scale_consistency(self):
        """スケール変換の一貫性テスト"""
        config = FloorMapConfig(
            scale_x_mm_per_px=28.1926406926406,
            scale_y_mm_per_px=28.241430700447,
        )

        # mm/px * px/m = mm/m = 1/1000 * 1000 = 1
        # つまり scale_mm_per_px * scale_px_per_m = 1000
        assert config.scale_x_mm_per_px * config.scale_x_px_per_m == pytest.approx(1000.0)
        assert config.scale_y_mm_per_px * config.scale_y_px_per_m == pytest.approx(1000.0)

    def test_scale_m_per_px_consistency(self):
        """メートルスケールの一貫性テスト"""
        config = FloorMapConfig(scale_x_mm_per_px=25.0, scale_y_mm_per_px=30.0)

        # scale_m_per_px = scale_mm_per_px / 1000
        assert config.scale_x_m_per_px * 1000 == pytest.approx(config.scale_x_mm_per_px)
        assert config.scale_y_m_per_px * 1000 == pytest.approx(config.scale_y_mm_per_px)

    def test_pixel_to_mm_conversion(self):
        """ピクセルからミリメートルへの変換テスト"""
        config = FloorMapConfig(scale_x_mm_per_px=25.0, scale_y_mm_per_px=30.0)

        # 100ピクセル = 100 * 25 = 2500mm (X軸)
        # 100ピクセル = 100 * 30 = 3000mm (Y軸)
        px = 100
        expected_x_mm = px * config.scale_x_mm_per_px
        expected_y_mm = px * config.scale_y_mm_per_px

        assert expected_x_mm == pytest.approx(2500.0)
        assert expected_y_mm == pytest.approx(3000.0)

    def test_mm_to_pixel_conversion(self):
        """ミリメートルからピクセルへの変換テスト"""
        config = FloorMapConfig(scale_x_mm_per_px=25.0, scale_y_mm_per_px=30.0)

        # 1000mm = 1000 / 25 = 40px (X軸)
        # 1000mm = 1000 / 30 = 33.33px (Y軸)
        mm = 1000
        expected_x_px = mm / config.scale_x_mm_per_px
        expected_y_px = mm / config.scale_y_mm_per_px

        assert expected_x_px == pytest.approx(40.0)
        assert expected_y_px == pytest.approx(33.333, rel=0.01)

    def test_meter_to_pixel_conversion(self):
        """メートルからピクセルへの変換テスト"""
        config = FloorMapConfig(scale_x_mm_per_px=25.0, scale_y_mm_per_px=25.0)

        # 1m = 40px (scale_px_per_m = 1000 / 25 = 40)
        meters = 5.0
        expected_px = meters * config.scale_x_px_per_m

        assert expected_px == pytest.approx(200.0)

    def test_from_config_type_conversion(self):
        """設定辞書からの型変換テスト"""
        config_dict = {
            "image_width": "1920",  # 文字列
            "image_height": 1080.5,  # float
            "image_origin_x": "10.5",  # 文字列
        }
        config = FloorMapConfig.from_config(config_dict)

        # intに変換される
        assert config.width_px == 1920
        assert config.height_px == 1080
        # floatに変換される
        assert config.origin_x_px == 10.5
