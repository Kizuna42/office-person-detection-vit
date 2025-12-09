"""Unit tests for PWA (Piecewise Affine) Transform."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.transform import FloorMapConfig, PiecewiseAffineTransformer, ThinPlateSplineTransformer


@pytest.fixture
def sample_correspondence_points() -> list[dict]:
    """テスト用の対応点（6点で三角形分割可能）"""
    return [
        {"src_point": [100.0, 100.0], "dst_point": [200.0, 200.0]},
        {"src_point": [500.0, 100.0], "dst_point": [600.0, 200.0]},
        {"src_point": [100.0, 400.0], "dst_point": [200.0, 500.0]},
        {"src_point": [500.0, 400.0], "dst_point": [600.0, 500.0]},
        {"src_point": [300.0, 250.0], "dst_point": [400.0, 350.0]},
        {"src_point": [300.0, 100.0], "dst_point": [400.0, 200.0]},
    ]


@pytest.fixture
def sample_floormap_config() -> FloorMapConfig:
    """テスト用のフロアマップ設定"""
    return FloorMapConfig(
        width_px=1000,
        height_px=800,
        scale_x_mm_per_px=10.0,
        scale_y_mm_per_px=10.0,
    )


@pytest.fixture
def sample_src_dst_arrays(sample_correspondence_points):
    """対応点をnumpy配列に変換"""
    src = np.array([p["src_point"] for p in sample_correspondence_points])
    dst = np.array([p["dst_point"] for p in sample_correspondence_points])
    return src, dst


class TestPiecewiseAffineTransformer:
    """PWA変換器のテスト"""

    def test_init(self, sample_src_dst_arrays, sample_floormap_config):
        """初期化が正しく動作する"""
        src, dst = sample_src_dst_arrays

        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        assert pwa.src_points.shape == (6, 2)
        assert pwa.dst_points.shape == (6, 2)
        assert len(pwa.affine_matrices) > 0
        assert pwa.delaunay is not None

    def test_init_insufficient_points(self, sample_floormap_config):
        """点が不足している場合はエラー"""
        src = np.array([[100, 100], [200, 200]])
        dst = np.array([[150, 150], [250, 250]])

        with pytest.raises(ValueError, match="最低3点"):
            PiecewiseAffineTransformer(src, dst, sample_floormap_config)

    def test_transform_pixel_interpolation(self, sample_src_dst_arrays, sample_floormap_config):
        """訓練点での変換は正確（補間）"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        # 訓練点での変換は正確であるべき
        for s, d in zip(src, dst, strict=False):
            result = pwa.transform_pixel((s[0], s[1]))
            assert result.is_valid
            assert result.floor_coords_px is not None
            # 補間なのでほぼ完璧
            assert abs(result.floor_coords_px[0] - d[0]) < 0.01
            assert abs(result.floor_coords_px[1] - d[1]) < 0.01

    def test_transform_pixel_extrapolation(self, sample_src_dst_arrays, sample_floormap_config):
        """三角形外の点は外挿される"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        # 三角形外の点
        result = pwa.transform_pixel((0.0, 0.0))
        assert result.is_valid
        assert result.is_extrapolated

    def test_transform_detection(self, sample_src_dst_arrays, sample_floormap_config):
        """BBoxの変換（足元点を使用）"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        # bbox = (x, y, width, height)
        # 足元点 = (x + width/2, y + height)
        bbox = (100.0, 50.0, 100.0, 50.0)
        result = pwa.transform_detection(bbox)

        assert result.is_valid
        # 足元点は (150, 100) → 変換後は (250, 200) 付近
        assert result.floor_coords_px is not None

    def test_transform_batch(self, sample_src_dst_arrays, sample_floormap_config):
        """バッチ変換"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        bboxes = [
            (100.0, 50.0, 100.0, 50.0),
            (200.0, 150.0, 80.0, 100.0),
        ]
        results = pwa.transform_batch(bboxes)

        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_evaluate_training_error(self, sample_src_dst_arrays, sample_floormap_config):
        """訓練誤差の評価"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        error = pwa.evaluate_training_error()

        assert "rmse" in error
        assert "max_error" in error
        # PWAは訓練データに対して完璧な補間
        assert error["rmse"] < 0.01

    def test_get_info(self, sample_src_dst_arrays, sample_floormap_config):
        """情報取得"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        info = pwa.get_info()

        assert info["method"] == "piecewise_affine"
        assert info["num_points"] == 6
        assert "num_triangles" in info
        assert "training_error" in info

    def test_save_load(self, sample_src_dst_arrays, sample_floormap_config):
        """モデルの保存と読み込み"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            pwa.save(model_path)

            loaded = PiecewiseAffineTransformer.load(model_path, sample_floormap_config)

            # 同じ変換結果になるべき
            for s in src:
                result1 = pwa.transform_pixel((s[0], s[1]))
                result2 = loaded.transform_pixel((s[0], s[1]))
                assert result1.floor_coords_px[0] == pytest.approx(result2.floor_coords_px[0])
                assert result1.floor_coords_px[1] == pytest.approx(result2.floor_coords_px[1])

    def test_from_correspondence_file(self, sample_correspondence_points, sample_floormap_config):
        """対応点ファイルからの作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "correspondence.json"
            with open(file_path, "w") as f:
                json.dump({"point_correspondences": sample_correspondence_points}, f)

            pwa = PiecewiseAffineTransformer.from_correspondence_file(file_path, sample_floormap_config)

            assert pwa.src_points.shape == (6, 2)

    def test_mm_coordinates(self, sample_src_dst_arrays, sample_floormap_config):
        """mm座標の計算"""
        src, dst = sample_src_dst_arrays
        pwa = PiecewiseAffineTransformer(src, dst, sample_floormap_config)

        result = pwa.transform_pixel((src[0][0], src[0][1]))

        assert result.floor_coords_mm is not None
        # scale = 10 mm/px
        assert result.floor_coords_mm[0] == pytest.approx(result.floor_coords_px[0] * 10.0)
        assert result.floor_coords_mm[1] == pytest.approx(result.floor_coords_px[1] * 10.0)


class TestThinPlateSplineTransformer:
    """TPS変換器のテスト"""

    def test_init(self, sample_src_dst_arrays, sample_floormap_config):
        """初期化が正しく動作する"""
        src, dst = sample_src_dst_arrays

        tps = ThinPlateSplineTransformer(src, dst, sample_floormap_config)

        assert tps.src_points.shape == (6, 2)
        assert tps.dst_points.shape == (6, 2)

    def test_transform_pixel_interpolation(self, sample_src_dst_arrays, sample_floormap_config):
        """訓練点での変換は正確"""
        src, dst = sample_src_dst_arrays
        tps = ThinPlateSplineTransformer(src, dst, sample_floormap_config)

        for s, d in zip(src, dst, strict=False):
            result = tps.transform_pixel((s[0], s[1]))
            assert result.is_valid
            assert result.floor_coords_px is not None
            # TPSも訓練点では正確
            assert abs(result.floor_coords_px[0] - d[0]) < 0.1
            assert abs(result.floor_coords_px[1] - d[1]) < 0.1

    def test_get_info(self, sample_src_dst_arrays, sample_floormap_config):
        """情報取得"""
        src, dst = sample_src_dst_arrays
        tps = ThinPlateSplineTransformer(src, dst, sample_floormap_config)

        info = tps.get_info()

        assert info["method"] == "thin_plate_spline"
        assert info["num_points"] == 6
        assert "training_error" in info

    def test_from_correspondence_file(self, sample_correspondence_points, sample_floormap_config):
        """対応点ファイルからの作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "correspondence.json"
            with open(file_path, "w") as f:
                json.dump({"point_correspondences": sample_correspondence_points}, f)

            tps = ThinPlateSplineTransformer.from_correspondence_file(file_path, sample_floormap_config)

            assert tps.src_points.shape == (6, 2)
