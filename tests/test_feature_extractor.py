"""Unit tests for feature extraction module."""

import numpy as np
import pytest

from src.tracking.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """FeatureExtractorのテスト"""

    def test_init(self):
        """初期化テスト"""
        extractor = FeatureExtractor()
        assert extractor is not None

    def test_normalize_features_empty(self):
        """空の特徴量配列の正規化テスト"""
        extractor = FeatureExtractor()
        features = np.array([])
        result = extractor.normalize_features(features)
        assert result.size == 0

    def test_normalize_features_1d(self):
        """1次元特徴量の正規化テスト"""
        extractor = FeatureExtractor()
        features = np.array([3.0, 4.0])
        result = extractor.normalize_features(features.reshape(1, -1))
        assert result.shape == (1, 2)
        # L2正規化されていることを確認
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_features_2d(self):
        """2次元特徴量の正規化テスト"""
        extractor = FeatureExtractor()
        features = np.array([[3.0, 4.0], [5.0, 12.0]])
        result = extractor.normalize_features(features)
        assert result.shape == (2, 2)
        # 各行がL2正規化されていることを確認
        for i in range(2):
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 1e-6

    def test_normalize_features_zero_vector(self):
        """ゼロベクトルの正規化テスト"""
        extractor = FeatureExtractor()
        features = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = extractor.normalize_features(features)
        assert result.shape == (2, 2)
        # ゼロベクトルはそのまま（1e-8で割るため）
        assert np.allclose(result[0], [0.0, 0.0])

    def test_extract_roi_features_basic(self):
        """基本的なROI特徴量抽出テスト"""
        extractor = FeatureExtractor()
        # 3次元特徴量マップ (H, W, feature_dim)
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        bboxes = [(10.0, 20.0, 30.0, 40.0)]  # (x, y, width, height)
        image_shape = (200, 200)  # (height, width)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (1, 256)
        # 正規化されていることを確認
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6 or norm < 1e-6  # ゼロベクトルの場合も許容

    def test_extract_roi_features_multiple_bboxes(self):
        """複数バウンディングボックスでのROI特徴量抽出テスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        bboxes = [
            (10.0, 20.0, 30.0, 40.0),
            (50.0, 60.0, 20.0, 30.0),
            (80.0, 90.0, 15.0, 25.0),
        ]
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (3, 256)
        # 各行が正規化されていることを確認
        for i in range(3):
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 1e-6 or norm < 1e-6

    def test_extract_roi_features_empty_bboxes(self):
        """空のバウンディングボックスリストでのテスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        bboxes = []
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (0, 256)

    def test_extract_roi_features_out_of_bounds(self):
        """画像範囲外のバウンディングボックスでのテスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        # 画像範囲外のバウンディングボックス
        bboxes = [(-10.0, -20.0, 30.0, 40.0), (500.0, 600.0, 20.0, 30.0)]
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (2, 256)

    def test_extract_roi_features_invalid_dimension(self):
        """無効な次元数の特徴量マップでのテスト"""
        extractor = FeatureExtractor()
        # 2次元配列（3次元が必要）
        encoder_features = np.random.rand(100, 100).astype(np.float32)
        bboxes = [(10.0, 20.0, 30.0, 40.0)]
        image_shape = (200, 200)

        with pytest.raises(ValueError, match="Expected 3D encoder features"):
            extractor.extract_roi_features(encoder_features, bboxes, image_shape)

    def test_extract_roi_features_small_bbox(self):
        """小さいバウンディングボックスでのテスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        # 非常に小さいバウンディングボックス
        bboxes = [(10.0, 20.0, 1.0, 1.0)]
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (1, 256)

    def test_extract_roi_features_large_bbox(self):
        """大きいバウンディングボックスでのテスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(100, 100, 256).astype(np.float32)
        # 画像全体をカバーする大きいバウンディングボックス
        bboxes = [(0.0, 0.0, 200.0, 200.0)]
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (1, 256)

    def test_extract_roi_features_edge_cases(self):
        """エッジケースのテスト"""
        extractor = FeatureExtractor()
        encoder_features = np.random.rand(50, 50, 128).astype(np.float32)
        # 様々なエッジケース
        bboxes = [
            (0.0, 0.0, 10.0, 10.0),  # 左上角
            (190.0, 190.0, 10.0, 10.0),  # 右下角
            (100.0, 100.0, 0.0, 0.0),  # 幅・高さが0
        ]
        image_shape = (200, 200)

        result = extractor.extract_roi_features(encoder_features, bboxes, image_shape)
        assert result.shape == (3, 128)

    def test_normalize_features_large_array(self):
        """大きな特徴量配列の正規化テスト"""
        extractor = FeatureExtractor()
        # 1000サンプル、512次元の特徴量
        features = np.random.rand(1000, 512).astype(np.float32)
        result = extractor.normalize_features(features)
        assert result.shape == (1000, 512)
        # 各行が正規化されていることを確認（サンプル数個）
        for i in range(0, 1000, 100):  # 10サンプルをチェック
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 1e-6 or norm < 1e-6
