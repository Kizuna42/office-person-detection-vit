"""Unit tests for feature visualization module."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.visualization.feature_visualizer import FeatureVisualizer


class TestFeatureVisualizer:
    """FeatureVisualizerのテスト"""

    def test_init_default(self):
        """デフォルトパラメータでの初期化テスト"""
        visualizer = FeatureVisualizer()
        assert visualizer.n_components == 2
        assert visualizer.perplexity == 30.0
        assert visualizer.random_state == 42

    def test_init_custom(self):
        """カスタムパラメータでの初期化テスト"""
        visualizer = FeatureVisualizer(n_components=3, perplexity=50.0, random_state=100)
        assert visualizer.n_components == 3
        assert visualizer.perplexity == 50.0
        assert visualizer.random_state == 100

    def test_visualize_features_tsne_empty(self):
        """空の特徴量でのt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.array([])
        result = visualizer.visualize_features_tsne(features)
        assert result.size == 0

    def test_visualize_features_tsne_single_sample(self):
        """単一サンプルでのt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(1, 10)
        result = visualizer.visualize_features_tsne(features)
        # サンプル数が少なすぎる場合は空配列を返す
        assert result.size == 0

    def test_visualize_features_tsne_2d(self):
        """2次元t-SNE可視化テスト"""
        visualizer = FeatureVisualizer(n_components=2)
        # 最低2サンプル必要
        features = np.random.rand(5, 10)
        result = visualizer.visualize_features_tsne(features)
        assert result.shape == (5, 2)

    def test_visualize_features_tsne_3d(self):
        """3次元t-SNE可視化テスト"""
        visualizer = FeatureVisualizer(n_components=3)
        features = np.random.rand(5, 10)
        result = visualizer.visualize_features_tsne(features)
        assert result.shape == (5, 3)

    def test_visualize_features_tsne_with_labels(self):
        """ラベル付きt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 20)
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        result = visualizer.visualize_features_tsne(features, labels=labels)
        assert result.shape == (10, 2)

    def test_visualize_features_tsne_with_track_ids(self):
        """トラックID付きt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 20)
        track_ids = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1]
        result = visualizer.visualize_features_tsne(features, track_ids=track_ids)
        assert result.shape == (10, 2)

    def test_visualize_features_tsne_with_output_path(self):
        """出力パス指定でのt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 20)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_tsne.png"
            result = visualizer.visualize_features_tsne(features, output_path=output_path)
            assert result.shape == (10, 2)
            # ファイルが作成されていることを確認
            assert output_path.exists()

    def test_visualize_features_tsne_invalid_dimension(self):
        """無効な次元数の特徴量でのテスト"""
        visualizer = FeatureVisualizer()
        # 1次元配列（2次元が必要）
        features = np.random.rand(10)
        with pytest.raises(ValueError, match="特徴量は2次元配列である必要があります"):
            visualizer.visualize_features_tsne(features)

    def test_visualize_features_tsne_large_perplexity(self):
        """perplexityが大きすぎる場合の調整テスト"""
        visualizer = FeatureVisualizer(perplexity=100.0)
        # サンプル数がperplexityより小さい場合、自動調整される
        features = np.random.rand(5, 10)
        result = visualizer.visualize_features_tsne(features)
        assert result.shape == (5, 2)

    def test_cluster_features_empty(self):
        """空の特徴量でのクラスタリングテスト"""
        visualizer = FeatureVisualizer()
        features = np.array([])
        labels, stats = visualizer.cluster_features(features)
        assert labels.size == 0
        assert stats == {}

    def test_cluster_features_basic(self):
        """基本的なクラスタリングテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(20, 10)
        labels, stats = visualizer.cluster_features(features, n_clusters=3)
        assert len(labels) == 20
        assert "n_clusters" in stats
        assert "n_samples" in stats
        assert stats["n_clusters"] == 3
        assert stats["n_samples"] == 20

    def test_cluster_features_auto_clusters(self):
        """自動クラスタ数決定のテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(50, 10)
        labels, stats = visualizer.cluster_features(features, n_clusters=None)
        assert len(labels) == 50
        assert "n_clusters" in stats
        assert stats["n_clusters"] >= 2

    def test_cluster_features_invalid_dimension(self):
        """無効な次元数の特徴量でのクラスタリングテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10)  # 1次元配列
        with pytest.raises(ValueError, match="特徴量は2次元配列である必要があります"):
            visualizer.cluster_features(features)

    def test_evaluate_clustering_quality_empty(self):
        """空の特徴量でのクラスタリング品質評価テスト"""
        visualizer = FeatureVisualizer()
        features = np.array([])
        track_ids = []
        result = visualizer.evaluate_clustering_quality(features, track_ids)
        assert result["same_cluster_ratio"] == 0.0
        assert result["avg_intra_cluster_similarity"] == 0.0
        assert result["avg_inter_cluster_similarity"] == 0.0

    def test_evaluate_clustering_quality_basic(self):
        """基本的なクラスタリング品質評価テスト"""
        visualizer = FeatureVisualizer()
        # 同じトラックIDの特徴量を近くに配置
        features = np.random.rand(20, 10)
        track_ids = [1] * 10 + [2] * 10
        result = visualizer.evaluate_clustering_quality(features, track_ids)
        assert "same_cluster_ratio" in result
        assert "avg_intra_cluster_similarity" in result
        assert "avg_inter_cluster_similarity" in result
        assert 0.0 <= result["same_cluster_ratio"] <= 1.0
        assert -1.0 <= result["avg_intra_cluster_similarity"] <= 1.0
        assert -1.0 <= result["avg_inter_cluster_similarity"] <= 1.0

    def test_evaluate_clustering_quality_single_track(self):
        """単一トラックIDでのクラスタリング品質評価テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 10)
        track_ids = [1] * 10
        result = visualizer.evaluate_clustering_quality(features, track_ids)
        assert "same_cluster_ratio" in result
        # 単一トラックの場合は評価が難しいが、エラーなく実行される

    def test_evaluate_clustering_quality_mismatched_length(self):
        """特徴量とトラックIDの長さが一致しない場合のテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 10)
        track_ids = [1, 2, 3]  # 長さが異なる
        # strict=Falseなのでエラーにならないが、警告が出る可能性がある
        result = visualizer.evaluate_clustering_quality(features, track_ids)
        assert "same_cluster_ratio" in result

    def test_cluster_features_small_n_clusters(self):
        """小さいクラスタ数でのテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 10)
        labels, stats = visualizer.cluster_features(features, n_clusters=2)
        assert len(labels) == 10
        assert stats["n_clusters"] == 2

    def test_cluster_features_large_n_clusters(self):
        """大きいクラスタ数でのテスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(20, 10)
        labels, stats = visualizer.cluster_features(features, n_clusters=10)
        assert len(labels) == 20
        assert stats["n_clusters"] == 10

    def test_visualize_features_tsne_custom_title(self):
        """カスタムタイトルでのt-SNE可視化テスト"""
        visualizer = FeatureVisualizer()
        features = np.random.rand(10, 20)
        result = visualizer.visualize_features_tsne(features, title="Custom Title")
        assert result.shape == (10, 2)
