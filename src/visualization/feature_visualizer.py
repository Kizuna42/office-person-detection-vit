"""Feature visualization module using t-SNE and clustering analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    from pathlib import Path

    from mpl_toolkits.mplot3d.axes3d import Axes3D

logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """特徴量可視化クラス

    t-SNE等を使用して特徴量を可視化し、クラスタリング評価を行います。
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42):
        """FeatureVisualizerを初期化

        Args:
            n_components: t-SNEの次元数（2または3）
            perplexity: t-SNEのperplexityパラメータ（サンプル数より小さい必要がある）
            random_state: 乱数シード
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        # t-SNEは後で初期化（サンプル数に応じてperplexityを調整）

        logger.info(f"FeatureVisualizer initialized: n_components={n_components}, perplexity={perplexity}")

    def visualize_features_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        track_ids: list[int] | None = None,
        output_path: Path | None = None,
        title: str = "Feature Visualization (t-SNE)",
    ) -> np.ndarray:
        """t-SNEを使用して特徴量を可視化

        Args:
            features: 特徴量配列 (num_samples, feature_dim)
            labels: ラベル配列（オプション、クラスタリング結果など）
            track_ids: トラックIDのリスト（オプション）
            output_path: 出力ファイルのパス（オプション）
            title: グラフのタイトル

        Returns:
            t-SNEで次元削減された特徴量 (num_samples, n_components)
        """
        if features.size == 0:
            logger.warning("特徴量が空です")
            return np.array([])

        if len(features.shape) != 2:
            raise ValueError(f"特徴量は2次元配列である必要があります: {features.shape}")

        logger.info(
            f"t-SNEで次元削減中: {features.shape[0]}サンプル, {features.shape[1]}次元 → {self.n_components}次元"
        )

        # perplexityをサンプル数に応じて調整（perplexity < n_samples が必要）
        n_samples = features.shape[0]
        if n_samples < 2:
            logger.warning("サンプル数が少なすぎます（t-SNEには最低2サンプル必要）")
            return np.array([])

        # perplexityはn_samples-1より小さい必要がある
        max_perplexity = max(1, n_samples - 1)
        adjusted_perplexity = min(self.perplexity, max_perplexity)
        if adjusted_perplexity != self.perplexity:
            logger.info(f"perplexityを調整: {self.perplexity} → {adjusted_perplexity} (サンプル数: {n_samples})")

        # t-SNEで次元削減
        tsne = TSNE(n_components=self.n_components, perplexity=adjusted_perplexity, random_state=self.random_state)
        embedded = tsne.fit_transform(features)

        # 可視化
        fig = plt.figure(figsize=(10, 8))
        if self.n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1], c=labels if labels is not None else track_ids, cmap="tab20", alpha=0.6
            )
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
        else:  # 3次元
            ax3d = cast("Axes3D", fig.add_subplot(111, projection="3d"))
            scatter = ax3d.scatter(
                embedded[:, 0],
                embedded[:, 1],
                embedded[:, 2],
                c=labels if labels is not None else track_ids,
                cmap="tab20",
                alpha=0.6,
            )
            ax3d.set_xlabel("t-SNE Component 1")
            ax3d.set_ylabel("t-SNE Component 2")
            ax3d.set_zlabel("t-SNE Component 3")
            ax = ax3d

        ax.set_title(title)
        colorbar_label = "Track ID" if track_ids is not None else "Cluster"
        plt.colorbar(scatter, ax=ax, label=colorbar_label)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"可視化結果を保存しました: {output_path}")

        plt.close(fig)

        return cast("np.ndarray", embedded)

    def cluster_features(
        self,
        features: np.ndarray,
        n_clusters: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """特徴量をクラスタリング

        Args:
            features: 特徴量配列 (num_samples, feature_dim)
            n_clusters: クラスタ数（Noneの場合は自動決定）

        Returns:
            (クラスタラベル, クラスタリング統計情報)
        """
        if features.size == 0:
            logger.warning("特徴量が空です")
            return np.array([]), {}

        if len(features.shape) != 2:
            raise ValueError(f"特徴量は2次元配列である必要があります: {features.shape}")

        # クラスタ数を自動決定（エルボー法の簡易版）
        if n_clusters is None:
            n_clusters = min(10, max(2, features.shape[0] // 10))

        logger.info(f"K-meansクラスタリング実行中: {features.shape[0]}サンプル, {n_clusters}クラスタ")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(features)

        # クラスタリング統計情報
        cluster_stats: dict[str, Any] = {
            "n_clusters": n_clusters,
            "n_samples": features.shape[0],
            "cluster_sizes": [int(np.sum(labels == cluster_id)) for cluster_id in range(n_clusters)],
            "inertia": float(kmeans.inertia_),
        }

        logger.info(f"クラスタリング完了: {n_clusters}クラスタ, 慣性={kmeans.inertia_:.2f}")

        return labels, cluster_stats

    def evaluate_clustering_quality(
        self,
        features: np.ndarray,
        track_ids: list[int],
    ) -> dict[str, float]:
        """クラスタリング品質を評価

        同一トラックIDの特徴量が同じクラスタに属するかを評価します。

        Args:
            features: 特徴量配列 (num_samples, feature_dim)
            track_ids: トラックIDのリスト

        Returns:
            評価結果の辞書:
                - same_cluster_ratio: 同一トラックIDが同じクラスタに属する割合
                - avg_intra_cluster_similarity: クラスタ内の平均類似度
                - avg_inter_cluster_similarity: クラスタ間の平均類似度
        """
        if features.size == 0 or len(track_ids) == 0:
            return {
                "same_cluster_ratio": 0.0,
                "avg_intra_cluster_similarity": 0.0,
                "avg_inter_cluster_similarity": 0.0,
            }

        # クラスタリング実行
        labels, _ = self.cluster_features(features)

        # 同一トラックIDが同じクラスタに属する割合を計算
        track_to_cluster = {}
        for track_id, cluster_id in zip(track_ids, labels, strict=False):
            if track_id not in track_to_cluster:
                track_to_cluster[track_id] = cluster_id
            elif track_to_cluster[track_id] != cluster_id:
                # 異なるクラスタに属している
                pass

        same_cluster_count = 0
        total_pairs = 0

        for track_id in set(track_ids):
            track_indices = [idx for idx, tid in enumerate(track_ids) if tid == track_id]
            if len(track_indices) < 2:
                continue

            cluster_ids = [labels[idx] for idx in track_indices]
            if len(set(cluster_ids)) == 1:  # 全て同じクラスタ
                same_cluster_count += len(track_indices) * (len(track_indices) - 1) // 2

            total_pairs += len(track_indices) * (len(track_indices) - 1) // 2

        same_cluster_ratio = same_cluster_count / total_pairs if total_pairs > 0 else 0.0

        # クラスタ内・クラスタ間の類似度を計算（コサイン類似度）
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(features)

        intra_cluster_similarities = []
        inter_cluster_similarities = []

        for idx_i in range(len(features)):
            for idx_j in range(idx_i + 1, len(features)):
                if labels[idx_i] == labels[idx_j]:
                    intra_cluster_similarities.append(similarity_matrix[idx_i, idx_j])
                else:
                    inter_cluster_similarities.append(similarity_matrix[idx_i, idx_j])

        avg_intra = np.mean(intra_cluster_similarities) if intra_cluster_similarities else 0.0
        avg_inter = np.mean(inter_cluster_similarities) if inter_cluster_similarities else 0.0

        result = {
            "same_cluster_ratio": float(same_cluster_ratio),
            "avg_intra_cluster_similarity": float(avg_intra),
            "avg_inter_cluster_similarity": float(avg_inter),
        }

        logger.info(
            f"クラスタリング品質評価: 同一クラスタ割合={same_cluster_ratio:.3f}, "
            f"クラスタ内類似度={avg_intra:.3f}, クラスタ間類似度={avg_inter:.3f}"
        )

        return result
