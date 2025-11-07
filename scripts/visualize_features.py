"""Feature visualization script using t-SNE."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.utils.feature_visualizer import FeatureVisualizer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_features_from_tracks(tracks_path: Path) -> tuple[np.ndarray, list[int]]:
    """トラックファイルから特徴量を読み込む

    Args:
        tracks_path: トラックファイルのパス（JSON形式）

    Returns:
        (特徴量配列, トラックIDリスト)
    """
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)

    features_list = []
    track_ids = []

    for track_data in data.get("tracks", []):
        track_id = track_data.get("track_id")
        features = track_data.get("features")

        if features is not None:
            features_list.append(np.array(features))
            track_ids.append(track_id)

    if not features_list:
        logger.warning("特徴量が見つかりませんでした")
        return np.array([]), []

    features_array = np.array(features_list)
    logger.info(f"特徴量を読み込みました: {features_array.shape[0]}サンプル, {features_array.shape[1]}次元")

    return features_array, track_ids


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="特徴量可視化スクリプト（t-SNE）")
    parser.add_argument("--tracks", type=str, required=True, help="トラックファイルのパス（JSON形式）")
    parser.add_argument("--output", type=str, default="feature_visualization.png", help="出力画像のパス")
    parser.add_argument("--output-tsne", type=str, help="t-SNE結果の出力パス（JSON形式、オプション）")
    parser.add_argument("--output-clusters", type=str, help="クラスタリング結果の出力パス（JSON形式、オプション）")
    parser.add_argument("--n-clusters", type=int, help="クラスタ数（オプション、自動決定の場合は指定不要）")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNEのperplexityパラメータ")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    logger.info("=" * 80)
    logger.info("特徴量可視化を開始")
    logger.info("=" * 80)

    # ファイルパスの確認
    tracks_path = Path(args.tracks)

    if not tracks_path.exists():
        logger.error(f"トラックファイルが見つかりません: {tracks_path}")
        return 1

    # 特徴量の読み込み
    logger.info(f"特徴量を読み込み中: {tracks_path}")
    features, track_ids = load_features_from_tracks(tracks_path)

    if features.size == 0:
        logger.error("特徴量が空です")
        return 1

    # FeatureVisualizerの初期化
    visualizer = FeatureVisualizer(perplexity=args.perplexity)

    # t-SNE可視化
    logger.info("t-SNEで次元削減中...")
    embedded = visualizer.visualize_features_tsne(
        features,
        track_ids=track_ids,
        output_path=Path(args.output),
        title="Feature Visualization (t-SNE)",
    )

    logger.info(f"t-SNE可視化完了: {args.output}")

    # t-SNE結果を保存（オプション）
    if args.output_tsne:
        tsne_data = {
            "embedded": embedded.tolist(),
            "track_ids": track_ids,
            "original_shape": features.shape,
        }
        with open(args.output_tsne, "w", encoding="utf-8") as f:
            json.dump(tsne_data, f, indent=2, ensure_ascii=False)
        logger.info(f"t-SNE結果を保存しました: {args.output_tsne}")

    # クラスタリング評価（オプション）
    if args.output_clusters:
        logger.info("クラスタリング評価を実行中...")
        labels, cluster_stats = visualizer.cluster_features(features, n_clusters=args.n_clusters)
        quality_metrics = visualizer.evaluate_clustering_quality(features, track_ids)

        cluster_data = {
            "cluster_labels": labels.tolist(),
            "cluster_stats": cluster_stats,
            "quality_metrics": quality_metrics,
            "track_ids": track_ids,
        }

        with open(args.output_clusters, "w", encoding="utf-8") as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)

        logger.info("=" * 80)
        logger.info("クラスタリング評価結果")
        logger.info("=" * 80)
        logger.info(f"  クラスタ数: {cluster_stats['n_clusters']}")
        logger.info(f"  クラスタサイズ: {cluster_stats['cluster_sizes']}")
        logger.info(f"  同一クラスタ割合: {quality_metrics['same_cluster_ratio']:.3f}")
        logger.info(f"  クラスタ内類似度: {quality_metrics['avg_intra_cluster_similarity']:.3f}")
        logger.info(f"  クラスタ間類似度: {quality_metrics['avg_inter_cluster_similarity']:.3f}")
        logger.info("=" * 80)
        logger.info(f"クラスタリング結果を保存しました: {args.output_clusters}")

    return 0


if __name__ == "__main__":
    exit(main())
