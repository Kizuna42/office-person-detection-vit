#!/usr/bin/env python
"""目標時刻との誤差測定と可視化ツール

抽出されたフレームの目標時刻との誤差を分析し、可視化します。
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import ConfigManager
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_extraction_results(csv_path: Path) -> list[dict]:
    """抽出結果CSVを読み込み

    Args:
        csv_path: extraction_results.csvのパス

    Returns:
        抽出結果のリスト
    """
    results = []

    if not csv_path.exists():
        logger.error(f"CSVファイルが見つかりません: {csv_path}")
        return results

    # ファイルの行数を取得（プログレスバー用）
    total_lines = sum(1 for _ in csv_path.open("r", encoding="utf-8"))

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_lines - 1, desc="CSV読み込み中"):  # -1はヘッダー行
            try:
                target_ts = datetime.strptime(row["target_timestamp"], "%Y/%m/%d %H:%M:%S")
                extracted_ts = datetime.strptime(row["extracted_timestamp"], "%Y/%m/%d %H:%M:%S")
                time_diff = float(row["time_diff_seconds"])

                results.append(
                    {
                        "target_timestamp": target_ts,
                        "extracted_timestamp": extracted_ts,
                        "time_diff_seconds": time_diff,
                        "frame_index": int(row["frame_index"]),
                        "confidence": float(row["confidence"]),
                    }
                )
            except Exception as e:
                logger.warning(f"行のパースに失敗: {row}, エラー: {e}")

    return results


def analyze_errors(results: list[dict], tolerance_seconds: float = 10.0) -> dict:
    """誤差を分析

    Args:
        results: 抽出結果のリスト
        tolerance_seconds: 許容誤差（秒）

    Returns:
        分析結果の辞書
    """
    if not results:
        return {}

    time_diffs = [abs(r["time_diff_seconds"]) for r in results]

    # 統計
    mean_error = np.mean(time_diffs)
    median_error = np.median(time_diffs)
    std_error = np.std(time_diffs)
    min_error = np.min(time_diffs)
    max_error = np.max(time_diffs)

    # ±10秒以内の達成率
    within_tolerance = sum(1 for d in time_diffs if d <= tolerance_seconds)
    tolerance_rate = (within_tolerance / len(time_diffs) * 100) if time_diffs else 0.0

    # 誤差分布の統計
    error_ranges = {
        "0-5秒": sum(1 for d in time_diffs if 0 <= d <= 5),
        "5-10秒": sum(1 for d in time_diffs if 5 < d <= 10),
        "10-30秒": sum(1 for d in time_diffs if 10 < d <= 30),
        "30-60秒": sum(1 for d in time_diffs if 30 < d <= 60),
        "60秒以上": sum(1 for d in time_diffs if d > 60),
    }

    analysis = {
        "total_count": len(results),
        "mean_error_seconds": mean_error,
        "median_error_seconds": median_error,
        "std_error_seconds": std_error,
        "min_error_seconds": min_error,
        "max_error_seconds": max_error,
        "within_tolerance_count": within_tolerance,
        "tolerance_rate_percent": tolerance_rate,
        "tolerance_seconds": tolerance_seconds,
        "error_distribution": error_ranges,
    }

    return analysis


def visualize_error_distribution(results: list[dict], output_path: Path, tolerance_seconds: float = 10.0):
    """誤差分布を可視化

    Args:
        results: 抽出結果のリスト
        output_path: 出力画像のパス
        tolerance_seconds: 許容誤差（秒）
    """
    if not results:
        logger.warning("可視化するデータがありません")
        return

    time_diffs = [r["time_diff_seconds"] for r in results]

    # 図の作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 誤差のヒストグラム
    ax1 = axes[0, 0]
    ax1.hist(time_diffs, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="green", linestyle="--", label="目標時刻")
    ax1.axvline(x=tolerance_seconds, color="red", linestyle="--", label=f"±{tolerance_seconds}秒")
    ax1.axvline(x=-tolerance_seconds, color="red", linestyle="--")
    ax1.set_xlabel("誤差（秒）")
    ax1.set_ylabel("頻度")
    ax1.set_title("目標時刻との誤差分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 時系列での誤差プロット
    ax2 = axes[0, 1]
    frame_indices = [r["frame_index"] for r in results]
    ax2.scatter(frame_indices, time_diffs, alpha=0.6, s=20)
    ax2.axhline(y=0, color="green", linestyle="--", label="目標時刻")
    ax2.axhline(y=tolerance_seconds, color="red", linestyle="--", label=f"±{tolerance_seconds}秒")
    ax2.axhline(y=-tolerance_seconds, color="red", linestyle="--")
    ax2.set_xlabel("フレーム番号")
    ax2.set_ylabel("誤差（秒）")
    ax2.set_title("時系列での誤差")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 誤差範囲の分布（円グラフ）
    ax3 = axes[1, 0]
    error_ranges = {
        "0-5秒": sum(1 for d in time_diffs if 0 <= abs(d) <= 5),
        "5-10秒": sum(1 for d in time_diffs if 5 < abs(d) <= 10),
        "10-30秒": sum(1 for d in time_diffs if 10 < abs(d) <= 30),
        "30-60秒": sum(1 for d in time_diffs if 30 < abs(d) <= 60),
        "60秒以上": sum(1 for d in time_diffs if abs(d) > 60),
    }
    labels = [k for k, v in error_ranges.items() if v > 0]
    sizes = [v for k, v in error_ranges.items() if v > 0]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#c0392b"]
    ax3.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[: len(labels)])
    ax3.set_title("誤差範囲の分布")

    # 4. 信頼度と誤差の関係
    ax4 = axes[1, 1]
    confidences = [r["confidence"] for r in results]
    ax4.scatter(confidences, [abs(d) for d in time_diffs], alpha=0.6, s=20)
    ax4.set_xlabel("信頼度")
    ax4.set_ylabel("誤差の絶対値（秒）")
    ax4.set_title("信頼度と誤差の関係")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"誤差分布グラフを保存しました: {output_path}")
    plt.close()


def print_analysis_report(analysis: dict):
    """分析レポートを出力

    Args:
        analysis: 分析結果の辞書
    """
    logger.info("=" * 80)
    logger.info("目標時刻との誤差分析レポート")
    logger.info("=" * 80)

    logger.info(f"総抽出数: {analysis.get('total_count', 0)}")
    logger.info("\n誤差統計:")
    logger.info(f"  平均誤差: {analysis.get('mean_error_seconds', 0):.2f}秒")
    logger.info(f"  中央値誤差: {analysis.get('median_error_seconds', 0):.2f}秒")
    logger.info(f"  標準偏差: {analysis.get('std_error_seconds', 0):.2f}秒")
    logger.info(f"  最小誤差: {analysis.get('min_error_seconds', 0):.2f}秒")
    logger.info(f"  最大誤差: {analysis.get('max_error_seconds', 0):.2f}秒")

    logger.info(f"\n許容誤差（±{analysis.get('tolerance_seconds', 10)}秒）:")
    logger.info(f"  達成数: {analysis.get('within_tolerance_count', 0)}/{analysis.get('total_count', 0)}")
    logger.info(f"  達成率: {analysis.get('tolerance_rate_percent', 0):.2f}%")

    logger.info("\n誤差分布:")
    error_dist = analysis.get("error_distribution", {})
    for range_name, count in error_dist.items():
        if count > 0:
            percentage = (count / analysis.get("total_count", 1)) * 100
            logger.info(f"  {range_name}: {count}件 ({percentage:.1f}%)")

    logger.info("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="目標時刻との誤差測定と可視化")
    parser.add_argument("--csv", type=str, help="extraction_results.csvのパス")
    parser.add_argument("--output-dir", type=str, help="出力ディレクトリ（CSVが未指定の場合）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--tolerance", type=float, default=10.0, help="許容誤差（秒、デフォルト: 10.0）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # CSVパスの決定
    if args.csv:
        csv_path = Path(args.csv)
    elif args.output_dir:
        csv_path = Path(args.output_dir) / "extracted_frames" / "extraction_results.csv"
    else:
        config = ConfigManager(args.config)
        output_dir = Path(config.get("output.directory", "output"))
        csv_path = output_dir / "extracted_frames" / "extraction_results.csv"

    # 結果読み込み
    results = load_extraction_results(csv_path)

    if not results:
        logger.warning("分析するデータがありません（CSVファイルが空またはヘッダーのみ）")
        logger.warning("これは、すべてのフレームでタイムスタンプ抽出に失敗したことを示しています")
        logger.info("=" * 80)
        logger.info("目標時刻との誤差分析レポート")
        logger.info("=" * 80)
        logger.info("総抽出数: 0")
        logger.info("すべてのフレームでタイムスタンプ抽出に失敗しました")
        logger.info("=" * 80)
        return 0

    # 誤差分析
    analysis = analyze_errors(results, args.tolerance)

    if not analysis:
        logger.error("分析に失敗しました")
        return 1

    # レポート出力
    print_analysis_report(analysis)

    # 可視化
    output_dir = csv_path.parent
    graph_path = output_dir / "timestamp_error_distribution.png"
    visualize_error_distribution(results, graph_path, args.tolerance)

    # 結果をJSONで保存
    import json

    report_path = output_dir / "timestamp_error_analysis.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n分析結果を保存しました: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
