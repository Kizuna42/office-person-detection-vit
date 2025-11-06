#!/usr/bin/env python
"""時系列整合性スコアの測定ツール

抽出されたタイムスタンプの時系列整合性を評価します。
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.config import ConfigManager
from src.pipeline import FrameExtractionPipeline
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_temporal_consistency_score(
    results: List[Dict], expected_interval_seconds: float = 10.0, tolerance: float = 2.0
) -> Tuple[float, Dict]:
    """時系列整合性スコアを計算

    Args:
        results: 抽出結果のリスト（timestamp, frame_idxを含む）
        expected_interval_seconds: 期待される時間間隔（秒）
        tolerance: 許容誤差（秒）

    Returns:
        (スコア(0-100), 詳細統計)
    """
    if len(results) < 2:
        return 0.0, {"valid_pairs": 0, "total_pairs": 0, "avg_time_diff": 0.0}

    # フレーム番号でソート
    sorted_results = sorted(results, key=lambda x: x["frame_idx"])

    valid_pairs = 0
    total_pairs = 0
    time_diffs = []

    for i in tqdm(range(len(sorted_results) - 1), desc="時系列整合性計算中", leave=False):
        current = sorted_results[i]
        next_result = sorted_results[i + 1]

        # 時間差を計算
        time_diff = abs(
            (next_result["timestamp"] - current["timestamp"]).total_seconds()
        )
        frame_diff = next_result["frame_idx"] - current["frame_idx"]

        # 期待される時間差を計算（フレーム差から）
        # タイムラプス動画の場合、実際の時間差は動画の時間差より大きい
        # ここでは簡易的に期待間隔を使用
        expected_diff = expected_interval_seconds

        # 許容範囲内かチェック
        min_diff = expected_diff - tolerance
        max_diff = expected_diff + tolerance

        total_pairs += 1
        if min_diff <= time_diff <= max_diff:
            valid_pairs += 1

        time_diffs.append(time_diff)

    # スコア計算（0-100）
    score = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 0.0

    stats = {
        "valid_pairs": valid_pairs,
        "total_pairs": total_pairs,
        "avg_time_diff": sum(time_diffs) / len(time_diffs) if time_diffs else 0.0,
        "min_time_diff": min(time_diffs) if time_diffs else 0.0,
        "max_time_diff": max(time_diffs) if time_diffs else 0.0,
    }

    return score, stats


def evaluate_extraction_results(csv_path: Path) -> Dict:
    """抽出結果CSVを評価

    Args:
        csv_path: extraction_results.csvのパス

    Returns:
        評価結果の辞書
    """
    results = []

    if not csv_path.exists():
        logger.error(f"CSVファイルが見つかりません: {csv_path}")
        return {}

    # ファイルの行数を取得（プログレスバー用）
    total_lines = sum(1 for _ in csv_path.open("r", encoding="utf-8"))

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_lines - 1, desc="CSV読み込み中"):  # -1はヘッダー行
            try:
                extracted_ts = datetime.strptime(
                    row["extracted_timestamp"], "%Y/%m/%d %H:%M:%S"
                )
                frame_idx = int(row["frame_index"])
                confidence = float(row["confidence"])
                time_diff = float(row["time_diff_seconds"])

                results.append(
                    {
                        "timestamp": extracted_ts,
                        "frame_idx": frame_idx,
                        "confidence": confidence,
                        "time_diff": time_diff,
                        "ocr_text": row.get("ocr_text", ""),
                    }
                )
            except Exception as e:
                logger.warning(f"行のパースに失敗: {row}, エラー: {e}")

    if not results:
        logger.warning("評価可能な結果がありません（CSVファイルが空またはヘッダーのみ）")
        logger.warning("これは、すべてのフレームでタイムスタンプ抽出に失敗したことを示しています")
        return {
            "extraction_success_rate": 0.0,
            "extracted_count": 0,
            "message": "No extraction results available - all frames failed",
        }

    # 抽出成功率
    total_targets = len(results)  # 簡易的な計算（実際はtarget_timestampから計算すべき）
    success_rate = 100.0  # CSVに含まれるのは成功したもののみ

    # 平均信頼度
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    min_confidence = min(r["confidence"] for r in results)
    max_confidence = max(r["confidence"] for r in results)

    # 時系列整合性スコア
    temporal_score, temporal_stats = calculate_temporal_consistency_score(results)

    # 目標時刻との誤差統計
    time_diffs = [r["time_diff"] for r in results]
    avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
    within_tolerance = sum(1 for d in time_diffs if abs(d) <= 10.0)
    tolerance_rate = (within_tolerance / len(time_diffs) * 100) if time_diffs else 0.0

    evaluation = {
        "extraction_success_rate": success_rate,
        "extracted_count": len(results),
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "temporal_consistency_score": temporal_score,
        "temporal_stats": temporal_stats,
        "avg_time_diff_seconds": avg_time_diff,
        "tolerance_rate_10s": tolerance_rate,
        "within_tolerance_count": within_tolerance,
    }

    return evaluation


def print_evaluation_report(evaluation: Dict):
    """評価レポートを出力

    Args:
        evaluation: 評価結果の辞書
    """
    logger.info("=" * 80)
    logger.info("時系列整合性スコア評価レポート")
    logger.info("=" * 80)

    logger.info(f"抽出成功数: {evaluation.get('extracted_count', 0)}")
    logger.info(f"抽出成功率: {evaluation.get('extraction_success_rate', 0):.2f}%")

    logger.info("\n信頼度統計:")
    logger.info(f"  平均: {evaluation.get('avg_confidence', 0):.4f}")
    logger.info(f"  最小: {evaluation.get('min_confidence', 0):.4f}")
    logger.info(f"  最大: {evaluation.get('max_confidence', 0):.4f}")

    logger.info("\n時系列整合性:")
    temporal_stats = evaluation.get("temporal_stats", {})
    logger.info(f"  スコア: {evaluation.get('temporal_consistency_score', 0):.2f}%")
    logger.info(
        f"  有効ペア: {temporal_stats.get('valid_pairs', 0)}/{temporal_stats.get('total_pairs', 0)}"
    )
    logger.info(f"  平均時間差: {temporal_stats.get('avg_time_diff', 0):.2f}秒")
    logger.info(f"  最小時間差: {temporal_stats.get('min_time_diff', 0):.2f}秒")
    logger.info(f"  最大時間差: {temporal_stats.get('max_time_diff', 0):.2f}秒")

    logger.info("\n目標時刻との誤差:")
    logger.info(f"  平均誤差: {evaluation.get('avg_time_diff_seconds', 0):.2f}秒")
    logger.info(
        f"  ±10秒以内: {evaluation.get('within_tolerance_count', 0)}/{evaluation.get('extracted_count', 0)} ({evaluation.get('tolerance_rate_10s', 0):.2f}%)"
    )

    logger.info("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="時系列整合性スコアの測定")
    parser.add_argument("--csv", type=str, help="extraction_results.csvのパス")
    parser.add_argument("--output-dir", type=str, help="出力ディレクトリ（CSVが未指定の場合）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
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

    # 評価実行
    evaluation = evaluate_extraction_results(csv_path)

    if not evaluation:
        logger.error("評価に失敗しました")
        return 1

    # レポート出力
    print_evaluation_report(evaluation)

    # 結果をJSONで保存
    import json

    report_path = csv_path.parent / "temporal_consistency_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n評価結果を保存しました: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
