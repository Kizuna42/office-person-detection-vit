#!/usr/bin/env python3
"""検出ベンチマークCLIランナー

使用例:
    python -m src.benchmark.detection_runner --gt output/labels/result_fixed.json --pred output/latest/02_detection/detections.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from src.evaluation.detection_benchmark import DetectionBenchmark, DetectionMetrics


def setup_logging(debug: bool = False) -> logging.Logger:
    """ロギング設定"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="検出ベンチマーク評価ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # COCO形式GTで評価
  python -m src.benchmark.detection_runner --gt output/labels/result_fixed.json --pred detections.json

  # 信頼度閾値指定
  python -m src.benchmark.detection_runner --gt gt.json --pred pred.json --conf-threshold 0.5

  # レポート出力
  python -m src.benchmark.detection_runner --gt gt.json --pred pred.json -o output/benchmark --report
        """,
    )

    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Ground Truthファイルパス（COCO形式JSON）",
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="予測結果ファイルパス（COCO形式JSON or coordinate_transformations.json）",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU閾値（デフォルト: 0.5）",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="信頼度閾値（デフォルト: 0.0）",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=0,
        help="personカテゴリID（デフォルト: 0）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="評価結果の出力ディレクトリ",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="診断ログ出力を無効化",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Markdownレポートを生成",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果をJSON形式で標準出力に出力",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード",
    )

    return parser.parse_args()


def print_metrics_table(metrics: DetectionMetrics) -> None:
    """メトリクスをテーブル形式で出力"""
    print("\n" + "=" * 60)
    print(" 検出評価結果")
    print("=" * 60)

    print(f"\n評価画像数: {metrics.num_images}")
    print(f"GT総数: {metrics.gt_count}")
    print(f"予測総数: {metrics.pred_count}")

    print("\n--- 基本メトリクス ---")
    print(f"  Precision:  {metrics.precision:>8.2%}")
    print(f"  Recall:     {metrics.recall:>8.2%}")
    print(f"  F1-Score:   {metrics.f1_score:>8.2%}")

    print("\n--- AP/mAP ---")
    print(f"  AP@50:      {metrics.ap_50:>8.2%}")
    print(f"  AP@75:      {metrics.ap_75:>8.2%}")
    print(f"  mAP:        {metrics.ap:>8.2%}")

    print("\n--- 詳細カウント ---")
    print(f"  TP:         {metrics.true_positives:>8d}")
    print(f"  FP:         {metrics.false_positives:>8d}")
    print(f"  FN:         {metrics.false_negatives:>8d}")

    print("\n" + "=" * 60)


def main() -> int:
    """メインエントリーポイント"""
    args = parse_args()
    logger = setup_logging(args.debug)

    # ファイル存在チェック
    if not args.gt.exists():
        logger.error("GTファイルが見つかりません: %s", args.gt)
        return 1

    if not args.pred.exists():
        logger.error("予測ファイルが見つかりません: %s", args.pred)
        return 1

    # ベンチマーク実行
    logger.info("検出ベンチマークを開始")
    logger.info("  GT: %s", args.gt)
    logger.info("  Pred: %s", args.pred)

    benchmark = DetectionBenchmark(
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.conf_threshold,
        person_category_id=args.category_id,
        output_diagnostics=not args.no_diagnostics,
    )

    try:
        metrics = benchmark.evaluate_from_files(
            gt_path=args.gt,
            pred_path=args.pred,
        )
    except Exception as e:
        logger.error("評価中にエラーが発生しました: %s", e)
        if args.debug:
            raise
        return 1

    # 結果出力
    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_metrics_table(metrics)

    # ファイル出力
    if args.output:
        paths = benchmark.export_results(
            metrics=metrics,
            output_dir=args.output,
            include_diagnostics=not args.no_diagnostics,
        )
        logger.info("結果を出力しました:")
        for name, path in paths.items():
            logger.info("  %s: %s", name, path)

        # レポート生成
        if args.report:
            report = benchmark.generate_report(metrics)
            report_path = args.output / "detection_report.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info("Markdownレポートを生成: %s", report_path)

    # サマリーログ
    logger.info("評価完了: %s", metrics.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
