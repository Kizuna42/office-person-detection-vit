"""トラッキングベンチマークCLIランナー

コマンドラインからトラッキング評価を実行するためのツール。

使用例:
    # 基本評価
    python -m src.benchmark.tracking_runner --gt output/labels/result_fixed.json --pred output/tracking.csv

    # Gold GT形式で評価
    python -m src.benchmark.tracking_runner --gt gt_tracking.json --pred tracking.csv --gt-format gold

    # 疎サンプリングモード
    python -m src.benchmark.tracking_runner --gt gt.json --pred pred.csv --sparse
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from src.evaluation.tracking_benchmark import TrackingBenchmark, TrackingMetrics


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
        description="トラッキングベンチマーク評価ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # COCO形式GTで評価
  python -m src.benchmark.tracking_runner --gt output/labels/result_fixed.json --pred output/tracking.csv

  # Gold GT形式で評価（person_id付き）
  python -m src.benchmark.tracking_runner --gt gt_tracking.json --pred tracking.csv --gt-format gold

  # 疎サンプリングモード（5分間隔）
  python -m src.benchmark.tracking_runner --gt gt.json --pred pred.csv --sparse

  # 診断ログ出力
  python -m src.benchmark.tracking_runner --gt gt.json --pred pred.csv -o output/benchmark
        """,
    )

    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Ground Truthファイルパス（COCO形式JSON または Gold形式JSON）",
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="予測結果ファイルパス（MOTChallenge形式CSV）",
    )
    parser.add_argument(
        "--gt-format",
        choices=["coco", "gold"],
        default="coco",
        help="GTファイルの形式（デフォルト: coco）",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU閾値（デフォルト: 0.5）",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="疎サンプリングモード（5分間隔評価）",
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
    parser.add_argument(
        "--category-id",
        type=int,
        default=0,
        help="GTのpersonカテゴリID（デフォルト: 0）",
    )

    return parser.parse_args()


def print_metrics_table(metrics: TrackingMetrics, sparse_mode: bool = False) -> None:
    """メトリクスをテーブル形式で出力"""
    print("\n" + "=" * 60)
    print(" トラッキング評価結果")
    print("=" * 60)

    mode = "疎サンプリング（5分間隔）" if sparse_mode else "密サンプリング（10秒間隔）"
    print(f"\n評価モード: {mode}")
    print(f"評価フレーム数: {metrics.num_frames}")
    print(f"フレーム間遷移数: {metrics.num_transitions}")

    print("\n--- MOT標準メトリクス ---")
    print(f"  MOTA:  {metrics.mota:>8.2%}")
    print(f"  IDF1:  {metrics.idf1:>8.2%}")
    print(f"  IDP:   {metrics.idp:>8.2%}")
    print(f"  IDR:   {metrics.idr:>8.2%}")
    print(f"  IDSW:  {metrics.idsw:>8d}")
    print(f"  FP:    {metrics.fp:>8d}")
    print(f"  FN:    {metrics.fn:>8d}")
    print(f"  GT:    {metrics.gt_count:>8d}")

    if sparse_mode:
        print("\n--- 疎サンプリング指標 ---")
        print(f"  IDSW/遷移: {metrics.idsw_per_transition:>8.4f}")

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
    logger.info("トラッキングベンチマークを開始")
    logger.info("  GT: %s (%s形式)", args.gt, args.gt_format)
    logger.info("  Pred: %s", args.pred)

    benchmark = TrackingBenchmark(
        iou_threshold=args.iou_threshold,
        sparse_mode=args.sparse,
        output_diagnostics=not args.no_diagnostics,
    )

    try:
        metrics = benchmark.evaluate_from_files(
            gt_path=args.gt,
            pred_path=args.pred,
            gt_format=args.gt_format,
            person_category_id=args.category_id,
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
        print_metrics_table(metrics, args.sparse)

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
            report_path = args.output / "tracking_report.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info("Markdownレポートを生成: %s", report_path)

    # サマリーログ
    logger.info("評価完了: %s", metrics.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
