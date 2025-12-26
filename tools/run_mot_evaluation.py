#!/usr/bin/env python3
"""
MOT評価パイプライン実行スクリプト。

MOTA/IDF1/HOTA等の指標算出、IDスイッチ箇所の可視化を行う。

使用例:
    python tools/run_mot_evaluation.py \\
        --gt output/ground_truth/gt_tracking.csv \\
        --pred output/sessions/latest/03_tracking/tracks_mot.csv \\
        --output output/evaluation/
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.mot_metrics import MOTMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_mot_csv(path: Path) -> pd.DataFrame:
    """MOTChallenge形式CSVを読み込み。

    ヘッダーなし形式とヘッダーあり形式の両方に対応。

    Args:
        path: CSVファイルパス

    Returns:
        標準化されたDataFrame (FrameId, Id, X, Y, Width, Height, Confidence)
    """
    # まずヘッダーありで試行
    try:
        df = pd.read_csv(path)
        if "FrameId" in df.columns or "frame" in df.columns:
            # ヘッダーあり形式
            column_map = {
                "frame": "FrameId",
                "id": "Id",
                "bb_left": "X",
                "bb_top": "Y",
                "bb_width": "Width",
                "bb_height": "Height",
                "conf": "Confidence",
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            if "Confidence" not in df.columns:
                df["Confidence"] = 1.0

            return df[["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"]]
    except Exception:
        pass

    # ヘッダーなし形式
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "FrameId",
            "Id",
            "X",
            "Y",
            "Width",
            "Height",
            "Confidence",
            "_x",
            "_y",
            "_z",
        ],
    )
    return df[["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"]]


def print_metrics_table(metrics: dict[str, float]) -> None:
    """メトリクスをテーブル形式で出力。"""
    print("\n" + "=" * 50)
    print("MOT Evaluation Results")
    print("=" * 50)

    # 主要指標
    print(f"\n{'Metric':<15} {'Value':>15}")
    print("-" * 30)
    print(f"{'MOTA':<15} {metrics.get('MOTA', 0.0):>14.2%}")
    print(f"{'IDF1':<15} {metrics.get('IDF1', 0.0):>14.2%}")
    print(f"{'IDP':<15} {metrics.get('IDP', 0.0):>14.2%}")
    print(f"{'IDR':<15} {metrics.get('IDR', 0.0):>14.2%}")

    if "HOTA" in metrics:
        print(f"{'HOTA':<15} {metrics.get('HOTA', 0.0):>14.2%}")
        print(f"{'DetA':<15} {metrics.get('DetA', 0.0):>14.2%}")
        print(f"{'AssA':<15} {metrics.get('AssA', 0.0):>14.2%}")

    # エラー指標
    print(f"\n{'Errors':<15}")
    print("-" * 30)
    print(f"{'ID Switches':<15} {int(metrics.get('IDSW', 0)):>15}")
    print(f"{'False Positives':<15} {int(metrics.get('FP', 0)):>15}")
    print(f"{'False Negatives':<15} {int(metrics.get('FN', 0)):>15}")
    print(f"{'GT Objects':<15} {int(metrics.get('GT', 0)):>15}")

    print("=" * 50 + "\n")


def save_json_report(metrics: dict[str, float], output_path: Path) -> None:
    """メトリクスをJSONレポートとして保存。"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("評価レポート保存: %s", output_path)


def generate_id_switch_report(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """IDスイッチ発生箇所の簡易レポートを生成。

    Note: 詳細なフレーム単位の分析はMOTAccumulatorのイベントログから抽出可能。
          ここでは、フレームごとのID分布の変化を可視化する簡易版を実装。
    """
    # フレームごとのユニークID数を集計
    gt_frame_ids = gt_df.groupby("FrameId")["Id"].apply(set).to_dict()
    pred_frame_ids = pred_df.groupby("FrameId")["Id"].apply(set).to_dict()

    frames = sorted(set(gt_frame_ids.keys()) | set(pred_frame_ids.keys()))

    report_lines = [
        "# ID Switch Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Frame-by-Frame ID Distribution",
        "",
        "| Frame | GT IDs | Pred IDs | GT Count | Pred Count |",
        "|-------|--------|----------|----------|------------|",
    ]

    for frame in frames[:50]:  # 最初の50フレームのみ表示
        gt_ids = gt_frame_ids.get(frame, set())
        pred_ids = pred_frame_ids.get(frame, set())
        gt_str = ",".join(map(str, sorted(gt_ids)))[:20] or "-"
        pred_str = ",".join(map(str, sorted(pred_ids)))[:20] or "-"
        report_lines.append(f"| {frame} | {gt_str} | {pred_str} | {len(gt_ids)} | {len(pred_ids)} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("IDスイッチレポート保存: %s", output_path)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="MOT評価パイプライン実行",
    )
    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Ground Truth CSVパス (MOTChallenge形式)",
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="予測結果CSVパス (MOTChallenge形式)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU閾値 (default: 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    """エントリーポイント。"""
    args = parse_args()

    if not args.gt.exists():
        logger.error("GTファイルが見つかりません: %s", args.gt)
        return 1

    if not args.pred.exists():
        logger.error("予測ファイルが見つかりません: %s", args.pred)
        return 1

    # データ読込
    logger.info("GT読込: %s", args.gt)
    gt_df = load_mot_csv(args.gt)
    logger.info("GT: %d annotations, %d frames", len(gt_df), gt_df["FrameId"].nunique())

    logger.info("Pred読込: %s", args.pred)
    pred_df = load_mot_csv(args.pred)
    logger.info("Pred: %d annotations, %d frames", len(pred_df), pred_df["FrameId"].nunique())

    # 評価実行
    metrics = MOTMetrics(iou_threshold=args.iou_threshold)
    results = metrics.evaluate_from_dataframes(gt_df, pred_df)

    # 結果出力
    print_metrics_table(results)

    output_dir = args.output
    save_json_report(results, output_dir / "metrics.json")
    generate_id_switch_report(gt_df, pred_df, output_dir / "id_switch_report.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
