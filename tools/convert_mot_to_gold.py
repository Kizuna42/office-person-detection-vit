#!/usr/bin/env python3
"""
MOT形式CSV（CVATからエクスポート）をGold GT形式JSONに変換するスクリプト。

使用方法:
    python tools/convert_mot_to_gold.py \
        --input output/ground_truth/gt_tracking_fixed.csv \
        --output output/ground_truth/gt_tracking.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOT形式CSVをGold GT JSONに変換",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="MOT形式CSVパス（CVATエクスポート）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力Gold GT JSONパス",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        logger.error("入力ファイルが見つかりません: %s", args.input)
        return 1

    # MOT CSVを読み込み
    # フォーマット: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
    try:
        df = pd.read_csv(
            args.input,
            header=None,
            names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
        )
    except Exception as e:
        logger.error("CSVの読み込みに失敗しました: %s", e)
        return 1

    # フレームごとにグループ化
    frames = []
    for frame_num, group in df.groupby("frame"):
        annotations = []
        for _, row in group.iterrows():
            annotations.append(
                {
                    "person_id": int(row["id"]),
                    "bbox": [
                        float(row["bb_left"]),
                        float(row["bb_top"]),
                        float(row["bb_width"]),
                        float(row["bb_height"]),
                    ],
                    "confidence": float(row["conf"]) if row["conf"] > 0 else 1.0,
                }
            )

        frames.append(
            {
                "frame_idx": int(frame_num) - 1,  # MOTは1-indexed、Gold GTは0-indexed
                "annotations": annotations,
            }
        )

    # ソート
    frames.sort(key=lambda f: f["frame_idx"])

    gold_data = {
        "version": "1.0",
        "description": "Converted from MOT CSV (CVAT export)",
        "frames": frames,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)

    logger.info("Gold GT JSONを出力しました: %s", args.output)
    logger.info("フレーム数: %d, 総アノテーション数: %d", len(frames), len(df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
