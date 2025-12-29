#!/usr/bin/env python3
"""
検出・トラッキング結果（coordinate_transformations.json）を
Gold GT形式（gt_tracking.json）に変換するスクリプト。

このスクリプトは、現在のパイプライン出力を初期GTとして利用し、
手作業でのアノテーション修正を効率化するために使用します。

使用方法:
    python tools/convert_to_gold_gt.py \
        --input output/latest/04_transform/coordinate_transformations.json \
        --output output/ground_truth/gt_tracking.json \
        --mot-output output/ground_truth/gt_tracking.csv

    # MOT形式出力オプション(--mot-output)を指定すると、CVAT等のツールで
    # インポート可能なMOT Challenge形式CSVも同時に生成します。
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="検出結果をGold GT形式に変換",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="入力JSONパス（coordinate_transformations.json）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力GT JSONパス",
    )
    parser.add_argument(
        "--mot-output",
        type=Path,
        help="MOT Challenge形式CSVの出力パス（CVAT等での修正用）",
    )
    return parser.parse_args()


def load_transform_json(path: Path) -> dict[str, Any]:
    """coordinate_transformations.jsonを読み込む"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def convert_to_gold(data: dict[str, Any]) -> dict[str, Any]:
    """データをGold GT形式に変換"""
    frames = []

    # data["frames"] はリスト
    input_frames = data.get("frames", [])

    for frame_data in input_frames:
        frame_idx = frame_data.get("idx")
        timestamp = frame_data.get("ts")
        detections = frame_data.get("det", [])

        annotations = []
        for det in detections:
            # coordinate_transformations.jsonのデテクション情報
            # bb: [x, y, w, h]
            # id: tracking id (存在しない場合もある)
            # zones: zone list

            bbox = det.get("bb")
            person_id = det.get("id")

            # IDがあり、bboxがある場合のみ追加
            if person_id is not None and bbox is not None:
                ann = {
                    "person_id": int(person_id),
                    "bbox": bbox,
                }

                # オプション情報
                if "zones" in det:
                    ann["zone_ids"] = det["zones"]
                if "conf" in det:
                    ann["confidence"] = det["conf"]

                annotations.append(ann)

        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "annotations": annotations,
            }
        )

    return {
        "version": "1.0",
        "description": "Generated from pipeline output",
        "frames": frames,
    }


def export_mot_csv(gold_data: dict[str, Any], output_path: Path) -> None:
    """GoldデータをMOT Challenge形式CSVとして出力"""
    rows = []

    for frame in gold_data["frames"]:
        frame_idx = frame["frame_idx"] + 1  # MOTは1-indexed

        for ann in frame["annotations"]:
            # frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            x, y, w, h = ann["bbox"]
            pid = ann["person_id"]
            conf = ann.get("confidence", 1.0)

            rows.append(
                {
                    "frame": frame_idx,
                    "id": pid,
                    "bb_left": x,
                    "bb_top": y,
                    "bb_width": w,
                    "bb_height": h,
                    "conf": conf,
                    "x": -1,
                    "y": -1,
                    "z": -1,
                }
            )

    if not rows:
        logger.warning("出力するデータがありません")
        return

    df = pd.DataFrame(rows)
    # カラム順序指定
    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=False, columns=cols)
    logger.info("MOT形式CSVを出力しました: %s", output_path)
    logger.info("このファイルはCVATなどのアノテーションツールでインポート可能です（Format: MOT 1.1）")


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        logger.error("入力ファイルが見つかりません: %s", args.input)
        return 1

    try:
        if args.output.exists():
            logger.warning("出力ファイルは上書きされます: %s", args.output)

        data = load_transform_json(args.input)
        gold_data = convert_to_gold(data)

        # Gold JSON出力
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(gold_data, f, indent=2, ensure_ascii=False)
        logger.info("Gold GT形式のJSONを出力しました: %s", args.output)

        # MOT CSV出力（オプション）
        if args.mot_output:
            export_mot_csv(gold_data, args.mot_output)

    except Exception as e:
        logger.exception("変換中にエラーが発生しました: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
