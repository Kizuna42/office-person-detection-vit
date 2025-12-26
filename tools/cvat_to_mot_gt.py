#!/usr/bin/env python3
"""
CVAT XMLエクスポートからMOTChallenge形式GT CSVへの変換スクリプト。

CVATでID修正後にエクスポートしたXMLを、評価エンジンで使用可能なMOT形式に変換する。

使用方法:
    python tools/cvat_to_mot_gt.py \\
        --input output/cvat/tracks_annotated.xml \\
        --output output/ground_truth/gt_tracking.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lxml import etree
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cvat_xml_to_mot_csv(cvat_path: Path, output_path: Path) -> None:
    """CVAT XML (video 1.1) → MOTChallenge GT CSV 変換。

    出力形式: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
    (MOTChallenge標準形式、ヘッダーなし)

    Args:
        cvat_path: CVAT XMLファイルパス
        output_path: 出力CSVパス
    """
    tree = etree.parse(str(cvat_path))
    root = tree.getroot()

    rows: list[list[float | int]] = []

    for track in root.findall(".//track"):
        track_id = int(track.get("id", "0"))

        for box in track.findall("box"):
            # outside="1" はそのフレームで不在を示す
            if box.get("outside") == "1":
                continue

            frame = int(box.get("frame", "0")) + 1  # MOTは1-indexed

            xtl = float(box.get("xtl", "0"))
            ytl = float(box.get("ytl", "0"))
            xbr = float(box.get("xbr", "0"))
            ybr = float(box.get("ybr", "0"))

            width = xbr - xtl
            height = ybr - ytl

            # conf, x, y, z は未使用値 (-1)
            rows.append([frame, track_id, xtl, ytl, width, height, 1.0, -1, -1, -1])

    # フレーム順、ID順でソート
    rows.sort(key=lambda x: (x[0], x[1]))

    # DataFrame作成・保存
    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=False)

    logger.info("MOT CSV出力完了: %s", output_path)
    logger.info("総アノテーション数: %d, ユニークトラック数: %d", len(df), df["id"].nunique())


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="CVAT XMLからMOTChallenge形式CSVへ変換",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CVAT XMLファイルパス",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力MOT CSVパス",
    )
    return parser.parse_args()


def main() -> int:
    """エントリーポイント。"""
    args = parse_args()

    if not args.input.exists():
        logger.error("入力ファイルが見つかりません: %s", args.input)
        return 1

    cvat_xml_to_mot_csv(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
