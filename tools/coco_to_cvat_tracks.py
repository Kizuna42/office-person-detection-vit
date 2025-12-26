#!/usr/bin/env python3
"""
COCO Detection形式からCVAT for Video 1.1 (XML)形式への変換スクリプト。

既存の検出結果をCVATにTrackとしてインポートするため、以下の処理を行う:
1. フレーム順に画像をソート
2. フレーム間IoUマッチングで仮Track IDを割当（Hungarian Algorithm）
3. CVAT for Video 1.1 XML形式で出力

使用方法:
    python tools/coco_to_cvat_tracks.py \\
        --input output/labels/result_fixed.json \\
        --images data/annotation_images \\
        --output output/cvat/tracks_initial.xml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import TypedDict

from lxml import etree
import numpy as np
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BBoxDict(TypedDict):
    """BBox情報を保持する辞書型。"""

    ann_id: int
    bbox: list[float]  # [x, y, w, h]
    track_id: int | None


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """2つのBBox間のIoUを計算。

    Args:
        box1: [x, y, w, h] 形式
        box2: [x, y, w, h] 形式

    Returns:
        IoU値 (0.0 - 1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 座標変換 [x, y, w, h] -> [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # 交差領域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # 合計領域
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(boxes1: list[list[float]], boxes2: list[list[float]]) -> np.ndarray:
    """2つのBBoxリスト間のIoU行列を計算。

    Args:
        boxes1: 前フレームのBBoxリスト
        boxes2: 現フレームのBBoxリスト

    Returns:
        IoU行列 (shape: len(boxes1) x len(boxes2))
    """
    n, m = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((n, m), dtype=np.float64)

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = compute_iou(box1, box2)

    return iou_matrix


def extract_frame_number(file_name: str) -> int:
    """ファイル名からフレーム番号を抽出。

    Args:
        file_name: 例 "frame_000006_0h00m06s.jpg"

    Returns:
        フレーム番号 (例: 6)
    """
    match = re.search(r"frame_(\d+)", file_name)
    if match:
        return int(match.group(1))
    return 0


def assign_temporary_track_ids(
    frame_bboxes: dict[int, list[BBoxDict]],
    iou_threshold: float = 0.5,
) -> dict[int, list[BBoxDict]]:
    """フレーム間IoUマッチングで仮Track IDを割当。

    Hungarian Algorithmを使用して、連続フレーム間でIoU >= iou_threshold の
    BBoxを同一トラックとして関連付ける。

    Args:
        frame_bboxes: フレームインデックス -> BBoxリストの辞書
        iou_threshold: マッチング閾値

    Returns:
        Track IDが割当済みのframe_bboxes
    """
    frames = sorted(frame_bboxes.keys())
    if not frames:
        return frame_bboxes

    next_track_id = 1

    # 前フレームのtrack_id -> bbox マッピング
    prev_track_to_bbox: dict[int, list[float]] = {}

    for frame_idx in frames:
        current_bboxes = frame_bboxes[frame_idx]

        if not prev_track_to_bbox:
            # 初回フレーム: 全BBoxに新規Track IDを割当
            for bbox_dict in current_bboxes:
                bbox_dict["track_id"] = next_track_id
                prev_track_to_bbox[next_track_id] = bbox_dict["bbox"]
                next_track_id += 1
            continue

        # IoUマトリクス計算
        prev_track_ids = list(prev_track_to_bbox.keys())
        prev_boxes = [prev_track_to_bbox[tid] for tid in prev_track_ids]
        curr_boxes = [b["bbox"] for b in current_bboxes]

        if curr_boxes and prev_boxes:
            iou_matrix = compute_iou_matrix(prev_boxes, curr_boxes)
            # コスト行列 (最小化のため 1 - IoU)
            cost_matrix = 1.0 - iou_matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_curr_indices: set[int] = set()

            for r, c in zip(row_ind, col_ind, strict=False):
                if iou_matrix[r, c] >= iou_threshold:
                    # マッチ成功: 既存Track IDを継続
                    current_bboxes[c]["track_id"] = prev_track_ids[r]
                    matched_curr_indices.add(c)

            # 未マッチのBBoxに新規Track IDを割当
            for idx, bbox_dict in enumerate(current_bboxes):
                if idx not in matched_curr_indices:
                    bbox_dict["track_id"] = next_track_id
                    next_track_id += 1
        else:
            # 前フレームまたは現フレームが空の場合
            for bbox_dict in current_bboxes:
                bbox_dict["track_id"] = next_track_id
                next_track_id += 1

        # 次フレーム用に更新
        prev_track_to_bbox = {
            b["track_id"]: b["bbox"]  # type: ignore[misc]
            for b in current_bboxes
            if b["track_id"] is not None
        }

    return frame_bboxes


def generate_cvat_xml(
    frame_bboxes: dict[int, list[BBoxDict]],
    images: list[dict],
    output_path: Path,
) -> None:
    """CVAT for Video 1.1 XML形式で出力。

    Args:
        frame_bboxes: Track ID割当済みのフレーム→BBox辞書
        images: COCO形式の画像情報リスト
        output_path: 出力XMLパス
    """
    # Track ID -> フレームごとのBBox情報を集約
    tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)

    for frame_idx, bboxes in frame_bboxes.items():
        for bbox_dict in bboxes:
            track_id = bbox_dict.get("track_id")
            if track_id is not None:
                tracks[track_id].append((frame_idx, bbox_dict["bbox"]))

    # XML構築
    root = etree.Element("annotations")

    # version
    version_elem = etree.SubElement(root, "version")
    version_elem.text = "1.1"

    # meta
    meta = etree.SubElement(root, "meta")
    task = etree.SubElement(meta, "task")

    name = etree.SubElement(task, "name")
    name.text = "MOT Annotation Task"

    size = etree.SubElement(task, "size")
    size.text = str(len(images))

    mode = etree.SubElement(task, "mode")
    mode.text = "annotation"

    labels = etree.SubElement(task, "labels")
    label = etree.SubElement(labels, "label")
    label_name = etree.SubElement(label, "name")
    label_name.text = "person"

    # tracks
    for track_id in sorted(tracks.keys()):
        track_data = tracks[track_id]
        track_elem = etree.SubElement(
            root,
            "track",
            id=str(track_id),
            label="person",
        )

        for frame_idx, bbox in sorted(track_data, key=lambda x: x[0]):
            x, y, w, h = bbox
            xtl, ytl = x, y
            xbr, ybr = x + w, y + h

            etree.SubElement(
                track_elem,
                "box",
                frame=str(frame_idx),
                keyframe="1",
                xtl=f"{xtl:.2f}",
                ytl=f"{ytl:.2f}",
                xbr=f"{xbr:.2f}",
                ybr=f"{ybr:.2f}",
                outside="0",
                occluded="0",
            )

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(root)
    tree.write(
        str(output_path),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )

    logger.info("CVAT XML出力完了: %s", output_path)
    logger.info("トラック数: %d, フレーム数: %d", len(tracks), len(images))


def coco_to_cvat_xml(
    coco_path: Path,
    images_dir: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
) -> None:
    """メイン変換処理。

    Args:
        coco_path: COCO Detection JSON パス
        images_dir: 画像ディレクトリ（ファイル存在確認用）
        output_path: 出力CVAT XMLパス
        iou_threshold: Track割当のIoU閾値
    """
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    # フレーム順ソート（ファイル名からフレーム番号抽出）
    images = sorted(
        coco.get("images", []),
        key=lambda x: extract_frame_number(x.get("file_name", "")),
    )

    # image_id -> frame_index マッピング
    image_id_to_frame: dict[int, int] = {}
    for idx, img in enumerate(images):
        image_id_to_frame[img["id"]] = idx

    # フレームごとのBBox集約
    frame_bboxes: dict[int, list[BBoxDict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in image_id_to_frame:
            continue

        frame_idx = image_id_to_frame[image_id]
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        frame_bboxes[frame_idx].append(
            {
                "ann_id": ann.get("id", -1),
                "bbox": [float(b) for b in bbox],
                "track_id": None,
            }
        )

    # IoUベースの仮ID割当
    logger.info("Track ID割当開始 (IoU閾値: %.2f)", iou_threshold)
    frame_bboxes = assign_temporary_track_ids(frame_bboxes, iou_threshold)

    # CVAT XML生成
    generate_cvat_xml(frame_bboxes, images, output_path)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="COCO Detection形式からCVAT for Video 1.1 XMLへ変換",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="COCO Detection JSON パス",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="画像ディレクトリ パス",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力CVAT XMLパス",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Track割当のIoU閾値 (default: 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    """エントリーポイント。"""
    args = parse_args()

    if not args.input.exists():
        logger.error("入力ファイルが見つかりません: %s", args.input)
        return 1

    if not args.images.is_dir():
        logger.error("画像ディレクトリが見つかりません: %s", args.images)
        return 1

    coco_to_cvat_xml(
        coco_path=args.input,
        images_dir=args.images,
        output_path=args.output,
        iou_threshold=args.iou_threshold,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
