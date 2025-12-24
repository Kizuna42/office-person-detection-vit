#!/usr/bin/env python3
"""COCOアノテーションをYOLO形式に変換するスクリプト.

YOLOv8のFine-tuning用にデータセットを準備する。
"""

import argparse
import json
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def coco_to_yolo_bbox(bbox: list, img_width: int, img_height: int) -> tuple:
    """COCO形式[x, y, w, h]をYOLO形式[x_center, y_center, w, h]（正規化）に変換."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def convert_annotations(
    coco_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
) -> None:
    """COCOアノテーションをYOLO形式に変換.

    Args:
        coco_json_path: COCO形式JSONのパス
        images_dir: 画像ディレクトリ
        output_dir: 出力ディレクトリ（YOLO形式）
        val_ratio: 検証データの割合
    """
    logger.info(f"COCO JSONを読み込み: {coco_json_path}")
    with open(coco_json_path, encoding="utf-8") as f:
        coco_data = json.load(f)

    # 出力ディレクトリ構造を作成
    train_images_dir = output_dir / "images" / "train"
    val_images_dir = output_dir / "images" / "val"
    train_labels_dir = output_dir / "labels" / "train"
    val_labels_dir = output_dir / "labels" / "val"

    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 画像IDからファイル名/サイズへのマッピング
    images_info = {}
    for img in coco_data.get("images", []):
        images_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # 画像IDごとにアノテーションを集約
    annotations_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # train/val分割
    image_ids = list(images_info.keys())
    val_count = max(1, int(len(image_ids) * val_ratio))
    val_ids = set(image_ids[-val_count:])
    train_ids = set(image_ids[:-val_count])

    logger.info(f"Train: {len(train_ids)}画像, Val: {len(val_ids)}画像")

    # 変換処理
    for img_id, info in images_info.items():
        file_name = info["file_name"]
        width = info["width"]
        height = info["height"]

        # 画像ファイルのパス
        src_image = images_dir / file_name
        if not src_image.exists():
            logger.warning(f"画像が見つかりません: {src_image}")
            continue

        # train/val振り分け
        if img_id in val_ids:
            dst_image = val_images_dir / file_name
            dst_label = val_labels_dir / (Path(file_name).stem + ".txt")
        else:
            dst_image = train_images_dir / file_name
            dst_label = train_labels_dir / (Path(file_name).stem + ".txt")

        # 画像をコピー
        shutil.copy2(src_image, dst_image)

        # ラベルファイルを作成
        annotations = annotations_by_image.get(img_id, [])
        with open(dst_label, "w", encoding="utf-8") as f:
            for ann in annotations:
                bbox = ann["bbox"]
                # YOLO形式: <class_id> <x_center> <y_center> <width> <height>
                x_c, y_c, w, h = coco_to_yolo_bbox(bbox, width, height)
                # class_id = 0 (person)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # data.yaml作成
    yaml_content = f"""# YOLOv8 Person Detection Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: person

# Number of classes
nc: 1
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info(f"data.yaml作成: {yaml_path}")
    logger.info("変換完了!")


def main():
    parser = argparse.ArgumentParser(description="COCO to YOLO形式変換")
    parser.add_argument("--coco", required=True, type=Path, help="COCO JSON path")
    parser.add_argument("--images", required=True, type=Path, help="画像ディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="検証データ割合")
    args = parser.parse_args()

    convert_annotations(args.coco, args.images, args.output, args.val_ratio)


if __name__ == "__main__":
    main()
