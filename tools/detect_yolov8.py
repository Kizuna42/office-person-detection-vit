#!/usr/bin/env python3
"""YOLOv8による検出を実行し、COCO形式で出力するスクリプト.

ベンチマーク評価用にDETRと同じ形式で出力する。
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_with_yolo(
    model_path: str,
    images_dir: Path,
    output_path: Path,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> None:
    """YOLOv8で検出を実行しCOCO形式で出力.

    Args:
        model_path: モデルパス（.pt）
        images_dir: 入力画像ディレクトリ
        output_path: 出力JSONパス
        conf_threshold: 信頼度閾値
        iou_threshold: NMS IoU閾値
    """
    logger.info(f"モデルをロード: {model_path}")
    model = YOLO(model_path)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    logger.info(f"画像数: {len(image_paths)}")

    coco_output = {
        "images": [],
        "categories": [{"id": 0, "name": "person"}],
        "annotations": [],
    }

    annotation_id = 0

    for image_id, image_path in enumerate(image_paths):
        logger.info(f"[{image_id + 1}/{len(image_paths)}] {image_path.name}")

        # 画像情報
        img = Image.open(image_path)
        coco_output["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": img.width,
                "height": img.height,
            }
        )

        # 検出実行
        results = model(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[0],  # person only
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # xyxy形式からxywh形式に変換
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])

                coco_output["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "score": conf,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        logger.info(f"  検出数: {len(boxes) if boxes is not None else 0}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    logger.info(f"結果を保存: {output_path}")
    logger.info(f"総検出数: {annotation_id}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 検出")
    parser.add_argument("--model", default="yolov8x.pt", help="モデルパス")
    parser.add_argument("--input", required=True, type=Path, help="入力画像ディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="出力JSONパス")
    parser.add_argument("--conf", type=float, default=0.5, help="信頼度閾値")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU閾値")
    args = parser.parse_args()

    detect_with_yolo(
        model_path=args.model,
        images_dir=args.input,
        output_path=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )


if __name__ == "__main__":
    main()
