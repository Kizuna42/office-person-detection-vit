#!/usr/bin/env python3
"""
GTフレーム画像に対して検出を実行し、COCO形式で結果を出力するスクリプト。

使用方法:
    python tools/detect_for_benchmark.py \
        --input data/annotation_images \
        --output output/benchmark/detections.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GTフレーム画像に対して検出を実行",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="入力画像ディレクトリ",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="出力COCO形式JSONパス",
    )
    parser.add_argument(
        "--model",
        default="facebook/detr-resnet-50",
        help="DETRモデル名（デフォルト: facebook/detr-resnet-50）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="信頼度閾値（デフォルト: 0.5）",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="デバイス（auto/mps/cuda/cpu）",
    )
    return parser.parse_args()


def load_model(model_name: str, device: str) -> tuple[DetrForObjectDetection, DetrImageProcessor]:
    """モデルをロード"""
    logger.info("モデルをロード中: %s", model_name)
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    logger.info("モデルロード完了（デバイス: %s）", device)
    return model, processor


def detect_persons(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    image_path: Path,
    device: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """画像から人物を検出"""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 後処理
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=False):
        # COCO: person = label 1
        if label.item() == 1:
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                    "score": float(score),
                }
            )

    return detections


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        logger.error("入力ディレクトリが見つかりません: %s", args.input)
        return 1

    # 画像ファイル一覧を取得
    image_paths = sorted(args.input.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(args.input.glob("*.png"))

    if not image_paths:
        logger.error("画像ファイルが見つかりません: %s", args.input)
        return 1

    logger.info("画像ファイル数: %d", len(image_paths))

    # モデルロード
    model, processor = load_model(args.model, args.device)

    # COCO形式の出力を構築
    coco_output: dict[str, Any] = {
        "images": [],
        "categories": [{"id": 0, "name": "person"}],
        "annotations": [],
    }

    annotation_id = 0

    for image_id, image_path in enumerate(image_paths):
        logger.info("[%d/%d] 処理中: %s", image_id + 1, len(image_paths), image_path.name)

        # 画像情報
        image = Image.open(image_path)
        coco_output["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image.width,
                "height": image.height,
            }
        )

        # 検出実行
        detections = detect_persons(model, processor, image_path, args.device, args.threshold)

        for det in detections:
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,  # person
                    "bbox": det["bbox"],
                    "area": det["bbox"][2] * det["bbox"][3],
                    "score": det["score"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        logger.info("  検出数: %d", len(detections))

    # 結果を保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    logger.info("検出結果を保存しました: %s", args.output)
    logger.info("総検出数: %d", annotation_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
