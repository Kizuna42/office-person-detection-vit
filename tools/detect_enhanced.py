#!/usr/bin/env python3
"""
改良版検出スクリプト - マルチスケール推論とSoft-NMS対応.

機能:
- マルチスケール推論（複数解像度で検出し結果を統合）
- Soft-NMS（重複抑制しつつ近接物体を保持）
- 画像端パディング（端部の検出漏れ対策）
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMSで重複検出を抑制しつつ近接物体を保持.

    Args:
        boxes: 形状 (N, 4) の配列 [x, y, w, h]
        scores: 形状 (N,) のスコア配列
        iou_threshold: ハードNMS適用のIoU閾値
        sigma: ガウシアン減衰のパラメータ
        score_threshold: 最終的なスコア閾値

    Returns:
        フィルタ後のボックスとスコア
    """
    if len(boxes) == 0:
        return boxes, scores

    # x,y,w,h → x1,y1,x2,y2
    boxes_xyxy = boxes.copy().astype(np.float64)
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]

    scores = scores.copy().astype(np.float64)
    indices = np.arange(len(scores))

    keep = []
    while len(indices) > 0:
        # 最大スコアのインデックス
        max_idx = np.argmax(scores[indices])
        max_pos = indices[max_idx]
        keep.append(max_pos)

        # 残りのボックスとのIoUを計算
        other_indices = np.delete(indices, max_idx)
        if len(other_indices) == 0:
            break

        max_box = boxes_xyxy[max_pos]
        other_boxes = boxes_xyxy[other_indices]

        # IoU計算
        xx1 = np.maximum(max_box[0], other_boxes[:, 0])
        yy1 = np.maximum(max_box[1], other_boxes[:, 1])
        xx2 = np.minimum(max_box[2], other_boxes[:, 2])
        yy2 = np.minimum(max_box[3], other_boxes[:, 3])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        area_max = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = area_max + area_others - inter_area

        iou = inter_area / np.maximum(union_area, 1e-6)

        # Soft-NMS: ガウシアン減衰
        decay = np.exp(-(iou**2) / sigma)
        scores[other_indices] *= decay

        # 閾値以上のみ残す
        valid_mask = scores[other_indices] >= score_threshold
        indices = other_indices[valid_mask]

    keep = np.array(keep)
    return boxes[keep], scores[keep]


def pad_image(image: np.ndarray, pad_ratio: float = 0.1) -> tuple[np.ndarray, tuple[int, int]]:
    """画像端をパディングして端部検出を改善.

    Args:
        image: 入力画像 (H, W, C)
        pad_ratio: パディング比率

    Returns:
        パディング画像と(pad_h, pad_w)
    """
    h, w = image.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w)


def adjust_boxes_for_padding(boxes: list[dict], pad_h: int, pad_w: int) -> list[dict]:
    """パディング分を差し引いて座標を補正."""
    adjusted = []
    for box in boxes:
        x, y, w, h = box["bbox"]
        adjusted.append(
            {
                "bbox": [x - pad_w, y - pad_h, w, h],
                "score": box["score"],
            }
        )
    return adjusted


def adjust_boxes_for_scale(boxes: list[dict], scale: float) -> list[dict]:
    """スケール分を補正."""
    adjusted = []
    for box in boxes:
        x, y, w, h = box["bbox"]
        adjusted.append(
            {
                "bbox": [x / scale, y / scale, w / scale, h / scale],
                "score": box["score"],
            }
        )
    return adjusted


class EnhancedDetector:
    """マルチスケール推論とSoft-NMS対応の検出器."""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: str = "cpu",
        threshold: float = 0.5,
        enable_multiscale: bool = False,
        enable_padding: bool = False,
        scales: list[float] | None = None,
    ):
        self.threshold = threshold
        self.enable_multiscale = enable_multiscale
        self.enable_padding = enable_padding
        self.scales = scales or [0.8, 1.0, 1.2]
        self.device = device

        logger.info("モデルをロード中: %s", model_name)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info("モデルロード完了（デバイス: %s）", device)

    def detect_single(self, image: Image.Image, threshold: float | None = None) -> list[dict[str, Any]]:
        """単一スケールで検出."""
        threshold = threshold or self.threshold

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
            0
        ]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=False):
            if label.item() == 1:  # person
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
        return detections

    def detect_multiscale(self, image_np: np.ndarray, threshold: float | None = None) -> list[dict[str, Any]]:
        """マルチスケール推論で検出."""
        all_detections = []

        for scale in self.scales:
            if scale == 1.0:
                scaled_np = image_np
            else:
                h, w = image_np.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_np = cv2.resize(image_np, (new_w, new_h))

            scaled_pil = Image.fromarray(cv2.cvtColor(scaled_np, cv2.COLOR_BGR2RGB))
            dets = self.detect_single(scaled_pil, threshold=threshold or self.threshold * 0.8)

            # スケール補正
            adjusted = adjust_boxes_for_scale(dets, scale)
            all_detections.extend(adjusted)

        # Soft-NMSで統合
        if all_detections:
            boxes = np.array([d["bbox"] for d in all_detections])
            scores = np.array([d["score"] for d in all_detections])
            boxes, scores = soft_nms(boxes, scores, score_threshold=self.threshold)
            all_detections = [{"bbox": boxes[i].tolist(), "score": float(scores[i])} for i in range(len(boxes))]

        return all_detections

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        """画像パスから検出."""
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            logger.warning("画像を読めません: %s", image_path)
            return []

        # パディング
        pad_h, pad_w = 0, 0
        if self.enable_padding:
            image_np, (pad_h, pad_w) = pad_image(image_np, pad_ratio=0.05)

        # マルチスケール or シングル
        if self.enable_multiscale:
            detections = self.detect_multiscale(image_np)
        else:
            image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            detections = self.detect_single(image_pil)

        # パディング補正
        if self.enable_padding:
            detections = adjust_boxes_for_padding(detections, pad_h, pad_w)

        # 画像範囲外を除去
        h, w = cv2.imread(str(image_path)).shape[:2]
        valid_detections = []
        for det in detections:
            x, y, bw, bh = det["bbox"]
            if x + bw > 0 and y + bh > 0 and x < w and y < h:
                det["bbox"] = [max(0, x), max(0, y), min(bw, w - x), min(bh, h - y)]
                valid_detections.append(det)

        return valid_detections


def main() -> int:
    parser = argparse.ArgumentParser(description="改良版検出スクリプト")
    parser.add_argument("--input", required=True, type=Path, help="入力画像ディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="出力JSONパス")
    parser.add_argument("--model", default="facebook/detr-resnet-50", help="モデル名")
    parser.add_argument("--threshold", type=float, default=0.7, help="信頼度閾値")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--multiscale", action="store_true", help="マルチスケール推論を有効化")
    parser.add_argument("--padding", action="store_true", help="画像端パディングを有効化")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("入力ディレクトリが見つかりません: %s", args.input)
        return 1

    image_paths = sorted(list(args.input.glob("*.jpg")) + list(args.input.glob("*.png")))
    if not image_paths:
        logger.error("画像ファイルが見つかりません: %s", args.input)
        return 1

    logger.info("画像ファイル数: %d", len(image_paths))
    logger.info("オプション: multiscale=%s, padding=%s, threshold=%.2f", args.multiscale, args.padding, args.threshold)

    detector = EnhancedDetector(
        model_name=args.model,
        device=args.device,
        threshold=args.threshold,
        enable_multiscale=args.multiscale,
        enable_padding=args.padding,
    )

    coco_output: dict[str, Any] = {
        "images": [],
        "categories": [{"id": 0, "name": "person"}],
        "annotations": [],
    }

    annotation_id = 0

    for image_id, image_path in enumerate(image_paths):
        logger.info("[%d/%d] 処理中: %s", image_id + 1, len(image_paths), image_path.name)

        image = Image.open(image_path)
        coco_output["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image.width,
                "height": image.height,
            }
        )

        detections = detector.detect(image_path)

        for det in detections:
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": det["bbox"],
                    "area": det["bbox"][2] * det["bbox"][3],
                    "score": det["score"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        logger.info("  検出数: %d", len(detections))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    logger.info("検出結果を保存しました: %s", args.output)
    logger.info("総検出数: %d", annotation_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
