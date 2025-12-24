#!/usr/bin/env python3
"""信頼度閾値の最適化分析ツール.

検出結果に対して様々な信頼度閾値を適用し、
Precision-Recallカーブを描画して最適閾値を探索する。
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    """JSONファイルを読み込む."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_iou(box1: list, box2: list) -> float:
    """2つのボックス間のIoUを計算. 形式: [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 交差領域
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_at_threshold(
    gt_data: dict,
    pred_data: dict,
    conf_threshold: float,
    iou_threshold: float = 0.5,
) -> dict:
    """指定閾値でのPrecision/Recallを計算."""
    # 画像名→IDマッピング
    gt_images = {img["file_name"]: img["id"] for img in gt_data.get("images", [])}
    pred_images = {img["file_name"]: img["id"] for img in pred_data.get("images", [])}

    # 共通画像を取得
    common_images = set(gt_images.keys()) & set(pred_images.keys())

    # GTアノテーションを画像IDでグループ化
    gt_by_image = {}
    for ann in gt_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # 予測を画像IDでグループ化
    pred_by_image = {}
    for ann in pred_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(ann)

    tp, fp, fn = 0, 0, 0

    for img_name in common_images:
        gt_img_id = gt_images[img_name]
        pred_img_id = pred_images[img_name]

        gt_boxes = gt_by_image.get(gt_img_id, [])
        pred_boxes = [p for p in pred_by_image.get(pred_img_id, []) if p.get("score", 1.0) >= conf_threshold]

        # マッチング
        matched_gt = set()
        for pred in sorted(pred_boxes, key=lambda x: -x.get("score", 1.0)):
            best_iou = 0.0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": conf_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="信頼度閾値最適化分析")
    parser.add_argument("--gt", required=True, help="GTラベルJSONパス")
    parser.add_argument("--pred", required=True, help="予測結果JSONパス")
    parser.add_argument("--output", "-o", default="output/threshold_analysis.png", help="出力グラフパス")
    parser.add_argument("--min-threshold", type=float, default=0.3, help="最小閾値")
    parser.add_argument("--max-threshold", type=float, default=0.9, help="最大閾値")
    parser.add_argument("--step", type=float, default=0.05, help="閾値ステップ")
    args = parser.parse_args()

    logger.info("データ読み込み中...")
    gt_data = load_json(args.gt)
    pred_data = load_json(args.pred)

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    results = []

    logger.info(f"閾値範囲: {args.min_threshold:.2f} - {args.max_threshold:.2f}")
    for thresh in thresholds:
        result = evaluate_at_threshold(gt_data, pred_data, thresh)
        results.append(result)
        logger.info(
            f"  thresh={thresh:.2f}: P={result['precision']:.3f}, " f"R={result['recall']:.3f}, F1={result['f1']:.3f}"
        )

    # 最適閾値を探索
    best_f1 = max(results, key=lambda x: x["f1"])
    logger.info(
        f"\n最適閾値 (F1最大): {best_f1['threshold']:.2f} "
        f"(P={best_f1['precision']:.3f}, R={best_f1['recall']:.3f}, F1={best_f1['f1']:.3f})"
    )

    # グラフ描画
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Precision-Recall vs Threshold
    threshs = [r["threshold"] for r in results]
    ax1.plot(threshs, [r["precision"] for r in results], "b-o", label="Precision", markersize=4)
    ax1.plot(threshs, [r["recall"] for r in results], "r-o", label="Recall", markersize=4)
    ax1.plot(threshs, [r["f1"] for r in results], "g-o", label="F1-Score", markersize=4)
    ax1.axvline(x=best_f1["threshold"], color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Metrics vs Confidence Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(args.min_threshold, args.max_threshold)
    ax1.set_ylim(0, 1)

    # Precision-Recall Curve
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    ax2.plot(recalls, precisions, "b-o", markersize=4)
    ax2.scatter([best_f1["recall"]], [best_f1["precision"]], color="red", s=100, zorder=5)
    ax2.annotate(
        f'Best F1\nthresh={best_f1["threshold"]:.2f}',
        (best_f1["recall"], best_f1["precision"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    logger.info(f"グラフを保存: {args.output}")

    # 結果をJSONでも保存
    json_output = Path(args.output).with_suffix(".json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_threshold": best_f1,
                "all_results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"結果JSONを保存: {json_output}")


if __name__ == "__main__":
    main()
