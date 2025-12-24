"""検出ベンチマーク評価モジュール

EvaluationModuleを拡張し、AP/mAP計算、診断ログ出力、
CLI対応を追加したベンチマークモジュール。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """検出評価メトリクス

    COCO形式の標準メトリクスを含む。
    """

    # === 基本メトリクス ===
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # === AP/mAP ===
    ap_50: float = 0.0  # AP@IoU=0.5
    ap_75: float = 0.0  # AP@IoU=0.75
    ap: float = 0.0  # AP@IoU=0.5:0.95 (COCO標準)

    # === 詳細カウント ===
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    gt_count: int = 0
    pred_count: int = 0

    # === 評価設定 ===
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.0
    num_images: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "ap_50": self.ap_50,
            "ap_75": self.ap_75,
            "ap": self.ap,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "num_images": self.num_images,
        }

    def summary(self) -> str:
        """サマリー文字列を生成"""
        return (
            f"Precision: {self.precision:.2%}, Recall: {self.recall:.2%}, "
            f"F1: {self.f1_score:.2%}, AP@50: {self.ap_50:.2%}, mAP: {self.ap:.2%}"
        )


@dataclass
class DetectionDiagnostics:
    """検出診断情報"""

    false_positives: list[dict[str, Any]] = field(default_factory=list)
    false_negatives: list[dict[str, Any]] = field(default_factory=list)
    true_positives: list[dict[str, Any]] = field(default_factory=list)
    low_confidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
            "low_confidence": self.low_confidence,
            "summary": {
                "total_false_positives": len(self.false_positives),
                "total_false_negatives": len(self.false_negatives),
                "total_true_positives": len(self.true_positives),
                "total_low_confidence": len(self.low_confidence),
            },
        }

    def export_jsonl(self, output_dir: Path) -> dict[str, Path]:
        """診断ログをJSONL形式でエクスポート"""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # False Positive ログ
        if self.false_positives:
            fp_path = output_dir / "detection_fp.jsonl"
            with fp_path.open("w", encoding="utf-8") as f:
                for fp in self.false_positives:
                    json.dump(fp, f, ensure_ascii=False)
                    f.write("\n")
            paths["false_positives"] = fp_path
            logger.info("FP診断ログを出力: %s (%d件)", fp_path, len(self.false_positives))

        # False Negative ログ
        if self.false_negatives:
            fn_path = output_dir / "detection_fn.jsonl"
            with fn_path.open("w", encoding="utf-8") as f:
                for fn in self.false_negatives:
                    json.dump(fn, f, ensure_ascii=False)
                    f.write("\n")
            paths["false_negatives"] = fn_path

        # Low Confidence ログ
        if self.low_confidence:
            lc_path = output_dir / "detection_low_conf.jsonl"
            with lc_path.open("w", encoding="utf-8") as f:
                for lc in self.low_confidence:
                    json.dump(lc, f, ensure_ascii=False)
                    f.write("\n")
            paths["low_confidence"] = lc_path

        return paths


class DetectionBenchmark:
    """検出ベンチマーク評価クラス

    EvaluationModuleを拡張し、以下の機能を追加:
    - AP/mAP計算（COCO形式）
    - 診断ログ出力
    - CLI対応
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
        person_category_id: int = 0,
        output_diagnostics: bool = True,
    ):
        """初期化

        Args:
            iou_threshold: IoU閾値（デフォルト: 0.5）
            confidence_threshold: 信頼度閾値（デフォルト: 0.0）
            person_category_id: personカテゴリID（デフォルト: 0）
            output_diagnostics: 診断ログを出力するか
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.person_category_id = person_category_id
        self.output_diagnostics = output_diagnostics
        self._diagnostics = DetectionDiagnostics()

        logger.info(
            "DetectionBenchmark initialized: IoU=%.2f, conf=%.2f, category_id=%d",
            iou_threshold,
            confidence_threshold,
            person_category_id,
        )

    def evaluate_from_files(
        self,
        gt_path: str | Path,
        pred_path: str | Path,
    ) -> DetectionMetrics:
        """ファイルから検出結果を評価

        Args:
            gt_path: Ground Truthファイルパス（COCO形式JSON）
            pred_path: 予測結果ファイルパス（COCO形式JSON or カスタム形式）

        Returns:
            評価メトリクス
        """
        gt_path = Path(gt_path)
        pred_path = Path(pred_path)

        # GT読み込み
        with gt_path.open(encoding="utf-8") as f:
            gt_data = json.load(f)

        # 予測結果読み込み
        with pred_path.open(encoding="utf-8") as f:
            pred_data = json.load(f)

        return self.evaluate(gt_data, pred_data)

    def evaluate(
        self,
        gt_data: dict[str, Any],
        pred_data: dict[str, Any],
    ) -> DetectionMetrics:
        """検出結果を評価

        Args:
            gt_data: Ground Truth（COCO形式）
            pred_data: 予測結果（COCO形式 or カスタム形式）

        Returns:
            評価メトリクス
        """
        self._diagnostics = DetectionDiagnostics()

        # データ構造を正規化
        gt_by_image = self._group_annotations_by_image(gt_data)
        pred_by_image = self._parse_predictions(pred_data)

        # 基本メトリクス計算
        tp, fp, fn = 0, 0, 0
        all_scores = []

        image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())

        for image_id in image_ids:
            gt_boxes = gt_by_image.get(image_id, [])
            pred_boxes = pred_by_image.get(image_id, [])

            # 信頼度でフィルタリング
            pred_boxes = [p for p in pred_boxes if p.get("score", 1.0) >= self.confidence_threshold]

            # マッチング
            matches, fp_list, fn_list = self._match_detections(gt_boxes, pred_boxes, image_id)

            tp += len(matches)
            fp += len(fp_list)
            fn += len(fn_list)

            # AP計算用にスコアを収集
            for match in matches:
                all_scores.append((match["score"], True))
            for fp_det in fp_list:
                all_scores.append((fp_det.get("score", 1.0), False))

            # 診断情報収集
            if self.output_diagnostics:
                self._diagnostics.true_positives.extend(matches)
                self._diagnostics.false_positives.extend(fp_list)
                self._diagnostics.false_negatives.extend(fn_list)

        # Precision/Recall/F1計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AP計算
        ap_50 = self._calculate_ap(all_scores)
        ap_75 = self._calculate_ap_at_iou(gt_by_image, pred_by_image, 0.75)
        ap = self._calculate_coco_ap(gt_by_image, pred_by_image)

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            ap_50=ap_50,
            ap_75=ap_75,
            ap=ap,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            gt_count=sum(len(boxes) for boxes in gt_by_image.values()),
            pred_count=sum(len(boxes) for boxes in pred_by_image.values()),
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
            num_images=len(image_ids),
        )

    def _group_annotations_by_image(self, data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        """画像ごとにアノテーションをグループ化"""
        result: dict[int, list[dict[str, Any]]] = {}

        for ann in data.get("annotations", []):
            if ann.get("category_id") != self.person_category_id:
                continue

            image_id = ann["image_id"]
            if image_id not in result:
                result[image_id] = []

            result[image_id].append(
                {
                    "bbox": ann["bbox"],
                    "id": ann.get("id"),
                }
            )

        return result

    def _parse_predictions(self, data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        """予測結果をパース"""
        result: dict[int, list[dict[str, Any]]] = {}

        # COCO形式（annotationsキー）
        if "annotations" in data:
            for ann in data["annotations"]:
                if ann.get("category_id") != self.person_category_id:
                    continue

                image_id = ann["image_id"]
                if image_id not in result:
                    result[image_id] = []

                result[image_id].append(
                    {
                        "bbox": ann["bbox"],
                        "score": ann.get("score", 1.0),
                        "id": ann.get("id"),
                    }
                )

        # カスタム形式（framesキー）
        elif "frames" in data:
            for frame in data["frames"]:
                image_id = frame.get("frame_idx", frame.get("idx", 0))
                if image_id not in result:
                    result[image_id] = []

                for det in frame.get("det", frame.get("detections", [])):
                    result[image_id].append(
                        {
                            "bbox": det.get("bb", det.get("bbox", [])),
                            "score": det.get("conf", det.get("confidence", 1.0)),
                            "id": det.get("id"),
                        }
                    )

        return result

    def _match_detections(
        self,
        gt_boxes: list[dict[str, Any]],
        pred_boxes: list[dict[str, Any]],
        image_id: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """GTと予測をマッチング"""
        matches = []
        fp_list = []
        fn_list = []

        if not gt_boxes and not pred_boxes:
            return matches, fp_list, fn_list

        # スコアで降順ソート
        pred_sorted = sorted(pred_boxes, key=lambda x: x.get("score", 0), reverse=True)
        matched_gt = set()

        for pred in pred_sorted:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = self._calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matches.append(
                    {
                        "image_id": image_id,
                        "pred_bbox": pred["bbox"],
                        "gt_bbox": gt_boxes[best_gt_idx]["bbox"],
                        "iou": best_iou,
                        "score": pred.get("score", 1.0),
                    }
                )
            else:
                fp_list.append(
                    {
                        "image_id": image_id,
                        "bbox": pred["bbox"],
                        "score": pred.get("score", 1.0),
                        "best_iou": best_iou,
                    }
                )

        # マッチしなかったGT
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                fn_list.append(
                    {
                        "image_id": image_id,
                        "bbox": gt["bbox"],
                        "gt_id": gt.get("id"),
                    }
                )

        return matches, fp_list, fn_list

    def _calculate_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """IoUを計算"""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 交差領域
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)

    def _calculate_ap(self, scores_matches: list[tuple[float, bool]]) -> float:
        """APを計算"""
        if not scores_matches:
            return 0.0

        # スコアで降順ソート
        sorted_data = sorted(scores_matches, key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        total_positives = sum(1 for _, is_tp in sorted_data if is_tp)
        if total_positives == 0:
            return 0.0

        for _score, is_tp in sorted_data:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_positives

            precisions.append(precision)
            recalls.append(recall)

        # 11点補間AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p_interp = 0.0
            for prec, rec in zip(precisions, recalls, strict=False):
                if rec >= t:
                    p_interp = max(p_interp, prec)
            ap += p_interp / 11

        return float(ap)

    def _calculate_ap_at_iou(
        self,
        gt_by_image: dict[int, list[dict[str, Any]]],
        pred_by_image: dict[int, list[dict[str, Any]]],
        iou_threshold: float,
    ) -> float:
        """指定IoU閾値でのAPを計算"""
        original_threshold = self.iou_threshold
        self.iou_threshold = iou_threshold

        all_scores = []
        for image_id in set(gt_by_image.keys()) | set(pred_by_image.keys()):
            gt_boxes = gt_by_image.get(image_id, [])
            pred_boxes = pred_by_image.get(image_id, [])

            matches, fp_list, _ = self._match_detections(gt_boxes, pred_boxes, image_id)

            for match in matches:
                all_scores.append((match["score"], True))
            for fp_det in fp_list:
                all_scores.append((fp_det.get("score", 1.0), False))

        self.iou_threshold = original_threshold
        return self._calculate_ap(all_scores)

    def _calculate_coco_ap(
        self,
        gt_by_image: dict[int, list[dict[str, Any]]],
        pred_by_image: dict[int, list[dict[str, Any]]],
    ) -> float:
        """COCO形式のmAP（AP@0.5:0.95）を計算"""
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []

        for iou_thresh in iou_thresholds:
            ap = self._calculate_ap_at_iou(gt_by_image, pred_by_image, iou_thresh)
            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0

    def get_diagnostics(self) -> DetectionDiagnostics:
        """診断情報を取得"""
        return self._diagnostics

    def export_results(
        self,
        metrics: DetectionMetrics,
        output_dir: Path,
        include_diagnostics: bool = True,
    ) -> dict[str, Path]:
        """評価結果をエクスポート"""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # メトリクスサマリー
        metrics_path = output_dir / "detection_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "iou_threshold": self.iou_threshold,
                    "confidence_threshold": self.confidence_threshold,
                    "metrics": metrics.to_dict(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        paths["metrics"] = metrics_path
        logger.info("検出メトリクスを出力: %s", metrics_path)

        # 診断ログ
        if include_diagnostics and self.output_diagnostics:
            diagnostics_dir = output_dir / "diagnostics"
            diag_paths = self._diagnostics.export_jsonl(diagnostics_dir)
            paths.update(diag_paths)

        return paths

    def generate_report(self, metrics: DetectionMetrics, title: str = "検出評価レポート") -> str:
        """Markdownレポートを生成"""
        report = f"""# {title}

## 評価設定

| 項目 | 値 |
|-----|-----|
| IoU閾値 | {self.iou_threshold} |
| 信頼度閾値 | {self.confidence_threshold} |
| 評価画像数 | {metrics.num_images} |
| GT総数 | {metrics.gt_count} |
| 予測総数 | {metrics.pred_count} |

## 基本メトリクス

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **Precision** | {metrics.precision:.2%} | 検出精度 |
| **Recall** | {metrics.recall:.2%} | 検出率 |
| **F1-Score** | {metrics.f1_score:.2%} | 調和平均 |

## AP/mAP (COCO形式)

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **AP@50** | {metrics.ap_50:.2%} | IoU=0.5 |
| **AP@75** | {metrics.ap_75:.2%} | IoU=0.75 |
| **mAP** | {metrics.ap:.2%} | AP@0.5:0.95 |

## 詳細カウント

| 項目 | 件数 |
|-----|------|
| True Positive | {metrics.true_positives} |
| False Positive | {metrics.false_positives} |
| False Negative | {metrics.false_negatives} |

## 診断サマリー

| 項目 | 件数 |
|-----|------|
| False Positive | {len(self._diagnostics.false_positives)} |
| False Negative | {len(self._diagnostics.false_negatives)} |
| Low Confidence | {len(self._diagnostics.low_confidence)} |
"""
        return report
