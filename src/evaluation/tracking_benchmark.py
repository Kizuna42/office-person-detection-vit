"""トラッキングベンチマーク評価モジュール

MOTMetricsを拡張し、疎サンプリング対応、診断ログ出力、
Gold GT形式対応を追加したベンチマークモジュール。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.mot_metrics import MOTMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrackingMetrics:
    """トラッキング評価メトリクス

    MOT標準メトリクスに加え、疎サンプリング固有の指標を含む。
    """

    # === MOT標準メトリクス ===
    mota: float = 0.0  # Multiple Object Tracking Accuracy
    idf1: float = 0.0  # ID F1 Score
    idp: float = 0.0  # ID Precision
    idr: float = 0.0  # ID Recall
    idsw: int = 0  # ID Switch回数
    fp: int = 0  # False Positives
    fn: int = 0  # False Negatives (Misses)
    gt_count: int = 0  # Ground Truth総数

    # === 疎サンプリング固有メトリクス ===
    num_frames: int = 0  # 評価対象フレーム数
    num_transitions: int = 0  # フレーム間遷移数
    idsw_per_transition: float = 0.0  # 遷移あたりのIDSW率

    # === 追加メトリクス ===
    motp: float = 0.0  # Multiple Object Tracking Precision
    mt: int = 0  # Mostly Tracked (80%以上追跡成功)
    ml: int = 0  # Mostly Lost (20%未満追跡成功)
    frag: int = 0  # Fragmentations

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "mota": self.mota,
            "idf1": self.idf1,
            "idp": self.idp,
            "idr": self.idr,
            "idsw": self.idsw,
            "fp": self.fp,
            "fn": self.fn,
            "gt_count": self.gt_count,
            "num_frames": self.num_frames,
            "num_transitions": self.num_transitions,
            "idsw_per_transition": self.idsw_per_transition,
            "motp": self.motp,
            "mt": self.mt,
            "ml": self.ml,
            "frag": self.frag,
        }

    def summary(self) -> str:
        """サマリー文字列を生成"""
        return (
            f"MOTA: {self.mota:.2%}, IDF1: {self.idf1:.2%}, "
            f"IDSW: {self.idsw} ({self.idsw_per_transition:.3f}/transition), "
            f"FP: {self.fp}, FN: {self.fn}"
        )


@dataclass
class IDSwitchEvent:
    """ID Switch イベント詳細"""

    frame_idx: int
    timestamp: str | None
    old_track_id: int
    new_track_id: int
    gt_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h
    reason: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingDiagnostics:
    """トラッキング診断情報"""

    id_switches: list[IDSwitchEvent] = field(default_factory=list)
    lost_tracks: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    missed_detections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id_switches": [
                {
                    "frame_idx": e.frame_idx,
                    "timestamp": e.timestamp,
                    "old_track_id": e.old_track_id,
                    "new_track_id": e.new_track_id,
                    "gt_id": e.gt_id,
                    "bbox": list(e.bbox),
                    "reason": e.reason,
                    "details": e.details,
                }
                for e in self.id_switches
            ],
            "lost_tracks": self.lost_tracks,
            "false_positives": self.false_positives,
            "missed_detections": self.missed_detections,
            "summary": {
                "total_id_switches": len(self.id_switches),
                "total_lost_tracks": len(self.lost_tracks),
                "total_false_positives": len(self.false_positives),
                "total_missed_detections": len(self.missed_detections),
            },
        }

    def export_jsonl(self, output_dir: Path) -> dict[str, Path]:
        """診断ログをJSONL形式でエクスポート"""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # ID Switch ログ
        if self.id_switches:
            idsw_path = output_dir / "id_switches.jsonl"
            with idsw_path.open("w", encoding="utf-8") as f:
                for event in self.id_switches:
                    json.dump(
                        {
                            "frame_idx": event.frame_idx,
                            "timestamp": event.timestamp,
                            "old_track_id": event.old_track_id,
                            "new_track_id": event.new_track_id,
                            "gt_id": event.gt_id,
                            "bbox": list(event.bbox),
                            "reason": event.reason,
                            "details": event.details,
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
            paths["id_switches"] = idsw_path
            logger.info("ID Switch診断ログを出力: %s (%d件)", idsw_path, len(self.id_switches))

        # Lost Track ログ
        if self.lost_tracks:
            lost_path = output_dir / "lost_tracks.jsonl"
            with lost_path.open("w", encoding="utf-8") as f:
                for track in self.lost_tracks:
                    json.dump(track, f, ensure_ascii=False)
                    f.write("\n")
            paths["lost_tracks"] = lost_path

        # False Positive ログ
        if self.false_positives:
            fp_path = output_dir / "false_positives.jsonl"
            with fp_path.open("w", encoding="utf-8") as f:
                for fp in self.false_positives:
                    json.dump(fp, f, ensure_ascii=False)
                    f.write("\n")
            paths["false_positives"] = fp_path

        # Missed Detection ログ
        if self.missed_detections:
            miss_path = output_dir / "missed_detections.jsonl"
            with miss_path.open("w", encoding="utf-8") as f:
                for miss in self.missed_detections:
                    json.dump(miss, f, ensure_ascii=False)
                    f.write("\n")
            paths["missed_detections"] = miss_path

        return paths


class TrackingBenchmark:
    """トラッキングベンチマーク評価クラス

    MOTMetricsを拡張し、以下の機能を追加:
    - 疎サンプリング対応メトリクス
    - 診断ログ出力
    - Gold GT形式対応（person_id付きアノテーション）
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        sparse_mode: bool = False,
        output_diagnostics: bool = True,
    ):
        """初期化

        Args:
            iou_threshold: IoU閾値（デフォルト: 0.5）
            sparse_mode: 疎サンプリングモード（5分間隔）
            output_diagnostics: 診断ログを出力するか
        """
        self.mot_metrics = MOTMetrics(iou_threshold=iou_threshold)
        self.iou_threshold = iou_threshold
        self.sparse_mode = sparse_mode
        self.output_diagnostics = output_diagnostics
        self._diagnostics = TrackingDiagnostics()

        mode_str = "sparse (5min)" if sparse_mode else "dense (10sec)"
        logger.info(
            "TrackingBenchmark initialized: IoU=%.2f, mode=%s, diagnostics=%s",
            iou_threshold,
            mode_str,
            output_diagnostics,
        )

    def evaluate(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        timestamps: dict[int, str] | None = None,
    ) -> TrackingMetrics:
        """トラッキング結果を評価

        Args:
            gt_df: Ground Truth DataFrame (FrameId, Id, X, Y, Width, Height)
            pred_df: 予測結果 DataFrame (同上)
            timestamps: フレームID→タイムスタンプのマッピング（診断用）

        Returns:
            評価メトリクス
        """
        # 基本メトリクスを取得
        basic_metrics = self.mot_metrics.evaluate_from_dataframes(gt_df, pred_df)

        # フレーム数と遷移数を計算
        frames = sorted(set(gt_df["FrameId"]).union(set(pred_df["FrameId"])))
        num_frames = len(frames)
        num_transitions = max(0, num_frames - 1)

        # 遷移あたりのIDSW率
        idsw = int(basic_metrics["IDSW"])
        idsw_per_transition = idsw / num_transitions if num_transitions > 0 else 0.0

        # 診断情報を収集
        if self.output_diagnostics:
            self._collect_diagnostics(gt_df, pred_df, timestamps)

        return TrackingMetrics(
            mota=basic_metrics["MOTA"],
            idf1=basic_metrics["IDF1"],
            idp=basic_metrics["IDP"],
            idr=basic_metrics["IDR"],
            idsw=idsw,
            fp=int(basic_metrics["FP"]),
            fn=int(basic_metrics["FN"]),
            gt_count=int(basic_metrics["GT"]),
            num_frames=num_frames,
            num_transitions=num_transitions,
            idsw_per_transition=idsw_per_transition,
        )

    def evaluate_from_files(
        self,
        gt_path: str | Path,
        pred_path: str | Path,
        gt_format: str = "coco",
        person_category_id: int = 0,
    ) -> TrackingMetrics:
        """ファイルからトラッキング結果を評価

        Args:
            gt_path: Ground Truthファイルパス
            pred_path: 予測結果ファイルパス（MOTChallenge形式CSV）
            gt_format: GTの形式 ("coco" | "gold")
            person_category_id: COCO形式のpersonカテゴリID（デフォルト: 0）

        Returns:
            評価メトリクス
        """
        # GT読み込み
        if gt_format == "gold":
            gt_df = self._load_gold_gt(gt_path)
        else:
            gt_df = self.mot_metrics.coco_to_mot_dataframe(gt_path, person_category_id=person_category_id)

        # 予測結果読み込み
        pred_df = self.mot_metrics.load_mot_predictions(pred_path)

        return self.evaluate(gt_df, pred_df)

    def _load_gold_gt(self, gt_path: str | Path) -> pd.DataFrame:
        """Gold GT形式（person_id付き）を読み込み

        Gold GT形式:
        {
            "frames": [
                {
                    "frame_idx": 0,
                    "timestamp": "2024-01-15T10:00:05",
                    "annotations": [
                        {"person_id": 1, "bbox": [x, y, w, h], "zone_id": "..."}
                    ]
                }
            ]
        }
        """
        gt_path = Path(gt_path)
        with gt_path.open(encoding="utf-8") as f:
            data = json.load(f)

        rows: list[dict[str, Any]] = []

        # Gold形式
        if "frames" in data:
            for frame_data in data["frames"]:
                frame_idx = frame_data.get("frame_idx", 0)
                for ann in frame_data.get("annotations", []):
                    person_id = ann.get("person_id", ann.get("id", -1))
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4:
                        rows.append(
                            {
                                "FrameId": frame_idx + 1,  # MOTChallenge: 1-indexed
                                "Id": int(person_id),
                                "X": float(bbox[0]),
                                "Y": float(bbox[1]),
                                "Width": float(bbox[2]),
                                "Height": float(bbox[3]),
                                "Confidence": 1.0,
                            }
                        )
        else:
            # COCO形式にフォールバック
            return self.mot_metrics.coco_to_mot_dataframe(gt_path)

        df = pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"])
        logger.info("Gold GTを読み込み: %d アノテーション, %d フレーム", len(rows), df["FrameId"].nunique())
        return df

    def _collect_diagnostics(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        timestamps: dict[int, str] | None = None,
    ) -> None:
        """診断情報を収集

        ID Switch、Lost Track、FP、FNの詳細を記録
        """
        self._diagnostics = TrackingDiagnostics()

        if gt_df.empty and pred_df.empty:
            return

        frames = sorted(set(gt_df["FrameId"]).union(set(pred_df["FrameId"])))
        pred_id_to_gt_id: dict[int, int] = {}  # 予測ID → GT IDのマッピング履歴

        for _frame_idx, frame in enumerate(frames):
            gt_frame = gt_df[gt_df["FrameId"] == frame]
            pred_frame = pred_df[pred_df["FrameId"] == frame]
            timestamp = timestamps.get(frame) if timestamps else None

            # マッチングを計算
            gt_boxes = gt_frame[["X", "Y", "Width", "Height"]].values if not gt_frame.empty else np.empty((0, 4))
            pred_boxes = pred_frame[["X", "Y", "Width", "Height"]].values if not pred_frame.empty else np.empty((0, 4))

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)
                gt_ids = gt_frame["Id"].tolist()
                pred_ids = pred_frame["Id"].tolist()

                # Greedy マッチング（簡易版）
                matched_gt = set()
                matched_pred = set()

                for _ in range(min(len(gt_ids), len(pred_ids))):
                    max_iou = 0
                    best_match = (-1, -1)
                    for gi, _gt_id in enumerate(gt_ids):
                        if gi in matched_gt:
                            continue
                        for pi, _pred_id in enumerate(pred_ids):
                            if pi in matched_pred:
                                continue
                            if iou_matrix[gi, pi] > max_iou:
                                max_iou = iou_matrix[gi, pi]
                                best_match = (gi, pi)

                    if max_iou >= self.iou_threshold:
                        gi, pi = best_match
                        matched_gt.add(gi)
                        matched_pred.add(pi)

                        gt_id = gt_ids[gi]
                        pred_id = pred_ids[pi]

                        # ID Switch検出
                        if pred_id in pred_id_to_gt_id:
                            prev_gt_id = pred_id_to_gt_id[pred_id]
                            if prev_gt_id != gt_id:
                                bbox = tuple(pred_boxes[pi])
                                self._diagnostics.id_switches.append(
                                    IDSwitchEvent(
                                        frame_idx=int(frame),
                                        timestamp=timestamp,
                                        old_track_id=pred_id,
                                        new_track_id=pred_id,  # 同じtrack_idが別のGT IDを追跡
                                        gt_id=gt_id,
                                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                                        reason="gt_id_change",
                                        details={"previous_gt_id": prev_gt_id, "iou": float(max_iou)},
                                    )
                                )
                        pred_id_to_gt_id[pred_id] = gt_id

                # False Positives (未マッチの予測)
                for pi, pred_id in enumerate(pred_ids):
                    if pi not in matched_pred:
                        self._diagnostics.false_positives.append(
                            {
                                "frame_idx": int(frame),
                                "timestamp": timestamp,
                                "pred_id": pred_id,
                                "bbox": pred_boxes[pi].tolist(),
                            }
                        )

                # Missed Detections (未マッチのGT)
                for gi, gt_id in enumerate(gt_ids):
                    if gi not in matched_gt:
                        self._diagnostics.missed_detections.append(
                            {
                                "frame_idx": int(frame),
                                "timestamp": timestamp,
                                "gt_id": gt_id,
                                "bbox": gt_boxes[gi].tolist(),
                            }
                        )

            elif len(pred_boxes) > 0:
                # 全て False Positive
                for _, row in pred_frame.iterrows():
                    self._diagnostics.false_positives.append(
                        {
                            "frame_idx": int(frame),
                            "timestamp": timestamp,
                            "pred_id": int(row["Id"]),
                            "bbox": [row["X"], row["Y"], row["Width"], row["Height"]],
                        }
                    )

            elif len(gt_boxes) > 0:
                # 全て Missed Detection
                for _, row in gt_frame.iterrows():
                    self._diagnostics.missed_detections.append(
                        {
                            "frame_idx": int(frame),
                            "timestamp": timestamp,
                            "gt_id": int(row["Id"]),
                            "bbox": [row["X"], row["Y"], row["Width"], row["Height"]],
                        }
                    )

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """IoU行列を計算

        Args:
            boxes1: (N, 4) [x, y, w, h]
            boxes2: (M, 4) [x, y, w, h]

        Returns:
            (N, M) IoU行列
        """
        n = len(boxes1)
        m = len(boxes2)
        iou_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """2つのBBoxのIoUを計算"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # (x, y, w, h) -> (x1, y1, x2, y2)
        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

        # 交差領域
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

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

    def get_diagnostics(self) -> TrackingDiagnostics:
        """診断情報を取得"""
        return self._diagnostics

    def export_results(
        self,
        metrics: TrackingMetrics,
        output_dir: Path,
        include_diagnostics: bool = True,
    ) -> dict[str, Path]:
        """評価結果をエクスポート

        Args:
            metrics: 評価メトリクス
            output_dir: 出力ディレクトリ
            include_diagnostics: 診断ログを含めるか

        Returns:
            出力ファイルパスの辞書
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # メトリクスサマリー
        metrics_path = output_dir / "tracking_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "sparse" if self.sparse_mode else "dense",
                    "iou_threshold": self.iou_threshold,
                    "metrics": metrics.to_dict(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        paths["metrics"] = metrics_path
        logger.info("トラッキングメトリクスを出力: %s", metrics_path)

        # 診断ログ
        if include_diagnostics and self.output_diagnostics:
            diagnostics_dir = output_dir / "diagnostics"
            diag_paths = self._diagnostics.export_jsonl(diagnostics_dir)
            paths.update(diag_paths)

        return paths

    def generate_report(self, metrics: TrackingMetrics, title: str = "トラッキング評価レポート") -> str:
        """Markdownレポートを生成"""
        mode = "疎サンプリング（5分間隔）" if self.sparse_mode else "密サンプリング（10秒間隔）"

        report = f"""# {title}

## 評価設定

| 項目 | 値 |
|-----|-----|
| 評価モード | {mode} |
| IoU閾値 | {self.iou_threshold} |
| 評価フレーム数 | {metrics.num_frames} |
| フレーム間遷移数 | {metrics.num_transitions} |

## MOT標準メトリクス

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **MOTA** | {metrics.mota:.2%} | Multiple Object Tracking Accuracy |
| **IDF1** | {metrics.idf1:.2%} | ID F1 Score |
| IDP | {metrics.idp:.2%} | ID Precision |
| IDR | {metrics.idr:.2%} | ID Recall |
| **IDSW** | {metrics.idsw} | ID Switch回数 |
| FP | {metrics.fp} | False Positives |
| FN | {metrics.fn} | False Negatives |
| GT | {metrics.gt_count} | Ground Truth総数 |

## 疎サンプリング指標

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **IDSW/遷移** | {metrics.idsw_per_transition:.4f} | 遷移あたりのID Switch率 |

## 診断サマリー

| 項目 | 件数 |
|-----|------|
| ID Switch | {len(self._diagnostics.id_switches)} |
| Lost Track | {len(self._diagnostics.lost_tracks)} |
| False Positive | {len(self._diagnostics.false_positives)} |
| Missed Detection | {len(self._diagnostics.missed_detections)} |
"""
        return report
