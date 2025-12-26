"""MOT (Multiple Object Tracking) metrics evaluation module."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MOTMetrics:
    """motmetricsを用いたMOT評価ユーティリティ."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        logger.info("MOTMetrics initialized with IoU threshold %.2f", iou_threshold)

    def evaluate_from_files(
        self, gt_coco_path: str | Path, pred_mot_path: str | Path, person_category_id: int = 1
    ) -> dict[str, float]:
        """COCO形式GTとMOT形式予測ファイルからメトリクスを算出する。"""
        gt_df = self.coco_to_mot_dataframe(gt_coco_path, person_category_id=person_category_id)
        pred_df = self.load_mot_predictions(pred_mot_path)
        return self.evaluate_from_dataframes(gt_df, pred_df)

    def evaluate_from_dataframes(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
        """MOT形式DataFrame同士からメトリクスを算出する。"""
        if gt_df.empty:
            logger.warning("Ground truth dataframe is empty; returning zeroed metrics.")
            return {
                "MOTA": 0.0,
                "IDF1": 0.0,
                "IDP": 0.0,
                "IDR": 0.0,
                "IDSW": 0.0,
                "FP": 0.0,
                "FN": 0.0,
                "GT": 0.0,
            }

        acc = mm.MOTAccumulator(auto_id=True)
        frames = sorted(set(gt_df["FrameId"]).union(set(pred_df["FrameId"])))

        for frame in frames:
            gt_frame = gt_df[gt_df["FrameId"] == frame]
            pred_frame = pred_df[pred_df["FrameId"] == frame]

            gt_ids = gt_frame["Id"].tolist()
            pred_ids = pred_frame["Id"].tolist()

            gt_boxes = gt_frame[["X", "Y", "Width", "Height"]].to_numpy() if not gt_frame.empty else np.empty((0, 4))
            pred_boxes = (
                pred_frame[["X", "Y", "Width", "Height"]].to_numpy() if not pred_frame.empty else np.empty((0, 4))
            )

            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
            if distances.size > 0:
                # iou_matrixは既に1-IoU（距離）を返す。しきい値を超える距離をNaNにする
                distances[distances > (1.0 - self.iou_threshold)] = np.nan

            acc.update(gt_ids, pred_ids, distances)

        metrics = ["mota", "idf1", "idp", "idr", "num_switches", "num_false_positives", "num_misses", "num_objects"]
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=metrics, name="acc").fillna(0.0)
        row = summary.loc["acc"]

        return {
            "MOTA": float(row["mota"]),
            "IDF1": float(row["idf1"]),
            "IDP": float(row["idp"]),
            "IDR": float(row["idr"]),
            "IDSW": float(row["num_switches"]),
            "FP": float(row["num_false_positives"]),
            "FN": float(row["num_misses"]),
            "GT": float(row["num_objects"]),
        }

    def coco_to_mot_dataframe(self, coco_path: str | Path, person_category_id: int = 1) -> pd.DataFrame:
        """COCO形式のGTをMOTChallenge互換DataFrameに変換する。"""
        coco_path = Path(coco_path)
        with coco_path.open(encoding="utf-8") as f:
            data = json.load(f)

        images = {img["id"]: img for img in data.get("images", [])}
        annotations = data.get("annotations", [])

        rows: list[dict[str, float | int]] = []
        for ann in annotations:
            if ann.get("category_id") != person_category_id:
                continue

            image_id = ann.get("image_id")
            image_info = images.get(image_id)
            if image_info is None:
                continue

            frame_id = int(image_info.get("id", 0)) + 1
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue

            rows.append(
                {
                    "FrameId": frame_id,
                    "Id": int(ann.get("id", -1)),
                    "X": float(bbox[0]),
                    "Y": float(bbox[1]),
                    "Width": float(bbox[2]),
                    "Height": float(bbox[3]),
                    "Confidence": 1.0,
                }
            )

        df = pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"])
        if df.empty:
            logger.warning("COCO GTから人物アノテーションが得られませんでした: %s", coco_path)
        return df

    def load_mot_predictions(self, pred_path: str | Path) -> pd.DataFrame:
        """MOTChallenge形式（ヘッダー付き）予測CSVを読み込みDataFrame化する。

        対応形式:
        - MOTChallenge標準: frame,id,bb_left,bb_top,bb_width,bb_height,conf,...
        - 本プロジェクト形式: track_id,frame_index,timestamp,x,y,zone_ids,confidence
        - transformed JSON: coordinate_transformations.json (別メソッドで変換)
        """
        pred_path = Path(pred_path)
        df = pd.read_csv(pred_path)

        # 本プロジェクトのtracks.csv形式を検出（bbox無し、中心点のみ）
        if "track_id" in df.columns and "frame_index" in df.columns and "bb_left" not in df.columns:
            logger.info("本プロジェクトのtracks.csv形式を検出: 中心点のみ（bbox推定）")
            # 固定サイズでbboxを推定（評価用）
            default_w, default_h = 40.0, 80.0
            df["FrameId"] = df["frame_index"] + 1  # 1-indexed
            df["Id"] = df["track_id"]
            df["X"] = df["x"] - default_w / 2
            df["Y"] = df["y"] - default_h / 2
            df["Width"] = default_w
            df["Height"] = default_h
            df["Confidence"] = df.get("confidence", 1.0)
            return df[["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"]]

        column_map = {
            "frame": "FrameId",
            "FrameId": "FrameId",
            "id": "Id",
            "Id": "Id",
            "bb_left": "X",
            "bb_top": "Y",
            "bb_width": "Width",
            "bb_height": "Height",
            "conf": "Confidence",
            "score": "Confidence",
        }
        renamed = {src: dst for src, dst in column_map.items() if src in df.columns}
        df = df.rename(columns=renamed)

        required_cols = ["FrameId", "Id", "X", "Y", "Width", "Height"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"予測ファイルに必要な列が不足しています: {col}")

        df["Confidence"] = df.get("Confidence", 1.0)
        return df[["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"]]

    def dataframe_to_csv(self, df: pd.DataFrame, output_path: Path) -> Path:
        """DataFrameをMOTChallenge形式で保存するヘルパー."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path
