"""Transform and zone classification phase of the pipeline.

Phase 3: ホモグラフィ変換によるカメラ座標→フロアマップ座標変換とゾーン判定
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from src.models import Detection, FrameResult
from src.pipeline.phases.base import BasePhase
from src.transform import FloorMapConfig, HomographyTransformer, TransformResult
from src.zone import ZoneClassifier

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from src.config import ConfigManager


class TransformPhase(BasePhase):
    """座標変換とゾーン判定フェーズ。

    ホモグラフィ行列を使用して、カメラ座標からフロアマップ座標への変換を実行し、
    各検出結果にゾーン情報を付与します。
    """

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化。

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.transformer: HomographyTransformer | None = None
        self.zone_classifier: ZoneClassifier | None = None

    def initialize(self) -> None:
        """座標変換器とゾーン分類器を初期化。"""
        self.log_phase_start("フェーズ3: 座標変換とゾーン判定 (homography)")

        # フロアマップ設定を読み込み
        floormap_config = self.config.get("floormap", {})
        fm_config = FloorMapConfig(
            width_px=int(floormap_config.get("image_width", 1878)),
            height_px=int(floormap_config.get("image_height", 1369)),
            origin_x_px=float(floormap_config.get("image_origin_x", 7.0)),
            origin_y_px=float(floormap_config.get("image_origin_y", 9.0)),
            scale_x_mm_per_px=float(floormap_config.get("image_x_mm_per_pixel", 28.1926406926406)),
            scale_y_mm_per_px=float(floormap_config.get("image_y_mm_per_pixel", 28.241430700447)),
        )

        # ホモグラフィ変換器を初期化
        homography_config = self.config.get("homography", {})
        matrix = homography_config.get("matrix")

        if matrix is None:
            raise ValueError("homography.matrix が設定されていません")

        H = np.array(matrix, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"ホモグラフィ行列は3x3である必要があります: {H.shape}")

        self.transformer = HomographyTransformer(H, fm_config)
        self.logger.info("HomographyTransformer initialized")
        self.logger.info(f"  Matrix:\n{H}")

        # Initialize Zone Classifier
        zones = self.config.get("zones", [])
        if not zones:
            self.logger.warning("ゾーン定義が設定されていません")

        self.zone_classifier = ZoneClassifier(zones, allow_overlap=False)
        self.logger.info(f"ZoneClassifier initialized with {len(zones)} zones.")

    def execute(self, detection_results: list[tuple[int, str, list[Detection]]]) -> list[FrameResult]:
        """座標変換とゾーン判定を実行。

        Args:
            detection_results: [(frame_num, timestamp, detections), ...]

        Returns:
            FrameResult のリスト
        """
        if self.transformer is None or self.zone_classifier is None:
            raise RuntimeError("Not initialized. Call initialize() first.")

        frame_results: list[FrameResult] = []

        # Statistics
        total_detections = 0
        transform_success = 0
        transform_errors = 0
        out_of_bounds_count = 0
        zone_classification_count = 0

        for frame_num, timestamp, detections in tqdm(detection_results, desc="座標変換・ゾーン判定中"):
            total_detections += len(detections)

            # バッチ処理で変換
            if detections:
                bboxes = [d.bbox for d in detections]
                results = self.transformer.transform_batch(bboxes)

                for detection, result in zip(detections, results, strict=True):
                    self._apply_transform_result(detection, result)

                    if result.is_valid:
                        transform_success += 1

                        if not result.is_within_bounds:
                            out_of_bounds_count += 1

                        # ゾーン判定
                        if result.floor_coords_px is not None:
                            zone_ids = self.zone_classifier.classify(result.floor_coords_px)
                            detection.zone_ids = zone_ids
                            if zone_ids:
                                zone_classification_count += 1
                    else:
                        transform_errors += 1

            frame_result = FrameResult(
                frame_number=frame_num,
                timestamp=timestamp,
                detections=detections,
                zone_counts={},
            )
            frame_results.append(frame_result)

        # Log statistics
        self._log_statistics(
            total_detections,
            transform_success,
            transform_errors,
            out_of_bounds_count,
            zone_classification_count,
        )

        return frame_results

    def _apply_transform_result(self, detection: Detection, result: TransformResult) -> None:
        """変換結果を Detection に適用。

        Args:
            detection: 検出結果
            result: 変換結果
        """
        if result.is_valid:
            detection.floor_coords = result.floor_coords_px
            detection.floor_coords_mm = result.floor_coords_mm
            # camera_coords は足元点として設定
            if detection.bbox:
                x, y, w, h = detection.bbox
                detection.camera_coords = (x + w / 2.0, y + h)
        else:
            detection.floor_coords = None
            detection.floor_coords_mm = None
            detection.zone_ids = []

    def _log_statistics(
        self,
        total: int,
        success: int,
        errors: int,
        out_of_bounds: int,
        zone_classified: int,
    ) -> None:
        """統計情報をログ出力。

        Args:
            total: 総検出数
            success: 成功数
            errors: エラー数
            out_of_bounds: 範囲外数
            zone_classified: ゾーン分類数
        """
        if total > 0:
            self.logger.info("=" * 80)
            self.logger.info("Phase 3 Statistics:")
            self.logger.info(f"  Total Detections: {total}")
            self.logger.info(f"  Transform Success: {success} ({success / total * 100:.1f}%)")
            self.logger.info(f"  Transform Errors: {errors} ({errors / total * 100:.1f}%)")
            self.logger.info(f"  Out of Bounds: {out_of_bounds} ({out_of_bounds / total * 100:.1f}%)")
            self.logger.info(f"  Zone Classified: {zone_classified} ({zone_classified / total * 100:.1f}%)")
            self.logger.info("=" * 80)

    def export_results(self, frame_results: list[FrameResult], output_path: Path) -> None:
        """座標変換結果をJSON形式で出力。

        Args:
            frame_results: フレーム結果のリスト
            output_path: 出力ディレクトリパス
        """
        coordinate_data = []
        for frame_result in frame_results:
            frame_data: dict[str, Any] = {
                "frame_number": frame_result.frame_number,
                "timestamp": frame_result.timestamp,
                "detections": [],
            }

            for detection in frame_result.detections:
                detection_data: dict[str, Any] = {
                    "bbox": {
                        "x": float(detection.bbox[0]),
                        "y": float(detection.bbox[1]),
                        "width": float(detection.bbox[2]),
                        "height": float(detection.bbox[3]),
                    },
                    "confidence": float(detection.confidence),
                }

                if detection.camera_coords is not None:
                    detection_data["camera_coords"] = {
                        "x": float(detection.camera_coords[0]),
                        "y": float(detection.camera_coords[1]),
                    }

                if detection.floor_coords is not None:
                    detection_data["floor_coords_px"] = {
                        "x": float(detection.floor_coords[0]),
                        "y": float(detection.floor_coords[1]),
                    }

                if detection.floor_coords_mm is not None:
                    detection_data["floor_coords_mm"] = {
                        "x": float(detection.floor_coords_mm[0]),
                        "y": float(detection.floor_coords_mm[1]),
                    }

                if detection.zone_ids:
                    detection_data["zone_ids"] = detection.zone_ids

                if hasattr(detection, "track_id") and detection.track_id is not None:
                    detection_data["track_id"] = detection.track_id

                frame_data["detections"].append(detection_data)

            coordinate_data.append(frame_data)

        coordinate_output_path = output_path / "coordinate_transformations.json"
        try:
            with open(coordinate_output_path, "w", encoding="utf-8") as f:
                json.dump(coordinate_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved coordinate transformations to {coordinate_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")

    def cleanup(self) -> None:
        """リソースを解放。"""
        self.transformer = None
        self.zone_classifier = None
