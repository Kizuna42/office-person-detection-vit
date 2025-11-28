"""Transform and zone classification phase of the pipeline.

Phase 3: 高精度座標変換とゾーン判定

変換方式:
- "homography": ホモグラフィ行列による2D→2D直接変換（高精度推奨）
- "pinhole": ピンホールカメラモデルによるレイキャスト
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from tqdm import tqdm

from src.models import Detection, FrameResult
from src.pipeline.phases.base import BasePhase
from src.transform import (
    CameraExtrinsics,
    CameraIntrinsics,
    FloorMapConfig,
    FloorMapTransformer,
    RayCaster,
    TransformResult,
    UnifiedTransformer,
)
from src.zone import ZoneClassifier

if TYPE_CHECKING:
    import logging

    from src.config import ConfigManager


class TransformerProtocol(Protocol):
    """変換器インターフェース"""

    def transform_pixel(self, image_point: tuple[float, float]) -> TransformResult: ...

    def transform_batch(self, bboxes: list[tuple[float, float, float, float]]) -> list[TransformResult]: ...


class PiecewiseAffineTransformer:
    """Piecewise Affine 変換器（高精度）。"""

    def __init__(
        self,
        model_path: str,
        floormap_config: FloorMapConfig,
    ):
        """初期化。

        Args:
            model_path: PKLモデルファイルパス
            floormap_config: フロアマップ設定
        """
        from src.transform.piecewise_affine import load_model

        self.model = load_model(model_path)
        self.floormap_config = floormap_config

    def transform_pixel(self, image_point: tuple[float, float]) -> TransformResult:
        """1点を変換。"""
        pt = np.array([[image_point[0], image_point[1]]])
        transformed = self.model.transform(pt)
        floor_x, floor_y = float(transformed[0, 0]), float(transformed[0, 1])

        is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

        floor_mm = (
            floor_x * self.floormap_config.scale_x_mm_per_px,
            floor_y * self.floormap_config.scale_y_mm_per_px,
        )

        return TransformResult(
            is_valid=True,
            floor_coords_px=(floor_x, floor_y),
            floor_coords_mm=floor_mm,
            world_coords_m=None,
            is_within_bounds=is_within,
        )

    def transform_batch(self, bboxes: list[tuple[float, float, float, float]]) -> list[TransformResult]:
        """バッチ変換（足元点を使用）。"""
        if not bboxes:
            return []

        # 足元点を計算
        foot_points = np.array([[x + w / 2, y + h] for x, y, w, h in bboxes], dtype=np.float64)

        # 変換
        transformed = self.model.transform(foot_points)

        results = []
        for i in range(len(bboxes)):
            floor_x, floor_y = float(transformed[i, 0]), float(transformed[i, 1])

            is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

            floor_mm = (
                floor_x * self.floormap_config.scale_x_mm_per_px,
                floor_y * self.floormap_config.scale_y_mm_per_px,
            )

            results.append(
                TransformResult(
                    is_valid=True,
                    floor_coords_px=(floor_x, floor_y),
                    floor_coords_mm=floor_mm,
                    world_coords_m=None,
                    is_within_bounds=is_within,
                )
            )

        return results


class HomographyTransformer:
    """ホモグラフィ変換器

    2D画像座標からフロアマップ座標への直接変換。
    """

    def __init__(
        self,
        homography_matrix: np.ndarray,
        floormap_config: FloorMapConfig,
    ):
        """初期化

        Args:
            homography_matrix: 3x3ホモグラフィ行列
            floormap_config: フロアマップ設定
        """
        self.H = homography_matrix.astype(np.float64)
        self.floormap_config = floormap_config

    def transform_pixel(self, image_point: tuple[float, float]) -> TransformResult:
        """1点を変換

        Args:
            image_point: 画像座標 (x, y)

        Returns:
            TransformResult
        """
        pt = np.array([[image_point[0], image_point[1], 1.0]])
        transformed = (self.H @ pt.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        floor_x, floor_y = float(transformed[0, 0]), float(transformed[0, 1])

        # 境界チェック
        is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

        # mm座標を計算
        floor_mm = (
            floor_x * self.floormap_config.scale_x_mm_per_px,
            floor_y * self.floormap_config.scale_y_mm_per_px,
        )

        return TransformResult(
            is_valid=True,
            floor_coords_px=(floor_x, floor_y),
            floor_coords_mm=floor_mm,
            world_coords_m=None,  # N/A for homography
            is_within_bounds=is_within,
        )

    def transform_batch(self, bboxes: list[tuple[float, float, float, float]]) -> list[TransformResult]:
        """バッチ変換（足元点を使用）

        Args:
            bboxes: バウンディングボックス [(x, y, w, h), ...]

        Returns:
            TransformResult のリスト
        """
        if not bboxes:
            return []

        # 足元点を計算
        foot_points = np.array([[x + w / 2, y + h] for x, y, w, h in bboxes], dtype=np.float64)

        # ホモグラフィ変換
        ones = np.ones((len(foot_points), 1))
        pts_h = np.hstack([foot_points, ones])
        transformed = (self.H @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        results = []
        for i in range(len(bboxes)):
            floor_x, floor_y = float(transformed[i, 0]), float(transformed[i, 1])

            is_within = 0 <= floor_x < self.floormap_config.width_px and 0 <= floor_y < self.floormap_config.height_px

            floor_mm = (
                floor_x * self.floormap_config.scale_x_mm_per_px,
                floor_y * self.floormap_config.scale_y_mm_per_px,
            )

            results.append(
                TransformResult(
                    is_valid=True,
                    floor_coords_px=(floor_x, floor_y),
                    floor_coords_mm=floor_mm,
                    world_coords_m=None,
                    is_within_bounds=is_within,
                )
            )

        return results


class TransformPhase(BasePhase):
    """座標変換とゾーン判定フェーズ

    変換方式:
    - "homography": ホモグラフィ行列による直接変換（高精度）
    - "pinhole": ピンホールカメラモデル（3D経由）
    """

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.transformer: TransformerProtocol | None = None
        self.zone_classifier: ZoneClassifier | None = None
        self.transform_method: str = "homography"

    def initialize(self) -> None:
        """座標変換器とゾーン分類器を初期化"""
        # 変換方式を取得
        transform_config = self.config.get("transform", {})
        self.transform_method = transform_config.get("method", "homography")

        self.log_phase_start(f"フェーズ3: 座標変換とゾーン判定 ({self.transform_method})")

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

        if self.transform_method == "piecewise_affine":
            self._initialize_piecewise_affine(fm_config)
        elif self.transform_method == "homography":
            self._initialize_homography(fm_config)
        else:
            self._initialize_pinhole(fm_config)

        # Initialize Zone Classifier
        zones = self.config.get("zones", [])
        if not zones:
            self.logger.warning("ゾーン定義が設定されていません")

        self.zone_classifier = ZoneClassifier(zones, allow_overlap=False)
        self.logger.info(f"ZoneClassifier initialized with {len(zones)} zones.")

    def _initialize_piecewise_affine(self, fm_config: FloorMapConfig) -> None:
        """Piecewise Affine変換器を初期化（高精度）"""
        transform_config = self.config.get("transform", {})
        model_path = transform_config.get("model_path", "output/calibration/best_transformer_pwa.pkl")

        if not Path(model_path).exists():
            raise ValueError(f"PWAモデルファイルが見つかりません: {model_path}")

        self.transformer = PiecewiseAffineTransformer(model_path, fm_config)
        self.logger.info("PiecewiseAffineTransformer initialized (RMSE: ~0px on training data)")
        self.logger.info(f"  Model: {model_path}")

    def _initialize_homography(self, fm_config: FloorMapConfig) -> None:
        """ホモグラフィ変換器を初期化"""
        homography_config = self.config.get("homography", {})
        matrix = homography_config.get("matrix")

        if matrix is None:
            raise ValueError("homography.matrix が設定されていません")

        H = np.array(matrix, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"ホモグラフィ行列は3x3である必要があります: {H.shape}")

        self.transformer = HomographyTransformer(H, fm_config)
        self.logger.info("HomographyTransformer initialized (RMSE: ~1.72px)")
        self.logger.info(f"  Matrix:\n{H}")

    def _initialize_pinhole(self, fm_config: FloorMapConfig) -> None:
        """ピンホールモデル変換器を初期化"""
        camera_params = self.config.get("camera_params", {})

        # Intrinsics
        intrinsics = CameraIntrinsics(
            fx=float(camera_params.get("focal_length_x", 1250.0)),
            fy=float(camera_params.get("focal_length_y", 1250.0)),
            cx=float(camera_params.get("center_x", 640.0)),
            cy=float(camera_params.get("center_y", 360.0)),
            image_width=int(camera_params.get("image_width", 1280)),
            image_height=int(camera_params.get("image_height", 720)),
            dist_coeffs=np.array(
                camera_params.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
        )

        # Extrinsics
        extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=float(camera_params.get("height_m", 2.2)),
            pitch_deg=float(camera_params.get("pitch_deg", 45.0)),
            yaw_deg=float(camera_params.get("yaw_deg", 0.0)),
            roll_deg=float(camera_params.get("roll_deg", 0.0)),
            camera_x_m=float(camera_params.get("camera_x_m", 0.0)),
            camera_y_m=float(camera_params.get("camera_y_m", 0.0)),
        )

        # RayCaster
        ray_caster = RayCaster(intrinsics, extrinsics)

        self.logger.info("RayCaster initialized (Pinhole Model):")
        self.logger.info(
            f"  Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
            f"cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}"
        )
        self.logger.info(
            f"  Extrinsics: height={camera_params.get('height_m', 2.2):.2f}m, "
            f"pitch={camera_params.get('pitch_deg', 45.0):.1f}deg, "
            f"yaw={camera_params.get('yaw_deg', 0.0):.1f}deg"
        )

        # カメラ位置（フロアマップ座標系）
        camera_position_px = (
            float(camera_params.get("position_x_px", 1200.0)),
            float(camera_params.get("position_y_px", 800.0)),
        )

        floormap_transformer = FloorMapTransformer(fm_config, camera_position_px)

        self.logger.info(
            f"FloorMapTransformer initialized: "
            f"camera_pos=({camera_position_px[0]:.1f}, {camera_position_px[1]:.1f}), "
            f"scale=({fm_config.scale_x_mm_per_px:.2f}, {fm_config.scale_y_mm_per_px:.2f}) mm/px"
        )

        # Create UnifiedTransformer
        self.transformer = UnifiedTransformer(ray_caster, floormap_transformer)

    def execute(self, detection_results: list[tuple[int, str, list[Detection]]]) -> list[FrameResult]:
        """座標変換とゾーン判定を実行

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
        """変換結果を Detection に適用

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
        """統計情報をログ出力

        Args:
            total: 総検出数
            success: 成功数
            errors: エラー数
            out_of_bounds: 範囲外数
            zone_classified: ゾーン分類数
        """
        if total > 0:
            self.logger.info("=" * 80)
            self.logger.info("Phase 3 Statistics (High-Precision Pipeline):")
            self.logger.info(f"  Total Detections: {total}")
            self.logger.info(f"  Transform Success: {success} ({success / total * 100:.1f}%)")
            self.logger.info(f"  Transform Errors (Horizon/Invalid): {errors} ({errors / total * 100:.1f}%)")
            self.logger.info(f"  Out of Bounds: {out_of_bounds} ({out_of_bounds / total * 100:.1f}%)")
            self.logger.info(f"  Zone Classified: {zone_classified} ({zone_classified / total * 100:.1f}%)")
            self.logger.info("=" * 80)

    def export_results(self, frame_results: list[FrameResult], output_path: Path) -> None:
        """座標変換結果をJSON形式で出力

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
        """リソースを解放"""
        self.transformer = None
        self.zone_classifier = None
