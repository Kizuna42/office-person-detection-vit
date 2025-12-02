"""Transform and zone classification phase of the pipeline.

Phase 3: 座標変換（PWA/TPS/ホモグラフィ）によるカメラ座標→フロアマップ座標変換とゾーン判定

オプションでレンズ歪み補正を適用できます。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm import tqdm

from src.calibration import LensDistortionCorrector
from src.models import Detection, FrameResult
from src.pipeline.phases.base import BasePhase
from src.transform import (
    FloorMapConfig,
    HomographyTransformer,
    PiecewiseAffineTransformer,
    PWATransformResult,
    ThinPlateSplineTransformer,
    TransformResult,
)
from src.zone import ZoneClassifier

if TYPE_CHECKING:
    import logging

    from src.config import ConfigManager


# 座標変換器の型エイリアス
TransformerType = HomographyTransformer | PiecewiseAffineTransformer | ThinPlateSplineTransformer


class TransformPhase(BasePhase):
    """座標変換とゾーン判定フェーズ。

    複数の変換手法をサポート:
    - homography: 単一ホモグラフィ行列（高速、精度低）
    - piecewise_affine: Piecewise Affine変換（高精度、推奨）
    - thin_plate_spline: TPS変換（最高精度、計算コスト高）
    """

    SUPPORTED_METHODS = ("homography", "piecewise_affine", "thin_plate_spline")

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化。

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.transformer: TransformerType | None = None
        self.zone_classifier: ZoneClassifier | None = None
        self.transform_method: str = "homography"
        self.distortion_corrector: LensDistortionCorrector | None = None

    def _init_distortion_corrector(self) -> LensDistortionCorrector | None:
        """レンズ歪み補正器を初期化

        config.yaml の transform.lens_distortion セクションから設定を読み込みます。

        Returns:
            LensDistortionCorrector または None（無効時）
        """
        transform_config = self.config.get("transform", {})
        distortion_config = transform_config.get("lens_distortion", {})

        if not distortion_config.get("enabled", False):
            self.logger.info("Lens distortion correction is disabled")
            return None

        try:
            # 歪み係数を取得
            k1 = float(distortion_config.get("k1", 0.0))
            k2 = float(distortion_config.get("k2", 0.0))
            k3 = float(distortion_config.get("k3", 0.0))
            p1 = float(distortion_config.get("p1", 0.0))
            p2 = float(distortion_config.get("p2", 0.0))

            # カメラ行列パラメータ
            camera_config = self.config.get("camera_params", {})
            fx = float(distortion_config.get("focal_length_x", camera_config.get("focal_length", 1250.0)))
            fy = float(distortion_config.get("focal_length_y", camera_config.get("focal_length", 1250.0)))
            cx = float(distortion_config.get("center_x", 640.0))
            cy = float(distortion_config.get("center_y", 360.0))

            # 画像サイズ
            image_width = int(distortion_config.get("image_width", 1280))
            image_height = int(distortion_config.get("image_height", 720))

            from src.calibration import CameraIntrinsics, DistortionParams

            intrinsics = CameraIntrinsics(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=image_width,
                height=image_height,
                distortion=DistortionParams(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2),
            )

            corrector = LensDistortionCorrector(intrinsics)

            if corrector.enabled:
                self.logger.info("Lens distortion correction enabled:")
                self.logger.info(f"  k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}")
                self.logger.info(f"  p1={p1:.6f}, p2={p2:.6f}")
                self.logger.info(f"  Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            else:
                self.logger.info("Lens distortion correction: all coefficients are zero (disabled)")
                return None

            return corrector

        except Exception as e:
            self.logger.warning(f"Failed to initialize lens distortion corrector: {e}")
            return None

    def _create_floormap_config(self) -> FloorMapConfig:
        """フロアマップ設定を作成"""
        floormap_config = self.config.get("floormap", {})
        return FloorMapConfig(
            width_px=int(floormap_config.get("image_width", 1878)),
            height_px=int(floormap_config.get("image_height", 1369)),
            origin_x_px=float(floormap_config.get("image_origin_x", 7.0)),
            origin_y_px=float(floormap_config.get("image_origin_y", 9.0)),
            scale_x_mm_per_px=float(floormap_config.get("image_x_mm_per_pixel", 28.1926406926406)),
            scale_y_mm_per_px=float(floormap_config.get("image_y_mm_per_pixel", 28.241430700447)),
        )

    def _init_homography_transformer(self, fm_config: FloorMapConfig) -> HomographyTransformer:
        """ホモグラフィ変換器を初期化"""
        homography_config = self.config.get("homography", {})
        matrix = homography_config.get("matrix")

        if matrix is None:
            raise ValueError("homography.matrix が設定されていません")

        H = np.array(matrix, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"ホモグラフィ行列は3x3である必要があります: {H.shape}")

        transformer = HomographyTransformer(H, fm_config)
        self.logger.info("HomographyTransformer initialized")
        self.logger.info(f"  Matrix:\n{H}")

        return transformer

    def _init_pwa_transformer(self, fm_config: FloorMapConfig) -> PiecewiseAffineTransformer:
        """PWA変換器を初期化"""
        transform_config = self.config.get("transform", {})
        model_path = transform_config.get("model_path")
        correspondence_file = self.config.get("calibration", {}).get("correspondence_file")

        # モデルファイルがあれば読み込み、なければ対応点から作成
        if model_path and Path(model_path).exists():
            self.logger.info(f"Loading PWA model from {model_path}")
            transformer = PiecewiseAffineTransformer.load(
                model_path,
                fm_config,
                distortion_corrector=self.distortion_corrector,
            )
        elif correspondence_file and Path(correspondence_file).exists():
            self.logger.info(f"Creating PWA transformer from {correspondence_file}")
            transformer = PiecewiseAffineTransformer.from_correspondence_file(
                correspondence_file,
                fm_config,
                distortion_corrector=self.distortion_corrector,
            )
            # モデルを保存
            if model_path:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                transformer.save(model_path)
                self.logger.info(f"PWA model saved to {model_path}")
        else:
            raise ValueError("PWA変換には model_path または correspondence_file が必要です")

        # 訓練誤差を表示
        info = transformer.get_info()
        training_error = info.get("training_error", {})
        self.logger.info(
            f"PiecewiseAffineTransformer initialized: {info['num_points']} points, {info['num_triangles']} triangles"
        )
        self.logger.info(
            f"  Training RMSE: {training_error.get('rmse', 0):.2f}px, Max: {training_error.get('max_error', 0):.2f}px"
        )

        return transformer

    def _init_tps_transformer(self, fm_config: FloorMapConfig) -> ThinPlateSplineTransformer:
        """TPS変換器を初期化"""
        correspondence_file = self.config.get("calibration", {}).get("correspondence_file")

        if not correspondence_file or not Path(correspondence_file).exists():
            raise ValueError("TPS変換には correspondence_file が必要です")

        self.logger.info(f"Creating TPS transformer from {correspondence_file}")
        transformer = ThinPlateSplineTransformer.from_correspondence_file(
            correspondence_file,
            fm_config,
            regularization=0.0,
            distortion_corrector=self.distortion_corrector,
        )

        info = transformer.get_info()
        training_error = info.get("training_error", {})
        self.logger.info(f"ThinPlateSplineTransformer initialized: {info['num_points']} points")
        self.logger.info(
            f"  Training RMSE: {training_error.get('rmse', 0):.2f}px, Max: {training_error.get('max_error', 0):.2f}px"
        )

        return transformer

    def initialize(self) -> None:
        """座標変換器とゾーン分類器を初期化。"""
        # 変換方式を取得
        transform_config = self.config.get("transform", {})
        self.transform_method = transform_config.get("method", "homography")

        if self.transform_method not in self.SUPPORTED_METHODS:
            self.logger.warning(f"Unknown transform method '{self.transform_method}', falling back to 'homography'")
            self.transform_method = "homography"

        self.log_phase_start(f"フェーズ3: 座標変換とゾーン判定 ({self.transform_method})")

        # レンズ歪み補正器を初期化（PWA/TPSでのみ有効）
        if self.transform_method in ["piecewise_affine", "thin_plate_spline"]:
            self.distortion_corrector = self._init_distortion_corrector()

        # フロアマップ設定
        fm_config = self._create_floormap_config()

        # 変換器を初期化
        if self.transform_method == "piecewise_affine":
            self.transformer = self._init_pwa_transformer(fm_config)
        elif self.transform_method == "thin_plate_spline":
            self.transformer = self._init_tps_transformer(fm_config)
        else:
            self.transformer = self._init_homography_transformer(fm_config)

        # ゾーン分類器を初期化
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
        extrapolated_count = 0

        for frame_num, timestamp, detections in tqdm(detection_results, desc="座標変換・ゾーン判定中"):
            total_detections += len(detections)

            # バッチ処理で変換
            if detections:
                bboxes = [d.bbox for d in detections]
                batch_results = cast(
                    "list[TransformResult | PWATransformResult]",
                    self.transformer.transform_batch(bboxes),
                )

                for detection, result in zip(detections, batch_results, strict=True):
                    self._apply_transform_result(detection, result)

                    if result.is_valid:
                        transform_success += 1

                        if not result.is_within_bounds:
                            out_of_bounds_count += 1

                        # PWAの外挿をカウント
                        if hasattr(result, "is_extrapolated") and result.is_extrapolated:
                            extrapolated_count += 1

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
            extrapolated_count,
        )

        return frame_results

    def _apply_transform_result(
        self,
        detection: Detection,
        result: TransformResult | PWATransformResult,
    ) -> None:
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
        extrapolated: int = 0,
    ) -> None:
        """統計情報をログ出力。

        Args:
            total: 総検出数
            success: 成功数
            errors: エラー数
            out_of_bounds: 範囲外数
            zone_classified: ゾーン分類数
            extrapolated: 外挿数（PWA/TPS）
        """
        if total > 0:
            self.logger.info("=" * 80)
            self.logger.info(f"Phase 3 Statistics ({self.transform_method}):")
            self.logger.info(f"  Total Detections: {total}")
            self.logger.info(f"  Transform Success: {success} ({success / total * 100:.1f}%)")
            self.logger.info(f"  Transform Errors: {errors} ({errors / total * 100:.1f}%)")
            self.logger.info(f"  Out of Bounds: {out_of_bounds} ({out_of_bounds / total * 100:.1f}%)")
            self.logger.info(f"  Zone Classified: {zone_classified} ({zone_classified / total * 100:.1f}%)")
            if extrapolated > 0:
                self.logger.info(f"  Extrapolated (PWA): {extrapolated} ({extrapolated / total * 100:.1f}%)")
            self.logger.info("=" * 80)

    def _round_coord(self, value: float, precision: int) -> float:
        """座標値を指定精度で丸める。

        Args:
            value: 座標値
            precision: 小数点以下の桁数

        Returns:
            丸められた座標値
        """
        return round(value, precision)

    def export_results(self, frame_results: list[FrameResult], output_path: Path) -> None:
        """座標変換結果をJSON形式で出力。

        Args:
            frame_results: フレーム結果のリスト
            output_path: 出力ディレクトリパス
        """
        # JSON最適化設定を取得
        json_opt = self.config.get("output.json_optimization", {})
        opt_enabled = json_opt.get("enabled", False)
        precision = json_opt.get("coordinate_precision", 1) if opt_enabled else 6
        compact_keys = json_opt.get("compact_keys", False) and opt_enabled
        exclude_px_coords = json_opt.get("exclude_px_coords", False) and opt_enabled

        coordinate_data = []
        for frame_result in frame_results:
            # フレームキー名
            frame_key = "idx" if compact_keys else "frame_number"
            ts_key = "ts" if compact_keys else "timestamp"
            det_key = "det" if compact_keys else "detections"

            frame_data: dict[str, Any] = {
                frame_key: frame_result.frame_number,
                ts_key: frame_result.timestamp,
                det_key: [],
            }

            for detection in frame_result.detections:
                # キー名の決定
                bbox_key = "bb" if compact_keys else "bbox"
                conf_key = "conf" if compact_keys else "confidence"
                cam_key = "cam" if compact_keys else "camera_coords"
                floor_px_key = "floor_px" if compact_keys else "floor_coords_px"
                floor_mm_key = "floor_mm" if compact_keys else "floor_coords_mm"
                zone_key = "zones" if compact_keys else "zone_ids"
                id_key = "id" if compact_keys else "track_id"

                # bbox: 配列形式でコンパクトに
                if compact_keys:
                    detection_data: dict[str, Any] = {
                        bbox_key: [
                            self._round_coord(detection.bbox[0], precision),
                            self._round_coord(detection.bbox[1], precision),
                            self._round_coord(detection.bbox[2], precision),
                            self._round_coord(detection.bbox[3], precision),
                        ],
                        conf_key: self._round_coord(detection.confidence, 2),
                    }
                else:
                    detection_data = {
                        bbox_key: {
                            "x": self._round_coord(detection.bbox[0], precision),
                            "y": self._round_coord(detection.bbox[1], precision),
                            "width": self._round_coord(detection.bbox[2], precision),
                            "height": self._round_coord(detection.bbox[3], precision),
                        },
                        conf_key: self._round_coord(detection.confidence, 3),
                    }

                if detection.camera_coords is not None:
                    if compact_keys:
                        detection_data[cam_key] = [
                            self._round_coord(detection.camera_coords[0], precision),
                            self._round_coord(detection.camera_coords[1], precision),
                        ]
                    else:
                        detection_data[cam_key] = {
                            "x": self._round_coord(detection.camera_coords[0], precision),
                            "y": self._round_coord(detection.camera_coords[1], precision),
                        }

                # floor_coords_px（オプションで除外可能）
                if detection.floor_coords is not None and not exclude_px_coords:
                    if compact_keys:
                        detection_data[floor_px_key] = [
                            self._round_coord(detection.floor_coords[0], precision),
                            self._round_coord(detection.floor_coords[1], precision),
                        ]
                    else:
                        detection_data[floor_px_key] = {
                            "x": self._round_coord(detection.floor_coords[0], precision),
                            "y": self._round_coord(detection.floor_coords[1], precision),
                        }

                if detection.floor_coords_mm is not None:
                    if compact_keys:
                        detection_data[floor_mm_key] = [
                            self._round_coord(detection.floor_coords_mm[0], precision),
                            self._round_coord(detection.floor_coords_mm[1], precision),
                        ]
                    else:
                        detection_data[floor_mm_key] = {
                            "x": self._round_coord(detection.floor_coords_mm[0], precision),
                            "y": self._round_coord(detection.floor_coords_mm[1], precision),
                        }

                if detection.zone_ids:
                    detection_data[zone_key] = detection.zone_ids

                if hasattr(detection, "track_id") and detection.track_id is not None:
                    detection_data[id_key] = detection.track_id

                frame_data[det_key].append(detection_data)

            coordinate_data.append(frame_data)

        # メタデータを追加
        method_key = "method" if compact_keys else "transform_method"
        info_key = "info" if compact_keys else "transformer_info"

        # transformer_infoから訓練誤差の詳細を除外してコンパクトに
        transformer_info = self.transformer.get_info() if self.transformer else {}
        if compact_keys and transformer_info:
            transformer_info = {
                "method": transformer_info.get("method", self.transform_method),
                "points": transformer_info.get("num_points", 0),
                "triangles": transformer_info.get("num_triangles", 0),
            }

        output_data = {
            method_key: self.transform_method,
            info_key: transformer_info,
            "frames": coordinate_data,
        }

        coordinate_output_path = output_path / "coordinate_transformations.json"
        try:
            # コンパクトモードではインデントを減らす
            indent = None if compact_keys else 2
            with open(coordinate_output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=indent, ensure_ascii=False, default=str)
            self.logger.info(f"Saved coordinate transformations to {coordinate_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")

    def cleanup(self) -> None:
        """リソースを解放。"""
        self.transformer = None
        self.zone_classifier = None
