"""Transform and zone classification phase of the pipeline."""

import json
from pathlib import Path

from tqdm import tqdm

from src.models import Detection, FrameResult
from src.pipeline.base_phase import BasePhase
from src.transform import CoordinateTransformer
from src.zone import ZoneClassifier


class TransformPhase(BasePhase):
    """座標変換とゾーン判定フェーズ"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.coordinate_transformer: CoordinateTransformer | None = None
        self.zone_classifier: ZoneClassifier | None = None

    def initialize(self) -> None:
        """座標変換器とゾーン分類器を初期化"""
        self.log_phase_start("フェーズ3: 座標変換とゾーン判定")

        # CoordinateTransformerの初期化
        homography_matrix = self.config.get("homography.matrix")
        floormap_config = self.config.get("floormap")

        if homography_matrix is None:
            raise ValueError("ホモグラフィ行列が設定されていません")

        self.coordinate_transformer = CoordinateTransformer(homography_matrix, floormap_config)
        self.logger.info("CoordinateTransformer initialized")

        # ZoneClassifierの初期化
        zones = self.config.get("zones", [])
        if not zones:
            self.logger.warning("ゾーン定義が設定されていません")

        self.zone_classifier = ZoneClassifier(zones, allow_overlap=False)
        self.logger.info(
            "ZoneClassifier initialized with %d zones (allow_overlap=%s)",
            len(zones),
            False,
        )

    def execute(self, detection_results: list[tuple[int, str, list[Detection]]]) -> list[FrameResult]:
        """座標変換とゾーン判定を実行

        Args:
            detection_results: 検出結果のリスト [(frame_num, timestamp, detections), ...]

        Returns:
            FrameResultのリスト
        """
        if self.coordinate_transformer is None or self.zone_classifier is None:
            raise RuntimeError("変換器または分類器が初期化されていません。initialize()を先に呼び出してください。")

        frame_results: list[FrameResult] = []

        # 統計情報の収集
        transform_errors = 0
        out_of_bounds_count = 0
        zone_classification_count = 0

        for frame_num, timestamp, detections in tqdm(detection_results, desc="座標変換・ゾーン判定中"):
            # 各検出結果に対して座標変換とゾーン判定を適用
            for detection in detections:
                try:
                    # フロアマップ座標に変換（ピクセル単位）
                    floor_coords = self.coordinate_transformer.transform(detection.camera_coords)
                    detection.floor_coords = floor_coords

                    # mm単位にも変換
                    floor_coords_mm = self.coordinate_transformer.pixel_to_mm(floor_coords)
                    detection.floor_coords_mm = floor_coords_mm

                    # 座標がフロアマップ範囲内かチェック
                    if not self.coordinate_transformer.is_within_bounds(floor_coords):
                        out_of_bounds_count += 1
                        self.logger.debug(
                            f"フレーム #{frame_num}: 座標が範囲外: "
                            f"カメラ座標={detection.camera_coords}, "
                            f"フロアマップ座標={floor_coords}"
                        )

                    # ゾーン判定
                    zone_ids = self.zone_classifier.classify(floor_coords)
                    detection.zone_ids = zone_ids
                    if zone_ids:
                        zone_classification_count += 1

                except ValueError as e:
                    # 座標変換エラー（詳細ログはCoordinateTransformerで出力済み）
                    transform_errors += 1
                    self.logger.warning(
                        f"フレーム #{frame_num}: 座標変換エラー - "
                        f"カメラ座標={detection.camera_coords}, "
                        f"エラー={e}"
                    )
                    detection.floor_coords = None
                    detection.floor_coords_mm = None
                    detection.zone_ids = []
                except Exception as e:
                    # その他の予期しないエラー
                    transform_errors += 1
                    self.logger.error(
                        f"フレーム #{frame_num}: 予期しないエラー - "
                        f"カメラ座標={detection.camera_coords}, "
                        f"エラー={type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    detection.floor_coords = None
                    detection.floor_coords_mm = None
                    detection.zone_ids = []

            # FrameResultを作成（集計は次のフェーズで実行）
            frame_result = FrameResult(
                frame_number=frame_num,
                timestamp=timestamp,
                detections=detections,
                zone_counts={},  # 後で集計
            )
            frame_results.append(frame_result)

        # 統計情報をログ出力
        total_detections = sum(len(detections) for _, _, detections in detection_results)
        if total_detections > 0:
            self.logger.info("=" * 80)
            self.logger.info("座標変換・ゾーン判定統計:")
            self.logger.info(f"  総検出数: {total_detections}")
            self.logger.info(f"  変換エラー数: {transform_errors} ({transform_errors/total_detections*100:.1f}%)")
            self.logger.info(f"  範囲外座標数: {out_of_bounds_count} ({out_of_bounds_count/total_detections*100:.1f}%)")
            self.logger.info(
                f"  ゾーン分類成功数: {zone_classification_count} ({zone_classification_count/total_detections*100:.1f}%)"
            )
            self.logger.info("=" * 80)

        self.logger.info(f"座標変換とゾーン判定が完了: {len(frame_results)}フレーム")

        return frame_results

    def export_results(self, frame_results: list[FrameResult], output_path: Path) -> None:
        """座標変換結果をJSON形式で出力

        Args:
            frame_results: FrameResultのリスト
            output_path: 出力ディレクトリ
        """
        coordinate_data = []
        for frame_result in frame_results:
            frame_data = {
                "frame_number": frame_result.frame_number,
                "timestamp": frame_result.timestamp,
                "detections": [],
            }

            for detection in frame_result.detections:
                detection_data = {
                    "bbox": {
                        "x": float(detection.bbox[0]),
                        "y": float(detection.bbox[1]),
                        "width": float(detection.bbox[2]),
                        "height": float(detection.bbox[3]),
                    },
                    "confidence": float(detection.confidence),
                    "camera_coords": {
                        "x": float(detection.camera_coords[0]),
                        "y": float(detection.camera_coords[1]),
                    },
                }

                # フロアマップ座標が存在する場合のみ追加
                if detection.floor_coords is not None:
                    detection_data["floor_coords"] = {
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

                frame_data["detections"].append(detection_data)

            coordinate_data.append(frame_data)

        # JSONファイルに出力
        coordinate_output_path = output_path / "coordinate_transformations.json"
        try:
            with open(coordinate_output_path, "w", encoding="utf-8") as f:
                json.dump(coordinate_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"座標変換結果をJSONに出力しました: {coordinate_output_path}")
            self.logger.info(f"  出力フレーム数: {len(coordinate_data)}")
            self.logger.info(f"  総検出数: {sum(len(f['detections']) for f in coordinate_data)}")
        except Exception as e:
            self.logger.error(f"座標変換結果のJSON出力に失敗しました: {e}")

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self.coordinate_transformer = None
        self.zone_classifier = None
