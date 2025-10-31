"""Manual pipeline runner for verification with a handful of frames.

指定した少数のフレームに対して検出・座標変換・集計・可視化を実行し、
モジュールの動作確認を行うためのユーティリティスクリプト。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregator import Aggregator
from src.config_manager import ConfigManager
from src.coordinate_transformer import CoordinateTransformer
from src.data_models import FrameResult
from src.floormap_visualizer import FloormapVisualizer
from src.visualizer import Visualizer
from src.video_processor import VideoProcessor
from src.vit_detector import ViTDetector
from src.zone_classifier import ZoneClassifier


logger = logging.getLogger(__name__)


# 既知のタイムライン（timestamp_verification_summary.json を参照）に基づく
# テスト用フレーム番号とタイムスタンプ
SAMPLE_FRAMES: List[Tuple[int, str]] = [
    (0, "16:05"),
    (30, "16:10"),
    (60, "16:15"),
]


def run_manual_pipeline(config_path: str = "config.yaml") -> None:
    """手動パイプラインを実行する。"""

    config = ConfigManager(config_path)

    output_dir = Path(config.get("output.directory", "output")).resolve()
    detections_dir = output_dir / "detections" / "manual"
    floormap_dir = output_dir / "floormaps" / "manual"
    graphs_dir = output_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)
    floormap_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # モジュールの初期化
    video_path = config.get("video.input_path")
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    detector = ViTDetector(
        model_name=config.get("detection.model_name"),
        confidence_threshold=config.get("detection.confidence_threshold", 0.5),
        device=config.get("detection.device"),
    )
    detector.load_model()

    coordinate_transformer = CoordinateTransformer(
        config.get("homography.matrix"),
        config.get("floormap"),
    )

    zones = config.get("zones", [])
    zone_classifier = ZoneClassifier(zones, allow_overlap=False)

    aggregator = Aggregator()
    visualizer = Visualizer(debug_mode=False)
    floormap_visualizer = FloormapVisualizer(
        config.get("floormap.image_path"),
        config.get("floormap"),
        zones,
        config.get("camera"),
    )

    frame_results: List[FrameResult] = []
    coordinate_records = []

    try:
        for frame_number, timestamp_label in SAMPLE_FRAMES:
            frame = video_processor.get_frame(frame_number)
            if frame is None:
                logger.warning("フレーム %d を取得できませんでした", frame_number)
                continue

            detections = detector.detect(frame)

            for detection in detections:
                try:
                    floor_coords = coordinate_transformer.transform(detection.camera_coords)
                    detection.floor_coords = floor_coords
                    detection.floor_coords_mm = coordinate_transformer.pixel_to_mm(floor_coords)
                    detection.zone_ids = zone_classifier.classify(floor_coords)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("座標変換に失敗: frame=%d err=%s", frame_number, exc)
                    detection.floor_coords = None
                    detection.floor_coords_mm = None
                    detection.zone_ids = []

            zone_counts = aggregator.aggregate_frame(timestamp_label, detections)

            frame_result = FrameResult(
                frame_number=frame_number,
                timestamp=timestamp_label,
                detections=detections,
                zone_counts=zone_counts,
            )
            frame_results.append(frame_result)

            # 検出結果の可視化
            detection_image = visualizer.draw_detections(frame, detections)
            visualizer.save_image(
                detection_image,
                str(detections_dir / f"detection_manual_{timestamp_label.replace(':', '')}.jpg"),
            )

            # フロアマップ上に描画
            floormap_image = floormap_visualizer.visualize_frame(frame_result)
            floormap_visualizer.save_visualization(
                floormap_image,
                str(floormap_dir / f"floormap_manual_{timestamp_label.replace(':', '')}.png"),
            )

            # 座標出力用に保存
            for det in detections:
                record = {
                    "frame_number": frame_number,
                    "timestamp": timestamp_label,
                    "confidence": det.confidence,
                    "bbox": {
                        "x": det.bbox[0],
                        "y": det.bbox[1],
                        "width": det.bbox[2],
                        "height": det.bbox[3],
                    },
                    "camera_coords": {
                        "x": det.camera_coords[0],
                        "y": det.camera_coords[1],
                    },
                    "floor_coords": {
                        "x": det.floor_coords[0],
                        "y": det.floor_coords[1],
                    }
                    if det.floor_coords is not None
                    else None,
                    "floor_coords_mm": {
                        "x": det.floor_coords_mm[0],
                        "y": det.floor_coords_mm[1],
                    }
                    if det.floor_coords_mm is not None
                    else None,
                    "zones": det.zone_ids,
                }
                coordinate_records.append(record)

        # 集計結果の可視化
        graphs_generated = {
            "time_series": visualizer.plot_time_series(
                aggregator,
                str(graphs_dir / "time_series_manual.png"),
            ),
            "statistics": visualizer.plot_zone_statistics(
                aggregator,
                str(graphs_dir / "statistics_manual.png"),
            ),
            "heatmap": visualizer.plot_heatmap(
                aggregator,
                str(graphs_dir / "heatmap_manual.png"),
            ),
        }

        logger.info("グラフ生成結果: %s", graphs_generated)

        # CSV/JSON 出力
        aggregator.export_csv(str(output_dir / "zone_counts_manual.csv"))
        with open(output_dir / "coordinate_transformations_manual.json", "w", encoding="utf-8") as fp:
            json.dump(coordinate_records, fp, indent=2, ensure_ascii=False)

    finally:
        video_processor.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_manual_pipeline()

