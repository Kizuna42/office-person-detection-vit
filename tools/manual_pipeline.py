"""Manual pipeline runner for verification with a handful of frames.

指定した少数のフレームに対して検出・座標変換・集計・可視化を実行し、
モジュールの動作確認を行うためのユーティリティスクリプト。

パイプラインクラスを使用して実装を簡潔化しています。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.pipeline import (
    AggregationPhase,
    DetectionPhase,
    TransformPhase,
    VisualizationPhase,
)
from src.utils import setup_logging, setup_output_directories
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


# 既知のタイムライン（timestamp_verification_summary.json を参照）に基づく
# テスト用フレーム番号とタイムスタンプ
SAMPLE_FRAMES: List[Tuple[int, str]] = [
    (0, "16:05"),
    (30, "16:10"),
    (60, "16:15"),
]


def run_manual_pipeline(config_path: str = "config.yaml") -> None:
    """手動パイプラインを実行する。

    Args:
        config_path: 設定ファイルのパス
    """
    # 設定の読み込み
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output")).resolve()
    manual_output_dir = output_dir / "manual"
    setup_output_directories(manual_output_dir)

    # ロギング設定
    setup_logging(debug_mode=False, output_dir=str(output_dir))
    logger.info("=" * 80)
    logger.info("手動パイプライン実行開始")
    logger.info("=" * 80)

    # ビデオプロセッサの初期化
    video_path = config.get("video.input_path")
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    detector = None

    try:
        # サンプルフレームの準備
        sample_frames: List[Tuple[int, str, np.ndarray]] = []
        for frame_number, timestamp_label in SAMPLE_FRAMES:
            frame = video_processor.get_frame(frame_number)
            if frame is None:
                logger.warning("フレーム %d を取得できませんでした", frame_number)
                continue
            sample_frames.append((frame_number, timestamp_label, frame))

        if not sample_frames:
            logger.error("有効なフレームがありません")
            return

        logger.info(f"処理対象フレーム数: {len(sample_frames)}")

        # フェーズ1: 人物検出
        detection_phase = DetectionPhase(config, logger)
        detection_phase.initialize()
        detector = detection_phase.detector

        detection_results = detection_phase.execute(sample_frames)
        detection_phase.log_statistics(detection_results, manual_output_dir)

        # フェーズ2: 座標変換とゾーン判定
        transform_phase = TransformPhase(config, logger)
        transform_phase.initialize()
        frame_results = transform_phase.execute(detection_results)
        transform_phase.export_results(frame_results, manual_output_dir)

        # フェーズ3: 集計
        aggregation_phase = AggregationPhase(config, logger)
        aggregator = aggregation_phase.execute(frame_results, manual_output_dir)

        # フェーズ4: 可視化
        visualization_phase = VisualizationPhase(config, logger)
        visualization_phase.execute(aggregator, frame_results, manual_output_dir)

        # 座標変換結果をJSON形式で出力（デバッグ用）
        coordinate_records = []
        for frame_result in frame_results:
            for det in frame_result.detections:
                record = {
                    "frame_number": frame_result.frame_number,
                    "timestamp": frame_result.timestamp,
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
                    "floor_coords": (
                        {"x": det.floor_coords[0], "y": det.floor_coords[1]}
                        if det.floor_coords is not None
                        else None
                    ),
                    "floor_coords_mm": (
                        {"x": det.floor_coords_mm[0], "y": det.floor_coords_mm[1]}
                        if det.floor_coords_mm is not None
                        else None
                    ),
                    "zones": det.zone_ids,
                }
                coordinate_records.append(record)

        coordinate_json_path = manual_output_dir / "coordinate_transformations_manual.json"
        with open(coordinate_json_path, "w", encoding="utf-8") as fp:
            json.dump(coordinate_records, fp, indent=2, ensure_ascii=False)
        logger.info(f"座標変換結果をJSONに出力しました: {coordinate_json_path}")

        logger.info("=" * 80)
        logger.info("手動パイプライン実行完了")
        logger.info("=" * 80)

    finally:
        video_processor.release()
        if detector is not None:
            from src.utils import cleanup_resources

            cleanup_resources(detector=detector, logger=logger)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_manual_pipeline()

