"""
Dagster PoC: 既存 PipelineOrchestrator を 1 ジョブとして包むだけの軽量セットアップ。

使い方:
  export POC_CONFIG=config/calibration_template.yaml
  dagit -m tools.dagster_poc

備考:
 - Dagster が未インストールの場合は ImportError を避け、起動時にヒントを表示する。
 - 実行負荷を抑えるため、動画は短尺/フレーム間引き設定を推奨。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dagster import Definitions, In, Out, job, op
except Exception as e:  # pragma: no cover - 任意依存
    raise SystemExit("Dagster がインストールされていません。pip install dagster dagit を実行してください。") from e

from src.config import ConfigManager
from src.pipeline import PipelineOrchestrator
from src.utils import setup_logging


@op(
    ins={"config_path": In(str, description="設定ファイルパス")},
    out=Out(str, description="生成された summary.json のパス"),
    description="PipelineOrchestrator をフル実行する PoC 用 op",
)
def run_full_pipeline(context, config_path: str) -> str:
    setup_logging(debug=False)
    logger = logging.getLogger("dagster_poc")
    config = ConfigManager(config_path)
    if not config.validate():
        raise ValueError("設定ファイルの検証に失敗しました")

    orchestrator = PipelineOrchestrator(config, logger)
    orchestrator.setup_output_directories(use_session_management=True, args={})

    video_path = config.get("video.input_path")
    extraction_results = orchestrator.extract_frames(video_path)
    sample_frames = orchestrator.prepare_frames_for_detection(extraction_results, video_path)
    detection_results, detector_phase = orchestrator.run_detection(sample_frames)
    tracked_results, _ = orchestrator.run_tracking(detection_results, sample_frames, detection_phase=detector_phase)
    frame_results, _ = orchestrator.run_transform(tracked_results)
    _, aggregator = orchestrator.run_aggregation(frame_results)
    orchestrator.run_visualization(aggregator, frame_results)
    orchestrator.save_session_summary(extraction_results, detection_results, frame_results, aggregator)
    summary_path = (
        orchestrator.session_dir / "summary.json" if orchestrator.session_dir else Path("output/summary.json")
    )
    context.log.info(f"summary: {summary_path}")
    return str(summary_path)


@job
def poc_job():
    cfg = os.environ.get("POC_CONFIG", "config/calibration_template.yaml")
    run_full_pipeline(config_path=cfg)


defs = Definitions(jobs=[poc_job])
