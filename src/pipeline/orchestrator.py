"""Pipeline orchestrator for coordinating all phases."""

from datetime import datetime
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.aggregation import Aggregator
from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.frame_extraction_pipeline import FrameExtractionPipeline
from src.pipeline.phases import (
    AggregationPhase,
    DetectionPhase,
    TrackingPhase,
    TransformPhase,
    VisualizationPhase,
)
from src.services.checkpoint_service import CheckpointService
from src.services.output_service import OutputService
from src.services.perf_service import PerformanceService
from src.utils import OutputManager, cleanup_resources, setup_output_directories
from src.video import VideoProcessor


class PipelineOrchestrator:
    """パイプライン全体を統括するオーケストレーター"""

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        self.config = config
        self.logger = logger
        self.session_dir: Path | None = None
        self.output_path: Path = Path(config.get("output.directory", "output"))
        # 互換性用の公開属性
        self.output_manager: OutputManager | None = None
        self.output_service = OutputService(self.logger, self.output_path)
        self.performance_service = PerformanceService()
        # 既存属性との互換性を維持
        self.performance_monitor = self.performance_service.monitor
        self.checkpoint_service: CheckpointService | None = None

    def setup_output_directories(self, use_session_management: bool, args: dict | None = None) -> None:
        """出力ディレクトリをセットアップ

        Args:
            use_session_management: セッション管理を使用するか
            args: コマンドライン引数（オプション、メタデータ保存用）
        """
        self.output_path = self.output_service.setup(use_session_management, self.config, args)
        self.session_dir = self.output_service.session_dir
        self.output_manager = self.output_service.output_manager

        if self.session_dir:
            self.checkpoint_service = CheckpointService(self.session_dir)
        else:
            # 従来の構造を維持するために空のチェックポイントサービスは持たない
            setup_output_directories(self.output_path)

    def get_phase_output_dir(self, phase_name: str) -> Path:
        """フェーズごとの出力ディレクトリを取得

        Args:
            phase_name: フェーズ名（例: "01_extraction"）

        Returns:
            出力ディレクトリのパス
        """
        if self.session_dir:
            return self.session_dir / phase_name
        return self.output_path

    def extract_frames(self, video_path: str, start_time: str | None = None, end_time: str | None = None) -> list[dict]:
        """フレーム抽出を実行

        Args:
            video_path: 動画ファイルのパス
            start_time: 開始時刻（HH:MM形式、オプション）
            end_time: 終了時刻（HH:MM形式、オプション）

        Returns:
            抽出結果のリスト
        """
        with self.performance_monitor.measure("01_extraction"):
            # 設定からパラメータを取得
            timestamp_config = self.config.get("timestamp", {})
            extraction_config = timestamp_config.get("extraction", {})
            sampling_config = timestamp_config.get("sampling", {})
            target_config = timestamp_config.get("target", {})
            ocr_config = self.config.get("ocr", {})

            # 開始・終了日時の取得
            start_datetime, end_datetime = self._parse_datetime_range(target_config, start_time, end_time)

            # 出力ディレクトリの決定
            extraction_output_dir = self.get_phase_output_dir("01_extraction")

            # パイプライン初期化
            pipeline = FrameExtractionPipeline(
                video_path=video_path,
                output_dir=str(extraction_output_dir),
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                interval_minutes=self.config.get("video.frame_interval_minutes", 5),
                tolerance_seconds=self.config.get("video.tolerance_seconds", 10.0),
                confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
                coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
                fine_search_window_seconds=sampling_config.get("search_window_seconds", 60.0),
                fine_interval_seconds=sampling_config.get("fine_interval_seconds", 0.1),
                fps=self.config.get("video.fps", 30.0),
                time_compression_ratio=self.config.get("video.time_compression_ratio", 1.0),
                roi_config=extraction_config.get("roi"),
                enabled_ocr_engines=ocr_config.get("engines"),
                use_improved_validator=extraction_config.get("use_improved_validator", False),
                base_tolerance_seconds=extraction_config.get("validator", {}).get("base_tolerance_seconds", 10.0),
                history_size=extraction_config.get("validator", {}).get("history_size", 10),
                z_score_threshold=extraction_config.get("validator", {}).get("z_score_threshold", 2.0),
                use_weighted_consensus=extraction_config.get("use_weighted_consensus", False),
                use_voting_consensus=extraction_config.get("use_voting_consensus", False),
            )

            # フレーム抽出モードに応じて実行
            extraction_mode = timestamp_config.get("extraction_mode", "manual_targets")
            if extraction_mode == "auto_targets":
                auto_targets_config = timestamp_config.get("auto_targets", {})
                self.logger.info("自動目標タイムスタンプ生成モードでフレーム抽出を実行します")
                results = pipeline.run_with_auto_targets(
                    max_frames=auto_targets_config.get("max_frames"),
                    disable_validation=auto_targets_config.get("disable_validation", False),
                )
            else:
                self.logger.info("手動目標タイムスタンプ指定モードでフレーム抽出を実行します")
                results = pipeline.run()

            # OCRキャッシュ統計をログ出力
            cache_stats = pipeline.extractor.get_cache_stats()
            if cache_stats["cache_hits"] + cache_stats["cache_misses"] > 0:
                self.logger.info("=" * 80)
                self.logger.info("OCRキャッシュ統計:")
                self.logger.info(f"  キャッシュサイズ: {cache_stats['cache_size']}エントリ")
                self.logger.info(f"  キャッシュヒット: {cache_stats['cache_hits']}回")
                self.logger.info(f"  キャッシュミス: {cache_stats['cache_misses']}回")
                self.logger.info(f"  ヒット率: {cache_stats['hit_rate']:.1f}%")
                self.logger.info("=" * 80)

            return results

    def prepare_frames_for_detection(
        self, extraction_results: list[dict], video_path: str
    ) -> list[tuple[int, str, np.ndarray]]:
        """検出用にフレームを準備

        Args:
            extraction_results: 抽出結果のリスト
            video_path: 動画ファイルのパス

        Returns:
            検出用フレームのリスト [(frame_idx, timestamp, frame), ...]
        """
        import cv2

        sample_frames = []
        video_processor = None

        try:
            for result in tqdm(extraction_results, desc="フレーム準備中"):
                frame = result.get("frame")

                # フレームがない場合、frame_pathから読み込みを試行
                if frame is None:
                    frame_path = result.get("frame_path")
                    if frame_path and Path(frame_path).exists():
                        frame = cv2.imread(frame_path)
                        self.logger.debug(f"フレームをディスクから読み込み: {frame_path}")

                # それでもない場合は動画から再取得
                if frame is None:
                    if video_processor is None:
                        video_processor = VideoProcessor(video_path)
                        video_processor.open()

                    frame = video_processor.get_frame(result["frame_idx"])
                    if frame is None:
                        self.logger.warning(f"フレーム {result['frame_idx']} を取得できませんでした")
                        continue

                timestamp_str = result["timestamp"].strftime("%Y/%m/%d %H:%M:%S")
                sample_frames.append((result["frame_idx"], timestamp_str, frame))

        finally:
            if video_processor:
                video_processor.release()

        self.logger.info(f"後続処理用フレーム準備完了: {len(sample_frames)}フレーム")
        return sample_frames

    def run_detection(
        self, sample_frames: list[tuple[int, str, np.ndarray]]
    ) -> tuple[list[tuple[int, str, list[Detection]]], DetectionPhase]:
        """人物検出を実行

        Args:
            sample_frames: 検出用フレームのリスト

        Returns:
            (検出結果のリスト, DetectionPhaseインスタンス)
        """
        with self.performance_monitor.measure("02_detection"):
            detection_phase = DetectionPhase(self.config, self.logger)
            detection_phase.initialize()

            output_dir = self.get_phase_output_dir("02_detection")
            detection_phase.output_path = output_dir

            detection_results = detection_phase.execute(sample_frames)
            detection_phase.log_statistics(detection_results, output_dir)

            # チェックポイント保存
            if self.checkpoint_service:
                self.checkpoint_service.save(
                    "02_detection",
                    {"total_detections": sum(len(dets) for _, _, dets in detection_results)},
                )

            return detection_results, detection_phase

    def run_tracking(
        self,
        detection_results: list[tuple[int, str, list[Detection]]],
        sample_frames: list[tuple[int, str, np.ndarray]],
        detection_phase: DetectionPhase | None = None,
    ) -> tuple[list[tuple[int, str, list[Detection]]], TrackingPhase]:
        """追跡処理を実行

        Args:
            detection_results: 検出結果のリスト
            sample_frames: サンプルフレームのリスト [(frame_num, timestamp, frame), ...]
            detection_phase: 検出フェーズインスタンス（検出器共有用、オプション）

        Returns:
            (track_idが割り当てられた検出結果のリスト, TrackingPhaseインスタンス)
        """
        # 追跡が無効な場合は検出結果をそのまま返す
        if not self.config.get("tracking.enabled", False):
            self.logger.info("追跡機能が無効です（tracking.enabled=false）")
            tracking_phase = TrackingPhase(self.config, self.logger)
            tracking_phase.initialize()  # 初期化のみ（実行はスキップ）
            return detection_results, tracking_phase

        with self.performance_monitor.measure("03_tracking"):
            tracking_phase = TrackingPhase(self.config, self.logger)

            # 検出フェーズから検出器を共有（メモリ効率化）
            if detection_phase is not None and detection_phase.detector is not None:
                tracking_phase.set_detector(detection_phase.detector)
                self.logger.info("検出器をPhase 2からPhase 3に共有しました")

            tracking_phase.initialize()

            output_dir = self.get_phase_output_dir("03_tracking")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 追跡実行（sample_framesをそのまま渡す）
            tracked_results = tracking_phase.execute(detection_results, sample_frames)

            # 結果をエクスポート
            tracking_phase.export_results(output_dir)

            # チェックポイント保存
            if self.checkpoint_service:
                self.checkpoint_service.save(
                    "03_tracking",
                    {"total_tracks": len(tracking_phase.get_tracks())},
                )

            return tracked_results, tracking_phase

    def run_transform(
        self, detection_results: list[tuple[int, str, list[Detection]]]
    ) -> tuple[list[FrameResult], TransformPhase]:
        """座標変換とゾーン判定を実行

        Args:
            detection_results: 検出結果のリスト

        Returns:
            (FrameResultのリスト, TransformPhaseインスタンス)
        """
        with self.performance_monitor.measure("04_transform"):
            transform_phase = TransformPhase(self.config, self.logger)
            transform_phase.initialize()

            output_dir = self.get_phase_output_dir("04_transform")
            frame_results = transform_phase.execute(detection_results)
            transform_phase.export_results(frame_results, output_dir)

            # チェックポイント保存
            if self.checkpoint_service:
                self.checkpoint_service.save(
                    "04_transform",
                    {"frames_processed": len(frame_results)},
                )

            return frame_results, transform_phase

    def run_aggregation(self, frame_results: list[FrameResult]) -> tuple[AggregationPhase, Aggregator]:
        """集計処理を実行

        Args:
            frame_results: FrameResultのリスト

        Returns:
            (AggregationPhaseインスタンス, Aggregatorインスタンス)
        """
        with self.performance_monitor.measure("05_aggregation"):
            aggregation_phase = AggregationPhase(self.config, self.logger)
            output_dir = self.get_phase_output_dir("05_aggregation")
            aggregator = aggregation_phase.execute(frame_results, output_dir)

            # チェックポイント保存
            if self.checkpoint_service:
                self.checkpoint_service.save(
                    "05_aggregation",
                    {"zones_count": len(aggregator.get_statistics())},
                )

            return aggregation_phase, aggregator

    def run_visualization(self, aggregator: Aggregator, frame_results: list[FrameResult]) -> None:
        """可視化を実行

        Args:
            aggregator: Aggregatorインスタンス
            frame_results: FrameResultのリスト
        """
        with self.performance_monitor.measure("06_visualization"):
            visualization_phase = VisualizationPhase(self.config, self.logger)
            output_dir = self.get_phase_output_dir("06_visualization")
            visualization_phase.execute(aggregator, frame_results, output_dir)

            # チェックポイント保存
            if self.checkpoint_service:
                self.checkpoint_service.save(
                    "06_visualization",
                    {"floormaps_generated": len(frame_results)},
                )

    def save_session_summary(
        self,
        extraction_results: list[dict],
        detection_results: list[tuple[int, str, list[Detection]]],
        frame_results: list[FrameResult],
        aggregator: Aggregator,
    ) -> None:
        """セッションサマリーを保存

        Args:
            extraction_results: 抽出結果のリスト
            detection_results: 検出結果のリスト
            frame_results: FrameResultのリスト
            aggregator: Aggregatorインスタンス
        """
        if not self.session_dir or not self.output_service.output_manager:
            return

        # パフォーマンスメトリクスを取得
        performance_summary = self.performance_service.summary()

        # サマリー用の主要統計（詳細はpipeline_checkpoint.jsonを参照）
        total_detections = sum(len(dets) for _, _, dets in detection_results)
        summary = {
            "status": "completed",
            "statistics": {
                "frames_extracted": len(extraction_results),
                "total_detections": total_detections,
                "avg_detections_per_frame": total_detections / len(detection_results) if detection_results else 0.0,
                "frames_processed": len(frame_results),
                "zones_count": len(aggregator.get_statistics()),
                "floormaps_generated": len(frame_results),
            },
            "performance": performance_summary,
        }
        self.output_service.save_summary(summary)
        self.logger.info(f"セッションサマリーを保存しました: {self.session_dir / 'summary.json'}")

    def _parse_datetime_range(
        self,
        target_config: dict,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        """日時範囲をパース

        Args:
            target_config: ターゲット設定
            start_time: 開始時刻（HH:MM形式、オプション）
            end_time: 終了時刻（HH:MM形式、オプション）

        Returns:
            (開始日時, 終了日時) のタプル
        """
        start_datetime = None
        end_datetime = None

        if target_config:
            start_str = target_config.get("start_datetime")
            end_str = target_config.get("end_datetime")
            if start_str:
                start_datetime = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            if end_str:
                end_datetime = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")

        # コマンドライン引数で上書き
        if start_time and start_datetime:
            hour, minute = map(int, start_time.split(":"))
            start_datetime = start_datetime.replace(hour=hour, minute=minute, second=0)

        if end_time and end_datetime:
            hour, minute = map(int, end_time.split(":"))
            end_datetime = end_datetime.replace(hour=hour, minute=minute, second=0)

        return start_datetime, end_datetime

    def cleanup(self, detector: DetectionPhase | None = None) -> None:
        """リソースをクリーンアップ

        Args:
            detector: DetectionPhaseインスタンス（オプション）
        """
        # 一時ファイルのクリーンアップ
        cleanup_temp_files = self.config.get("output.cleanup_temp_files", True)
        if cleanup_temp_files and self.session_dir:
            temp_frames_dir = self.session_dir / "01_extraction" / "_temp_frames"
            if temp_frames_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_frames_dir)
                    self.logger.info(f"一時ファイルを削除しました: {temp_frames_dir}")
                except Exception as e:
                    self.logger.warning(f"一時ファイルの削除に失敗しました: {e}")

        cleanup_resources(
            video_processor=None,
            detector=detector.detector if detector else None,
            logger=self.logger,
        )
