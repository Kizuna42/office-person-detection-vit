"""Pipeline orchestrator for coordinating all phases."""

from datetime import datetime
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.aggregation import Aggregator
from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.aggregation_phase import AggregationPhase
from src.pipeline.detection_phase import DetectionPhase
from src.pipeline.frame_extraction_pipeline import FrameExtractionPipeline
from src.pipeline.transform_phase import TransformPhase
from src.pipeline.visualization_phase import VisualizationPhase
from src.utils import OutputManager, PerformanceMonitor, cleanup_resources, setup_output_directories
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
        self.output_manager: OutputManager | None = None
        self.session_dir: Path | None = None
        self.output_path: Path = Path(config.get("output.directory", "output"))
        self.performance_monitor = PerformanceMonitor()

    def setup_output_directories(self, use_session_management: bool, args: dict | None = None) -> None:
        """出力ディレクトリをセットアップ

        Args:
            use_session_management: セッション管理を使用するか
            args: コマンドライン引数（オプション、メタデータ保存用）
        """
        if use_session_management:
            self.output_manager = OutputManager(self.output_path)
            self.session_dir = self.output_manager.create_session()
            self.logger.info(f"セッション管理を有効化しました: {self.session_dir.name}")

            # メタデータを保存
            config_dict = self.config.config if hasattr(self.config, "config") else {}
            args_dict = vars(args) if args else {}
            self.output_manager.save_metadata(self.session_dir, config_dict, args_dict)
            self.output_path = self.session_dir
        else:
            self.logger.info("セッション管理は無効です（従来の出力構造を使用）")
            # 従来のディレクトリ構造を作成
            setup_output_directories(self.output_path)

        self.logger.info(f"出力ディレクトリ: {self.output_path.absolute()}")

    def get_phase_output_dir(self, phase_name: str) -> Path:
        """フェーズごとの出力ディレクトリを取得

        Args:
            phase_name: フェーズ名（例: "phase1_extraction"）

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
        with self.performance_monitor.measure("phase1_extraction"):
            # 設定からパラメータを取得
            timestamp_config = self.config.get("timestamp", {})
            extraction_config = timestamp_config.get("extraction", {})
            sampling_config = timestamp_config.get("sampling", {})
            target_config = timestamp_config.get("target", {})
            ocr_config = self.config.get("ocr", {})

            # 開始・終了日時の取得
            start_datetime, end_datetime = self._parse_datetime_range(target_config, start_time, end_time)

            # 出力ディレクトリの決定
            extraction_output_dir = self.get_phase_output_dir("phase1_extraction")

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
        sample_frames = []
        video_processor = None

        try:
            for result in tqdm(extraction_results, desc="フレーム準備中"):
                frame = result.get("frame")
                if frame is None:
                    # フレームが保存されていない場合は動画から再取得
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
        with self.performance_monitor.measure("phase2_detection"):
            detection_phase = DetectionPhase(self.config, self.logger)
            detection_phase.initialize()

            output_dir = self.get_phase_output_dir("phase2_detection")
            detection_phase.output_path = output_dir

            detection_results = detection_phase.execute(sample_frames)
            detection_phase.log_statistics(detection_results, output_dir)

            return detection_results, detection_phase

    def run_transform(
        self, detection_results: list[tuple[int, str, list[Detection]]]
    ) -> tuple[list[FrameResult], TransformPhase]:
        """座標変換とゾーン判定を実行

        Args:
            detection_results: 検出結果のリスト

        Returns:
            (FrameResultのリスト, TransformPhaseインスタンス)
        """
        with self.performance_monitor.measure("phase3_transform"):
            transform_phase = TransformPhase(self.config, self.logger)
            transform_phase.initialize()

            output_dir = self.get_phase_output_dir("phase3_transform")
            frame_results = transform_phase.execute(detection_results)
            transform_phase.export_results(frame_results, output_dir)

            return frame_results, transform_phase

    def run_aggregation(self, frame_results: list[FrameResult]) -> tuple[AggregationPhase, Aggregator]:
        """集計処理を実行

        Args:
            frame_results: FrameResultのリスト

        Returns:
            (AggregationPhaseインスタンス, Aggregatorインスタンス)
        """
        with self.performance_monitor.measure("phase4_aggregation"):
            aggregation_phase = AggregationPhase(self.config, self.logger)
            output_dir = self.get_phase_output_dir("phase4_aggregation")
            aggregator = aggregation_phase.execute(frame_results, output_dir)

            return aggregation_phase, aggregator

    def run_visualization(self, aggregator: Aggregator, frame_results: list[FrameResult]) -> None:
        """可視化を実行

        Args:
            aggregator: Aggregatorインスタンス
            frame_results: FrameResultのリスト
        """
        with self.performance_monitor.measure("phase5_visualization"):
            visualization_phase = VisualizationPhase(self.config, self.logger)
            output_dir = self.get_phase_output_dir("phase5_visualization")
            visualization_phase.execute(aggregator, frame_results, output_dir)

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
        if not self.session_dir or not self.output_manager:
            return

        # パフォーマンスメトリクスを取得
        performance_summary = self.performance_monitor.get_summary()

        summary = {
            "status": "completed",
            "phases": {
                "extraction": {
                    "frames_extracted": len(extraction_results),
                    "success_rate": 1.0 if extraction_results else 0.0,
                },
                "detection": {
                    "total_detections": sum(len(dets) for _, _, dets in detection_results),
                    "avg_per_frame": sum(len(dets) for _, _, dets in detection_results) / len(detection_results)
                    if detection_results
                    else 0.0,
                },
                "transform": {"frames_processed": len(frame_results)},
                "aggregation": {"zones_count": len(aggregator.get_statistics())},
                "visualization": {
                    "graphs_generated": 2,  # time_series, statistics
                    "floormaps_generated": len(frame_results),
                },
            },
            "performance": performance_summary,
        }
        self.output_manager.save_summary(self.session_dir, summary)
        self.output_manager.update_latest_link(self.session_dir)
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
        cleanup_resources(
            video_processor=None,
            detector=detector.detector if detector else None,
            logger=self.logger,
        )
