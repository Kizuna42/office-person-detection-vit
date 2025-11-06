#!/usr/bin/env python
"""
オフィス人物検出システム - メインエントリーポイント

Vision Transformer (ViT) ベースの物体検出モデルを使用して、
オフィス内の定点カメラ映像から人物を検出し、
フロアマップ上でのゾーン別人数集計を実行します。
"""

import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.cli import parse_arguments
from src.config import ConfigManager
from src.evaluation import run_evaluation
from src.pipeline import (AggregationPhase, DetectionPhase,
                          FrameExtractionPipeline, TransformPhase,
                          VisualizationPhase)
from src.utils import (OutputManager, cleanup_resources, setup_logging,
                       setup_mps_compatibility, setup_output_directories)


def main():
    """メイン処理"""
    # MPS互換性設定を適用（警告抑制）
    setup_mps_compatibility()

    # コマンドライン引数のパース
    args = parse_arguments()

    # 初期ロギング設定（設定ファイル読み込み前）
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("オフィス人物検出システム 起動")
    logger.info("=" * 80)

    detector = None

    try:
        # 設定ファイルの読み込み
        logger.info(f"設定ファイルを読み込んでいます: {args.config}")
        config = ConfigManager(args.config)

        # 設定の検証
        if not config.validate():
            logger.error("設定ファイルの検証に失敗しました")
            return 1

        # デバッグモードの場合、設定を上書き
        if args.debug:
            config.set("output.debug_mode", True)
            logger.info("デバッグモードが有効になりました")

        # ロギングを再設定（出力ディレクトリを反映）
        output_dir = config.get("output.directory", "output")
        setup_logging(args.debug, output_dir)
        logger = logging.getLogger(__name__)

        # 出力ディレクトリの作成
        output_path = Path(output_dir)

        # セッション管理の初期化
        use_session_management = config.get("output.use_session_management", False)
        session_dir = None
        output_manager = None

        if use_session_management:
            # セッション管理が有効な場合: OutputManagerが必要なディレクトリのみ作成
            # （sessions/, archive/, shared/のみ。detections/, floormaps/, graphs/は作成しない）
            output_manager = OutputManager(output_path)
            session_dir = output_manager.create_session()
            logger.info(f"セッション管理を有効化しました: {session_dir.name}")

            # メタデータを保存
            # ConfigManagerの内部設定を取得
            config_dict = config.config if hasattr(config, "config") else {}
            args_dict = vars(args) if args else {}
            output_manager.save_metadata(session_dir, config_dict, args_dict)

            # 実際の出力先をセッションディレクトリに変更
            output_path = session_dir
        else:
            # セッション管理が無効な場合: 従来のディレクトリ構造を作成
            logger.info("セッション管理は無効です（従来の出力構造を使用）")
            setup_output_directories(output_path)

        logger.info(f"出力ディレクトリ: {output_path.absolute()}")

        # ファインチューニングモード
        if args.fine_tune:
            logger.warning("ファインチューニングモードは未実装です")
            return 1

        # ========================================
        # フェーズ1: フレーム抽出（5分刻みタイムスタンプベース）
        # ========================================
        from datetime import datetime

        video_path = config.get("video.input_path")
        # セッション管理が有効な場合はphase1_extractionを使用（frames/はパイプライン内で作成）
        if use_session_management and session_dir:
            frame_extraction_output_dir = session_dir / "phase1_extraction"
            frame_extraction_csv_dir = session_dir / "phase1_extraction"
        else:
            # セッション管理が無効な場合のみ従来のディレクトリを使用
            frame_extraction_output_dir = output_path / "extracted_frames"
            frame_extraction_csv_dir = output_path / "extracted_frames"

        # 設定からパラメータを取得
        timestamp_config = config.get("timestamp", {})
        extraction_config = timestamp_config.get("extraction", {})
        sampling_config = timestamp_config.get("sampling", {})
        target_config = timestamp_config.get("target", {})
        ocr_config = config.get("ocr", {})

        # 開始・終了日時の取得
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
        if args.start_time:
            # HH:MM形式をdatetimeに変換（開始日時が設定されていればその日付を使用）
            if start_datetime:
                hour, minute = map(int, args.start_time.split(":"))
                start_datetime = start_datetime.replace(
                    hour=hour, minute=minute, second=0
                )

        if args.end_time:
            if end_datetime:
                hour, minute = map(int, args.end_time.split(":"))
                end_datetime = end_datetime.replace(hour=hour, minute=minute, second=0)

        # 改善機能の設定を取得（デフォルトはFalseで既存実装を使用）
        validator_config = extraction_config.get("validator", {})
        use_improved_validator = extraction_config.get("use_improved_validator", False)
        use_weighted_consensus = extraction_config.get("use_weighted_consensus", False)
        use_voting_consensus = extraction_config.get("use_voting_consensus", False)

        # パイプライン初期化
        pipeline = FrameExtractionPipeline(
            video_path=video_path,
            output_dir=str(frame_extraction_csv_dir),  # CSVはphase1_extractionディレクトリに保存
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            interval_minutes=config.get("video.frame_interval_minutes", 5),
            tolerance_seconds=config.get("video.tolerance_seconds", 10.0),
            confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
            coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
            fine_search_window_seconds=sampling_config.get(
                "search_window_seconds", 60.0
            ),
            fine_interval_seconds=sampling_config.get("fine_interval_seconds", 0.1),
            fps=config.get("video.fps", 30.0),
            roi_config=extraction_config.get("roi"),
            enabled_ocr_engines=ocr_config.get("engines"),
            use_improved_validator=use_improved_validator,
            base_tolerance_seconds=validator_config.get("base_tolerance_seconds", 10.0),
            history_size=validator_config.get("history_size", 10),
            z_score_threshold=validator_config.get("z_score_threshold", 2.0),
            use_weighted_consensus=use_weighted_consensus,
            use_voting_consensus=use_voting_consensus,
        )

        # フレーム抽出モードに応じて実行方法を選択
        extraction_mode = timestamp_config.get("extraction_mode", "manual_targets")
        auto_targets_config = timestamp_config.get("auto_targets", {})

        if extraction_mode == "auto_targets":
            logger.info("自動目標タイムスタンプ生成モードでフレーム抽出を実行します")
            max_frames = auto_targets_config.get("max_frames")
            disable_validation = auto_targets_config.get("disable_validation", False)
            extraction_results = pipeline.run_with_auto_targets(
                max_frames=max_frames, disable_validation=disable_validation
            )
        else:
            logger.info("手動目標タイムスタンプ指定モードでフレーム抽出を実行します")
            extraction_results = pipeline.run()

        if not extraction_results:
            logger.error("フレーム抽出に失敗しました")
            return 1

        logger.info(f"フレーム抽出完了: {len(extraction_results)}フレーム")

        # タイムスタンプOCRのみの場合はここで終了
        if args.timestamps_only:
            logger.info("=" * 80)
            logger.info("タイムスタンプOCR処理が正常に完了しました")
            logger.info(f"抽出フレーム数: {len(extraction_results)}")
            logger.info(f"出力ディレクトリ: {frame_extraction_output_dir.absolute()}")
            logger.info("=" * 80)
            return 0

        # 抽出結果を後続処理用の形式に変換
        # DetectionPhaseは (frame_num, timestamp_str, frame) のタプルリストを期待
        sample_frames = []
        for result in tqdm(extraction_results, desc="フレーム準備中"):
            frame = result.get("frame")
            if frame is None:
                # フレームが保存されていない場合は動画から再取得
                from src.video import VideoProcessor

                video_processor = VideoProcessor(video_path)
                video_processor.open()
                frame = video_processor.get_frame(result["frame_idx"])
                video_processor.release()
                if frame is None:
                    logger.warning(f"フレーム {result['frame_idx']} を取得できませんでした")
                    continue

            timestamp_str = result["timestamp"].strftime("%Y/%m/%d %H:%M:%S")
            sample_frames.append((result["frame_idx"], timestamp_str, frame))

        logger.info(f"後続処理用フレーム準備完了: {len(sample_frames)}フレーム")

        # ========================================
        # フェーズ2: ViT人物検出
        # ========================================
        detection_phase = DetectionPhase(config, logger)
        detection_phase.initialize()
        detector = detection_phase.detector

        # セッション管理が有効な場合はphase2_detectionディレクトリを使用
        if use_session_management and session_dir:
            detection_output_dir = session_dir / "phase2_detection"
        else:
            detection_output_dir = output_path

        # DetectionPhaseにoutput_pathを設定（検出画像の保存先）
        detection_phase.output_path = detection_output_dir

        detection_results = detection_phase.execute(sample_frames)
        detection_phase.log_statistics(detection_results, detection_output_dir)

        # ========================================
        # フェーズ3: 座標変換とゾーン判定
        # ========================================
        transform_phase = TransformPhase(config, logger)
        transform_phase.initialize()

        # セッション管理が有効な場合はphase3_transformディレクトリを使用
        if use_session_management and session_dir:
            transform_output_dir = session_dir / "phase3_transform"
        else:
            transform_output_dir = output_path

        frame_results = transform_phase.execute(detection_results)
        transform_phase.export_results(frame_results, transform_output_dir)

        # ========================================
        # フェーズ4: 集計とレポート生成
        # ========================================
        # セッション管理が有効な場合はphase4_aggregationディレクトリを使用
        if use_session_management and session_dir:
            aggregation_output_dir = session_dir / "phase4_aggregation"
        else:
            aggregation_output_dir = output_path

        aggregation_phase = AggregationPhase(config, logger)
        aggregator = aggregation_phase.execute(frame_results, aggregation_output_dir)

        # ========================================
        # フェーズ5: 可視化
        # ========================================
        # セッション管理が有効な場合はphase5_visualizationディレクトリを使用
        if use_session_management and session_dir:
            visualization_output_dir = session_dir / "phase5_visualization"
        else:
            visualization_output_dir = output_path

        visualization_phase = VisualizationPhase(config, logger)
        visualization_phase.execute(aggregator, frame_results, visualization_output_dir)

        # ========================================
        # 精度評価（オプション）
        # ========================================
        if args.evaluate:
            run_evaluation(detection_results, config, logger)

        # セッション管理が有効な場合はサマリーを保存
        if use_session_management and session_dir and output_manager:
            summary = {
                "status": "completed",
                "phases": {
                    "extraction": {
                        "frames_extracted": len(extraction_results),
                        "success_rate": 1.0 if extraction_results else 0.0,
                    },
                    "detection": {
                        "total_detections": sum(
                            len(dets) for _, _, dets in detection_results
                        ),
                        "avg_per_frame": sum(
                            len(dets) for _, _, dets in detection_results
                        )
                        / len(detection_results)
                        if detection_results
                        else 0.0,
                    },
                    "transform": {
                        "frames_processed": len(frame_results),
                    },
                    "aggregation": {
                        "zones_count": len(aggregator.get_statistics()),
                    },
                    "visualization": {
                        "graphs_generated": 2,  # time_series, statistics
                        "floormaps_generated": len(frame_results),
                    },
                },
            }
            output_manager.save_summary(session_dir, summary)
            output_manager.update_latest_link(session_dir)
            logger.info(f"セッションサマリーを保存しました: {session_dir / 'summary.json'}")

        logger.info("=" * 80)
        logger.info("処理が正常に完了しました")
        if use_session_management and session_dir:
            logger.info(f"セッションディレクトリ: {session_dir.absolute()}")
        logger.info("=" * 80)

        return 0

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    finally:
        cleanup_resources(video_processor=None, detector=detector, logger=logger)


if __name__ == "__main__":
    sys.exit(main())
