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

from src.cli import parse_arguments
from src.config import ConfigManager
from src.evaluation import run_evaluation
from src.pipeline import (
    AggregationPhase,
    DetectionPhase,
    FrameExtractionPipeline,
    TransformPhase,
    VisualizationPhase,
)
from src.utils import (
    cleanup_resources,
    setup_logging,
    setup_mps_compatibility,
    setup_output_directories,
)


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
            config.set('output.debug_mode', True)
            logger.info("デバッグモードが有効になりました")
        
        # ロギングを再設定（出力ディレクトリを反映）
        output_dir = config.get('output.directory', 'output')
        setup_logging(args.debug, output_dir)
        logger = logging.getLogger(__name__)
        
        # 出力ディレクトリの作成
        output_path = Path(output_dir)
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
        
        video_path = config.get('video.input_path')
        frame_extraction_output_dir = output_path / 'extracted_frames'
        
        # 設定からパラメータを取得
        timestamp_config = config.get('timestamp', {})
        extraction_config = timestamp_config.get('extraction', {})
        sampling_config = timestamp_config.get('sampling', {})
        target_config = timestamp_config.get('target', {})
        ocr_config = config.get('ocr', {})
        
        # 開始・終了日時の取得
        start_datetime = None
        end_datetime = None
        if target_config:
            start_str = target_config.get('start_datetime')
            end_str = target_config.get('end_datetime')
            if start_str:
                start_datetime = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            if end_str:
                end_datetime = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
        
        # コマンドライン引数で上書き
        if args.start_time:
            # HH:MM形式をdatetimeに変換（開始日時が設定されていればその日付を使用）
            if start_datetime:
                hour, minute = map(int, args.start_time.split(':'))
                start_datetime = start_datetime.replace(hour=hour, minute=minute, second=0)
        
        if args.end_time:
            if end_datetime:
                hour, minute = map(int, args.end_time.split(':'))
                end_datetime = end_datetime.replace(hour=hour, minute=minute, second=0)
        
        # パイプライン初期化
        pipeline = FrameExtractionPipeline(
            video_path=video_path,
            output_dir=str(frame_extraction_output_dir),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            interval_minutes=config.get('video.frame_interval_minutes', 5),
            tolerance_seconds=config.get('video.tolerance_seconds', 10.0),
            confidence_threshold=extraction_config.get('confidence_threshold', 0.7),
            coarse_interval_seconds=sampling_config.get('coarse_interval_seconds', 10.0),
            fine_search_window_seconds=sampling_config.get('search_window_seconds', 30.0),
            fps=config.get('video.fps', 30.0),
            roi_config=extraction_config.get('roi'),
            enabled_ocr_engines=ocr_config.get('engines')
        )
        
        # フレーム抽出実行
        extraction_results = pipeline.run()
        
        if not extraction_results:
            logger.error("フレーム抽出に失敗しました")
            return 1
        
        # 抽出結果を後続処理用の形式に変換
        # DetectionPhaseは (frame_num, timestamp_str, frame) のタプルリストを期待
        sample_frames = []
        for result in extraction_results:
            frame = result.get('frame')
            if frame is None:
                # フレームが保存されていない場合は動画から再取得
                from src.video import VideoProcessor
                video_processor = VideoProcessor(video_path)
                video_processor.open()
                frame = video_processor.get_frame(result['frame_idx'])
                video_processor.release()
                if frame is None:
                    logger.warning(f"フレーム {result['frame_idx']} を取得できませんでした")
                    continue
            
            timestamp_str = result['timestamp'].strftime('%Y/%m/%d %H:%M:%S')
            sample_frames.append((
                result['frame_idx'],
                timestamp_str,
                frame
            ))
        
        logger.info(f"フレーム抽出完了: {len(sample_frames)}フレーム")
        
        # ========================================
        # フェーズ2: ViT人物検出
        # ========================================
        detection_phase = DetectionPhase(config, logger)
        detection_phase.initialize()
        detector = detection_phase.detector
        
        detection_results = detection_phase.execute(sample_frames)
        detection_phase.log_statistics(detection_results, output_path)
        
        # ========================================
        # フェーズ3: 座標変換とゾーン判定
        # ========================================
        transform_phase = TransformPhase(config, logger)
        transform_phase.initialize()
        
        frame_results = transform_phase.execute(detection_results)
        transform_phase.export_results(frame_results, output_path)
        
        # ========================================
        # フェーズ4: 集計とレポート生成
        # ========================================
        aggregation_phase = AggregationPhase(config, logger)
        aggregator = aggregation_phase.execute(frame_results, output_path)
        
        # ========================================
        # フェーズ5: 可視化
        # ========================================
        visualization_phase = VisualizationPhase(config, logger)
        visualization_phase.execute(aggregator, frame_results, output_path)
        
        # ========================================
        # 精度評価（オプション）
        # ========================================
        if args.evaluate:
            run_evaluation(detection_results, config, logger)
        
        logger.info("=" * 80)
        logger.info("処理が正常に完了しました")
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
        cleanup_resources(
            video_processor=None,
            detector=detector,
            logger=logger
        )


if __name__ == "__main__":
    sys.exit(main())
