#!/usr/bin/env python
"""
オフィス人物検出システム - メインエントリーポイント

Vision Transformer (ViT) ベースの物体検出モデルを使用して、
オフィス内の定点カメラ映像から人物を検出し、
フロアマップ上でのゾーン別人数集計を実行します。
"""

import logging
import sys

from src.cli import parse_arguments
from src.config import ConfigManager
from src.evaluation import run_evaluation
from src.pipeline import PipelineOrchestrator
from src.utils import setup_logging, setup_mps_compatibility


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

    detector_phase = None

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

        # パイプラインオーケストレーターの初期化
        orchestrator = PipelineOrchestrator(config, logger)

        # 出力ディレクトリのセットアップ
        use_session_management = config.get("output.use_session_management", False)
        orchestrator.setup_output_directories(use_session_management, args)

        # ファインチューニングモード
        if args.fine_tune:
            logger.warning("ファインチューニングモードは未実装です")
            return 1

        # フェーズ1: フレーム抽出
        video_path = config.get("video.input_path")
        extraction_results = orchestrator.extract_frames(video_path, start_time=args.start_time, end_time=args.end_time)

        if not extraction_results:
            logger.error("フレーム抽出に失敗しました")
            return 1

        logger.info(f"フレーム抽出完了: {len(extraction_results)}フレーム")

        # タイムスタンプOCRのみの場合はここで終了
        if args.timestamps_only:
            logger.info("=" * 80)
            logger.info("タイムスタンプOCR処理が正常に完了しました")
            logger.info(f"抽出フレーム数: {len(extraction_results)}")
            logger.info(f"出力ディレクトリ: {orchestrator.get_phase_output_dir('phase1_extraction').absolute()}")
            logger.info("=" * 80)
            return 0

        # フェーズ2-5: 検出→変換→集計→可視化
        sample_frames = orchestrator.prepare_frames_for_detection(extraction_results, video_path)
        detection_results, detector_phase = orchestrator.run_detection(sample_frames)
        frame_results, _ = orchestrator.run_transform(detection_results)
        _, aggregator = orchestrator.run_aggregation(frame_results)
        orchestrator.run_visualization(aggregator, frame_results)

        # 精度評価（オプション）
        if args.evaluate:
            run_evaluation(detection_results, config, logger)

        # セッションサマリーの保存
        orchestrator.save_session_summary(extraction_results, detection_results, frame_results, aggregator)

        logger.info("=" * 80)
        logger.info("処理が正常に完了しました")
        if orchestrator.session_dir:
            logger.info(f"セッションディレクトリ: {orchestrator.session_dir.absolute()}")
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
        if "orchestrator" in locals() and detector_phase:
            orchestrator.cleanup(detector_phase)
        elif "orchestrator" in locals():
            orchestrator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
