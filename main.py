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
from typing import List, Tuple

from src.cli import parse_arguments
from src.config import ConfigManager
from src.detection import ViTDetector
from src.evaluation import EvaluationModule
from src.models import Detection, FrameResult
from src.pipeline import (
    AggregationPhase,
    DetectionPhase,
    FrameSamplingPhase,
    TimestampOCRMode,
    TransformPhase,
    VisualizationPhase,
)
from src.utils import setup_logging
from src.utils.memory_utils import cleanup_resources


def run_evaluation(
    detection_results: List[Tuple[int, str, List[Detection]]],
    config: ConfigManager,
    logger: logging.Logger
) -> None:
    """精度評価を実行
    
    Args:
        detection_results: 検出結果のリスト
        config: ConfigManager インスタンス
        logger: ロガー
    """
    logger.info("=" * 60)
    logger.info("精度評価を開始します")
    logger.info("=" * 60)
    
    try:
        # 評価モジュールの初期化
        gt_path = config.get('evaluation.ground_truth_path')
        iou_threshold = config.get('evaluation.iou_threshold', 0.5)
        
        evaluator = EvaluationModule(gt_path, iou_threshold)
        
        # 検出結果を辞書形式に変換
        detections_dict = {}
        for frame_num, timestamp, detections in detection_results:
            filename = f"frame_{frame_num:06d}_{timestamp.replace(':', 'h')}m.jpg"
            detections_dict[filename] = detections
        
        # 評価実行
        metrics = evaluator.evaluate(detections_dict)
        
        # レポート出力
        output_dir = Path(config.get('output.directory', 'output'))
        evaluator.export_report(metrics, str(output_dir / 'evaluation_report.csv'), format='csv')
        evaluator.export_report(metrics, str(output_dir / 'evaluation_report.json'), format='json')
        
        logger.info("=" * 60)
        logger.info("評価結果:")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  F1-score: {metrics.f1_score:.4f}")
        logger.info(f"  True Positives: {metrics.true_positives}")
        logger.info(f"  False Positives: {metrics.false_positives}")
        logger.info(f"  False Negatives: {metrics.false_negatives}")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Ground Truthファイルが見つかりません: {e}")
    except Exception as e:
        logger.error(f"評価処理中にエラーが発生しました: {e}", exc_info=True)


def setup_output_directories(output_dir: Path) -> None:
    """出力ディレクトリを作成
    
    Args:
        output_dir: 出力ディレクトリのパス
    """
    for subdir in ['detections', 'floormaps', 'graphs', 'labels']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def main():
    """メイン処理"""
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
        
        # タイムスタンプOCRのみモード
        if args.timestamps_only:
            timestamp_mode = TimestampOCRMode(config, logger)
            return timestamp_mode.execute(
                start_time=args.start_time,
                end_time=args.end_time
            )
        
        # ========================================
        # フェーズ1: フレームサンプリング
        # ========================================
        frame_sampling_phase = FrameSamplingPhase(config, logger)
        try:
            sample_frames = frame_sampling_phase.execute(
                start_time=args.start_time,
                end_time=args.end_time
            )
        finally:
            frame_sampling_phase.cleanup()
        
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
            video_processor=None,  # FrameSamplingPhaseで管理されているため
            detector=detector,
            logger=logger
        )


if __name__ == "__main__":
    sys.exit(main())
