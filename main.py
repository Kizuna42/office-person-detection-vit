#!/usr/bin/env python
"""
オフィス人物検出システム - メインエントリーポイント

Vision Transformer (ViT) ベースの物体検出モデルを使用して、
オフィス内の定点カメラ映像から人物を検出し、
フロアマップ上でのゾーン別人数集計を実行します。
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config_manager import ConfigManager
from src.video_processor import VideoProcessor
from src.timestamp_extractor import TimestampExtractor
from src.frame_sampler import FrameSampler


# ロギング設定
def setup_logging(debug_mode: bool = False):
    """ロギングを設定する
    
    Args:
        debug_mode: デバッグモードの場合True
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # コンソール出力
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # ファイル出力（オプション）
    log_dir = Path('output')
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / 'detection.log', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def parse_arguments():
    """コマンドライン引数をパースする
    
    Returns:
        パース済み引数
    """
    parser = argparse.ArgumentParser(
        description='オフィス人物検出システム - ViTベースの人物検出とゾーン別集計'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='設定ファイルのパス（デフォルト: config.yaml）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードで実行（詳細ログ、中間結果出力）'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='精度評価モードで実行（Ground Truthとの比較）'
    )
    
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='ファインチューニングモードで実行'
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        help='開始時刻を指定（HH:MM形式）、指定しない場合は自動検出'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        help='終了時刻を指定（HH:MM形式）、指定しない場合は自動検出'
    )
    
    return parser.parse_args()


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_arguments()
    
    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("オフィス人物検出システム 起動")
    logger.info("=" * 60)
    
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
        
        # 出力ディレクトリの作成
        output_dir = Path(config.get('output.directory', 'output'))
        for subdir in ['detections', 'floormaps', 'graphs', 'labels']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("フェーズ1: フレームサンプリング")
        logger.info("=" * 60)
        
        # 動画処理の初期化
        video_path = config.get('video.input_path')
        logger.info(f"動画ファイル: {video_path}")
        
        video_processor = VideoProcessor(video_path)
        video_processor.open()
        
        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        
        # フレームサンプラーの初期化
        interval_minutes = config.get('video.frame_interval_minutes', 5)
        tolerance_seconds = config.get('video.tolerance_seconds', 10)
        frame_sampler = FrameSampler(interval_minutes, tolerance_seconds)
        
        # フレームサンプリング実行
        logger.info("フレームサンプリングを開始します...")
        sample_frames = frame_sampler.extract_sample_frames(
            video_processor,
            timestamp_extractor,
            start_time=args.start_time,
            end_time=args.end_time
        )
        
        logger.info(f"サンプルフレーム数: {len(sample_frames)}個")
        
        # リソース解放
        video_processor.release()
        
        if not sample_frames:
            logger.error("サンプルフレームが抽出できませんでした")
            return 1
        
        # TODO: フェーズ2以降の実装
        logger.info("=" * 60)
        logger.info("フェーズ2以降は実装予定")
        logger.info("=" * 60)
        logger.info("実装予定:")
        logger.info("  - フェーズ2: ViT人物検出")
        logger.info("  - フェーズ3: 座標変換とゾーン判定")
        logger.info("  - フェーズ4: 集計とレポート生成")
        logger.info("  - フェーズ5: 可視化")
        
        if args.evaluate:
            logger.info("\n精度評価モードは実装予定です")
        
        if args.fine_tune:
            logger.info("\nファインチューニングモードは実装予定です")
        
        logger.info("=" * 60)
        logger.info("処理が完了しました")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        return 1
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

