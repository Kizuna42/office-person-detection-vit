#!/usr/bin/env python
"""Phase 2: 初期評価・デバッグ用の検証スクリプト

実動作確認、精度評価、問題特定を支援するツール
"""

import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import time

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psutil  # noqa: E402

from src.config import ConfigManager  # noqa: E402
from src.pipeline import FrameExtractionPipeline  # noqa: E402
from src.utils import setup_logging, setup_output_directories  # noqa: E402

logger = logging.getLogger(__name__)


def measure_memory_usage():
    """現在のメモリ使用量を取得（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def verify_sample_video(video_path: str, config_path: str = "config.yaml"):
    """サンプル動画での動作確認

    Args:
        video_path: 動画ファイルのパス
        config_path: 設定ファイルのパス
    """
    logger.info("=" * 80)
    logger.info("Phase 2.1.1: サンプル動画での動作確認")
    logger.info("=" * 80)

    # 設定読み込み
    config = ConfigManager(config_path)

    # 出力ディレクトリ設定
    output_dir = Path(config.get("output.directory", "output"))
    frame_extraction_output_dir = output_dir / "extracted_frames"
    setup_output_directories(frame_extraction_output_dir)

    # 設定からパラメータを取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    sampling_config = timestamp_config.get("sampling", {})
    target_config = timestamp_config.get("target", {})
    ocr_config = config.get("ocr", {})

    # 短い時間範囲でテスト（最初の30分間）
    start_datetime = None
    end_datetime = None
    if target_config:
        start_str = target_config.get("start_datetime")
        if start_str:
            start_datetime = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            end_datetime = start_datetime + timedelta(minutes=30)  # 30分間のみ

    # メモリ使用量の記録
    memory_before = measure_memory_usage()
    start_time = time.time()

    try:
        # パイプライン初期化
        pipeline = FrameExtractionPipeline(
            video_path=video_path,
            output_dir=str(frame_extraction_output_dir),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            interval_minutes=config.get("video.frame_interval_minutes", 5),
            tolerance_seconds=config.get("video.tolerance_seconds", 10.0),
            confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
            coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
            fine_search_window_seconds=sampling_config.get("search_window_seconds", 60.0),
            fine_interval_seconds=sampling_config.get("fine_interval_seconds", 0.1),
            fps=config.get("video.fps", 30.0),
            roi_config=extraction_config.get("roi"),
            enabled_ocr_engines=ocr_config.get("engines"),
        )

        # 実行
        results = pipeline.run()

        # 処理時間とメモリ使用量
        elapsed_time = time.time() - start_time
        memory_after = measure_memory_usage()

        # 結果の検証
        logger.info("=" * 80)
        logger.info("動作確認結果")
        logger.info("=" * 80)
        logger.info("✓ エラーなく完了: 成功")
        logger.info(f"✓ 抽出フレーム数: {len(results)}")
        logger.info(f"✓ 処理時間: {elapsed_time:.2f}秒")
        logger.info(
            f"✓ メモリ使用量: {memory_before:.1f}MB → {memory_after:.1f}MB (増加: {memory_after - memory_before:.1f}MB)"
        )

        # 出力ファイルの確認
        csv_path = frame_extraction_output_dir / "extraction_results.csv"
        if csv_path.exists():
            logger.info(f"✓ CSV出力: {csv_path}")
        else:
            logger.warning(f"✗ CSV出力が見つかりません: {csv_path}")

        frame_count = len(list(frame_extraction_output_dir.glob("frame_*.jpg")))
        logger.info(f"✓ 保存されたフレーム数: {frame_count}")

        # ログ出力の確認
        log_path = output_dir / "system.log"
        if log_path.exists():
            logger.info(f"✓ ログファイル: {log_path}")
            # エラーログの確認
            with log_path.open("r", encoding="utf-8") as f:
                log_content = f.read()
                error_count = log_content.count("ERROR")
                warning_count = log_content.count("WARNING")
                logger.info(f"  エラーログ数: {error_count}, 警告ログ数: {warning_count}")

        logger.info("=" * 80)

        return len(results) > 0

    except Exception as e:
        logger.error(f"動作確認中にエラーが発生しました: {e}", exc_info=True)
        return False


def verify_production_video(video_path: str, duration_seconds: int = 3600, config_path: str = "config.yaml"):
    """本番動画での試験実行（最初の指定時間のみ）

    Args:
        video_path: 動画ファイルのパス
        duration_seconds: 処理する時間（秒）
        config_path: 設定ファイルのパス
    """
    logger.info("=" * 80)
    logger.info(f"Phase 2.1.2: 本番動画での試験実行（最初の{duration_seconds}秒）")
    logger.info("=" * 80)

    # 設定読み込み
    config = ConfigManager(config_path)

    # 出力ディレクトリ設定
    output_dir = Path(config.get("output.directory", "output"))
    frame_extraction_output_dir = output_dir / "extracted_frames"
    setup_output_directories(frame_extraction_output_dir)

    # 設定からパラメータを取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    sampling_config = timestamp_config.get("sampling", {})
    target_config = timestamp_config.get("target", {})
    ocr_config = config.get("ocr", {})

    # 時間範囲の設定
    start_datetime = None
    end_datetime = None
    if target_config:
        start_str = target_config.get("start_datetime")
        if start_str:
            start_datetime = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            end_datetime = start_datetime + timedelta(seconds=duration_seconds)

    # メモリ使用量の記録
    memory_before = measure_memory_usage()
    start_time = time.time()

    try:
        # パイプライン初期化
        pipeline = FrameExtractionPipeline(
            video_path=video_path,
            output_dir=str(frame_extraction_output_dir),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            interval_minutes=config.get("video.frame_interval_minutes", 5),
            tolerance_seconds=config.get("video.tolerance_seconds", 10.0),
            confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
            coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
            fine_search_window_seconds=sampling_config.get("search_window_seconds", 60.0),
            fine_interval_seconds=sampling_config.get("fine_interval_seconds", 0.1),
            fps=config.get("video.fps", 30.0),
            roi_config=extraction_config.get("roi"),
            enabled_ocr_engines=ocr_config.get("engines"),
        )

        # 実行
        results = pipeline.run()

        # 処理時間とメモリ使用量
        elapsed_time = time.time() - start_time
        memory_after = measure_memory_usage()

        # 結果の検証
        logger.info("=" * 80)
        logger.info("試験実行結果")
        logger.info("=" * 80)
        logger.info(f"✓ 処理時間: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分)")
        logger.info(
            f"✓ メモリ使用量: {memory_before:.1f}MB → {memory_after:.1f}MB (増加: {memory_after - memory_before:.1f}MB)"
        )
        logger.info(f"✓ 抽出フレーム数: {len(results)}")
        if len(results) > 0:
            avg_time_per_frame = elapsed_time / len(results)
            logger.info(f"✓ フレームあたりの処理時間: {avg_time_per_frame:.2f}秒")
        logger.info("=" * 80)

        return len(results) > 0

    except Exception as e:
        logger.error(f"試験実行中にエラーが発生しました: {e}", exc_info=True)
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Phase 2: 初期評価・デバッグ用の検証スクリプト")
    parser.add_argument(
        "--mode",
        choices=["sample", "production"],
        required=True,
        help="実行モード: sample=サンプル動画, production=本番動画",
    )
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--duration", type=int, default=3600, help="本番動画モードでの処理時間（秒、デフォルト: 3600）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 動画パスの取得
    if args.video:
        video_path = args.video
    else:
        config = ConfigManager(args.config)
        video_path = config.get("video.input_path")

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    # モードに応じて実行
    if args.mode == "sample":
        success = verify_sample_video(video_path, args.config)
    else:
        success = verify_production_video(video_path, args.duration, args.config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
