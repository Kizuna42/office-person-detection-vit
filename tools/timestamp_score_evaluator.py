#!/usr/bin/env python3
"""タイムスタンプ抽出スコア評価ツール

動画から10秒間隔で100フレームを抽出し、各フレームからOCRでタイムスタンプを抽出。
抽出成功率と時系列整合性スコアを算出してコンソールに出力する。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import VideoProcessor
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_extraction_rate(
    results: List[Tuple[bool, Optional[str], float]]
) -> Dict[str, float]:
    """抽出成功率を計算

    Args:
        results: [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト

    Returns:
        抽出成功率の統計情報
    """
    total = len(results)
    success_count = sum(1 for success, _, _ in results if success)
    success_rate = (success_count / total * 100.0) if total > 0 else 0.0

    # 信頼度統計（成功したもののみ）
    confidences = [conf for success, _, conf in results if success]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0

    return {
        "total_frames": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "average_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
    }


def calculate_temporal_consistency(
    results: List[Tuple[bool, Optional[str], float]]
) -> Dict[str, float]:
    """時系列整合性スコアを計算

    連続フレーム間のタイムスタンプ差を計算し、期待値（約10秒）との整合性を評価する。

    Args:
        results: [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト

    Returns:
        時系列整合性スコアの統計情報
    """
    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
    EXPECTED_INTERVAL_SECONDS = 10.0
    TOLERANCE_MIN = 8.0  # 許容範囲の下限（秒）
    TOLERANCE_MAX = 12.0  # 許容範囲の上限（秒）

    # タイムスタンプをdatetimeに変換
    timestamps: List[Optional[datetime]] = []
    for success, timestamp_str, _ in results:
        if success and timestamp_str:
            try:
                dt = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
                timestamps.append(dt)
            except ValueError:
                timestamps.append(None)
        else:
            timestamps.append(None)

    # 連続フレーム間の時間差を計算
    time_diffs: List[float] = []
    valid_pairs = 0
    consistency_count = 0

    for i in range(len(timestamps) - 1):
        if timestamps[i] is not None and timestamps[i + 1] is not None:
            diff_seconds = abs((timestamps[i + 1] - timestamps[i]).total_seconds())
            time_diffs.append(diff_seconds)
            valid_pairs += 1

            # 期待範囲内かチェック
            if TOLERANCE_MIN <= diff_seconds <= TOLERANCE_MAX:
                consistency_count += 1

    # 統計を計算
    if not time_diffs:
        return {
            "consistency_score": 0.0,
            "average_time_diff": 0.0,
            "min_time_diff": 0.0,
            "max_time_diff": 0.0,
            "valid_pairs": 0,
        }

    avg_time_diff = sum(time_diffs) / len(time_diffs)
    min_time_diff = min(time_diffs)
    max_time_diff = max(time_diffs)
    consistency_score = (
        (consistency_count / valid_pairs * 100.0) if valid_pairs > 0 else 0.0
    )

    return {
        "consistency_score": consistency_score,
        "average_time_diff": avg_time_diff,
        "min_time_diff": min_time_diff,
        "max_time_diff": max_time_diff,
        "valid_pairs": valid_pairs,
        "consistency_count": consistency_count,
    }


def extract_frames_with_timestamps(
    video_processor: VideoProcessor,
    timestamp_extractor: TimestampExtractor,
    interval_seconds: float = 10.0,
    num_frames: int = 100,
) -> List[Tuple[bool, Optional[str], float]]:
    """動画から指定間隔でフレームを抽出し、タイムスタンプを抽出

    Args:
        video_processor: VideoProcessorインスタンス
        timestamp_extractor: TimestampExtractorインスタンス
        interval_seconds: フレーム抽出間隔（秒）
        num_frames: 抽出するフレーム数

    Returns:
        [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト
    """
    if video_processor.fps is None or video_processor.fps <= 0:
        raise ValueError("動画のFPSが取得できません")

    # フレーム間隔を計算
    frame_interval = int(video_processor.fps * interval_seconds)
    logger.info(f"フレーム抽出間隔: {frame_interval}フレーム ({interval_seconds}秒)")

    results: List[Tuple[bool, Optional[str], float]] = []

    # 動画を先頭に戻す
    video_processor.reset()

    for i in range(num_frames):
        frame_number = i * frame_interval

        # フレーム数が範囲外の場合は終了
        if (
            video_processor.total_frames
            and frame_number >= video_processor.total_frames
        ):
            logger.warning(
                f"フレーム {frame_number} が範囲外です（総フレーム数: {video_processor.total_frames}）"
            )
            break

        # フレームを取得
        frame = video_processor.get_frame(frame_number)
        if frame is None:
            logger.warning(f"フレーム {frame_number} の取得に失敗しました")
            results.append((False, None, 0.0))
            continue

        # タイムスタンプを抽出（信頼度付き）
        timestamp, confidence = timestamp_extractor.extract_with_confidence(frame)
        success = timestamp is not None

        if success:
            logger.debug(
                f"フレーム {frame_number}: タイムスタンプ抽出成功 - {timestamp} (信頼度: {confidence:.2f})"
            )
        else:
            logger.debug(f"フレーム {frame_number}: タイムスタンプ抽出失敗")

        results.append((success, timestamp, confidence))

        # 進捗表示（10フレームごと）
        if (i + 1) % 10 == 0:
            logger.info(f"進捗: {i + 1}/{num_frames} フレーム処理済み")

    return results


def print_results(
    extraction_stats: Dict[str, float], consistency_stats: Dict[str, float]
) -> None:
    """結果をコンソールに出力

    Args:
        extraction_stats: 抽出成功率の統計情報
        consistency_stats: 時系列整合性スコアの統計情報
    """
    print("=" * 80)
    print("タイムスタンプ抽出スコア評価結果")
    print("=" * 80)
    print()

    # 抽出成功率
    print("【抽出成功率】")
    print(f"  処理フレーム数: {int(extraction_stats['total_frames'])}")
    print(f"  抽出成功数: {int(extraction_stats['success_count'])}")
    print(f"  抽出成功率: {extraction_stats['success_rate']:.2f}%")
    print()

    # 信頼度統計
    print("【信頼度統計】")
    print(f"  平均信頼度: {extraction_stats['average_confidence']:.4f}")
    print(f"  最小信頼度: {extraction_stats['min_confidence']:.4f}")
    print(f"  最大信頼度: {extraction_stats['max_confidence']:.4f}")
    print()

    # 時系列整合性
    print("【時系列整合性スコア】")
    print(f"  整合性スコア: {consistency_stats['consistency_score']:.2f}%")
    print(f"  有効ペア数: {int(consistency_stats['valid_pairs'])}")
    print(f"  期待範囲内ペア数: {int(consistency_stats.get('consistency_count', 0))}")
    print()

    if consistency_stats["valid_pairs"] > 0:
        print("【時間差統計】")
        print(f"  平均時間差: {consistency_stats['average_time_diff']:.2f}秒")
        print(f"  最小時間差: {consistency_stats['min_time_diff']:.2f}秒")
        print(f"  最大時間差: {consistency_stats['max_time_diff']:.2f}秒")
        print()

    print("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="タイムスタンプ抽出スコア評価ツール"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="動画ファイルのパス（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="フレーム抽出間隔（秒、デフォルト: 10.0）",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="抽出するフレーム数（デフォルト: 100）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("タイムスタンプ抽出スコア評価ツール 起動")
    logger.info("=" * 80)

    video_processor = None

    try:
        # 設定ファイルの読み込み
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return 1

        config = ConfigManager(str(config_path))
        logger.info(f"設定ファイルを読み込みました: {config_path}")

        # 動画ファイルのパスを決定
        video_path = args.video or config.get("video.input_path")
        if not video_path:
            logger.error("動画ファイルのパスが指定されていません")
            return 1

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"動画ファイルが見つかりません: {video_path}")
            return 1

        logger.info(f"動画ファイル: {video_path}")

        # 動画処理の初期化
        video_processor = VideoProcessor(str(video_path))
        if not video_processor.open():
            logger.error("動画ファイルを開けませんでした")
            return 1

        logger.info(f"動画情報:")
        logger.info(f"  解像度: {video_processor.width}x{video_processor.height}")
        logger.info(f"  FPS: {video_processor.fps}")
        logger.info(f"  総フレーム数: {video_processor.total_frames}")

        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        logger.info("タイムスタンプ抽出器を初期化しました")

        # フレーム抽出とタイムスタンプ抽出
        logger.info(f"フレーム抽出を開始します（間隔: {args.interval}秒、フレーム数: {args.num_frames}）")
        results = extract_frames_with_timestamps(
            video_processor,
            timestamp_extractor,
            interval_seconds=args.interval,
            num_frames=args.num_frames,
        )

        logger.info(f"フレーム抽出完了: {len(results)}フレーム処理済み")

        # スコア算出
        logger.info("スコアを算出しています...")
        extraction_stats = calculate_extraction_rate(results)
        consistency_stats = calculate_temporal_consistency(results)

        # 結果を出力
        print_results(extraction_stats, consistency_stats)

        logger.info("処理が正常に完了しました")
        return 0

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"値エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    finally:
        if video_processor is not None:
            try:
                video_processor.release()
            except Exception as e:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")


if __name__ == "__main__":
    sys.exit(main())

