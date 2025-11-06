#!/usr/bin/env python3
"""動画の開始時刻・終了時刻を分析し、OCR読み取り結果と比較するツール"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager
from src.timestamp import TimestampExtractor
from src.video.video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> dict:
    """動画のメタデータを取得する

    Args:
        video_path: 動画ファイルのパス

    Returns:
        メタデータの辞書
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 動画の長さ（秒）
    duration_seconds = total_frames / fps if fps > 0 else 0
    duration_timedelta = timedelta(seconds=duration_seconds)

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "duration_timedelta": duration_timedelta,
    }


def extract_timestamp_from_frame(
    video_processor: VideoProcessor,
    frame_number: int,
    timestamp_extractor: TimestampExtractor,
) -> tuple[str | None, float]:
    """指定フレームからタイムスタンプを抽出する

    Args:
        video_processor: VideoProcessorインスタンス
        frame_number: フレーム番号
        timestamp_extractor: TimestampExtractorインスタンス

    Returns:
        (タイムスタンプ文字列, 信頼度) のタプル
    """
    try:
        frame = video_processor.get_frame(frame_number)
        if frame is None:
            return None, 0.0

        timestamp, confidence = timestamp_extractor.extract_with_confidence(frame)
        return timestamp, confidence
    except Exception as e:
        logger.error(f"フレーム {frame_number} のタイムスタンプ抽出に失敗: {e}")
        return None, 0.0


def analyze_video_timestamps(
    video_path: str,
    config_path: str,
    sample_frames: int = 10,
) -> dict:
    """動画の開始・終了時刻を分析する

    Args:
        video_path: 動画ファイルのパス
        config_path: 設定ファイルのパス
        sample_frames: サンプリングするフレーム数（最初と最後から）

    Returns:
        分析結果の辞書
    """
    logger.info("=" * 80)
    logger.info("動画タイムスタンプ分析ツール")
    logger.info("=" * 80)

    # メタデータ取得
    logger.info("動画メタデータを取得中...")
    metadata = get_video_metadata(video_path)
    logger.info(f"  解像度: {metadata['width']}x{metadata['height']}")
    logger.info(f"  FPS: {metadata['fps']:.2f}")
    logger.info(f"  総フレーム数: {metadata['total_frames']}")
    logger.info(f"  動画の長さ: {metadata['duration_seconds']:.2f}秒 ({metadata['duration_timedelta']})")

    # VideoProcessorとTimestampExtractorを初期化
    logger.info("動画処理器を初期化中...")
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    config = ConfigManager(config_path)
    timestamp_extractor = TimestampExtractor()

    # 最初の数フレームからタイムスタンプを抽出
    logger.info(f"最初の{sample_frames}フレームからタイムスタンプを抽出中...")
    first_timestamps = []
    for i in range(min(sample_frames, metadata["total_frames"])):
        timestamp, confidence = extract_timestamp_from_frame(
            video_processor, i, timestamp_extractor
        )
        if timestamp:
            first_timestamps.append((i, timestamp, confidence))
            logger.info(f"  フレーム {i}: {timestamp} (信頼度: {confidence:.2f})")

    # 最後の数フレームからタイムスタンプを抽出
    logger.info(f"最後の{sample_frames}フレームからタイムスタンプを抽出中...")
    last_timestamps = []
    last_frame_start = max(0, metadata["total_frames"] - sample_frames)
    for i in range(last_frame_start, metadata["total_frames"]):
        timestamp, confidence = extract_timestamp_from_frame(
            video_processor, i, timestamp_extractor
        )
        if timestamp:
            last_timestamps.append((i, timestamp, confidence))
            logger.info(f"  フレーム {i}: {timestamp} (信頼度: {confidence:.2f})")

    video_processor.release()

    # 結果をまとめる
    result = {
        "metadata": {
            "fps": metadata["fps"],
            "total_frames": metadata["total_frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "duration_seconds": metadata["duration_seconds"],
            "duration_formatted": str(metadata["duration_timedelta"]),
        },
        "first_timestamps": [
            {"frame": f, "timestamp": t, "confidence": c}
            for f, t, c in first_timestamps
        ],
        "last_timestamps": [
            {"frame": f, "timestamp": t, "confidence": c}
            for f, t, c in last_timestamps
        ],
    }

    # 開始時刻と終了時刻を推定
    if first_timestamps:
        first_ts_str = first_timestamps[0][1]
        try:
            start_dt = datetime.strptime(first_ts_str, "%Y/%m/%d %H:%M:%S")
            result["estimated_start"] = start_dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            result["estimated_start"] = None
    else:
        result["estimated_start"] = None

    if last_timestamps:
        last_ts_str = last_timestamps[-1][1]
        try:
            end_dt = datetime.strptime(last_ts_str, "%Y/%m/%d %H:%M:%S")
            result["estimated_end"] = end_dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            result["estimated_end"] = None
    else:
        result["estimated_end"] = None

    # メタデータから計算した終了時刻（開始時刻が取得できた場合）
    if result["estimated_start"]:
        start_dt = datetime.strptime(result["estimated_start"], "%Y/%m/%d %H:%M:%S")
        calculated_end_dt = start_dt + metadata["duration_timedelta"]
        result["calculated_end_from_metadata"] = calculated_end_dt.strftime(
            "%Y/%m/%d %H:%M:%S"
        )
    else:
        result["calculated_end_from_metadata"] = None

    return result


def print_comparison(result: dict) -> None:
    """結果を比較して表示する

    Args:
        result: 分析結果の辞書
    """
    print("\n" + "=" * 80)
    print("分析結果")
    print("=" * 80)

    print("\n【動画メタデータ】")
    print(f"  解像度: {result['metadata']['width']}x{result['metadata']['height']}")
    print(f"  FPS: {result['metadata']['fps']:.2f}")
    print(f"  総フレーム数: {result['metadata']['total_frames']:,}")
    print(f"  動画の長さ: {result['metadata']['duration_formatted']}")

    print("\n【OCR読み取り結果（最初のフレーム）】")
    if result["first_timestamps"]:
        for ts_info in result["first_timestamps"]:
            print(
                f"  フレーム {ts_info['frame']:6d}: {ts_info['timestamp']:20s} "
                f"(信頼度: {ts_info['confidence']:.2f})"
            )
        print(f"\n  → 推定開始時刻: {result['estimated_start']}")
    else:
        print("  （タイムスタンプを読み取れませんでした）")

    print("\n【OCR読み取り結果（最後のフレーム）】")
    if result["last_timestamps"]:
        for ts_info in result["last_timestamps"]:
            print(
                f"  フレーム {ts_info['frame']:6d}: {ts_info['timestamp']:20s} "
                f"(信頼度: {ts_info['confidence']:.2f})"
            )
        print(f"\n  → 推定終了時刻: {result['estimated_end']}")
    else:
        print("  （タイムスタンプを読み取れませんでした）")

    print("\n【比較】")
    if result["estimated_start"] and result["estimated_end"]:
        start_dt = datetime.strptime(result["estimated_start"], "%Y/%m/%d %H:%M:%S")
        end_dt = datetime.strptime(result["estimated_end"], "%Y/%m/%d %H:%M:%S")
        ocr_duration = end_dt - start_dt
        metadata_duration = timedelta(seconds=result["metadata"]["duration_seconds"])

        print(f"  OCR開始時刻:     {result['estimated_start']}")
        print(f"  OCR終了時刻:     {result['estimated_end']}")
        print(f"  OCR動画の長さ:   {ocr_duration}")

        if result["calculated_end_from_metadata"]:
            print(
                f"  メタデータから計算した終了時刻: {result['calculated_end_from_metadata']}"
            )
            calc_end_dt = datetime.strptime(
                result["calculated_end_from_metadata"], "%Y/%m/%d %H:%M:%S"
            )
            diff = (end_dt - calc_end_dt).total_seconds()
            print(f"  差: {diff:.2f}秒")

        print(f"\n  メタデータの動画の長さ: {metadata_duration}")
        diff_seconds = abs((ocr_duration - metadata_duration).total_seconds())
        print(f"  差: {diff_seconds:.2f}秒")

        if diff_seconds < 60:
            print("  ✅ OCR読み取り結果とメタデータは一致しています")
        elif diff_seconds < 300:
            print("  ⚠️  OCR読み取り結果とメタデータに若干の差があります")
        else:
            print("  ❌ OCR読み取り結果とメタデータに大きな差があります")
    else:
        print("  （開始時刻または終了時刻が取得できませんでした）")


def main() -> int:
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="動画の開始時刻・終了時刻を分析し、OCR読み取り結果と比較する"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="動画ファイルのパス（設定ファイルの値を使用する場合は省略）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=10,
        help="最初と最後からサンプリングするフレーム数（デフォルト: 10）",
    )

    args = parser.parse_args()

    # 設定ファイルの読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    config = ConfigManager(str(config_path))

    # 動画ファイルのパスを決定
    video_path = args.video or config.get("video.input_path")
    if not video_path:
        logger.error("動画ファイルのパスが指定されていません")
        return 1

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    try:
        # 分析実行
        result = analyze_video_timestamps(
            str(video_path), str(config_path), args.sample_frames
        )

        # 結果表示
        print_comparison(result)

        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


