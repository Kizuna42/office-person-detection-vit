#!/usr/bin/env python3
"""施策3: 日付部分の独立検証 シンプル検証ツール

単体テスト、統合テスト、精度評価を実施します。
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import VideoProcessor
from src.utils import setup_logging

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"


def validate_date_format(date_str: str) -> Tuple[bool, Optional[str]]:
    """日付形式の妥当性をチェック"""
    try:
        year, month, day = date_str.split("/")
        year_i = int(year)
        month_i = int(month)
        day_i = int(day)

        if not (2000 <= year_i <= 2100):
            return False, f"年が範囲外: {year_i}"
        if not (1 <= month_i <= 12):
            return False, f"月が範囲外: {month_i}"
        if not (1 <= day_i <= 31):
            return False, f"日が範囲外: {day_i}"

        try:
            datetime(year_i, month_i, day_i)
        except ValueError:
            return False, f"無効な日付: {date_str}"

        return True, None
    except (ValueError, AttributeError) as e:
        return False, f"日付形式が不正: {e}"


def extract_date_from_timestamp(timestamp: str) -> Optional[str]:
    """タイムスタンプから日付を抽出"""
    try:
        dt = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        return dt.strftime("%Y/%m/%d")
    except ValueError:
        return None


def unit_test(timestamp_extractor: TimestampExtractor, test_frames: List[np.ndarray]) -> Dict:
    """単体テスト: 日付・時刻の独立抽出の精度を測定"""
    logger.info("=" * 80)
    logger.info("単体テスト: 日付・時刻の独立抽出の精度測定")
    logger.info("=" * 80)

    results = {
        "total_frames": len(test_frames),
        "date_extraction_success": 0,
        "time_extraction_success": 0,
        "combined_extraction_success": 0,
        "date_validation_success": 0,
        "date_errors": [],
    }

    for i, frame in enumerate(test_frames):
        if frame is None or frame.size == 0:
            continue

        # ROI領域を抽出
        x, y, w, h = timestamp_extractor.roi
        frame_height, frame_width = frame.shape[:2]
        if x + w > frame_width or y + h > frame_height:
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)

        roi_image = frame[y : y + h, x : x + w]
        if roi_image.size == 0:
            continue

        # 日付部分の独立抽出
        date_str, date_conf = timestamp_extractor._extract_date_independently(roi_image)
        if date_str:
            results["date_extraction_success"] += 1
            is_valid, error_msg = validate_date_format(date_str)
            if is_valid:
                results["date_validation_success"] += 1
            else:
                results["date_errors"].append({"frame": i, "date": date_str, "error": error_msg})
        else:
            logger.debug(f"フレーム {i}: 日付抽出失敗")

        # 時刻部分の独立抽出
        time_str, time_conf = timestamp_extractor._extract_time_independently(roi_image)
        if time_str:
            results["time_extraction_success"] += 1

        # 結合抽出（extract()メソッドを使用）
        combined_timestamp = timestamp_extractor.extract(frame, frame_index=i)
        if combined_timestamp:
            results["combined_extraction_success"] += 1
            # 日付の妥当性チェック
            extracted_date = extract_date_from_timestamp(combined_timestamp)
            if extracted_date:
                is_valid, error_msg = validate_date_format(extracted_date)
                if not is_valid:
                    results["date_errors"].append({
                        "frame": i,
                        "timestamp": combined_timestamp,
                        "date": extracted_date,
                        "error": error_msg,
                    })

        if (i + 1) % 10 == 0:
            logger.info(f"進捗: {i + 1}/{len(test_frames)} フレーム処理済み")

    # 統計を計算
    total = results["total_frames"]
    results["statistics"] = {
        "date_extraction_rate": (results["date_extraction_success"] / total * 100.0) if total > 0 else 0.0,
        "time_extraction_rate": (results["time_extraction_success"] / total * 100.0) if total > 0 else 0.0,
        "combined_extraction_rate": (results["combined_extraction_success"] / total * 100.0) if total > 0 else 0.0,
        "date_validation_rate": (results["date_validation_success"] / results["date_extraction_success"] * 100.0) if results["date_extraction_success"] > 0 else 0.0,
    }

    return results


def integration_test(
    video_processor: VideoProcessor,
    timestamp_extractor: TimestampExtractor,
    sample_interval: int = 30,
) -> Dict:
    """統合テスト: 実際の動画で全フレームを処理"""
    logger.info("=" * 80)
    logger.info("統合テスト: 実際の動画で全フレームを処理")
    logger.info("=" * 80)

    results = {
        "total_frames": 0,
        "processed_frames": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "date_errors": [],
        "temporal_inconsistencies": [],
    }

    video_processor.reset()
    frame_count = 0
    last_timestamp: Optional[datetime] = None

    while True:
        ret, frame = video_processor.read_next_frame()
        if not ret or frame is None:
            break

        results["total_frames"] += 1

        if frame_count % sample_interval == 0:
            try:
                timestamp = timestamp_extractor.extract(frame, frame_index=frame_count)
                results["processed_frames"] += 1

                if timestamp:
                    try:
                        timestamp_dt = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
                        results["successful_extractions"] += 1

                        # 日付の妥当性チェック
                        extracted_date = extract_date_from_timestamp(timestamp)
                        if extracted_date:
                            is_valid, error_msg = validate_date_format(extracted_date)
                            if not is_valid:
                                results["date_errors"].append({
                                    "frame": frame_count,
                                    "timestamp": timestamp,
                                    "date": extracted_date,
                                    "error": error_msg,
                                })

                        # 時系列整合性チェック
                        if last_timestamp is not None:
                            days_diff = abs((timestamp_dt.date() - last_timestamp.date()).days)
                            if days_diff >= 1:
                                results["temporal_inconsistencies"].append({
                                    "frame": frame_count,
                                    "timestamp": timestamp,
                                    "last_timestamp": last_timestamp.strftime(TIMESTAMP_FORMAT),
                                    "days_diff": days_diff,
                                })

                        last_timestamp = timestamp_dt
                    except ValueError:
                        results["failed_extractions"] += 1
                else:
                    results["failed_extractions"] += 1
            except Exception as e:
                logger.error(f"フレーム {frame_count} の処理中にエラー: {e}")
                results["failed_extractions"] += 1

        frame_count += 1
        if frame_count % 1000 == 0:
            logger.info(f"進捗: {frame_count}フレーム処理済み")

    # 統計を計算
    processed = results["processed_frames"]
    successful = results["successful_extractions"]
    results["statistics"] = {
        "extraction_rate": (successful / processed * 100.0) if processed > 0 else 0.0,
        "date_error_rate": (len(results["date_errors"]) / successful * 100.0) if successful > 0 else 0.0,
        "temporal_inconsistency_rate": (len(results["temporal_inconsistencies"]) / successful * 100.0) if successful > 0 else 0.0,
    }

    return results


def print_results(unit_results: Optional[Dict], integration_results: Optional[Dict]) -> None:
    """結果を出力"""
    print("=" * 80)
    print("施策3: 日付部分の独立検証 検証結果")
    print("=" * 80)
    print()

    if unit_results:
        stats = unit_results["statistics"]
        print("【単体テスト結果】")
        print(f"  処理フレーム数: {unit_results['total_frames']}")
        print(f"  日付抽出成功率: {stats['date_extraction_rate']:.2f}%")
        print(f"  時刻抽出成功率: {stats['time_extraction_rate']:.2f}%")
        print(f"  結合抽出成功率: {stats['combined_extraction_rate']:.2f}%")
        print(f"  日付妥当性チェック成功率: {stats['date_validation_rate']:.2f}%")
        if unit_results["date_errors"]:
            print(f"  日付エラー数: {len(unit_results['date_errors'])}")
        print()

    if integration_results:
        stats = integration_results["statistics"]
        print("【統合テスト結果】")
        print(f"  総フレーム数: {integration_results['total_frames']}")
        print(f"  処理フレーム数: {integration_results['processed_frames']}")
        print(f"  抽出成功数: {integration_results['successful_extractions']}")
        print(f"  抽出失敗数: {integration_results['failed_extractions']}")
        print(f"  抽出成功率: {stats['extraction_rate']:.2f}%")
        print(f"  日付誤認識率: {stats['date_error_rate']:.2f}%")
        print(f"  時系列不整合率: {stats['temporal_inconsistency_rate']:.2f}%")
        if integration_results["date_errors"]:
            print(f"  日付エラー詳細（最初の5件）:")
            for error in integration_results["date_errors"][:5]:
                print(f"    フレーム {error['frame']}: {error.get('timestamp', 'N/A')} - {error['error']}")
        print()

    print("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="施策3: 日付部分の独立検証 シンプル検証ツール")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--video", type=str, default=None, help="動画ファイルのパス")
    parser.add_argument("--test-type", type=str, choices=["unit", "integration", "all"], default="all")
    parser.add_argument("--num-test-frames", type=int, default=50, help="単体テストで使用するフレーム数")
    parser.add_argument("--sample-interval", type=int, default=100, help="統合テストのサンプリング間隔")
    parser.add_argument("--output", type=str, default="output/strategy3_validation", help="結果出力ディレクトリ")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    setup_logging(args.debug)
    logger.info("=" * 80)
    logger.info("施策3: 日付部分の独立検証 シンプル検証ツール")
    logger.info("=" * 80)

    video_processor = None

    try:
        config = ConfigManager(args.config)
        video_path = args.video or config.get("video.input_path")
        if not video_path or not Path(video_path).exists():
            logger.error(f"動画ファイルが見つかりません: {video_path}")
            return 1

        video_processor = VideoProcessor(str(video_path))
        if not video_processor.open():
            logger.error("動画ファイルを開けませんでした")
            return 1

        logger.info(f"動画: {video_path} ({video_processor.total_frames}フレーム)")

        timestamp_extractor = TimestampExtractor()
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        # 単体テスト
        if args.test_type in ["unit", "all"]:
            logger.info("単体テストを開始...")
            test_frames = []
            video_processor.reset()
            frame_count = 0
            while len(test_frames) < args.num_test_frames:
                ret, frame = video_processor.read_next_frame()
                if not ret or frame is None:
                    break
                if frame_count % 10 == 0:
                    test_frames.append(frame.copy())
                frame_count += 1
                if frame_count >= args.num_test_frames * 10:
                    break

            logger.info(f"テスト用フレーム: {len(test_frames)}フレーム抽出")
            unit_results = unit_test(timestamp_extractor, test_frames)
            all_results["unit_test"] = unit_results

        # 統合テスト
        if args.test_type in ["integration", "all"]:
            logger.info("統合テストを開始...")
            integration_results = integration_test(video_processor, timestamp_extractor, args.sample_interval)
            all_results["integration_test"] = integration_results

        # 結果を出力
        print_results(
            all_results.get("unit_test"),
            all_results.get("integration_test"),
        )

        # JSONに保存
        output_json = output_dir / "validation_results.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"結果を保存: {output_json}")

        return 0

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        return 1
    finally:
        if video_processor:
            video_processor.release()


if __name__ == "__main__":
    sys.exit(main())

