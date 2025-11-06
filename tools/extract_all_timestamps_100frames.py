"""先頭100フレーム内でタイムスタンプが読み取れるフレームを全て抽出するスクリプト

タイムスタンプ検証を緩和して、先頭100フレーム内でタイムスタンプが
読み取れるフレームを全て抽出し、結果を確認します。
"""

import csv
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2
from src.utils import setup_logging, setup_output_directories
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


def extract_all_timestamps_from_frames(
    video_path: str,
    max_frames: int = 100,
    config_path: str = "config.yaml",
    output_dir: str = "output/extracted_frames_test",
):
    """先頭100フレーム内でタイムスタンプが読み取れるフレームを全て抽出

    Args:
        video_path: 動画ファイルのパス
        max_frames: 最大処理フレーム数（デフォルト: 100）
        config_path: 設定ファイルのパス
        output_dir: 出力ディレクトリ
    """
    # 設定の読み込み
    config = ConfigManager(config_path)
    output_path = Path(output_dir)
    setup_output_directories(output_path)

    # ロギング設定
    setup_logging(debug_mode=True, output_dir=str(output_path.parent))
    logger.info("=" * 80)
    logger.info(f"先頭{max_frames}フレーム内でタイムスタンプ抽出開始")
    logger.info("=" * 80)

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return []

    # 動画の情報を取得
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"動画ファイルを開けません: {video_path}")
        return []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    logger.info(f"動画情報: 総フレーム数={total_video_frames}, FPS={fps:.2f}")
    logger.info(f"処理対象: 先頭{max_frames}フレーム（{max_frames/fps:.2f}秒分）")

    # 設定からパラメータを取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    ocr_config = config.get("ocr", {})

    # タイムスタンプ抽出器を初期化（検証を無効化）
    # 信頼度閾値を下げ、タイムスタンプ検証を無効化
    # 検証を無効化するために、カスタムバリデーターを作成
    from src.timestamp.timestamp_validator import TemporalValidator

    class NoOpValidator:
        """検証をスキップするダミーバリデーター"""

        def validate(self, timestamp, frame_idx):
            return True, 1.0, "Validation disabled"

        def reset(self):
            pass

    extractor = TimestampExtractorV2(
        confidence_threshold=0.3,  # 低めの閾値で多くのフレームを抽出
        roi_config=extraction_config.get("roi"),
        fps=fps,
        enabled_ocr_engines=ocr_config.get("engines"),
        use_improved_validator=False,  # 検証を無効化
    )
    # 検証を無効化
    extractor.validator = NoOpValidator()

    # ビデオプロセッサを初期化
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    # ステップ1: 先頭100フレームからタイムスタンプを全て抽出
    all_extracted_frames = []

    try:
        logger.info("ステップ1: 先頭100フレームからタイムスタンプを抽出中...")
        for frame_idx in tqdm(
            range(min(max_frames, total_video_frames)), desc="フレーム処理中"
        ):
            frame = video_processor.get_frame(frame_idx)
            if frame is None:
                logger.warning(f"フレーム{frame_idx}を取得できませんでした")
                continue

            # タイムスタンプ抽出（検証を緩和）
            result = extractor.extract(frame, frame_idx)

            if result and result.get("timestamp"):
                timestamp = result["timestamp"]
                confidence = result.get("confidence", 0.0)
                ocr_text = result.get("ocr_text", "")

                all_extracted_frames.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp": timestamp,
                        "confidence": confidence,
                        "ocr_text": ocr_text,
                        "frame": frame,  # 後で保存するために保持
                    }
                )

                logger.debug(
                    f"フレーム{frame_idx}: {timestamp} "
                    f"(信頼度={confidence:.2f}, OCR={ocr_text})"
                )

    finally:
        video_processor.release()
        extractor.reset_validator()

    if not all_extracted_frames:
        logger.error("タイムスタンプを抽出できたフレームがありません")
        return []

    # ステップ2: 5分刻みの目標タイムスタンプを生成
    first_timestamp = all_extracted_frames[0]["timestamp"]
    last_timestamp = all_extracted_frames[-1]["timestamp"]

    # 先頭フレームの時刻を5分刻みに切り上げ（例: 16:04:16 -> 16:05:00）
    start_minute = (first_timestamp.minute // 5 + 1) * 5
    if start_minute >= 60:
        start_minute = 0
        start_hour = first_timestamp.hour + 1
    else:
        start_hour = first_timestamp.hour

    start_target = first_timestamp.replace(
        hour=start_hour, minute=start_minute, second=0, microsecond=0
    )

    # 終了時刻を5分刻みに切り下げ
    end_minute = (last_timestamp.minute // 5) * 5
    end_target = last_timestamp.replace(minute=end_minute, second=0, microsecond=0)

    # 5分刻みの目標タイムスタンプを生成
    target_timestamps = []
    current = start_target
    while current <= end_target:
        target_timestamps.append(current)
        current += timedelta(minutes=5)

    logger.info(f"\nステップ2: 5分刻みの目標タイムスタンプを生成")
    logger.info(
        f"  抽出範囲: {first_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ～ {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"  目標タイムスタンプ: {len(target_timestamps)}個")
    for i, ts in enumerate(target_timestamps[:5], 1):
        logger.info(f"    [{i}] {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    if len(target_timestamps) > 5:
        logger.info(f"    ... (他{len(target_timestamps)-5}個)")

    # ステップ3: 各目標タイムスタンプに最も近いフレームを選択
    logger.info("\nステップ3: 各目標タイムスタンプに最も近いフレームを選択中...")
    selected_frames = []
    tolerance_seconds = 60.0  # ±60秒以内のフレームを許容

    for target_ts in target_timestamps:
        best_frame = None
        min_diff = float("inf")

        for extracted in all_extracted_frames:
            timestamp = extracted["timestamp"]
            diff = abs((timestamp - target_ts).total_seconds())

            if diff < min_diff and diff <= tolerance_seconds:
                min_diff = diff
                best_frame = extracted

        if best_frame:
            selected_frames.append(
                {
                    "target_timestamp": target_ts,
                    "frame_index": best_frame["frame_index"],
                    "timestamp": best_frame["timestamp"],
                    "time_diff_seconds": min_diff,
                    "confidence": best_frame["confidence"],
                    "ocr_text": best_frame["ocr_text"],
                    "frame": best_frame["frame"],
                }
            )
            logger.info(
                f"  目標: {target_ts.strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"フレーム{best_frame['frame_index']}: {best_frame['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"(差={min_diff:.1f}秒)"
            )
        else:
            logger.warning(
                f"  目標: {target_ts.strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"±{tolerance_seconds}秒以内のフレームが見つかりませんでした"
            )

    # ステップ4: 選択されたフレームのみを保存
    logger.info(f"\nステップ4: {len(selected_frames)}枚のフレームを保存中...")
    results = []
    saved_count = 0

    for selected in selected_frames:
        timestamp_str = selected["timestamp"].strftime("%Y%m%d_%H%M%S")
        target_str = selected["target_timestamp"].strftime("%Y%m%d_%H%M%S")
        output_path_frame = (
            output_path / f"frame_{target_str}_idx{selected['frame_index']}.jpg"
        )

        cv2.imwrite(str(output_path_frame), selected["frame"])
        saved_count += 1

        results.append(
            {
                "target_timestamp": selected["target_timestamp"],
                "frame_index": selected["frame_index"],
                "timestamp": selected["timestamp"],
                "timestamp_str": timestamp_str,
                "time_diff_seconds": selected["time_diff_seconds"],
                "confidence": selected["confidence"],
                "ocr_text": selected["ocr_text"],
                "frame_path": str(output_path_frame),
            }
        )

        logger.info(
            f"  保存: {output_path_frame.name} "
            f"(フレーム{selected['frame_index']}, 差={selected['time_diff_seconds']:.1f}秒)"
        )

    # 結果をCSVで保存
    csv_path = output_path / "five_minute_interval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_timestamp",
            "extracted_timestamp",
            "frame_index",
            "time_diff_seconds",
            "confidence",
            "ocr_text",
            "frame_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "target_timestamp": r["target_timestamp"].strftime(
                        "%Y/%m/%d %H:%M:%S"
                    ),
                    "extracted_timestamp": r["timestamp"].strftime("%Y/%m/%d %H:%M:%S"),
                    "frame_index": r["frame_index"],
                    "time_diff_seconds": f"{r['time_diff_seconds']:.2f}",
                    "confidence": f"{r['confidence']:.4f}",
                    "ocr_text": r["ocr_text"],
                    "frame_path": r["frame_path"],
                }
            )

    logger.info("=" * 80)
    logger.info("抽出結果サマリー")
    logger.info("=" * 80)
    logger.info(f"処理フレーム数: {min(max_frames, total_video_frames)}")
    logger.info(f"タイムスタンプ抽出成功: {len(all_extracted_frames)}フレーム")
    logger.info(f"5分刻み目標タイムスタンプ: {len(target_timestamps)}個")
    logger.info(f"選択されたフレーム: {len(selected_frames)}枚")
    logger.info(f"保存されたフレーム画像: {saved_count}枚")
    logger.info(f"結果CSV: {csv_path}")

    # 選択されたフレームの詳細を表示
    if results:
        logger.info("\n選択されたフレーム（5分刻み）:")
        for i, r in enumerate(results, 1):
            logger.info(
                f"  [{i}] 目標: {r['target_timestamp'].strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"フレーム{r['frame_index']}: {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"(差={r['time_diff_seconds']:.1f}秒, 信頼度={r['confidence']:.2f})"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="先頭100フレーム内でタイムスタンプが読み取れるフレームを全て抽出")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=100, help="最大処理フレーム数（デフォルト: 100）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/extracted_frames_test",
        help="出力ディレクトリ（デフォルト: output/extracted_frames_test）",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 設定から動画パスを取得
    config = ConfigManager(args.config)
    video_path = config.get("video.input_path")

    results = extract_all_timestamps_from_frames(
        video_path=video_path,
        max_frames=args.max_frames,
        config_path=args.config,
        output_dir=args.output_dir,
    )

    logger.info(f"\n✅ 抽出完了: {len(results)}フレームのタイムスタンプを抽出しました")
