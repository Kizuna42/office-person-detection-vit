#!/usr/bin/env python
"""精度ベンチマークツール

各改善施策の前後で精度を測定し、改善効果を評価します。
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.config import ConfigManager
from src.pipeline import FrameExtractionPipeline
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def run_benchmark(
    video_path: str,
    start_time: str,
    end_time: str,
    config_path: str = "config.yaml",
) -> dict:
    """精度ベンチマークを実行

    Args:
        video_path: 動画ファイルのパス
        start_time: 開始時刻（HH:MM形式）
        end_time: 終了時刻（HH:MM形式）
        config_path: 設定ファイルのパス

    Returns:
        ベンチマーク結果の辞書
    """
    from datetime import datetime

    config = ConfigManager(config_path)

    # 開始・終了日時の取得
    timestamp_config = config.get("timestamp", {})
    target_config = timestamp_config.get("target", {})

    start_datetime = None
    end_datetime = None
    if target_config:
        start_str = target_config.get("start_datetime")
        if start_str:
            base_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            hour, minute = map(int, start_time.split(":"))
            start_datetime = base_date.replace(hour=hour, minute=minute, second=0)

            hour, minute = map(int, end_time.split(":"))
            end_datetime = base_date.replace(hour=hour, minute=minute, second=0)

    # パイプライン初期化
    extraction_config = timestamp_config.get("extraction", {})
    sampling_config = timestamp_config.get("sampling", {})
    ocr_config = config.get("ocr", {})

    output_dir = Path(config.get("output.directory", "output")) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FrameExtractionPipeline(
        video_path=video_path,
        output_dir=str(output_dir),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        interval_minutes=config.get("video.frame_interval_minutes", 5),
        tolerance_seconds=config.get("video.tolerance_seconds", 10.0),
        confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
        coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 10.0),
        fine_search_window_seconds=sampling_config.get("search_window_seconds", 30.0),
        fps=config.get("video.fps", 30.0),
        roi_config=extraction_config.get("roi"),
        enabled_ocr_engines=ocr_config.get("engines"),
    )

    # フレーム抽出実行
    extraction_results = pipeline.run()

    if not extraction_results:
        return {
            "success": False,
            "total_frames": 0,
            "extracted_frames": 0,
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "valid_timestamps": 0,
            "invalid_timestamps": 0,
        }

    # 結果を分析
    total_expected = len(extraction_results)
    extracted_count = len([r for r in extraction_results if r.get("timestamp")])
    confidences = [r.get("confidence", 0.0) for r in extraction_results if r.get("confidence", 0.0) > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # 有効なタイムスタンプをカウント（形式チェック）
    valid_count = 0
    invalid_count = 0
    for r in tqdm(extraction_results, desc="タイムスタンプ検証中", leave=False):
        timestamp = r.get("timestamp")
        if timestamp:
            try:
                # タイムスタンプが正しくパースできるかチェック
                if isinstance(timestamp, datetime):
                    valid_count += 1
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
        else:
            invalid_count += 1

    success_rate = (extracted_count / total_expected * 100) if total_expected > 0 else 0.0

    result = {
        "success": True,
        "total_frames": total_expected,
        "extracted_frames": extracted_count,
        "success_rate": success_rate,
        "avg_confidence": avg_confidence,
        "valid_timestamps": valid_count,
        "invalid_timestamps": invalid_count,
        "timestamp": datetime.now().isoformat(),
    }

    return result


def compare_results(baseline: dict, improved: dict) -> dict:
    """ベースラインと改善後の結果を比較

    Args:
        baseline: ベースライン結果
        improved: 改善後結果

    Returns:
        比較結果の辞書
    """
    comparison = {
        "success_rate": {
            "baseline": baseline.get("success_rate", 0.0),
            "improved": improved.get("success_rate", 0.0),
            "delta": improved.get("success_rate", 0.0) - baseline.get("success_rate", 0.0),
            "is_improved": improved.get("success_rate", 0.0) > baseline.get("success_rate", 0.0),
        },
        "avg_confidence": {
            "baseline": baseline.get("avg_confidence", 0.0),
            "improved": improved.get("avg_confidence", 0.0),
            "delta": improved.get("avg_confidence", 0.0) - baseline.get("avg_confidence", 0.0),
            "is_improved": improved.get("avg_confidence", 0.0) > baseline.get("avg_confidence", 0.0),
        },
        "valid_timestamps": {
            "baseline": baseline.get("valid_timestamps", 0),
            "improved": improved.get("valid_timestamps", 0),
            "delta": improved.get("valid_timestamps", 0) - baseline.get("valid_timestamps", 0),
            "is_improved": improved.get("valid_timestamps", 0) > baseline.get("valid_timestamps", 0),
        },
    }

    # 総合評価（すべての指標が改善または維持）
    overall_improved = (
        (
            comparison["success_rate"]["is_improved"]
            or comparison["success_rate"]["delta"] >= -1.0  # 1%以内の低下は許容
        )
        and (
            comparison["avg_confidence"]["is_improved"]
            or comparison["avg_confidence"]["delta"] >= -0.01  # 0.01以内の低下は許容
        )
        and (
            comparison["valid_timestamps"]["is_improved"]
            or comparison["valid_timestamps"]["delta"] >= -1  # 1件以内の低下は許容
        )
    )

    comparison["overall_improved"] = overall_improved

    return comparison


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="精度ベンチマークツール")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--start-time", type=str, default="16:05", help="開始時刻（HH:MM形式）")
    parser.add_argument("--end-time", type=str, default="16:20", help="終了時刻（HH:MM形式）")
    parser.add_argument("--output", type=str, help="結果出力ファイル（JSON）")
    parser.add_argument("--baseline", type=str, help="ベースライン結果ファイル（比較用）")
    parser.add_argument("--compare", type=str, help="比較対象結果ファイル")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 比較モード
    if args.baseline and args.compare:
        logger.info("=" * 80)
        logger.info("ベンチマーク結果の比較")
        logger.info("=" * 80)

        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)

        with open(args.compare, encoding="utf-8") as f:
            improved = json.load(f)

        comparison = compare_results(baseline, improved)

        success_rate_baseline = baseline.get("success_rate", 0.0)
        success_rate_improved = improved.get("success_rate", 0.0)
        success_rate_delta = comparison["success_rate"]["delta"]
        logger.info(
            f"抽出成功率: {success_rate_baseline:.2f}% → " f"{success_rate_improved:.2f}% (Δ{success_rate_delta:+.2f}%)"
        )
        avg_conf_baseline = baseline.get("avg_confidence", 0.0)
        avg_conf_improved = improved.get("avg_confidence", 0.0)
        avg_conf_delta = comparison["avg_confidence"]["delta"]
        logger.info(f"平均信頼度: {avg_conf_baseline:.4f} → " f"{avg_conf_improved:.4f} (Δ{avg_conf_delta:+.4f})")
        valid_ts_baseline = baseline.get("valid_timestamps", 0)
        valid_ts_improved = improved.get("valid_timestamps", 0)
        valid_ts_delta = comparison["valid_timestamps"]["delta"]
        logger.info(f"有効タイムスタンプ: {valid_ts_baseline} → " f"{valid_ts_improved} (Δ{valid_ts_delta:+d})")
        logger.info("")

        if comparison["overall_improved"]:
            logger.info("✅ 改善が確認されました")
        else:
            logger.info("❌ 改善が見られませんでした")

        logger.info("=" * 80)

        # 比較結果を保存
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            logger.info(f"比較結果を保存しました: {args.output}")

        return 0 if comparison["overall_improved"] else 1

    # ベンチマーク実行
    config = ConfigManager(args.config)

    video_path = args.video if args.video else config.get("video.input_path")

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    logger.info("=" * 80)
    logger.info("精度ベンチマーク実行")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"時間範囲: {args.start_time} - {args.end_time}")
    logger.info("=" * 80)

    result = run_benchmark(
        video_path=video_path,
        start_time=args.start_time,
        end_time=args.end_time,
        config_path=args.config,
    )

    logger.info("=" * 80)
    logger.info("ベンチマーク結果")
    logger.info("=" * 80)
    logger.info(f"総フレーム数: {result.get('total_frames', 0)}")
    logger.info(f"抽出成功数: {result.get('extracted_frames', 0)}")
    logger.info(f"抽出成功率: {result.get('success_rate', 0.0):.2f}%")
    logger.info(f"平均信頼度: {result.get('avg_confidence', 0.0):.4f}")
    logger.info(f"有効タイムスタンプ: {result.get('valid_timestamps', 0)}")
    logger.info(f"無効タイムスタンプ: {result.get('invalid_timestamps', 0)}")
    logger.info("=" * 80)

    # 結果を保存
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"結果を保存しました: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
