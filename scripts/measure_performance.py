"""パフォーマンス測定スクリプト

実際の動画データで処理時間とメモリ使用量を測定します。
"""

import argparse
import json
import logging
from pathlib import Path
import time

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutilがインストールされていません。メモリ測定ができません。")
import numpy as np

from src.config import ConfigManager
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def measure_memory_usage() -> float:
    """現在のメモリ使用量を測定（MB単位）

    Returns:
        メモリ使用量（MB）
    """
    if psutil is None:
        return 0.0
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB単位に変換


def measure_processing_time(
    video_path: str,
    config: ConfigManager,
    max_frames: int | None = None,
) -> dict[str, float]:
    """処理時間を測定

    Args:
        video_path: 動画ファイルのパス
        config: ConfigManagerインスタンス
        max_frames: 最大処理フレーム数（Noneの場合は全フレーム）

    Returns:
        処理時間の辞書（秒単位）
    """
    logger.info("=" * 80)
    logger.info("パフォーマンス測定を開始")
    logger.info("=" * 80)

    # 初期メモリ使用量
    initial_memory = measure_memory_usage()
    logger.info(f"初期メモリ使用量: {initial_memory:.2f} MB")

    # パイプラインの初期化
    logger.info("パイプラインを初期化中...")
    orchestrator = PipelineOrchestrator(config, logger)
    orchestrator.setup_output_directories(use_session_management=False)

    # 各フェーズの処理時間を測定
    phase_times = {}

    # Phase 1: フレーム抽出
    logger.info("Phase 1: フレーム抽出を実行中...")
    start_time = time.time()
    extraction_results = orchestrator.extract_frames(video_path)
    phase_times["phase1_extraction"] = time.time() - start_time
    logger.info(f"  Phase 1完了: {phase_times['phase1_extraction']:.2f} 秒")

    if max_frames:
        extraction_results = extraction_results[:max_frames]

    # Phase 2: 検出
    logger.info("Phase 2: 人物検出を実行中...")
    start_time = time.time()
    detection_results, sample_frames = orchestrator.run_detection(extraction_results)
    phase_times["phase2_detection"] = time.time() - start_time
    logger.info(f"  Phase 2完了: {phase_times['phase2_detection']:.2f} 秒")

    # Phase 2.5: 追跡（有効な場合）
    if config.get("tracking.enabled", False):
        logger.info("Phase 2.5: オブジェクト追跡を実行中...")
        start_time = time.time()
        tracked_results, _ = orchestrator.run_tracking(detection_results, sample_frames)
        phase_times["phase2.5_tracking"] = time.time() - start_time
        logger.info(f"  Phase 2.5完了: {phase_times['phase2.5_tracking']:.2f} 秒")
        detection_results = tracked_results

    # Phase 3: 座標変換
    logger.info("Phase 3: 座標変換を実行中...")
    start_time = time.time()
    transform_results = orchestrator.run_transform(detection_results)
    phase_times["phase3_transform"] = time.time() - start_time
    logger.info(f"  Phase 3完了: {phase_times['phase3_transform']:.2f} 秒")

    # Phase 4: 集計
    logger.info("Phase 4: 集計を実行中...")
    start_time = time.time()
    aggregation_results = orchestrator.run_aggregation(transform_results)
    phase_times["phase4_aggregation"] = time.time() - start_time
    logger.info(f"  Phase 4完了: {phase_times['phase4_aggregation']:.2f} 秒")

    # Phase 5: 可視化
    logger.info("Phase 5: 可視化を実行中...")
    start_time = time.time()
    orchestrator.run_visualization(transform_results, aggregation_results)
    phase_times["phase5_visualization"] = time.time() - start_time
    logger.info(f"  Phase 5完了: {phase_times['phase5_visualization']:.2f} 秒")

    # 最終メモリ使用量
    final_memory = measure_memory_usage()
    logger.info(f"最終メモリ使用量: {final_memory:.2f} MB")

    # 総処理時間
    total_time = sum(phase_times.values())
    phase_times["total"] = total_time

    # メモリ増加量
    memory_increase = final_memory - initial_memory

    return {
        "phase_times": phase_times,
        "memory": {
            "initial_mb": initial_memory,
            "final_mb": final_memory,
            "increase_mb": memory_increase,
        },
        "num_frames": len(extraction_results),
    }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="パフォーマンス測定スクリプト")
    parser.add_argument("--video", type=str, required=True, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--output", type=str, default="performance_metrics.json", help="出力ファイルのパス")
    parser.add_argument("--max-frames", type=int, help="最大処理フレーム数（テスト用）")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    # ファイルパスの確認
    video_path = Path(args.video)
    config_path = Path(args.config)

    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    # 設定の読み込み
    config = ConfigManager(str(config_path))

    # パフォーマンス測定
    try:
        performance_metrics = measure_processing_time(str(video_path), config, args.max_frames)
    except Exception as e:
        logger.error(f"パフォーマンス測定中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # 結果の表示
    logger.info("=" * 80)
    logger.info("パフォーマンス測定結果")
    logger.info("=" * 80)
    logger.info(f"  処理フレーム数: {performance_metrics['num_frames']}")
    logger.info(f"  総処理時間: {performance_metrics['phase_times']['total']:.2f} 秒")
    logger.info(
        f"  フレームあたりの処理時間: {performance_metrics['phase_times']['total'] / performance_metrics['num_frames']:.2f} 秒/フレーム"
    )
    logger.info("")
    logger.info("  フェーズ別処理時間:")
    for phase, time_sec in performance_metrics["phase_times"].items():
        if phase != "total":
            percentage = (time_sec / performance_metrics["phase_times"]["total"]) * 100
            logger.info(f"    {phase}: {time_sec:.2f} 秒 ({percentage:.1f}%)")
    logger.info("")
    logger.info("  メモリ使用量:")
    logger.info(f"    初期: {performance_metrics['memory']['initial_mb']:.2f} MB")
    logger.info(f"    最終: {performance_metrics['memory']['final_mb']:.2f} MB")
    logger.info(f"    増加量: {performance_metrics['memory']['increase_mb']:.2f} MB")
    logger.info("=" * 80)

    # 目標値との比較
    target_time_per_frame = 2.0  # 秒/フレーム
    target_memory_increase = 12.0 * 1024  # MB（12GB）

    avg_time_per_frame = performance_metrics["phase_times"]["total"] / performance_metrics["num_frames"]

    logger.info("目標値との比較:")
    logger.info(
        f"  処理時間目標: {target_time_per_frame:.1f} 秒/フレーム {'✅ 達成' if avg_time_per_frame <= target_time_per_frame else '❌ 未達成'}"
    )
    logger.info(
        f"  メモリ増加目標: {target_memory_increase:.0f} MB {'✅ 達成' if performance_metrics['memory']['increase_mb'] <= target_memory_increase else '❌ 未達成'}"
    )

    # 結果をJSONファイルに保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        **performance_metrics,
        "targets": {
            "time_per_frame_seconds": target_time_per_frame,
            "memory_increase_mb": target_memory_increase,
        },
        "achieved": {
            "time_per_frame": avg_time_per_frame <= target_time_per_frame,
            "memory_increase": performance_metrics["memory"]["increase_mb"] <= target_memory_increase,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"測定結果を保存しました: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
