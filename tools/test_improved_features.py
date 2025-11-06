#!/usr/bin/env python
"""改善機能の精度テストツール

統合した改善機能（TemporalValidatorV2、重み付けスキーム、投票ロジック）の
精度をテストし、既存実装と比較します。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


def test_extractor(
    video_path: str,
    frame_indices: List[int],
    extractor: TimestampExtractorV2,
    config_name: str,
) -> Dict:
    """抽出器をテストして結果を返す

    Args:
        video_path: 動画ファイルのパス
        frame_indices: テストするフレーム番号のリスト
        extractor: テストする抽出器
        config_name: 設定名（ログ用）

    Returns:
        テスト結果の辞書
    """
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    results = {
        "config": config_name,
        "total_frames": len(frame_indices),
        "success_count": 0,
        "failure_count": 0,
        "confidences": [],
        "timestamps": [],
    }

    try:
        for frame_idx in frame_indices:
            # 各フレームのテスト前にバリデーターをリセット
            # タイムラプス動画では連続フレーム間の時間差が大きいため、
            # 各フレームを独立してテストする
            extractor.reset_validator()

            frame = video_processor.get_frame(frame_idx)
            if frame is None:
                logger.warning(f"Frame {frame_idx} not found")
                results["failure_count"] += 1
                continue

            result = extractor.extract(frame, frame_idx)
            if result:
                results["success_count"] += 1
                results["confidences"].append(result["confidence"])
                results["timestamps"].append(result["timestamp"])
            else:
                results["failure_count"] += 1

    finally:
        video_processor.release()

    # 統計を計算
    if results["confidences"]:
        results["avg_confidence"] = np.mean(results["confidences"])
        results["min_confidence"] = np.min(results["confidences"])
        results["max_confidence"] = np.max(results["confidences"])
    else:
        results["avg_confidence"] = 0.0
        results["min_confidence"] = 0.0
        results["max_confidence"] = 0.0

    results["success_rate"] = (
        results["success_count"] / results["total_frames"] * 100
        if results["total_frames"] > 0
        else 0.0
    )

    return results


def compare_configurations(
    video_path: str,
    frame_indices: List[int],
    config: Dict,
    output_dir: Path,
) -> Dict:
    """複数の設定を比較

    Args:
        video_path: 動画ファイルのパス
        frame_indices: テストするフレーム番号のリスト
        config: 設定辞書
        output_dir: 出力ディレクトリ

    Returns:
        比較結果の辞書
    """
    extraction_config = config.get("timestamp", {}).get("extraction", {})
    roi_config = extraction_config.get("roi")
    ocr_config = config.get("ocr", {})
    enabled_engines = ocr_config.get("engines", ["tesseract"])

    # ベースライン設定（既存実装）
    baseline_extractor = TimestampExtractorV2(
        confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
        roi_config=roi_config,
        fps=config.get("video", {}).get("fps", 30.0),
        enabled_ocr_engines=enabled_engines,
        use_improved_validator=False,
        use_weighted_consensus=False,
        use_voting_consensus=False,
    )

    # 改善機能1: TemporalValidatorV2のみ
    validator_v2_extractor = TimestampExtractorV2(
        confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
        roi_config=roi_config,
        fps=config.get("video", {}).get("fps", 30.0),
        enabled_ocr_engines=enabled_engines,
        use_improved_validator=True,
        base_tolerance_seconds=extraction_config.get("validator", {}).get(
            "base_tolerance_seconds", 10.0
        ),
        history_size=extraction_config.get("validator", {}).get("history_size", 10),
        z_score_threshold=extraction_config.get("validator", {}).get(
            "z_score_threshold", 2.0
        ),
        use_weighted_consensus=False,
        use_voting_consensus=False,
    )

    # 改善機能2: 重み付けスキーム
    weighted_extractor = TimestampExtractorV2(
        confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
        roi_config=roi_config,
        fps=config.get("video", {}).get("fps", 30.0),
        enabled_ocr_engines=enabled_engines,
        use_improved_validator=False,
        use_weighted_consensus=True,
        use_voting_consensus=False,
    )

    # 改善機能3: すべて有効
    all_improved_extractor = TimestampExtractorV2(
        confidence_threshold=extraction_config.get("confidence_threshold", 0.7),
        roi_config=roi_config,
        fps=config.get("video", {}).get("fps", 30.0),
        enabled_ocr_engines=enabled_engines,
        use_improved_validator=True,
        base_tolerance_seconds=extraction_config.get("validator", {}).get(
            "base_tolerance_seconds", 10.0
        ),
        history_size=extraction_config.get("validator", {}).get("history_size", 10),
        z_score_threshold=extraction_config.get("validator", {}).get(
            "z_score_threshold", 2.0
        ),
        use_weighted_consensus=True,
        use_voting_consensus=False,
    )

    # 各設定でテスト
    logger.info("=" * 80)
    logger.info("ベースライン（既存実装）をテスト中...")
    baseline_results = test_extractor(
        video_path, frame_indices, baseline_extractor, "Baseline"
    )

    logger.info("=" * 80)
    logger.info("TemporalValidatorV2をテスト中...")
    validator_v2_results = test_extractor(
        video_path, frame_indices, validator_v2_extractor, "TemporalValidatorV2"
    )

    logger.info("=" * 80)
    logger.info("重み付けスキームをテスト中...")
    weighted_results = test_extractor(
        video_path, frame_indices, weighted_extractor, "Weighted Consensus"
    )

    logger.info("=" * 80)
    logger.info("すべての改善機能をテスト中...")
    all_improved_results = test_extractor(
        video_path, frame_indices, all_improved_extractor, "All Improved"
    )

    # 結果を比較
    comparison = {
        "baseline": baseline_results,
        "validator_v2": validator_v2_results,
        "weighted": weighted_results,
        "all_improved": all_improved_results,
    }

    # 結果を表示
    logger.info("=" * 80)
    logger.info("比較結果")
    logger.info("=" * 80)

    for name, result in comparison.items():
        logger.info(f"\n{result['config']}:")
        logger.info(
            f"  成功率: {result['success_rate']:.2f}% ({result['success_count']}/{result['total_frames']})"
        )
        logger.info(
            f"  平均信頼度: {result['avg_confidence']:.4f} "
            f"(min: {result['min_confidence']:.4f}, max: {result['max_confidence']:.4f})"
        )

    # 改善判定
    baseline_success_rate = baseline_results["success_rate"]
    baseline_avg_confidence = baseline_results["avg_confidence"]

    improvements = {}
    for name, result in comparison.items():
        if name == "baseline":
            continue

        success_improvement = result["success_rate"] - baseline_success_rate
        confidence_improvement = result["avg_confidence"] - baseline_avg_confidence

        improvements[name] = {
            "success_rate_change": success_improvement,
            "confidence_change": confidence_improvement,
            "is_improved": success_improvement >= 0
            and confidence_improvement >= -0.05,  # 信頼度が5%以上低下しなければ改善とみなす
        }

    logger.info("\n" + "=" * 80)
    logger.info("改善判定")
    logger.info("=" * 80)

    for name, improvement in improvements.items():
        status = "✅ 改善" if improvement["is_improved"] else "❌ 精度低下"
        logger.info(f"\n{comparison[name]['config']}: {status}")
        logger.info(f"  成功率の変化: {improvement['success_rate_change']:+.2f}%")
        logger.info(f"  信頼度の変化: {improvement['confidence_change']:+.4f}")

    # 結果をJSONに保存
    import json

    output_file = output_dir / "improved_features_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "comparison": comparison,
                "improvements": improvements,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"\n結果を保存しました: {output_file}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="改善機能の精度テストツール")
    parser.add_argument(
        "--video",
        type=str,
        default="input/merged_moviefiles.mov",
        help="テストする動画ファイルのパス",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/improved_features_test",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="テストするフレーム番号のリスト（指定しない場合は最初の10フレーム）",
    )

    args = parser.parse_args()

    # ログ設定
    setup_logging()

    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.config

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # フレーム番号の決定
    if args.frames:
        frame_indices = args.frames
    else:
        # デフォルト: 最初の10フレーム
        frame_indices = list(range(0, 10))

    logger.info(f"テストフレーム: {frame_indices}")

    # 比較実行
    comparison = compare_configurations(args.video, frame_indices, config, output_dir)

    # 推奨設定を決定
    logger.info("\n" + "=" * 80)
    logger.info("推奨設定")
    logger.info("=" * 80)

    baseline = comparison["baseline"]
    best_config = "baseline"
    best_score = baseline["success_rate"] * 0.5 + baseline["avg_confidence"] * 50

    for name, result in comparison.items():
        if name == "baseline":
            continue
        score = result["success_rate"] * 0.5 + result["avg_confidence"] * 50
        if score > best_score:
            best_score = score
            best_config = name

    logger.info(f"推奨設定: {comparison[best_config]['config']}")
    logger.info(f"  成功率: {comparison[best_config]['success_rate']:.2f}%")
    logger.info(f"  平均信頼度: {comparison[best_config]['avg_confidence']:.4f}")


if __name__ == "__main__":
    main()
