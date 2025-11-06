#!/usr/bin/env python
"""高度な前処理手法のテストツール

複数の前処理戦略を試行し、最も効果的な手法を特定します。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.ocr_engine import MultiEngineOCR
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


def preprocess_strategy_1_upscale(roi: np.ndarray) -> np.ndarray:
    """戦略1: 拡大 + 標準前処理"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_strategy_2_invert(roi: np.ndarray) -> np.ndarray:
    """戦略2: 拡大 + 反転二値化"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def preprocess_strategy_3_adaptive(roi: np.ndarray) -> np.ndarray:
    """戦略3: 拡大 + 適応的閾値処理"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 適応的閾値処理
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def preprocess_strategy_4_morphology(roi: np.ndarray) -> np.ndarray:
    """戦略4: 拡大 + モルフォロジー強化"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # モルフォロジー演算
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # 文字を太くする
    kernel_thicken = np.ones((2, 1), np.uint8)
    binary = cv2.dilate(binary, kernel_thicken, iterations=1)

    return binary


def preprocess_strategy_5_grayscale_only(roi: np.ndarray) -> np.ndarray:
    """戦略5: 拡大 + グレースケールのみ（二値化なし）"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 二値化せずにグレースケールのまま返す
    return enhanced


def preprocess_strategy_6_original(roi: np.ndarray) -> np.ndarray:
    """戦略6: 元の画像を拡大のみ（前処理なし）"""
    h, w = roi.shape[:2]
    scale = max(300 / w, 300 / h, 1.0)
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    return gray


def test_all_strategies(
    video_path: str,
    roi_config: dict,
    frame_indices: List[int],
    output_dir: Path,
):
    """すべての前処理戦略をテスト"""
    # サンプルフレームを抽出
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    try:
        frames = []
        for idx in tqdm(frame_indices, desc="フレーム抽出中"):
            frame = video_processor.get_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
    finally:
        video_processor.release()

    if not frames:
        logger.error("フレームを抽出できませんでした")
        return

    # ROI抽出器を初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)

    # OCRエンジンを初期化
    ocr_engine = MultiEngineOCR(enabled_engines=["tesseract"])

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 前処理戦略のリスト
    strategies = [
        ("strategy_1_upscale", preprocess_strategy_1_upscale),
        ("strategy_2_invert", preprocess_strategy_2_invert),
        ("strategy_3_adaptive", preprocess_strategy_3_adaptive),
        ("strategy_4_morphology", preprocess_strategy_4_morphology),
        ("strategy_5_grayscale_only", preprocess_strategy_5_grayscale_only),
        ("strategy_6_original", preprocess_strategy_6_original),
    ]

    results = []

    # 各戦略でテスト
    for strategy_name, strategy_func in tqdm(strategies, desc="前処理戦略テスト中"):
        logger.info(f"戦略: {strategy_name}")

        strategy_results = []

        for frame_idx, frame in tqdm(frames, desc=f"  {strategy_name}処理中", leave=False):
            # ROI抽出
            roi, roi_coords = roi_extractor.extract_roi(frame)

            # 前処理
            preprocessed = strategy_func(roi)

            # OCR実行
            ocr_text, ocr_confidence = ocr_engine.extract_with_consensus(preprocessed)

            strategy_results.append(
                {
                    "frame_idx": frame_idx,
                    "ocr_text": ocr_text,
                    "confidence": ocr_confidence,
                }
            )

            # 前処理済み画像を保存
            preprocessed_path = (
                output_dir / f"{strategy_name}_frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(str(preprocessed_path), preprocessed)

        # 平均信頼度を計算
        avg_confidence = (
            sum(r["confidence"] for r in strategy_results) / len(strategy_results)
            if strategy_results
            else 0.0
        )
        success_count = sum(1 for r in strategy_results if r["confidence"] > 0.0)

        results.append(
            {
                "strategy": strategy_name,
                "avg_confidence": avg_confidence,
                "success_count": success_count,
                "total_count": len(strategy_results),
                "results": strategy_results,
            }
        )

        logger.info(
            f"  平均信頼度: {avg_confidence:.4f}, 成功数: {success_count}/{len(strategy_results)}"
        )

    # 結果を比較
    logger.info("=" * 80)
    logger.info("前処理戦略比較結果")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"{result['strategy']}:")
        logger.info(f"  平均信頼度: {result['avg_confidence']:.4f}")
        logger.info(f"  成功数: {result['success_count']}/{result['total_count']}")
        logger.info("")

    # 最良の戦略を特定
    best_result = max(results, key=lambda x: (x["success_count"], x["avg_confidence"]))

    logger.info("=" * 80)
    logger.info(f"最良の戦略: {best_result['strategy']}")
    logger.info(f"  平均信頼度: {best_result['avg_confidence']:.4f}")
    logger.info(f"  成功数: {best_result['success_count']}/{best_result['total_count']}")
    logger.info("=" * 80)

    # 結果をJSONで保存
    import json

    report_path = output_dir / "advanced_preprocessing_comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n比較結果を保存しました: {report_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="高度な前処理手法のテスト")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--output-dir", type=str, help="出力ディレクトリ")
    parser.add_argument(
        "--frame-indices",
        type=str,
        default="0,1000,2000,3000,4000",
        help="テストするフレーム番号（カンマ区切り）",
    )
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 設定読み込み
    config = ConfigManager(args.config)

    # 動画パスの取得
    if args.video:
        video_path = args.video
    else:
        video_path = config.get("video.input_path")

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    # ROI設定の取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    roi_config = extraction_config.get("roi", {})

    # フレーム番号のパース
    frame_indices = [int(x.strip()) for x in args.frame_indices.split(",")]

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config.get("output.directory", "output")) / "advanced_preprocessing"
        )

    # テスト実行
    logger.info("=" * 80)
    logger.info("高度な前処理手法のテスト")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info("=" * 80)

    test_all_strategies(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
