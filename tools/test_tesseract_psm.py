#!/usr/bin/env python
"""Tesseract PSMモードのテストツール

異なる--psmモード（6, 7, 8, 13）を試行し、精度を比較します。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.timestamp.timestamp_parser import TimestampParser
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.error("pytesseract is not available")


def test_psm_mode(
    roi: np.ndarray,
    psm_mode: int,
    whitelist: str = "0123456789/: ",
) -> tuple:
    """指定されたPSMモードでOCRを実行

    Args:
        roi: 前処理済みROI画像
        psm_mode: PSMモード（6, 7, 8, 13など）
        whitelist: 許可文字リスト

    Returns:
        (OCRテキスト, 信頼度) のタプル
    """
    if not TESSERACT_AVAILABLE:
        return "", 0.0

    config = f"--psm {psm_mode} --oem 3 -c tessedit_char_whitelist={whitelist}"

    try:
        # OCR実行
        text = pytesseract.image_to_string(roi, config=config).strip()

        # 信頼度を取得（詳細データから）
        try:
            data = pytesseract.image_to_data(
                roi, config=config, output_type=pytesseract.Output.DICT
            )
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            )
        except:
            # フォールバック: テキストの長さから推定（簡易的な方法）
            avg_confidence = min(len(text) / 20.0, 1.0) if text else 0.0

        return text, avg_confidence
    except Exception as e:
        logger.warning(f"PSM {psm_mode}でOCR実行中にエラー: {e}")
        return "", 0.0


def test_all_psm_modes(
    video_path: str,
    roi_config: dict,
    frame_indices: List[int],
    output_dir: Path,
) -> Dict:
    """すべてのPSMモードをテスト

    Args:
        video_path: 動画ファイルのパス
        roi_config: ROI設定
        frame_indices: テストするフレーム番号のリスト
        output_dir: 出力ディレクトリ

    Returns:
        テスト結果の辞書
    """
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
        return {}

    # ROI抽出器を初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)

    # タイムスタンプパーサーを初期化
    timestamp_parser = TimestampParser()

    # テストするPSMモード
    psm_modes = [6, 7, 8, 13]

    results = {}

    # 各PSMモードでテスト
    for psm_mode in tqdm(psm_modes, desc="PSMモードテスト中"):
        logger.info(f"PSMモード {psm_mode} をテスト中...")

        mode_results = []
        parse_success_count = 0

        for frame_idx, frame in tqdm(frames, desc=f"  PSM {psm_mode}処理中", leave=False):
            # ROI抽出
            roi, roi_coords = roi_extractor.extract_roi(frame)

            # 前処理
            preprocessed = roi_extractor.preprocess_roi(roi)

            # OCR実行
            ocr_text, ocr_confidence = test_psm_mode(preprocessed, psm_mode)

            # パース試行
            parsed_timestamp = None
            try:
                parsed_timestamp = timestamp_parser.parse(ocr_text)
                if parsed_timestamp:
                    parse_success_count += 1
            except:
                pass

            mode_results.append(
                {
                    "frame_idx": frame_idx,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_confidence,
                    "parsed": parsed_timestamp is not None,
                }
            )

        # 統計を計算
        avg_confidence = (
            sum(r["ocr_confidence"] for r in mode_results) / len(mode_results)
            if mode_results
            else 0.0
        )
        parse_success_rate = (
            (parse_success_count / len(mode_results) * 100) if mode_results else 0.0
        )

        results[psm_mode] = {
            "avg_confidence": avg_confidence,
            "parse_success_rate": parse_success_rate,
            "parse_success_count": parse_success_count,
            "total_count": len(mode_results),
            "results": mode_results,
        }

        logger.info(
            f"  PSM {psm_mode}: 平均信頼度={avg_confidence:.4f}, パース成功率={parse_success_rate:.2f}%"
        )

    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Tesseract PSMモードのテスト")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument(
        "--frame-indices",
        type=str,
        default="0,1000,2000,3000,4000",
        help="テストするフレーム番号（カンマ区切り）",
    )
    parser.add_argument("--output", type=str, help="結果出力ファイル（JSON）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    if not TESSERACT_AVAILABLE:
        logger.error("pytesseractが利用できません")
        return 1

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
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
    else:
        output_dir = Path(config.get("output.directory", "output")) / "psm_test"
        output_path = output_dir / "psm_test_results.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # テスト実行
    logger.info("=" * 80)
    logger.info("Tesseract PSMモードのテスト")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info("=" * 80)

    results = test_all_psm_modes(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        output_dir=output_dir,
    )

    # 結果を比較
    logger.info("=" * 80)
    logger.info("PSMモード比較結果")
    logger.info("=" * 80)

    best_psm = None
    best_score = -1.0

    for psm_mode, result in results.items():
        # スコア = パース成功率 * 0.7 + 平均信頼度 * 0.3
        score = (result["parse_success_rate"] / 100.0) * 0.7 + result[
            "avg_confidence"
        ] * 0.3
        logger.info(f"PSM {psm_mode}:")
        logger.info(f"  平均信頼度: {result['avg_confidence']:.4f}")
        logger.info(f"  パース成功率: {result['parse_success_rate']:.2f}%")
        logger.info(f"  総合スコア: {score:.4f}")
        logger.info("")

        if score > best_score:
            best_score = score
            best_psm = psm_mode

    logger.info("=" * 80)
    if best_psm:
        logger.info(f"最良のPSMモード: {best_psm} (スコア: {best_score:.4f})")
    logger.info("=" * 80)

    # 結果をJSONで保存
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n結果を保存しました: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
