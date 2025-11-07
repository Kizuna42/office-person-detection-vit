#!/usr/bin/env python
"""前処理パラメータのチューニングツール

CLAHE、二値化、ノイズ除去などの前処理パラメータを調整し、
A/Bテストで効果を測定します。
"""

import argparse
import logging
from pathlib import Path
import sys

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


def preprocess_roi_with_params(
    roi: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    use_otsu: bool = True,
    threshold_value: int = 127,
    denoise_h: float = 10.0,
    sharpen_strength: float = 1.0,
) -> np.ndarray:
    """指定されたパラメータでROIを前処理

    Args:
        roi: ROI画像（BGR形式）
        clahe_clip_limit: CLAHEのclipLimit
        clahe_tile_size: CLAHEのtileGridSize
        use_otsu: Otsu法を使用するか
        threshold_value: 固定閾値（use_otsu=Falseの場合）
        denoise_h: ノイズ除去の強度
        sharpen_strength: シャープ化の強度

    Returns:
        前処理済み画像
    """
    # グレースケール化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(gray)

    # 二値化
    if use_otsu:
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    denoised = cv2.fastNlMeansDenoising(binary, h=denoise_h)

    # シャープ化
    if sharpen_strength > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * sharpen_strength
        kernel[1, 1] = 9 - 8 * sharpen_strength
        sharpened = cv2.filter2D(denoised, -1, kernel)
    else:
        sharpened = denoised

    return sharpened


def test_preprocessing_parameters(
    video_path: str,
    roi_config: dict,
    frame_indices: list[int],
    param_sets: list[dict],
    output_dir: Path,
):
    """前処理パラメータのA/Bテスト

    Args:
        video_path: 動画ファイルのパス
        roi_config: ROI設定
        frame_indices: テストするフレーム番号のリスト
        param_sets: テストするパラメータセットのリスト
        output_dir: 出力ディレクトリ
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
        return

    # ROI抽出器を初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)

    # OCRエンジンを初期化（テスト用）
    ocr_engine = MultiEngineOCR(enabled_engines=["tesseract"])

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 各パラメータセットでテスト
    for param_idx, params in enumerate(tqdm(param_sets, desc="パラメータセットテスト中")):
        logger.info(f"パラメータセット {param_idx + 1}/{len(param_sets)}: {params}")

        param_results = []

        for frame_idx, frame in tqdm(frames, desc=f"  セット{param_idx+1}処理中", leave=False):
            # ROI抽出
            roi, roi_coords = roi_extractor.extract_roi(frame)

            # 前処理
            preprocessed = preprocess_roi_with_params(roi, **params)

            # OCR実行
            ocr_text, ocr_confidence = ocr_engine.extract_with_consensus(preprocessed)

            param_results.append(
                {
                    "frame_idx": frame_idx,
                    "ocr_text": ocr_text,
                    "confidence": ocr_confidence,
                }
            )

            # 前処理済み画像を保存
            param_name = f"param_{param_idx+1}"
            preprocessed_path = output_dir / f"{param_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(preprocessed_path), preprocessed)

        # 平均信頼度を計算
        avg_confidence = sum(r["confidence"] for r in param_results) / len(param_results) if param_results else 0.0

        results.append(
            {
                "params": params,
                "avg_confidence": avg_confidence,
                "results": param_results,
            }
        )

        logger.info(f"  平均信頼度: {avg_confidence:.4f}")

    # 結果を比較
    logger.info("=" * 80)
    logger.info("前処理パラメータ比較結果")
    logger.info("=" * 80)

    for i, result in enumerate(results):
        logger.info(f"パラメータセット {i+1}:")
        logger.info(f"  パラメータ: {result['params']}")
        logger.info(f"  平均信頼度: {result['avg_confidence']:.4f}")
        logger.info("")

    # 最良のパラメータセットを特定
    best_result = max(results, key=lambda x: x["avg_confidence"])
    best_idx = results.index(best_result)

    logger.info("=" * 80)
    logger.info(f"最良のパラメータセット: {best_idx + 1}")
    logger.info(f"  パラメータ: {best_result['params']}")
    logger.info(f"  平均信頼度: {best_result['avg_confidence']:.4f}")
    logger.info("=" * 80)

    # 結果をJSONで保存
    import json

    report_path = output_dir / "preprocessing_comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n比較結果を保存しました: {report_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="前処理パラメータのチューニング")
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
    video_path = args.video if args.video else config.get("video.input_path")

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    # ROI設定の取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    roi_config = extraction_config.get("roi", {})

    # フレーム番号のパース
    frame_indices = [int(x.strip()) for x in args.frame_indices.split(",")]

    # テストするパラメータセット
    param_sets = [
        # デフォルト
        {
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": (8, 8),
            "use_otsu": True,
            "denoise_h": 10.0,
            "sharpen_strength": 1.0,
        },
        # 高コントラスト
        {
            "clahe_clip_limit": 3.0,
            "clahe_tile_size": (8, 8),
            "use_otsu": True,
            "denoise_h": 10.0,
            "sharpen_strength": 1.0,
        },
        # 強ノイズ除去
        {
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": (8, 8),
            "use_otsu": True,
            "denoise_h": 15.0,
            "sharpen_strength": 1.0,
        },
        # 強シャープ化
        {
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": (8, 8),
            "use_otsu": True,
            "denoise_h": 10.0,
            "sharpen_strength": 1.5,
        },
    ]

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("output.directory", "output")) / "preprocessing_tuning"

    # テスト実行
    logger.info("=" * 80)
    logger.info("前処理パラメータのチューニング")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info(f"パラメータセット数: {len(param_sets)}")
    logger.info("=" * 80)

    test_preprocessing_parameters(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        param_sets=param_sets,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
