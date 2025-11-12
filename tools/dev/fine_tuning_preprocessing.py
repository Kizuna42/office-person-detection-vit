#!/usr/bin/env python
"""前処理パラメータの細かい最適化ツール

拡大サイズ、CLAHE、モルフォロジー、傾き補正などのパラメータを
細かく調整して最適な組み合わせを探索します。
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


def deskew_image(image: np.ndarray, max_angle: float = 5.0) -> tuple[np.ndarray, float]:
    """画像の傾きを補正

    Args:
        image: 入力画像（二値画像）
        max_angle: 最大補正角度（度）

    Returns:
        (補正後の画像, 検出された角度) のタプル
    """
    # エッジ検出
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Hough線変換で傾きを検出
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) == 0:
        return image, 0.0

    # 角度を計算
    angles = []
    for line in lines[:20]:  # 最初の20本のみ使用
        if isinstance(line, list | tuple) and len(line) >= 2:
            if not isinstance(line[0], list | np.ndarray):
                _rho, theta = line[0], line[1]
            else:
                _rho, theta = line[0]
        else:
            _rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if -max_angle <= angle <= max_angle:
            angles.append(angle)

    if not angles:
        return image, 0.0

    # 中央値を角度として使用
    angle = np.median(angles)

    if abs(angle) < 0.1:
        return image, 0.0

    # 回転
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle


def preprocess_with_fine_params(
    roi: np.ndarray,
    min_size: int = 200,
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    use_gaussian_blur: bool = True,
    blur_kernel_size: int = 3,
    threshold_method: str = "otsu_auto",  # "otsu_auto", "otsu", "otsu_inv", "adaptive"
    morphology_close_kernel: tuple[int, int] = (2, 2),
    morphology_open_kernel: tuple[int, int] = (2, 2),
    dilate_kernel: tuple[int, int] = (2, 1),
    dilate_iterations: int = 1,
    sharpen_strength: float = 0.0,
    apply_deskew: bool = False,
) -> np.ndarray:
    """細かく調整されたパラメータでROIを前処理

    Args:
        roi: ROI画像（BGR形式）
        min_size: 最小拡大サイズ
        clahe_clip_limit: CLAHEのclipLimit
        clahe_tile_size: CLAHEのtileGridSize
        use_gaussian_blur: ガウシアンブラーを使用するか
        blur_kernel_size: ブラーのカーネルサイズ
        threshold_method: 二値化方法
        morphology_close_kernel: クロージングのカーネルサイズ
        morphology_open_kernel: オープニングのカーネルサイズ
        dilate_kernel: 膨張のカーネルサイズ
        dilate_iterations: 膨張の反復回数
        sharpen_strength: シャープ化の強度（0.0で無効）
        apply_deskew: 傾き補正を適用するか

    Returns:
        前処理済み画像
    """
    # ROI画像を拡大
    h, w = roi.shape[:2]
    scale = max(min_size / w, min_size / h, 1.0)
    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # グレースケール化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

    # ガウシアンブラー
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0) if use_gaussian_blur else gray

    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(blurred)

    # 二値化
    if threshold_method == "otsu_auto":
        # 通常と反転の両方を試す
        _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white_pixels1 = np.sum(binary1 == 255)
        white_pixels2 = np.sum(binary2 == 255)
        binary = binary1 if white_pixels1 > white_pixels2 else binary2
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "otsu_inv":
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshold_method == "none":
        # 二値化せずにグレースケールのまま
        binary = enhanced
    else:
        binary = enhanced

    # モルフォロジー演算
    if morphology_close_kernel[0] > 0 and morphology_close_kernel[1] > 0:
        kernel_close = np.ones(morphology_close_kernel, np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    if morphology_open_kernel[0] > 0 and morphology_open_kernel[1] > 0:
        kernel_open = np.ones(morphology_open_kernel, np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # 膨張（文字を太くする）
    if dilate_kernel[0] > 0 and dilate_kernel[1] > 0 and dilate_iterations > 0:
        kernel_dilate = np.ones(dilate_kernel, np.uint8)
        binary = cv2.dilate(binary, kernel_dilate, iterations=dilate_iterations)

    # 傾き補正
    if apply_deskew:
        binary, angle = deskew_image(binary)
        if abs(angle) > 0.1:
            logger.debug(f"傾き補正適用: {angle:.2f}度")

    # シャープ化
    if sharpen_strength > 0:
        kernel_sharpen = np.array(
            [
                [0, -sharpen_strength, 0],
                [-sharpen_strength, 1 + 4 * sharpen_strength, -sharpen_strength],
                [0, -sharpen_strength, 0],
            ]
        )
        binary = cv2.filter2D(binary, -1, kernel_sharpen)

    return binary


def test_fine_tuning(
    video_path: str,
    roi_config: dict,
    frame_indices: list[int],
    output_dir: Path,
):
    """細かいパラメータ調整のテスト"""
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

    # テストするパラメータセット
    param_sets = [
        # ベースライン（現在の実装）
        {
            "name": "baseline",
            "min_size": 200,
            "clahe_clip_limit": 3.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # 拡大サイズを増やす
        {
            "name": "larger_scale_300",
            "min_size": 300,
            "clahe_clip_limit": 3.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # 拡大サイズをさらに増やす
        {
            "name": "larger_scale_400",
            "min_size": 400,
            "clahe_clip_limit": 3.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # CLAHEを強化
        {
            "name": "stronger_clahe",
            "min_size": 300,
            "clahe_clip_limit": 4.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # モルフォロジーを強化
        {
            "name": "stronger_morphology",
            "min_size": 300,
            "clahe_clip_limit": 3.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (3, 3),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (3, 1),
            "dilate_iterations": 2,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # 傾き補正を追加
        {
            "name": "with_deskew",
            "min_size": 300,
            "clahe_clip_limit": 3.0,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.0,
            "apply_deskew": True,
        },
        # グレースケールのみ（二値化なし）
        {
            "name": "grayscale_only",
            "min_size": 300,
            "clahe_clip_limit": 3.0,
            "threshold_method": "none",
            "morphology_close_kernel": (0, 0),
            "morphology_open_kernel": (0, 0),
            "dilate_kernel": (0, 0),
            "dilate_iterations": 0,
            "sharpen_strength": 0.0,
            "apply_deskew": False,
        },
        # 最適化された組み合わせ
        {
            "name": "optimized_combination",
            "min_size": 350,
            "clahe_clip_limit": 3.5,
            "threshold_method": "otsu_auto",
            "morphology_close_kernel": (2, 2),
            "morphology_open_kernel": (2, 2),
            "dilate_kernel": (2, 1),
            "dilate_iterations": 1,
            "sharpen_strength": 0.3,
            "apply_deskew": True,
        },
    ]

    results = []

    # 各パラメータセットでテスト
    for params in tqdm(param_sets, desc="パラメータセットテスト中"):
        name = params.pop("name")
        logger.info(f"パラメータセット: {name}")

        param_results = []

        for frame_idx, frame in tqdm(frames, desc=f"  {name}処理中", leave=False):
            # ROI抽出
            roi, _roi_coords = roi_extractor.extract_roi(frame)

            # 前処理
            preprocessed = preprocess_with_fine_params(roi, **params)

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
            preprocessed_path = output_dir / f"{name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(preprocessed_path), preprocessed)

        # 平均信頼度を計算
        avg_confidence = sum(r["confidence"] for r in param_results) / len(param_results) if param_results else 0.0
        success_count = sum(1 for r in param_results if r["confidence"] > 0.0)

        # OCRテキストの正確性を評価（簡単なチェック）
        valid_count = 0
        for r in param_results:
            text = r["ocr_text"]
            # タイムスタンプ形式（YYYY/MM/DD HH:MM:SS）に近いかチェック
            if len(text) >= 15 and "/" in text and ":" in text:
                valid_count += 1

        results.append(
            {
                "name": name,
                "params": params,
                "avg_confidence": avg_confidence,
                "success_count": success_count,
                "valid_count": valid_count,
                "total_count": len(param_results),
                "results": param_results,
            }
        )

        logger.info(
            f"  平均信頼度: {avg_confidence:.4f}, 成功数: {success_count}/{len(param_results)}, 有効形式: {valid_count}/{len(param_results)}"
        )

    # 結果を比較
    logger.info("=" * 80)
    logger.info("細かいパラメータ調整結果")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"{result['name']}:")
        logger.info(f"  平均信頼度: {result['avg_confidence']:.4f}")
        logger.info(f"  成功数: {result['success_count']}/{result['total_count']}")
        logger.info(f"  有効形式: {result['valid_count']}/{result['total_count']}")
        logger.info("")

    # 最良のパラメータセットを特定（信頼度と有効形式の両方を考慮）
    best_result = max(results, key=lambda x: (x["valid_count"], x["avg_confidence"]))

    logger.info("=" * 80)
    logger.info(f"最良のパラメータセット: {best_result['name']}")
    logger.info(f"  平均信頼度: {best_result['avg_confidence']:.4f}")
    logger.info(f"  成功数: {best_result['success_count']}/{best_result['total_count']}")
    logger.info(f"  有効形式: {best_result['valid_count']}/{best_result['total_count']}")
    logger.info(f"  パラメータ: {best_result['params']}")
    logger.info("=" * 80)

    # 結果をJSONで保存
    import json

    report_path = output_dir / "fine_tuning_comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n比較結果を保存しました: {report_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="前処理パラメータの細かい最適化")
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

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("output.directory", "output")) / "fine_tuning_preprocessing"

    # テスト実行
    logger.info("=" * 80)
    logger.info("前処理パラメータの細かい最適化")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info("=" * 80)

    test_fine_tuning(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
