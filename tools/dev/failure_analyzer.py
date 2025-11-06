#!/usr/bin/env python
"""OCR失敗ケースの自動分析ツール

抽出失敗したフレームを分析し、失敗理由を自動分類して改善策を提案します。
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from tqdm import tqdm

from src.config import ConfigManager
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """画像品質メトリクスを計算

    Args:
        image: 入力画像（グレースケールまたはBGR）

    Returns:
        品質メトリクスの辞書
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    metrics = {}

    # 1. コントラスト（標準偏差）
    metrics["contrast"] = float(np.std(gray))

    # 2. シャープネス（ラプラシアン分散）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics["sharpness"] = float(laplacian.var())

    # 3. 平均輝度
    metrics["brightness"] = float(np.mean(gray))

    # 4. ノイズレベル（高周波成分の推定）
    # Sobelフィルタでエッジ検出し、その分散をノイズ指標とする
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    metrics["noise_level"] = float(np.std(sobel_mag))

    # 5. 品質スコア（0-100）
    # コントラストとシャープネスを組み合わせた総合スコア
    contrast_score = min(metrics["contrast"] / 50.0, 1.0) * 50  # 最大50点
    sharpness_score = min(metrics["sharpness"] / 500.0, 1.0) * 50  # 最大50点
    metrics["quality_score"] = contrast_score + sharpness_score

    return metrics


def classify_failure_reason(
    row: Dict, quality_metrics: Dict[str, float], log_messages: List[str]
) -> Tuple[str, List[str]]:
    """失敗理由を分類し、改善策を提案

    Args:
        row: CSV行のデータ
        quality_metrics: 画像品質メトリクス
        log_messages: 関連するログメッセージ

    Returns:
        (失敗理由, 改善策のリスト)
    """
    reason = "unknown"
    recommendations = []

    # ログメッセージから失敗理由を推測
    log_text = " ".join(log_messages).lower()

    # 1. 画像品質が低い場合
    if quality_metrics["quality_score"] < 30:
        reason = "low_image_quality"
        recommendations.append("画像品質が低いです。前処理パラメータ（CLAHE、二値化）の調整を検討してください")
        if quality_metrics["contrast"] < 20:
            recommendations.append("コントラストが低いです。CLAHEのclipLimitを増やすことを検討してください")
        if quality_metrics["sharpness"] < 100:
            recommendations.append("画像がぼやけています。ROI領域の位置確認と前処理の強化を検討してください")
        if quality_metrics["brightness"] < 50:
            recommendations.append("画像が暗すぎます。明度調整の強化を検討してください")
        elif quality_metrics["brightness"] > 200:
            recommendations.append("画像が明るすぎます。明度調整の見直しを検討してください")

    # 2. OCR失敗
    elif "ocr failed" in log_text or "no ocr engines" in log_text:
        reason = "ocr_failure"
        recommendations.append("OCRエンジンがテキストを読み取れませんでした")
        if quality_metrics["quality_score"] < 50:
            recommendations.append("画像品質の改善によりOCR精度が向上する可能性があります")
        recommendations.append("OCRエンジンの設定（PSMモード、文字ホワイトリスト）の見直しを検討してください")

    # 3. パース失敗
    elif "parse failed" in log_text or "日の解析に失敗" in log_text:
        reason = "parse_failure"
        recommendations.append("OCR結果のパースに失敗しました")
        recommendations.append("OCR結果の前処理（正規化、誤認識補正）の強化を検討してください")
        recommendations.append("fuzzy_parseの誤認識補正ルールの追加を検討してください")

    # 4. 信頼度不足
    elif "low confidence" in log_text or "confidence" in log_text:
        reason = "low_confidence"
        recommendations.append("総合信頼度が閾値を下回りました")
        if quality_metrics["quality_score"] < 50:
            recommendations.append("画像品質の改善により信頼度が向上する可能性があります")
        recommendations.append("信頼度閾値の見直し（一時的に下げる）を検討してください")
        recommendations.append("OCRエンジンのアンサンブル強化を検討してください")

    # 5. 時系列不整合
    elif "temporal" in log_text or "時系列" in log_text or "backward" in log_text:
        reason = "temporal_inconsistency"
        recommendations.append("時系列検証に失敗しました")
        recommendations.append("前後フレームとの整合性を確認してください")
        recommendations.append("時系列許容範囲の調整を検討してください")

    # 6. ROI抽出失敗
    elif "empty roi" in log_text or "roi" in log_text.lower():
        reason = "roi_extraction_failure"
        recommendations.append("ROI領域の抽出に失敗しました")
        recommendations.append("config.yamlのROI設定を確認してください")
        recommendations.append("タイムスタンプ位置が動画内で変動していないか確認してください")

    # デフォルト: 画像品質が低い可能性
    else:
        if quality_metrics["quality_score"] < 50:
            reason = "low_image_quality"
            recommendations.append("画像品質が低い可能性があります")
        else:
            reason = "unknown"
            recommendations.append("ログを確認して詳細な原因を特定してください")

    return reason, recommendations


def analyze_failures(
    csv_path: Path, video_path: Path, output_dir: Path, log_path: Optional[Path] = None
) -> Dict:
    """失敗ケースを分析

    Args:
        csv_path: 抽出結果CSVファイルのパス
        video_path: 動画ファイルのパス
        output_dir: 出力ディレクトリ
        log_path: ログファイルのパス（オプション）

    Returns:
        分析結果の辞書
    """
    logger.info("=" * 80)
    logger.info("失敗ケース分析を開始")
    logger.info("=" * 80)

    # 出力ディレクトリ作成
    failed_frames_dir = output_dir / "failed_frames"
    failed_frames_dir.mkdir(parents=True, exist_ok=True)

    # CSV読み込み
    results = []
    target_frames = set()
    successful_frames = set()

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                target_frames.add(int(row.get("target_frame_idx", 0)))
                if row.get("timestamp") and row.get("timestamp") != "":
                    successful_frames.add(int(row.get("frame_idx", 0)))

    # 失敗フレームを特定
    failed_frame_indices = sorted(target_frames - successful_frames)
    logger.info(f"抽出成功: {len(successful_frames)}フレーム")
    logger.info(f"抽出失敗: {len(failed_frame_indices)}フレーム")

    # ログファイルから関連メッセージを抽出
    log_messages_by_frame = defaultdict(list)
    if log_path and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                # フレーム番号を抽出
                import re

                frame_match = re.search(r"Frame (\d+)", line)
                if frame_match:
                    frame_idx = int(frame_match.group(1))
                    if frame_idx in failed_frame_indices:
                        log_messages_by_frame[frame_idx].append(line.strip())

    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"動画ファイルを開けません: {video_path}")
        return {}

    # 失敗フレームを分析
    failure_stats = defaultdict(int)
    failure_details = []
    analyzed_count = 0

    max_frames_to_analyze = min(len(failed_frame_indices), 50)
    for frame_idx in tqdm(
        failed_frame_indices[:max_frames_to_analyze], desc="失敗フレーム分析中"
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # ROI領域を抽出（簡易版）
        # 実際のROI設定はconfig.yamlから読み込む必要がある
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.02) : int(h * 0.12), int(w * 0.7) : int(w * 0.98)]

        # 画像品質評価
        quality_metrics = calculate_image_quality_metrics(roi)

        # 失敗理由分類
        log_messages = log_messages_by_frame.get(frame_idx, [])
        reason, recommendations = classify_failure_reason(
            {}, quality_metrics, log_messages
        )

        failure_stats[reason] += 1

        # 失敗フレームを保存
        output_path = failed_frames_dir / f"failed_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame)

        # ROI領域も保存
        roi_output_path = failed_frames_dir / f"failed_roi_{frame_idx:06d}.jpg"
        cv2.imwrite(str(roi_output_path), roi)

        failure_details.append(
            {
                "frame_idx": frame_idx,
                "reason": reason,
                "quality_score": quality_metrics["quality_score"],
                "contrast": quality_metrics["contrast"],
                "sharpness": quality_metrics["sharpness"],
                "brightness": quality_metrics["brightness"],
                "recommendations": recommendations,
                "log_messages": log_messages[:3],  # 最初の3つだけ
            }
        )

        analyzed_count += 1

    cap.release()

    # 結果をまとめる
    analysis_result = {
        "total_failed": len(failed_frame_indices),
        "analyzed": analyzed_count,
        "failure_stats": dict(failure_stats),
        "failure_details": failure_details[:20],  # 最初の20件だけ
        "recommendations_summary": generate_summary_recommendations(
            failure_stats, failure_details
        ),
    }

    # 結果をJSONで保存
    import json

    result_path = output_dir / "failure_analysis.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 80)
    logger.info("失敗ケース分析完了")
    logger.info("=" * 80)
    logger.info(f"失敗フレーム数: {len(failed_frame_indices)}")
    logger.info(f"分析フレーム数: {analyzed_count}")
    logger.info(f"失敗理由の内訳:")
    for reason, count in sorted(failure_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}件")
    logger.info(f"結果を保存: {result_path}")
    logger.info(f"失敗フレーム画像: {failed_frames_dir}")

    return analysis_result


def generate_summary_recommendations(
    failure_stats: Dict[str, int], failure_details: List[Dict]
) -> List[str]:
    """総合的な改善策を生成

    Args:
        failure_stats: 失敗理由別の統計
        failure_details: 失敗詳細のリスト

    Returns:
        改善策のリスト
    """
    recommendations = []
    total = sum(failure_stats.values())

    if total == 0:
        return ["失敗フレームがありません"]

    # 最も多い失敗理由に基づいて推奨
    sorted_reasons = sorted(failure_stats.items(), key=lambda x: -x[1])
    top_reason, top_count = sorted_reasons[0]
    top_ratio = top_count / total

    if top_ratio > 0.5:
        if top_reason == "low_image_quality":
            recommendations.append("【優先】画像品質が低いフレームが多数あります。前処理パラメータの最適化を強く推奨します")
        elif top_reason == "ocr_failure":
            recommendations.append("【優先】OCR失敗が多数発生しています。OCRエンジンの設定見直しを強く推奨します")
        elif top_reason == "parse_failure":
            recommendations.append("【優先】パース失敗が多数発生しています。OCR結果の前処理強化を強く推奨します")
        elif top_reason == "low_confidence":
            recommendations.append("【優先】信頼度不足が多数発生しています。信頼度閾値の見直しを検討してください")

    # 平均品質スコアに基づく推奨
    if failure_details:
        avg_quality = np.mean([d["quality_score"] for d in failure_details])
        if avg_quality < 40:
            recommendations.append("【推奨】平均画像品質スコアが低いです。前処理パラメータの調整を推奨します")

    # その他の推奨
    recommendations.append("【参考】失敗フレーム画像を確認し、共通パターン（暗い、ぼやけ、ノイズ等）を特定してください")
    recommendations.append("【参考】ログファイルを確認し、詳細なエラーメッセージを確認してください")

    return recommendations


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="OCR失敗ケースの自動分析ツール")
    parser.add_argument(
        "--csv",
        type=str,
        default="output/extracted_frames/extraction_results.csv",
        help="抽出結果CSVファイルのパス",
    )
    parser.add_argument(
        "--video", type=str, default="input/merged_moviefiles.mov", help="動画ファイルのパス"
    )
    parser.add_argument(
        "--output", type=str, default="output/diagnostics", help="出力ディレクトリ"
    )
    parser.add_argument(
        "--log", type=str, default="output/system.log", help="ログファイルのパス（オプション）"
    )
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)

    # パス変換
    csv_path = Path(args.csv)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    log_path = Path(args.log) if args.log else None

    # 実行
    result = analyze_failures(csv_path, video_path, output_dir, log_path)

    # 結果を表示
    if result:
        print("\n" + "=" * 80)
        print("分析結果サマリー")
        print("=" * 80)
        print(f"失敗フレーム数: {result['total_failed']}")
        print(f"分析フレーム数: {result['analyzed']}")
        print("\n失敗理由の内訳:")
        for reason, count in sorted(
            result["failure_stats"].items(), key=lambda x: -x[1]
        ):
            print(f"  {reason}: {count}件")
        print("\n推奨改善策:")
        for rec in result["recommendations_summary"]:
            print(f"  {rec}")
        print("=" * 80)


if __name__ == "__main__":
    main()
