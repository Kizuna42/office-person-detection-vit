#!/usr/bin/env python3
"""動画の開始時刻・終了時刻を分析し、OCR読み取り結果と比較するツール

予測OCR出力と参照（ゴールド）テキストを比較して、タイムスタンプ整合性と内容一致を評価します。
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager
from src.timestamp import TimestampExtractor
from src.utils.text_metrics import (
    calculate_cer,
    calculate_token_metrics,
    calculate_wer,
)
from src.video.video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> dict:
    """動画のメタデータを取得する

    Args:
        video_path: 動画ファイルのパス

    Returns:
        メタデータの辞書
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 動画の長さ（秒）
    duration_seconds = total_frames / fps if fps > 0 else 0
    duration_timedelta = timedelta(seconds=duration_seconds)

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "duration_timedelta": duration_timedelta,
    }


def extract_timestamp_from_frame(
    video_processor: VideoProcessor,
    frame_number: int,
    timestamp_extractor: TimestampExtractor,
) -> tuple[str | None, float]:
    """指定フレームからタイムスタンプを抽出する

    Args:
        video_processor: VideoProcessorインスタンス
        frame_number: フレーム番号
        timestamp_extractor: TimestampExtractorインスタンス

    Returns:
        (タイムスタンプ文字列, 信頼度) のタプル
    """
    try:
        frame = video_processor.get_frame(frame_number)
        if frame is None:
            return None, 0.0

        timestamp, confidence = timestamp_extractor.extract_with_confidence(frame)
        return timestamp, confidence
    except Exception as e:
        logger.error(f"フレーム {frame_number} のタイムスタンプ抽出に失敗: {e}")
        return None, 0.0


def analyze_video_timestamps(
    video_path: str,
    config_path: str,
    sample_frames: int = 10,
) -> dict:
    """動画の開始・終了時刻を分析する

    Args:
        video_path: 動画ファイルのパス
        config_path: 設定ファイルのパス
        sample_frames: サンプリングするフレーム数（最初と最後から）

    Returns:
        分析結果の辞書
    """
    logger.info("=" * 80)
    logger.info("動画タイムスタンプ分析ツール")
    logger.info("=" * 80)

    # メタデータ取得
    logger.info("動画メタデータを取得中...")
    metadata = get_video_metadata(video_path)
    logger.info(f"  解像度: {metadata['width']}x{metadata['height']}")
    logger.info(f"  FPS: {metadata['fps']:.2f}")
    logger.info(f"  総フレーム数: {metadata['total_frames']}")
    logger.info(f"  動画の長さ: {metadata['duration_seconds']:.2f}秒 ({metadata['duration_timedelta']})")

    # VideoProcessorとTimestampExtractorを初期化
    logger.info("動画処理器を初期化中...")
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    config = ConfigManager(config_path)
    timestamp_extractor = TimestampExtractor()

    # 最初の数フレームからタイムスタンプを抽出
    logger.info(f"最初の{sample_frames}フレームからタイムスタンプを抽出中...")
    first_timestamps = []
    for i in range(min(sample_frames, metadata["total_frames"])):
        timestamp, confidence = extract_timestamp_from_frame(
            video_processor, i, timestamp_extractor
        )
        if timestamp:
            first_timestamps.append((i, timestamp, confidence))
            logger.info(f"  フレーム {i}: {timestamp} (信頼度: {confidence:.2f})")

    # 最後の数フレームからタイムスタンプを抽出
    logger.info(f"最後の{sample_frames}フレームからタイムスタンプを抽出中...")
    last_timestamps = []
    last_frame_start = max(0, metadata["total_frames"] - sample_frames)
    for i in range(last_frame_start, metadata["total_frames"]):
        timestamp, confidence = extract_timestamp_from_frame(
            video_processor, i, timestamp_extractor
        )
        if timestamp:
            last_timestamps.append((i, timestamp, confidence))
            logger.info(f"  フレーム {i}: {timestamp} (信頼度: {confidence:.2f})")

    video_processor.release()

    # 結果をまとめる
    result = {
        "metadata": {
            "fps": metadata["fps"],
            "total_frames": metadata["total_frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "duration_seconds": metadata["duration_seconds"],
            "duration_formatted": str(metadata["duration_timedelta"]),
        },
        "first_timestamps": [
            {"frame": f, "timestamp": t, "confidence": c}
            for f, t, c in first_timestamps
        ],
        "last_timestamps": [
            {"frame": f, "timestamp": t, "confidence": c}
            for f, t, c in last_timestamps
        ],
    }

    # 開始時刻と終了時刻を推定
    if first_timestamps:
        first_ts_str = first_timestamps[0][1]
        try:
            start_dt = datetime.strptime(first_ts_str, "%Y/%m/%d %H:%M:%S")
            result["estimated_start"] = start_dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            result["estimated_start"] = None
    else:
        result["estimated_start"] = None

    if last_timestamps:
        last_ts_str = last_timestamps[-1][1]
        try:
            end_dt = datetime.strptime(last_ts_str, "%Y/%m/%d %H:%M:%S")
            result["estimated_end"] = end_dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            result["estimated_end"] = None
    else:
        result["estimated_end"] = None

    # メタデータから計算した終了時刻（開始時刻が取得できた場合）
    if result["estimated_start"]:
        start_dt = datetime.strptime(result["estimated_start"], "%Y/%m/%d %H:%M:%S")
        calculated_end_dt = start_dt + metadata["duration_timedelta"]
        result["calculated_end_from_metadata"] = calculated_end_dt.strftime(
            "%Y/%m/%d %H:%M:%S"
        )
    else:
        result["calculated_end_from_metadata"] = None

    return result


def print_comparison(result: dict) -> None:
    """結果を比較して表示する

    Args:
        result: 分析結果の辞書
    """
    print("\n" + "=" * 80)
    print("分析結果")
    print("=" * 80)

    print("\n【動画メタデータ】")
    print(f"  解像度: {result['metadata']['width']}x{result['metadata']['height']}")
    print(f"  FPS: {result['metadata']['fps']:.2f}")
    print(f"  総フレーム数: {result['metadata']['total_frames']:,}")
    print(f"  動画の長さ: {result['metadata']['duration_formatted']}")

    print("\n【OCR読み取り結果（最初のフレーム）】")
    if result["first_timestamps"]:
        for ts_info in result["first_timestamps"]:
            print(
                f"  フレーム {ts_info['frame']:6d}: {ts_info['timestamp']:20s} "
                f"(信頼度: {ts_info['confidence']:.2f})"
            )
        print(f"\n  → 推定開始時刻: {result['estimated_start']}")
    else:
        print("  （タイムスタンプを読み取れませんでした）")

    print("\n【OCR読み取り結果（最後のフレーム）】")
    if result["last_timestamps"]:
        for ts_info in result["last_timestamps"]:
            print(
                f"  フレーム {ts_info['frame']:6d}: {ts_info['timestamp']:20s} "
                f"(信頼度: {ts_info['confidence']:.2f})"
            )
        print(f"\n  → 推定終了時刻: {result['estimated_end']}")
    else:
        print("  （タイムスタンプを読み取れませんでした）")

    print("\n【評価】")
    if result["estimated_start"] and result["estimated_end"]:
        start_dt = datetime.strptime(result["estimated_start"], "%Y/%m/%d %H:%M:%S")
        end_dt = datetime.strptime(result["estimated_end"], "%Y/%m/%d %H:%M:%S")
        ocr_duration = end_dt - start_dt
        ocr_duration_seconds = ocr_duration.total_seconds()
        metadata_duration = timedelta(seconds=result["metadata"]["duration_seconds"])

        print(f"  OCR開始時刻:     {result['estimated_start']}")
        print(f"  OCR終了時刻:     {result['estimated_end']}")
        print(f"  OCR読み取り結果の撮影期間: {ocr_duration} ({ocr_duration_seconds:.2f}秒)")

        # タイムラプス動画の特性を説明
        print(f"\n  【タイムラプス動画の特性】")
        print(f"  動画の長さ（メタデータ）: {metadata_duration} ({result['metadata']['duration_seconds']:.2f}秒)")
        print(f"  実際の撮影期間（OCR読み取り結果）: {ocr_duration} ({ocr_duration_seconds:.2f}秒)")
        
        if ocr_duration_seconds > result["metadata"]["duration_seconds"] * 2:
            print(f"  → タイムラプス動画であることが確認されました")
            print(f"  → 時間圧縮率: 約 {ocr_duration_seconds / result['metadata']['duration_seconds']:.1f}倍")

        # 評価基準: OCR読み取り結果の開始時刻と終了時刻の差が妥当か
        # （実際の撮影期間は事前に分からないため、ここでは情報表示のみ）
        print(f"\n  【評価基準】")
        print(f"  開始・終了時刻の精度: OCR読み取り結果の開始時刻と終了時刻の差が、")
        print(f"  実際の撮影期間と一致しているかを評価します（誤差 ≤60秒が目標）")
        print(f"  → この評価には、実際の撮影期間（Ground Truth）が必要です")

        # 参考情報: メタデータから計算した終了時刻（評価基準ではない）
        if result["calculated_end_from_metadata"]:
            print(f"\n  【参考情報（評価基準ではない）】")
            print(f"  メタデータから計算した終了時刻: {result['calculated_end_from_metadata']}")
            print(f"  （動画の長さを開始時刻に加算したもの）")
            calc_end_dt = datetime.strptime(
                result["calculated_end_from_metadata"], "%Y/%m/%d %H:%M:%S"
            )
            diff = (end_dt - calc_end_dt).total_seconds()
            print(f"  OCR終了時刻との差: {diff:.2f}秒")
            print(f"  → 注意: タイムラプス動画では、この差は正常です")
            print(f"  → 評価基準として使用するのは不適切です")
    else:
        print("  （開始時刻または終了時刻が取得できませんでした）")


def main() -> int:
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="動画の開始時刻・終了時刻を分析し、OCR読み取り結果と比較する"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="動画ファイルのパス（設定ファイルの値を使用する場合は省略）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=10,
        help="最初と最後からサンプリングするフレーム数（デフォルト: 10）",
    )

    args = parser.parse_args()

    # 設定ファイルの読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    config = ConfigManager(str(config_path))

    # 動画ファイルのパスを決定
    video_path = args.video or config.get("video.input_path")
    if not video_path:
        logger.error("動画ファイルのパスが指定されていません")
        return 1

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    try:
        # 分析実行
        result = analyze_video_timestamps(
            str(video_path), str(config_path), args.sample_frames
        )

        # 結果表示
        print_comparison(result)

        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


def load_ocr_data(file_path: str) -> List[Dict]:
    """OCR出力データを読み込む（JSONまたはTSV形式）

    Args:
        file_path: ファイルパス

    Returns:
        レコードのリスト。各レコードは {segment_id, start_sec, end_sec, text, ocr_confidence?} を含む
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # リスト形式または辞書形式に対応
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "segments" in data:
                return data["segments"]
            else:
                raise ValueError(f"JSON形式が不正です: {file_path}")
    elif path.suffix.lower() == ".tsv":
        records = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                record = {
                    "segment_id": row.get("segment_id", ""),
                    "start_sec": float(row.get("start_sec", 0)),
                    "end_sec": float(row.get("end_sec", 0)),
                    "text": row.get("text", ""),
                }
                if "ocr_confidence" in row:
                    record["ocr_confidence"] = float(row["ocr_confidence"])
                records.append(record)
        return records
    else:
        raise ValueError(f"サポートされていないファイル形式: {path.suffix}")


def parse_timestamp_to_seconds(timestamp_str: str) -> Optional[float]:
    """タイムスタンプ文字列を秒数に変換

    Args:
        timestamp_str: タイムスタンプ文字列（例: "2025/08/26 16:04:16"）

    Returns:
        秒数（エポックからの経過秒数）、またはNone
    """
    try:
        dt = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        return None


def evaluate_timestamp_consistency(
    pred_records: List[Dict],
    ref_records: List[Dict],
    tolerance_seconds: List[float] = [0.5, 1.0, 2.0, 5.0],
) -> Dict:
    """タイムスタンプ整合性を評価

    Args:
        pred_records: 予測OCRレコード
        ref_records: 参照レコード
        tolerance_seconds: 許容ズレのリスト（秒）

    Returns:
        タイムスタンプ整合性評価結果
    """
    # segment_idでマッチング
    ref_by_id = {r.get("segment_id", ""): r for r in ref_records}
    pred_by_id = {r.get("segment_id", ""): r for r in pred_records}

    results = {
        "total_segments": len(ref_records),
        "matched_segments": 0,
        "unmatched_pred": 0,
        "unmatched_ref": 0,
        "by_tolerance": {},
    }

    # 各許容ズレで評価
    for tol in tolerance_seconds:
        within_tolerance = 0
        time_diffs = []

        for segment_id, ref_record in ref_by_id.items():
            if segment_id not in pred_by_id:
                continue

            pred_record = pred_by_id[segment_id]
            ref_text = ref_record.get("text", "")
            pred_text = pred_record.get("text", "")

            # タイムスタンプを秒数に変換
            ref_sec = parse_timestamp_to_seconds(ref_text)
            pred_sec = parse_timestamp_to_seconds(pred_text)

            if ref_sec is None or pred_sec is None:
                continue

            time_diff = abs(pred_sec - ref_sec)
            time_diffs.append(time_diff)

            if time_diff <= tol:
                within_tolerance += 1

        results["by_tolerance"][str(tol)] = {
            "within_tolerance": within_tolerance,
            "total_matched": len(time_diffs),
            "rate": (
                within_tolerance / len(time_diffs) if time_diffs else 0.0
            ),
            "avg_time_diff": sum(time_diffs) / len(time_diffs) if time_diffs else 0.0,
            "min_time_diff": min(time_diffs) if time_diffs else 0.0,
            "max_time_diff": max(time_diffs) if time_diffs else 0.0,
        }

    # マッチング統計
    matched_ids = set(ref_by_id.keys()) & set(pred_by_id.keys())
    results["matched_segments"] = len(matched_ids)
    results["unmatched_pred"] = len(set(pred_by_id.keys()) - matched_ids)
    results["unmatched_ref"] = len(set(ref_by_id.keys()) - matched_ids)

    return results


def evaluate_content_accuracy(
    pred_records: List[Dict], ref_records: List[Dict]
) -> Dict:
    """内容一致評価（CER, WER, トークン単位のPrecision/Recall/F1）

    Args:
        pred_records: 予測OCRレコード
        ref_records: 参照レコード

    Returns:
        内容一致評価結果
    """
    # segment_idでマッチング
    ref_by_id = {r.get("segment_id", ""): r for r in ref_records}
    pred_by_id = {r.get("segment_id", ""): r for r in pred_records}

    matched_pairs = []
    for segment_id in set(ref_by_id.keys()) & set(pred_by_id.keys()):
        ref_text = ref_by_id[segment_id].get("text", "")
        pred_text = pred_by_id[segment_id].get("text", "")
        matched_pairs.append((ref_text, pred_text))

    if not matched_pairs:
        return {
            "total_pairs": 0,
            "cer": {"cer": 1.0},
            "wer": {"wer": 1.0},
            "token_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        }

    # 全体のCER/WERを計算
    total_cer = {"cer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0, "total_chars": 0}
    total_wer = {"wer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0, "total_words": 0}
    total_token_metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "total_ref_tokens": 0,
        "total_hyp_tokens": 0,
    }

    for ref_text, pred_text in matched_pairs:
        cer_result = calculate_cer(ref_text, pred_text)
        wer_result = calculate_wer(ref_text, pred_text)
        token_result = calculate_token_metrics(ref_text, pred_text)

        total_cer["cer"] += cer_result["cer"]
        total_cer["substitutions"] += cer_result["substitutions"]
        total_cer["insertions"] += cer_result["insertions"]
        total_cer["deletions"] += cer_result["deletions"]
        total_cer["total_chars"] += cer_result["total_chars"]

        total_wer["wer"] += wer_result["wer"]
        total_wer["substitutions"] += wer_result["substitutions"]
        total_wer["insertions"] += wer_result["insertions"]
        total_wer["deletions"] += wer_result["deletions"]
        total_wer["total_words"] += wer_result["total_words"]

        total_token_metrics["true_positives"] += token_result["true_positives"]
        total_token_metrics["false_positives"] += token_result["false_positives"]
        total_token_metrics["false_negatives"] += token_result["false_negatives"]
        total_token_metrics["total_ref_tokens"] += token_result["total_ref_tokens"]
        total_token_metrics["total_hyp_tokens"] += token_result["total_hyp_tokens"]

    # 平均を計算
    num_pairs = len(matched_pairs)
    total_cer["cer"] /= num_pairs
    total_wer["wer"] /= num_pairs

    # トークン単位のPrecision/Recall/F1を計算
    tp = total_token_metrics["true_positives"]
    fp = total_token_metrics["false_positives"]
    fn = total_token_metrics["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    token_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **total_token_metrics,
    }

    return {
        "total_pairs": num_pairs,
        "cer": total_cer,
        "wer": total_wer,
        "token_metrics": token_metrics,
    }


def extract_critical_errors(
    pred_records: List[Dict],
    ref_records: List[Dict],
    tolerance_seconds: float = 1.0,
) -> Dict:
    """重要エラーを抽出

    Args:
        pred_records: 予測OCRレコード
        ref_records: 参照レコード
        tolerance_seconds: タイムスタンプ許容ズレ（秒）

    Returns:
        重要エラーのリスト
    """
    ref_by_id = {r.get("segment_id", ""): r for r in ref_records}
    pred_by_id = {r.get("segment_id", ""): r for r in pred_records}

    errors = []

    for segment_id, ref_record in ref_by_id.items():
        ref_text = ref_record.get("text", "")
        pred_record = pred_by_id.get(segment_id)

        if not pred_record:
            errors.append({
                "type": "missing_segment",
                "segment_id": segment_id,
                "ref_text": ref_text,
                "pred_text": None,
            })
            continue

        pred_text = pred_record.get("text", "")

        # タイムスタンプの時間ずれ
        ref_sec = parse_timestamp_to_seconds(ref_text)
        pred_sec = parse_timestamp_to_seconds(pred_text)

        if ref_sec is not None and pred_sec is not None:
            time_diff = abs(pred_sec - ref_sec)
            if time_diff > tolerance_seconds:
                errors.append({
                    "type": "time_mismatch",
                    "segment_id": segment_id,
                    "ref_text": ref_text,
                    "pred_text": pred_text,
                    "time_diff_seconds": time_diff,
                })

        # 誤認識が意味変化を招く箇所（数字の誤認識など）
        if ref_text != pred_text:
            # 数字の誤認識をチェック
            ref_digits = "".join(c for c in ref_text if c.isdigit())
            pred_digits = "".join(c for c in pred_text if c.isdigit())
            if ref_digits != pred_digits:
                errors.append({
                    "type": "digit_mismatch",
                    "segment_id": segment_id,
                    "ref_text": ref_text,
                    "pred_text": pred_text,
                    "ref_digits": ref_digits,
                    "pred_digits": pred_digits,
                })

        # 空白/改行の失敗
        ref_normalized = " ".join(ref_text.split())
        pred_normalized = " ".join(pred_text.split())
        if ref_normalized != pred_normalized:
            # 空白の問題は他のエラーと重複する可能性があるため、軽微なものは除外
            if len(ref_text) > 0 and len(pred_text) > 0:
                cer_result = calculate_cer(ref_text, pred_text)
                if cer_result["cer"] > 0.1:  # 10%以上の文字エラー
                    errors.append({
                        "type": "text_mismatch",
                        "segment_id": segment_id,
                        "ref_text": ref_text,
                        "pred_text": pred_text,
                        "cer": cer_result["cer"],
                    })

    return {"errors": errors, "total_errors": len(errors)}


def analyze_ocr_accuracy(
    pred_path: str,
    ref_path: str,
    tolerance_seconds: List[float] = [0.5, 1.0, 2.0, 5.0],
) -> Dict:
    """OCR精度を分析

    Args:
        pred_path: 予測OCR出力ファイル（JSONまたはTSV）
        ref_path: 参照（ゴールド）テキストファイル（JSONまたはTSV）
        tolerance_seconds: タイムスタンプ許容ズレのリスト（秒）

    Returns:
        分析結果の辞書
    """
    logger.info("=" * 80)
    logger.info("OCR精度分析ツール")
    logger.info("=" * 80)

    # データ読み込み
    logger.info(f"予測OCR出力を読み込み中: {pred_path}")
    pred_records = load_ocr_data(pred_path)
    logger.info(f"  読み込み完了: {len(pred_records)}レコード")

    logger.info(f"参照テキストを読み込み中: {ref_path}")
    ref_records = load_ocr_data(ref_path)
    logger.info(f"  読み込み完了: {len(ref_records)}レコード")

    # タイムスタンプ整合性評価
    logger.info("タイムスタンプ整合性を評価中...")
    timestamp_consistency = evaluate_timestamp_consistency(
        pred_records, ref_records, tolerance_seconds
    )

    # 内容一致評価
    logger.info("内容一致を評価中...")
    content_accuracy = evaluate_content_accuracy(pred_records, ref_records)

    # 重要エラー抽出
    logger.info("重要エラーを抽出中...")
    critical_errors = extract_critical_errors(
        pred_records, ref_records, tolerance_seconds=max(tolerance_seconds)
    )

    # 結果をまとめる
    result = {
        "overall": {
            "total_pred_segments": len(pred_records),
            "total_ref_segments": len(ref_records),
            "matched_segments": timestamp_consistency["matched_segments"],
            "timestamp_consistency": timestamp_consistency,
            "content_accuracy": content_accuracy,
            "critical_errors": {
                "total": critical_errors["total_errors"],
                "by_type": {},
            },
        },
        "by_segment": [],
        "error_samples": {
            "time_mismatch": [],
            "digit_mismatch": [],
            "text_mismatch": [],
            "missing_segment": [],
        },
    }

    # セグメント単位の詳細
    ref_by_id = {r.get("segment_id", ""): r for r in ref_records}
    pred_by_id = {r.get("segment_id", ""): r for r in pred_records}

    for segment_id in set(ref_by_id.keys()) | set(pred_by_id.keys()):
        ref_record = ref_by_id.get(segment_id, {})
        pred_record = pred_by_id.get(segment_id, {})

        ref_text = ref_record.get("text", "")
        pred_text = pred_record.get("text", "")

        segment_result = {
            "segment_id": segment_id,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "matched": segment_id in ref_by_id and segment_id in pred_by_id,
        }

        if segment_id in ref_by_id and segment_id in pred_by_id:
            # タイムスタンプ整合性
            ref_sec = parse_timestamp_to_seconds(ref_text)
            pred_sec = parse_timestamp_to_seconds(pred_text)
            if ref_sec is not None and pred_sec is not None:
                segment_result["time_diff_seconds"] = abs(pred_sec - ref_sec)

            # 内容一致
            cer_result = calculate_cer(ref_text, pred_text)
            wer_result = calculate_wer(ref_text, pred_text)
            token_result = calculate_token_metrics(ref_text, pred_text)

            segment_result["cer"] = cer_result["cer"]
            segment_result["wer"] = wer_result["wer"]
            segment_result["token_precision"] = token_result["precision"]
            segment_result["token_recall"] = token_result["recall"]
            segment_result["token_f1"] = token_result["f1"]

        result["by_segment"].append(segment_result)

    # エラーサンプルを分類
    for error in critical_errors["errors"]:
        error_type = error["type"]
        if error_type in result["error_samples"]:
            result["error_samples"][error_type].append(error)

    # エラー種別ごとの集計
    for error in critical_errors["errors"]:
        error_type = error["type"]
        result["overall"]["critical_errors"]["by_type"][error_type] = (
            result["overall"]["critical_errors"]["by_type"].get(error_type, 0) + 1
        )

    return result


def main() -> int:
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="動画の開始時刻・終了時刻を分析し、OCR読み取り結果と比較する。"
        "または、予測OCR出力と参照テキストを比較して精度を評価する。"
    )
    parser.add_argument(
        "--pred",
        type=str,
        help="予測OCR出力ファイル（JSONまたはTSV）",
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="参照（ゴールド）テキストファイル（JSONまたはTSV）",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="出力JSONファイルパス",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
        help="タイムスタンプ許容ズレ（秒）のリスト（デフォルト: 0.5 1.0 2.0 5.0）",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="動画ファイルのパス（設定ファイルの値を使用する場合は省略）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=10,
        help="最初と最後からサンプリングするフレーム数（デフォルト: 10）",
    )

    args = parser.parse_args()

    # OCR精度評価モード
    if args.pred and args.ref:
        try:
            result = analyze_ocr_accuracy(
                args.pred, args.ref, tolerance_seconds=args.tolerance
            )

            # JSON出力
            if args.out:
                output_path = Path(args.out)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"分析結果を保存しました: {output_path}")
            else:
                # コンソール出力
                print("\n" + "=" * 80)
                print("OCR精度分析結果")
                print("=" * 80)
                print(json.dumps(result, ensure_ascii=False, indent=2))

            return 0

        except Exception as e:
            logger.error(f"エラーが発生しました: {e}", exc_info=True)
            return 1

    # 従来の動画分析モード
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    config = ConfigManager(str(config_path))

    video_path = args.video or config.get("video.input_path")
    if not video_path:
        logger.error("動画ファイルのパスが指定されていません")
        return 1

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    try:
        result = analyze_video_timestamps(
            str(video_path), str(config_path), args.sample_frames
        )
        print_comparison(result)
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


