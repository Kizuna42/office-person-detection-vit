#!/usr/bin/env python3
"""タイムスタンプ抽出スコア評価ツール

動画から10秒間隔で100フレームを抽出し、各フレームからOCRでタイムスタンプを抽出。
抽出成功率と時系列整合性スコアを算出してコンソールに出力する。

または、分析結果（JSON）を読み込み、詳細スコア計算とMarkdownレポートを生成する。
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import VideoProcessor
from src.utils import setup_logging, setup_mps_compatibility

logger = logging.getLogger(__name__)


def calculate_extraction_rate(
    results: List[Tuple[bool, Optional[str], float]]
) -> Dict[str, float]:
    """抽出成功率を計算

    Args:
        results: [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト

    Returns:
        抽出成功率の統計情報
    """
    total = len(results)
    success_count = sum(1 for success, _, _ in results if success)
    success_rate = (success_count / total * 100.0) if total > 0 else 0.0

    # 信頼度統計（成功したもののみ）
    confidences = [conf for success, _, conf in results if success]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0

    return {
        "total_frames": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "average_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
    }


def calculate_temporal_consistency(
    results: List[Tuple[bool, Optional[str], float]]
) -> Dict[str, float]:
    """時系列整合性スコアを計算

    連続フレーム間のタイムスタンプ差を計算し、期待値（約10秒）との整合性を評価する。

    Args:
        results: [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト

    Returns:
        時系列整合性スコアの統計情報
    """
    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
    EXPECTED_INTERVAL_SECONDS = 10.0
    TOLERANCE_MIN = 8.0  # 許容範囲の下限（秒）
    TOLERANCE_MAX = 12.0  # 許容範囲の上限（秒）

    # タイムスタンプをdatetimeに変換
    timestamps: List[Optional[datetime]] = []
    for success, timestamp_str, _ in results:
        if success and timestamp_str:
            try:
                dt = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
                timestamps.append(dt)
            except ValueError:
                timestamps.append(None)
        else:
            timestamps.append(None)

    # 連続フレーム間の時間差を計算
    time_diffs: List[float] = []
    valid_pairs = 0
    consistency_count = 0

    for i in range(len(timestamps) - 1):
        if timestamps[i] is not None and timestamps[i + 1] is not None:
            diff_seconds = abs((timestamps[i + 1] - timestamps[i]).total_seconds())
            time_diffs.append(diff_seconds)
            valid_pairs += 1

            # 期待範囲内かチェック
            if TOLERANCE_MIN <= diff_seconds <= TOLERANCE_MAX:
                consistency_count += 1

    # 統計を計算
    if not time_diffs:
        return {
            "consistency_score": 0.0,
            "average_time_diff": 0.0,
            "min_time_diff": 0.0,
            "max_time_diff": 0.0,
            "valid_pairs": 0,
        }

    avg_time_diff = sum(time_diffs) / len(time_diffs)
    min_time_diff = min(time_diffs)
    max_time_diff = max(time_diffs)
    consistency_score = (
        (consistency_count / valid_pairs * 100.0) if valid_pairs > 0 else 0.0
    )

    return {
        "consistency_score": consistency_score,
        "average_time_diff": avg_time_diff,
        "min_time_diff": min_time_diff,
        "max_time_diff": max_time_diff,
        "valid_pairs": valid_pairs,
        "consistency_count": consistency_count,
    }


def extract_frames_with_timestamps(
    video_processor: VideoProcessor,
    timestamp_extractor: TimestampExtractor,
    interval_seconds: float = 10.0,
    num_frames: int = 100,
) -> List[Tuple[bool, Optional[str], float]]:
    """動画から指定間隔でフレームを抽出し、タイムスタンプを抽出

    Args:
        video_processor: VideoProcessorインスタンス
        timestamp_extractor: TimestampExtractorインスタンス
        interval_seconds: フレーム抽出間隔（秒）
        num_frames: 抽出するフレーム数

    Returns:
        [(成功フラグ, タイムスタンプ, 信頼度), ...] のリスト
    """
    if video_processor.fps is None or video_processor.fps <= 0:
        raise ValueError("動画のFPSが取得できません")

    # フレーム間隔を計算
    frame_interval = int(video_processor.fps * interval_seconds)
    logger.info(f"フレーム抽出間隔: {frame_interval}フレーム ({interval_seconds}秒)")

    results: List[Tuple[bool, Optional[str], float]] = []

    # 動画を先頭に戻す
    video_processor.reset()

    for i in range(num_frames):
        frame_number = i * frame_interval

        # フレーム数が範囲外の場合は終了
        if (
            video_processor.total_frames
            and frame_number >= video_processor.total_frames
        ):
            logger.warning(
                f"フレーム {frame_number} が範囲外です（総フレーム数: {video_processor.total_frames}）"
            )
            break

        # フレームを取得
        frame = video_processor.get_frame(frame_number)
        if frame is None:
            logger.warning(f"フレーム {frame_number} の取得に失敗しました")
            results.append((False, None, 0.0))
            continue

        # タイムスタンプを抽出（信頼度付き）
        timestamp, confidence = timestamp_extractor.extract_with_confidence(frame)
        success = timestamp is not None

        if success:
            logger.debug(
                f"フレーム {frame_number}: タイムスタンプ抽出成功 - {timestamp} (信頼度: {confidence:.2f})"
            )
        else:
            logger.debug(f"フレーム {frame_number}: タイムスタンプ抽出失敗")

        results.append((success, timestamp, confidence))

        # 進捗表示（10フレームごと）
        if (i + 1) % 10 == 0:
            logger.info(f"進捗: {i + 1}/{num_frames} フレーム処理済み")

    return results


def print_results(
    extraction_stats: Dict[str, float], consistency_stats: Dict[str, float]
) -> None:
    """結果をコンソールに出力

    Args:
        extraction_stats: 抽出成功率の統計情報
        consistency_stats: 時系列整合性スコアの統計情報
    """
    print("=" * 80)
    print("タイムスタンプ抽出スコア評価結果")
    print("=" * 80)
    print()

    # 抽出成功率
    print("【抽出成功率】")
    print(f"  処理フレーム数: {int(extraction_stats['total_frames'])}")
    print(f"  抽出成功数: {int(extraction_stats['success_count'])}")
    print(f"  抽出成功率: {extraction_stats['success_rate']:.2f}%")
    print()

    # 信頼度統計
    print("【信頼度統計】")
    print(f"  平均信頼度: {extraction_stats['average_confidence']:.4f}")
    print(f"  最小信頼度: {extraction_stats['min_confidence']:.4f}")
    print(f"  最大信頼度: {extraction_stats['max_confidence']:.4f}")
    print()

    # 時系列整合性
    print("【時系列整合性スコア】")
    print(f"  整合性スコア: {consistency_stats['consistency_score']:.2f}%")
    print(f"  有効ペア数: {int(consistency_stats['valid_pairs'])}")
    print(f"  期待範囲内ペア数: {int(consistency_stats.get('consistency_count', 0))}")
    print(f"  期待範囲: 8.0-12.0秒（10秒間隔での抽出を想定）")
    print()

    if consistency_stats["valid_pairs"] > 0:
        print("【時間差統計】")
        print(f"  平均時間差: {consistency_stats['average_time_diff']:.2f}秒")
        print(f"  最小時間差: {consistency_stats['min_time_diff']:.2f}秒")
        print(f"  最大時間差: {consistency_stats['max_time_diff']:.2f}秒")
        print()
        
        # タイムラプス動画の特性を説明
        if consistency_stats['average_time_diff'] > 30:
            print("【注意】")
            print("  タイムラプス動画の場合、フレーム間の実際の時間差は")
            print("  動画の長さと実際の撮影期間の比率に依存します。")
            print("  評価ツールは10秒間隔でフレームを抽出しているため、")
            print("  期待範囲（8.0-12.0秒）は適切です。")
            print()

    print("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="タイムスタンプ抽出スコア評価ツール"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="動画ファイルのパス（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="フレーム抽出間隔（秒、デフォルト: 10.0）",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="抽出するフレーム数（デフォルト: 100）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("タイムスタンプ抽出スコア評価ツール 起動")
    logger.info("=" * 80)

    video_processor = None

    try:
        # 設定ファイルの読み込み
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return 1

        config = ConfigManager(str(config_path))
        logger.info(f"設定ファイルを読み込みました: {config_path}")

        # 動画ファイルのパスを決定
        video_path = args.video or config.get("video.input_path")
        if not video_path:
            logger.error("動画ファイルのパスが指定されていません")
            return 1

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"動画ファイルが見つかりません: {video_path}")
            return 1

        logger.info(f"動画ファイル: {video_path}")

        # 動画処理の初期化
        video_processor = VideoProcessor(str(video_path))
        if not video_processor.open():
            logger.error("動画ファイルを開けませんでした")
            return 1

        logger.info(f"動画情報:")
        logger.info(f"  解像度: {video_processor.width}x{video_processor.height}")
        logger.info(f"  FPS: {video_processor.fps}")
        logger.info(f"  総フレーム数: {video_processor.total_frames}")

        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        logger.info("タイムスタンプ抽出器を初期化しました")

        # フレーム抽出とタイムスタンプ抽出
        logger.info(f"フレーム抽出を開始します（間隔: {args.interval}秒、フレーム数: {args.num_frames}）")
        results = extract_frames_with_timestamps(
            video_processor,
            timestamp_extractor,
            interval_seconds=args.interval,
            num_frames=args.num_frames,
        )

        logger.info(f"フレーム抽出完了: {len(results)}フレーム処理済み")

        # スコア算出
        logger.info("スコアを算出しています...")
        extraction_stats = calculate_extraction_rate(results)
        consistency_stats = calculate_temporal_consistency(results)

        # 結果を出力
        print_results(extraction_stats, consistency_stats)

        logger.info("処理が正常に完了しました")
        return 0

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"値エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    finally:
        if video_processor is not None:
            try:
                video_processor.release()
            except Exception as e:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")


def apply_score_correction(
    base_score: float, time_diff_seconds: float, tolerance: float
) -> float:
    """タイムスタンプ誤差がある場合のスコア補正

    Args:
        base_score: ベーススコア（0.0-1.0）
        time_diff_seconds: タイムスタンプ誤差（秒）
        tolerance: 許容ズレ（秒）

    Returns:
        補正後のスコア
    """
    if time_diff_seconds <= tolerance:
        return base_score

    # 許容ズレを超えた場合、誤差に応じて減点
    excess = time_diff_seconds - tolerance
    penalty = min(excess / (tolerance * 10), 0.5)  # 最大50%減点
    return max(base_score - penalty, 0.0)


def calculate_detailed_scores(analysis_result: Dict, tolerance: float = 0.5) -> Dict:
    """詳細スコアを計算（タイムスタンプ誤差補正を含む）

    Args:
        analysis_result: 分析結果（video_timestamp_analyzer.pyの出力）
        tolerance: タイムスタンプ許容ズレ（秒）

    Returns:
        詳細スコアの辞書
    """
    overall = analysis_result.get("overall", {})
    by_segment = analysis_result.get("by_segment", [])

    # タイムスタンプ整合性スコア
    timestamp_consistency = overall.get("timestamp_consistency", {})
    tolerance_key = str(tolerance)
    if tolerance_key in timestamp_consistency.get("by_tolerance", {}):
        ts_stats = timestamp_consistency["by_tolerance"][tolerance_key]
        timestamp_score = ts_stats.get("rate", 0.0)
    else:
        timestamp_score = 0.0

    # 内容一致スコア
    content_accuracy = overall.get("content_accuracy", {})
    cer = content_accuracy.get("cer", {}).get("cer", 1.0)
    wer = content_accuracy.get("wer", {}).get("wer", 1.0)
    token_metrics = content_accuracy.get("token_metrics", {})
    token_f1 = token_metrics.get("f1", 0.0)

    # 内容スコア（CER, WER, Token F1の平均）
    content_score = (1.0 - cer) * 0.4 + (1.0 - wer) * 0.4 + token_f1 * 0.2

    # セグメント単位でスコア補正を適用
    corrected_scores = []
    for segment in by_segment:
        if not segment.get("matched", False):
            continue

        base_score = (
            segment.get("token_f1", 0.0) * 0.5
            + (1.0 - segment.get("cer", 1.0)) * 0.3
            + (1.0 - segment.get("wer", 1.0)) * 0.2
        )

        time_diff = segment.get("time_diff_seconds", 0.0)
        corrected_score = apply_score_correction(base_score, time_diff, tolerance)
        corrected_scores.append(corrected_score)

    # 補正後の平均スコア
    corrected_content_score = (
        sum(corrected_scores) / len(corrected_scores) if corrected_scores else 0.0
    )

    # 総合スコア（タイムスタンプ整合性40%、内容一致60%）
    overall_score = timestamp_score * 0.4 + corrected_content_score * 0.6

    return {
        "timestamp_score": timestamp_score,
        "content_score": content_score,
        "corrected_content_score": corrected_content_score,
        "overall_score": overall_score,
        "cer": cer,
        "wer": wer,
        "token_f1": token_f1,
        "tolerance_seconds": tolerance,
    }


def get_top_samples(
    by_segment: List[Dict], metric: str, top_n: int = 10, reverse: bool = True
) -> List[Dict]:
    """上位N件のサンプルを取得

    Args:
        by_segment: セグメント単位の結果
        metric: ソートに使用する指標（"token_f1", "cer", "wer", "time_diff_seconds"など）
        top_n: 取得件数
        reverse: Trueなら降順、Falseなら昇順

    Returns:
        上位N件のサンプル
    """
    # マッチしたセグメントのみを対象
    matched_segments = [s for s in by_segment if s.get("matched", False)]

    if not matched_segments:
        return []

    # ソート
    sorted_segments = sorted(
        matched_segments,
        key=lambda x: x.get(metric, 0.0) if reverse else -x.get(metric, 0.0),
        reverse=reverse,
    )

    return sorted_segments[:top_n]


def generate_markdown_report(
    analysis_result: Dict, detailed_scores: Dict, tolerance: float = 0.5
) -> str:
    """Markdownレポートを生成

    Args:
        analysis_result: 分析結果
        detailed_scores: 詳細スコア
        tolerance: タイムスタンプ許容ズレ（秒）

    Returns:
        Markdownレポート文字列
    """
    overall = analysis_result.get("overall", {})
    by_segment = analysis_result.get("by_segment", [])
    error_samples = analysis_result.get("error_samples", {})

    report = []
    report.append("# OCR精度評価レポート\n")
    report.append(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**タイムスタンプ許容ズレ**: ±{tolerance}秒\n")
    report.append("\n---\n")

    # エグゼクティブサマリー
    report.append("## エグゼクティブサマリー\n")
    report.append(f"**総合スコア**: {detailed_scores['overall_score']:.2%}\n")
    report.append(f"- タイムスタンプ整合性スコア: {detailed_scores['timestamp_score']:.2%}\n")
    report.append(f"- 内容一致スコア（補正前）: {detailed_scores['content_score']:.2%}\n")
    report.append(f"- 内容一致スコア（補正後）: {detailed_scores['corrected_content_score']:.2%}\n")
    report.append("\n---\n")

    # 定量結果
    report.append("## 定量結果\n\n")
    report.append("### タイムスタンプ整合性\n")
    timestamp_consistency = overall.get("timestamp_consistency", {})
    tolerance_key = str(tolerance)
    if tolerance_key in timestamp_consistency.get("by_tolerance", {}):
        ts_stats = timestamp_consistency["by_tolerance"][tolerance_key]
        report.append(f"- 許容範囲内率: {ts_stats.get('rate', 0.0):.2%}\n")
        report.append(f"- 平均時間差: {ts_stats.get('avg_time_diff', 0.0):.2f}秒\n")
        report.append(f"- 最小時間差: {ts_stats.get('min_time_diff', 0.0):.2f}秒\n")
        report.append(f"- 最大時間差: {ts_stats.get('max_time_diff', 0.0):.2f}秒\n")
    report.append("\n")

    report.append("### 内容一致評価\n")
    content_accuracy = overall.get("content_accuracy", {})
    report.append(f"- **CER（文字エラー率）**: {detailed_scores['cer']:.4f}\n")
    report.append(f"- **WER（単語エラー率）**: {detailed_scores['wer']:.4f}\n")
    report.append(f"- **Token F1**: {detailed_scores['token_f1']:.4f}\n")
    report.append("\n")

    cer_stats = content_accuracy.get("cer", {})
    wer_stats = content_accuracy.get("wer", {})
    token_stats = content_accuracy.get("token_metrics", {})

    report.append("#### 詳細統計\n")
    report.append(f"- 文字レベル - 置換: {cer_stats.get('substitutions', 0)}, "
                  f"挿入: {cer_stats.get('insertions', 0)}, "
                  f"削除: {cer_stats.get('deletions', 0)}\n")
    report.append(f"- 単語レベル - 置換: {wer_stats.get('substitutions', 0)}, "
                  f"挿入: {wer_stats.get('insertions', 0)}, "
                  f"削除: {wer_stats.get('deletions', 0)}\n")
    report.append(f"- トークンレベル - TP: {token_stats.get('true_positives', 0)}, "
                  f"FP: {token_stats.get('false_positives', 0)}, "
                  f"FN: {token_stats.get('false_negatives', 0)}\n")
    report.append("\n---\n")

    # 改善優先度付き課題一覧
    report.append("## 改善優先度付き課題一覧\n\n")
    critical_errors = overall.get("critical_errors", {})
    total_errors = critical_errors.get("total", 0)
    errors_by_type = critical_errors.get("by_type", {})

    if total_errors > 0:
        report.append(f"**総エラー数**: {total_errors}\n\n")
        report.append("| エラー種別 | 件数 | 優先度 |\n")
        report.append("|-----------|------|--------|\n")

        # 優先度を決定（件数が多いほど高優先度）
        sorted_errors = sorted(
            errors_by_type.items(), key=lambda x: x[1], reverse=True
        )
        for error_type, count in sorted_errors:
            priority = "高" if count >= total_errors * 0.3 else "中" if count >= total_errors * 0.1 else "低"
            report.append(f"| {error_type} | {count} | {priority} |\n")
    else:
        report.append("エラーは検出されませんでした。\n")
    report.append("\n---\n")

    # サンプル表示
    report.append("## サンプル表示\n\n")

    # 正解例（Token F1が高い順）
    report.append("### 正解例（上位10件）\n\n")
    top_correct = get_top_samples(by_segment, "token_f1", top_n=10, reverse=True)
    if top_correct:
        report.append("| Segment ID | 参照テキスト | 予測テキスト | Token F1 | CER | WER |\n")
        report.append("|-----------|-------------|-------------|----------|-----|-----|\n")
        for seg in top_correct:
            ref_text = seg.get("ref_text", "")[:30] + "..." if len(seg.get("ref_text", "")) > 30 else seg.get("ref_text", "")
            pred_text = seg.get("pred_text", "")[:30] + "..." if len(seg.get("pred_text", "")) > 30 else seg.get("pred_text", "")
            report.append(
                f"| {seg.get('segment_id', '')} | {ref_text} | {pred_text} | "
                f"{seg.get('token_f1', 0.0):.4f} | {seg.get('cer', 0.0):.4f} | "
                f"{seg.get('wer', 0.0):.4f} |\n"
            )
    else:
        report.append("正解例が見つかりませんでした。\n")
    report.append("\n")

    # 失敗例（Token F1が低い順、またはエラーサンプル）
    report.append("### 失敗例（上位10件）\n\n")
    top_errors = get_top_samples(by_segment, "token_f1", top_n=10, reverse=False)
    if top_errors:
        report.append("| Segment ID | 参照テキスト | 予測テキスト | Token F1 | CER | WER | 時間差（秒） |\n")
        report.append("|-----------|-------------|-------------|----------|-----|-----|------------|\n")
        for seg in top_errors:
            ref_text = seg.get("ref_text", "")[:30] + "..." if len(seg.get("ref_text", "")) > 30 else seg.get("ref_text", "")
            pred_text = seg.get("pred_text", "")[:30] + "..." if len(seg.get("pred_text", "")) > 30 else seg.get("pred_text", "")
            time_diff = seg.get("time_diff_seconds", 0.0)
            report.append(
                f"| {seg.get('segment_id', '')} | {ref_text} | {pred_text} | "
                f"{seg.get('token_f1', 0.0):.4f} | {seg.get('cer', 0.0):.4f} | "
                f"{seg.get('wer', 0.0):.4f} | {time_diff:.2f} |\n"
            )
    else:
        report.append("失敗例が見つかりませんでした。\n")
    report.append("\n")

    # 重要エラーサンプル
    if error_samples:
        report.append("### 重要エラーサンプル\n\n")
        for error_type, errors in error_samples.items():
            if errors:
                report.append(f"#### {error_type}\n\n")
                # 上位5件のみ表示
                for error in errors[:5]:
                    report.append(f"- **Segment ID**: {error.get('segment_id', '')}\n")
                    report.append(f"  - 参照: {error.get('ref_text', '')}\n")
                    report.append(f"  - 予測: {error.get('pred_text', '')}\n")
                    if "time_diff_seconds" in error:
                        report.append(f"  - 時間差: {error['time_diff_seconds']:.2f}秒\n")
                    report.append("\n")
    report.append("\n---\n")

    # 次のアクション提案
    report.append("## 次のアクション提案\n\n")
    if detailed_scores["overall_score"] < 0.5:
        report.append("1. **緊急**: タイムスタンプ抽出の成功率を向上させる\n")
        report.append("2. **緊急**: OCR前処理パラメータの最適化\n")
        report.append("3. **高**: タイムスタンプ整合性の改善\n")
    elif detailed_scores["overall_score"] < 0.8:
        report.append("1. **高**: 内容一致率の向上（CER/WERの改善）\n")
        report.append("2. **中**: タイムスタンプ誤差の削減\n")
        report.append("3. **低**: エラーケースの詳細分析\n")
    else:
        report.append("1. **低**: 残存エラーの詳細分析\n")
        report.append("2. **低**: さらなる精度向上のための最適化\n")
    report.append("\n")

    return "".join(report)


def evaluate_from_analysis(
    input_path: str, tolerance: float = 0.5, output_path: Optional[str] = None
) -> Dict:
    """分析結果から詳細スコアを計算してレポートを生成

    Args:
        input_path: 分析結果JSONファイルパス
        tolerance: タイムスタンプ許容ズレ（秒）
        output_path: 出力JSONファイルパス（オプション）

    Returns:
        評価結果の辞書
    """
    logger.info("=" * 80)
    logger.info("OCR精度評価ツール（分析結果から）")
    logger.info("=" * 80)

    # 分析結果を読み込み
    logger.info(f"分析結果を読み込み中: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        analysis_result = json.load(f)
    logger.info("読み込み完了")

    # 詳細スコアを計算
    logger.info(f"詳細スコアを計算中（許容ズレ: ±{tolerance}秒）...")
    detailed_scores = calculate_detailed_scores(analysis_result, tolerance)

    # Markdownレポートを生成
    logger.info("Markdownレポートを生成中...")
    markdown_report = generate_markdown_report(
        analysis_result, detailed_scores, tolerance
    )

    # 結果をまとめる
    result = {
        "detailed_scores": detailed_scores,
        "markdown_report": markdown_report,
        "analysis_result": analysis_result,
    }

    # JSON出力
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"評価結果を保存しました: {output_file}")

        # Markdownレポートも別ファイルで保存
        markdown_path = output_file.with_suffix(".md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        logger.info(f"Markdownレポートを保存しました: {markdown_path}")
    else:
        # コンソール出力
        print("\n" + "=" * 80)
        print("詳細スコア")
        print("=" * 80)
        print(json.dumps(detailed_scores, ensure_ascii=False, indent=2))
        print("\n" + "=" * 80)
        print("Markdownレポート")
        print("=" * 80)
        print(markdown_report)

    return result


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="タイムスタンプ抽出スコア評価ツール"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="分析結果JSONファイルパス（video_timestamp_analyzer.pyの出力）",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="タイムスタンプ許容ズレ（秒、デフォルト: 0.5）",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="出力JSONファイルパス",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="動画ファイルのパス（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="フレーム抽出間隔（秒、デフォルト: 10.0）",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="抽出するフレーム数（デフォルト: 100）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化",
    )

    args = parser.parse_args()

    # 分析結果から評価するモード
    if args.input:
        try:
            evaluate_from_analysis(
                args.input, tolerance=args.tolerance, output_path=args.out
            )
            return 0
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}", exc_info=True)
            return 1

    # MPS互換性設定を適用（警告抑制）
    setup_mps_compatibility()

    # 従来の動画分析モード
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("タイムスタンプ抽出スコア評価ツール 起動")
    logger.info("=" * 80)

    video_processor = None

    try:
        # 設定ファイルの読み込み
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return 1

        config = ConfigManager(str(config_path))
        logger.info(f"設定ファイルを読み込みました: {config_path}")

        # 動画ファイルのパスを決定
        video_path = args.video or config.get("video.input_path")
        if not video_path:
            logger.error("動画ファイルのパスが指定されていません")
            return 1

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"動画ファイルが見つかりません: {video_path}")
            return 1

        logger.info(f"動画ファイル: {video_path}")

        # 動画処理の初期化
        video_processor = VideoProcessor(str(video_path))
        if not video_processor.open():
            logger.error("動画ファイルを開けませんでした")
            return 1

        logger.info(f"動画情報:")
        logger.info(f"  解像度: {video_processor.width}x{video_processor.height}")
        logger.info(f"  FPS: {video_processor.fps}")
        logger.info(f"  総フレーム数: {video_processor.total_frames}")

        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        logger.info("タイムスタンプ抽出器を初期化しました")

        # フレーム抽出とタイムスタンプ抽出
        logger.info(f"フレーム抽出を開始します（間隔: {args.interval}秒、フレーム数: {args.num_frames}）")
        results = extract_frames_with_timestamps(
            video_processor,
            timestamp_extractor,
            interval_seconds=args.interval,
            num_frames=args.num_frames,
        )

        logger.info(f"フレーム抽出完了: {len(results)}フレーム処理済み")

        # スコア算出
        logger.info("スコアを算出しています...")
        extraction_stats = calculate_extraction_rate(results)
        consistency_stats = calculate_temporal_consistency(results)

        # 結果を出力
        print_results(extraction_stats, consistency_stats)

        logger.info("処理が正常に完了しました")
        return 0

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"値エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    finally:
        if video_processor is not None:
            try:
                video_processor.release()
            except Exception as e:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")


if __name__ == "__main__":
    sys.exit(main())

