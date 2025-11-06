#!/usr/bin/env python3
"""OCR精度評価ツール（統合版）

動画から直接タイムスタンプを抽出し、時系列整合性を評価してレポートを生成します。
正解データ（ゴールドデータ）は不要で、タイムラプス動画の特性を考慮した評価を行います。
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.video import VideoProcessor
from src.video.frame_sampler import FrameSampler
from src.utils import setup_logging, setup_mps_compatibility

logger = logging.getLogger(__name__)


def extract_timestamps_from_video(
    video_processor: VideoProcessor,
    timestamp_extractor: TimestampExtractor,
    frame_sampler: FrameSampler,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    scan_interval: int = 30,
) -> List[Dict]:
    """動画からタイムスタンプを抽出

    Args:
        video_processor: VideoProcessorインスタンス
        timestamp_extractor: TimestampExtractorインスタンス
        frame_sampler: FrameSamplerインスタンス
        start_time: 開始時刻（オプション）
        end_time: 終了時刻（オプション）
        scan_interval: スキャン間隔（フレーム数、デフォルト: 30）

    Returns:
        抽出結果のリスト。各要素は {frame_number, timestamp, confidence, video_sec} を含む
    """
    logger.info(f"フレームサンプリングを開始します...（スキャン間隔: {scan_interval}フレーム）")
    margin_minutes = 10
    sample_frames = frame_sampler.extract_sample_frames(
        video_processor,
        timestamp_extractor,
        start_time=start_time,
        end_time=end_time,
        margin_minutes=margin_minutes,
        scan_interval=scan_interval,
    )

    logger.info(f"サンプルフレーム数: {len(sample_frames)}個")

    results = []
    for frame_num, ts_str, frame in sample_frames:
        # 信頼度付きで再抽出（既に抽出済みだが、信頼度を取得するため）
        timestamp, confidence = timestamp_extractor.extract_with_confidence(frame)

        # 動画内の位置（秒）を計算
        if video_processor.fps and video_processor.fps > 0:
            video_sec = frame_num / video_processor.fps
        else:
            video_sec = 0.0

        results.append({
            "frame_number": frame_num,
            "timestamp": timestamp if timestamp else ts_str,
            "confidence": confidence,
            "video_sec": video_sec,
            "extracted": timestamp is not None,
        })

    return results


def evaluate_temporal_consistency(
    results: List[Dict],
    expected_interval_seconds: float = 300.0,  # 5分 = 300秒
    tolerance_seconds: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
) -> Dict:
    """時系列整合性を評価

    Args:
        results: 抽出結果のリスト
        expected_interval_seconds: 期待される時間間隔（秒、デフォルト: 300秒=5分）
        tolerance_seconds: 許容ズレのリスト（秒）

    Returns:
        時系列整合性評価結果
    """
    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"

    # タイムスタンプをdatetimeに変換
    timestamps: List[Optional[datetime]] = []
    for result in results:
        if result.get("extracted", False) and result.get("timestamp"):
            try:
                dt = datetime.strptime(result["timestamp"], TIMESTAMP_FORMAT)
                timestamps.append(dt)
            except ValueError:
                timestamps.append(None)
        else:
            timestamps.append(None)

    # 連続フレーム間の時間差を計算
    time_diffs: List[float] = []
    valid_pairs = 0

    for i in range(len(timestamps) - 1):
        if timestamps[i] is not None and timestamps[i + 1] is not None:
            diff_seconds = abs((timestamps[i + 1] - timestamps[i]).total_seconds())
            time_diffs.append(diff_seconds)
            valid_pairs += 1

    # 各許容ズレで評価
    by_tolerance = {}
    for tol in tolerance_seconds:
        within_tolerance = sum(1 for diff in time_diffs if abs(diff - expected_interval_seconds) <= tol)
        by_tolerance[str(tol)] = {
            "within_tolerance": within_tolerance,
            "total_pairs": valid_pairs,
            "rate": within_tolerance / valid_pairs if valid_pairs > 0 else 0.0,
        }

    # 統計を計算
    if not time_diffs:
        return {
            "total_extracted": 0,
            "valid_pairs": 0,
            "by_tolerance": by_tolerance,
            "avg_time_diff": 0.0,
            "min_time_diff": 0.0,
            "max_time_diff": 0.0,
            "std_time_diff": 0.0,
        }

    avg_time_diff = sum(time_diffs) / len(time_diffs)
    min_time_diff = min(time_diffs)
    max_time_diff = max(time_diffs)
    std_time_diff = (
        sum((d - avg_time_diff) ** 2 for d in time_diffs) / len(time_diffs)
    ) ** 0.5

    return {
        "total_extracted": sum(1 for r in results if r.get("extracted", False)),
        "valid_pairs": valid_pairs,
        "by_tolerance": by_tolerance,
        "avg_time_diff": avg_time_diff,
        "min_time_diff": min_time_diff,
        "max_time_diff": max_time_diff,
        "std_time_diff": std_time_diff,
        "expected_interval_seconds": expected_interval_seconds,
    }


def calculate_extraction_rate(results: List[Dict]) -> Dict:
    """抽出成功率を計算

    Args:
        results: 抽出結果のリスト

    Returns:
        抽出成功率の統計情報
    """
    total = len(results)
    success_count = sum(1 for r in results if r.get("extracted", False))

    # 信頼度統計（成功したもののみ）
    confidences = [r.get("confidence", 0.0) for r in results if r.get("extracted", False)]

    return {
        "total_frames": total,
        "success_count": success_count,
        "success_rate": (success_count / total * 100.0) if total > 0 else 0.0,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
    }


def generate_markdown_report(
    extraction_stats: Dict,
    consistency_stats: Dict,
    video_metadata: Dict,
    tolerance: float = 0.5,
) -> str:
    """Markdownレポートを生成

    Args:
        extraction_stats: 抽出成功率の統計情報
        consistency_stats: 時系列整合性の統計情報
        video_metadata: 動画メタデータ
        tolerance: タイムスタンプ許容ズレ（秒）

    Returns:
        Markdownレポート文字列
    """
    report = []
    report.append("# OCR精度評価レポート（タイムラプス動画）\n")
    report.append(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**タイムスタンプ許容ズレ**: ±{tolerance}秒\n")
    report.append("\n---\n")

    # 動画情報
    report.append("## 動画情報\n\n")
    report.append(f"- **解像度**: {video_metadata.get('width', 'N/A')}×{video_metadata.get('height', 'N/A')}\n")
    report.append(f"- **フレームレート**: {video_metadata.get('fps', 0.0):.2f} FPS\n")
    report.append(f"- **総フレーム数**: {video_metadata.get('total_frames', 0):,}\n")
    report.append(f"- **動画の長さ**: {video_metadata.get('duration_seconds', 0.0):.2f}秒\n")
    report.append("\n---\n")

    # エグゼクティブサマリー
    report.append("## エグゼクティブサマリー\n\n")
    success_rate = extraction_stats.get("success_rate", 0.0)
    consistency_rate = consistency_stats.get("by_tolerance", {}).get(
        str(tolerance), {}
    ).get("rate", 0.0)
    overall_score = (success_rate * 0.5 + consistency_rate * 100.0 * 0.5) / 100.0

    report.append(f"**総合スコア**: {overall_score:.2%}\n")
    report.append(f"- 抽出成功率: {success_rate:.2f}%\n")
    report.append(f"- 時系列整合性（±{tolerance}秒）: {consistency_rate:.2%}\n")
    report.append("\n---\n")

    # 抽出成功率
    report.append("## 抽出成功率\n\n")
    report.append(f"- **処理フレーム数**: {extraction_stats.get('total_frames', 0)}\n")
    report.append(f"- **抽出成功数**: {extraction_stats.get('success_count', 0)}\n")
    report.append(f"- **抽出成功率**: {success_rate:.2f}%\n")
    report.append("\n")

    report.append("### 信頼度統計\n")
    report.append(f"- 平均信頼度: {extraction_stats.get('average_confidence', 0.0):.4f}\n")
    report.append(f"- 最小信頼度: {extraction_stats.get('min_confidence', 0.0):.4f}\n")
    report.append(f"- 最大信頼度: {extraction_stats.get('max_confidence', 0.0):.4f}\n")
    report.append("\n---\n")

    # 時系列整合性
    report.append("## 時系列整合性評価\n\n")
    report.append(f"- **有効ペア数**: {consistency_stats.get('valid_pairs', 0)}\n")
    report.append(f"- **期待時間間隔**: {consistency_stats.get('expected_interval_seconds', 300.0):.0f}秒（5分）\n")
    report.append("\n")

    if consistency_stats.get("valid_pairs", 0) > 0:
        report.append("### 時間差統計\n")
        report.append(f"- 平均時間差: {consistency_stats.get('avg_time_diff', 0.0):.2f}秒\n")
        report.append(f"- 最小時間差: {consistency_stats.get('min_time_diff', 0.0):.2f}秒\n")
        report.append(f"- 最大時間差: {consistency_stats.get('max_time_diff', 0.0):.2f}秒\n")
        report.append(f"- 標準偏差: {consistency_stats.get('std_time_diff', 0.0):.2f}秒\n")
        report.append("\n")

        report.append("### 許容ズレ別の整合性率\n\n")
        report.append("| 許容ズレ | 整合性率 | 整合性数/総数 |\n")
        report.append("|---------|---------|-------------|\n")
        for tol_str, stats in sorted(
            consistency_stats.get("by_tolerance", {}).items(), key=lambda x: float(x[0])
        ):
            rate = stats.get("rate", 0.0)
            within = stats.get("within_tolerance", 0)
            total = stats.get("total_pairs", 0)
            report.append(f"| ±{tol_str}秒 | {rate:.2%} | {within}/{total} |\n")
    else:
        report.append("有効なペアが見つかりませんでした。\n")
    report.append("\n---\n")

    # 改善提案
    report.append("## 改善提案\n\n")
    if success_rate < 90.0:
        report.append("1. **緊急**: タイムスタンプ抽出の成功率を向上させる（目標: ≥90%）\n")
        report.append("   - OCR前処理パラメータの最適化\n")
        report.append("   - ROI領域の設定見直し\n")
    if consistency_rate < 0.8:
        report.append("2. **高**: 時系列整合性の改善（目標: ≥80%）\n")
        report.append("   - タイムスタンプ抽出の精度向上\n")
        report.append("   - 時系列補正ロジックの強化\n")
    if extraction_stats.get("average_confidence", 0.0) < 0.8:
        report.append("3. **中**: 信頼度の向上（目標: ≥0.8）\n")
        report.append("   - OCRエンジンの設定最適化\n")
        report.append("   - 複数OCRエンジンのアンサンブル強化\n")
    if success_rate >= 90.0 and consistency_rate >= 0.8:
        report.append("1. **低**: さらなる精度向上のための最適化\n")
    report.append("\n")

    return "".join(report)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="OCR精度評価ツール（統合版） - 動画から直接タイムスタンプを抽出して評価"
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
        "--tolerance",
        type=float,
        default=0.5,
        help="タイムスタンプ許容ズレ（秒、デフォルト: 0.5）",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=5,
        help="サンプリング間隔（分、デフォルト: 5）",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=30,
        help="スキャン間隔（フレーム数、デフォルト: 30）。1/100にする場合は3000を指定",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="出力JSONファイルパス（オプション）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化",
    )

    args = parser.parse_args()

    # MPS互換性設定を適用（警告抑制）
    setup_mps_compatibility()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("OCR精度評価ツール（統合版） 起動")
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

        # 動画メタデータを取得
        video_metadata = {
            "width": video_processor.width,
            "height": video_processor.height,
            "fps": video_processor.fps,
            "total_frames": video_processor.total_frames,
            "duration_seconds": (
                video_processor.total_frames / video_processor.fps
                if video_processor.fps and video_processor.fps > 0
                else 0.0
            ),
        }

        logger.info(f"動画情報:")
        logger.info(f"  解像度: {video_metadata['width']}x{video_metadata['height']}")
        logger.info(f"  FPS: {video_metadata['fps']}")
        logger.info(f"  総フレーム数: {video_metadata['total_frames']}")
        logger.info(f"  動画の長さ: {video_metadata['duration_seconds']:.2f}秒")

        # タイムスタンプ抽出器の初期化
        confidence_threshold = config.get(
            "timestamp.extraction.confidence_threshold", 0.2
        )
        timestamp_extractor = TimestampExtractor(
            confidence_threshold=confidence_threshold
        )
        logger.info("タイムスタンプ抽出器を初期化しました")

        # フレームサンプラーの初期化
        tolerance_seconds = config.get("video.tolerance_seconds", 10)
        frame_sampler = FrameSampler(
            interval_minutes=args.interval_minutes, tolerance_seconds=tolerance_seconds
        )
        logger.info(
            f"フレームサンプラーを初期化しました（間隔: {args.interval_minutes}分、許容誤差: ±{tolerance_seconds}秒）"
        )

        # タイムスタンプ抽出
        logger.info("タイムスタンプ抽出を開始します...")
        results = extract_timestamps_from_video(
            video_processor,
            timestamp_extractor,
            frame_sampler,
            scan_interval=args.scan_interval,
        )

        logger.info(f"タイムスタンプ抽出完了: {len(results)}フレーム処理済み")

        # 統計を計算
        logger.info("統計を計算中...")
        extraction_stats = calculate_extraction_rate(results)
        expected_interval = args.interval_minutes * 60.0  # 分を秒に変換
        consistency_stats = evaluate_temporal_consistency(
            results,
            expected_interval_seconds=expected_interval,
            tolerance_seconds=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        # Markdownレポートを生成
        logger.info("Markdownレポートを生成中...")
        markdown_report = generate_markdown_report(
            extraction_stats, consistency_stats, video_metadata, args.tolerance
        )

        # 結果をまとめる
        result = {
            "video_metadata": video_metadata,
            "extraction_stats": extraction_stats,
            "consistency_stats": consistency_stats,
            "results": results,
            "markdown_report": markdown_report,
        }

        # 出力
        if args.out:
            output_file = Path(args.out)
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
            print("評価結果")
            print("=" * 80)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("\n" + "=" * 80)
            print("Markdownレポート")
            print("=" * 80)
            print(markdown_report)

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

