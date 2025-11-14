#!/usr/bin/env python3
"""ベースラインレポート生成スクリプト

評価結果を読み込み、Markdown形式のレポートを生成します。
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def format_status(achieved: bool) -> str:
    """達成状況をフォーマット

    Args:
        achieved: 達成状況

    Returns:
        フォーマットされた文字列
    """
    return "✅ 達成" if achieved else "❌ 未達成"


def generate_report(session_id: str, config_path: str, output_path: Path | None = None) -> str:
    """ベースラインレポートを生成

    Args:
        session_id: セッションID
        config_path: 設定ファイルのパス
        output_path: 出力ファイルのパス（オプション、指定しない場合はセッションディレクトリに保存）

    Returns:
        レポートの内容（Markdown形式）
    """
    logger.info("=" * 80)
    logger.info("ベースラインレポートを生成中")
    logger.info(f"セッションID: {session_id}")
    logger.info("=" * 80)

    # 設定ファイルから情報を取得
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output"))
    session_dir = output_dir / "sessions" / session_id

    if not session_dir.exists():
        raise FileNotFoundError(f"セッションディレクトリが見つかりません: {session_dir}")

    # 評価結果を読み込み
    metrics_path = session_dir / "baseline_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"評価結果ファイルが見つかりません: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    # メタデータを読み込み（オプション）
    metadata_path = session_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    # レポートを生成
    report_lines = []

    # ヘッダー
    report_lines.append("# ベースライン評価レポート")
    report_lines.append("")
    report_lines.append(f"**セッションID**: `{session_id}`")
    report_lines.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if metadata.get("timestamp"):
        report_lines.append(f"**実行日時**: {metadata['timestamp']}")
    report_lines.append("")

    # 実行概要
    report_lines.append("## 実行概要")
    report_lines.append("")
    pipeline = metrics.get("pipeline", {})
    report_lines.append(f"- **処理フレーム数**: {pipeline.get('num_frames', 'N/A')}")
    report_lines.append(f"- **総処理時間**: {pipeline.get('total_time_seconds', 0.0):.2f} 秒")
    report_lines.append("")

    # MOTメトリクス
    report_lines.append("## MOTメトリクス")
    report_lines.append("")
    mot_metrics = metrics.get("mot_metrics", {})
    if mot_metrics.get("available"):
        targets = metrics.get("targets", {})
        achieved = metrics.get("achieved", {})

        report_lines.append("### 結果")
        report_lines.append("")
        report_lines.append("| メトリクス | 値 | 目標値 | 達成状況 |")
        report_lines.append("|-----------|-----|--------|----------|")
        report_lines.append(
            f"| MOTA | {mot_metrics.get('MOTA', 0.0):.3f} | {targets.get('MOTA', 0.7):.1f} | {format_status(achieved.get('MOTA', False))} |"
        )
        report_lines.append(
            f"| IDF1 | {mot_metrics.get('IDF1', 0.0):.3f} | {targets.get('IDF1', 0.8):.1f} | {format_status(achieved.get('IDF1', False))} |"
        )
        report_lines.append(f"| ID Switches | {mot_metrics.get('ID_Switches', 0)} | - | - |")
        report_lines.append("")

        report_lines.append("### 詳細")
        report_lines.append("")
        report_lines.append(f"- **MOTA (Multiple Object Tracking Accuracy)**: {mot_metrics.get('MOTA', 0.0):.3f}")
        report_lines.append("  - 範囲: 0.0-1.0（高いほど良い）")
        report_lines.append("  - 目標値: ≥ 0.7")
        report_lines.append("")
        report_lines.append(f"- **IDF1 (Identity F1-score)**: {mot_metrics.get('IDF1', 0.0):.3f}")
        report_lines.append("  - 範囲: 0.0-1.0（高いほど良い）")
        report_lines.append("  - 目標値: ≥ 0.8")
        report_lines.append("")
        report_lines.append(f"- **ID Switches**: {mot_metrics.get('ID_Switches', 0)}")
        report_lines.append("  - トラックIDが誤って切り替わった回数（少ないほど良い）")
        report_lines.append("")
    else:
        report_lines.append("**データなし**: Ground Truthトラックファイルが指定されていないか、存在しません。")
        report_lines.append("")
        report_lines.append(
            "MOTメトリクスを評価するには、`--gt`オプションでGround Truthトラックファイルを指定してください。"
        )
        report_lines.append("")

    # 再投影誤差
    report_lines.append("## 再投影誤差")
    report_lines.append("")
    reprojection_error = metrics.get("reprojection_error", {})
    if reprojection_error.get("available"):
        targets = metrics.get("targets", {})
        achieved = metrics.get("achieved", {})

        report_lines.append("### 結果")
        report_lines.append("")
        report_lines.append("| メトリクス | 値 (px) | 目標値 (px) | 達成状況 |")
        report_lines.append("|-----------|---------|-------------|----------|")
        report_lines.append(
            f"| 平均誤差 | {reprojection_error.get('mean_error', 0.0):.2f} | {targets.get('mean_error', 2.0):.1f} | {format_status(achieved.get('mean_error', False))} |"
        )
        report_lines.append(
            f"| 最大誤差 | {reprojection_error.get('max_error', 0.0):.2f} | {targets.get('max_error', 4.0):.1f} | {format_status(achieved.get('max_error', False))} |"
        )
        report_lines.append(f"| 最小誤差 | {reprojection_error.get('min_error', 0.0):.2f} | - | - |")
        report_lines.append(f"| 標準偏差 | {reprojection_error.get('std_error', 0.0):.2f} | - | - |")
        report_lines.append("")

        report_lines.append("### 詳細")
        report_lines.append("")
        report_lines.append(
            f"- **平均誤差**: {reprojection_error.get('mean_error', 0.0):.2f} ピクセル（目標: ≤ {targets.get('mean_error', 2.0):.1f} px）"
        )
        report_lines.append(
            f"- **最大誤差**: {reprojection_error.get('max_error', 0.0):.2f} ピクセル（目標: ≤ {targets.get('max_error', 4.0):.1f} px）"
        )
        report_lines.append("")
        if reprojection_error.get("error_map_path"):
            report_lines.append(f"- **誤差マップ**: `{reprojection_error['error_map_path']}`")
            report_lines.append("")
    else:
        report_lines.append("**データなし**: 対応点ファイルが指定されていないか、存在しません。")
        report_lines.append("")
        report_lines.append("再投影誤差を評価するには、`--points`オプションで対応点ファイルを指定してください。")
        report_lines.append("")

    # パフォーマンス
    report_lines.append("## パフォーマンス")
    report_lines.append("")
    performance = metrics.get("performance", {})
    if performance.get("available"):
        targets = metrics.get("targets", {})
        achieved = metrics.get("achieved", {})

        report_lines.append("### 結果")
        report_lines.append("")
        report_lines.append("| メトリクス | 値 | 目標値 | 達成状況 |")
        report_lines.append("|-----------|-----|--------|----------|")
        report_lines.append(
            f"| 処理時間/フレーム | {performance.get('time_per_frame_seconds', 0.0):.2f} 秒 | {targets.get('time_per_frame', 2.0):.1f} 秒 | {format_status(achieved.get('time_per_frame', False))} |"
        )
        report_lines.append(
            f"| メモリ増加 | {performance.get('memory_increase_mb', 0.0):.0f} MB | {targets.get('memory_mb', 12288):.0f} MB | {format_status(achieved.get('memory', False))} |"
        )
        report_lines.append("")

        report_lines.append("### 詳細")
        report_lines.append("")
        report_lines.append(f"- **総処理時間**: {performance.get('total_time_seconds', 0.0):.2f} 秒")
        report_lines.append(f"- **処理フレーム数**: {performance.get('num_frames', 0)}")
        report_lines.append(
            f"- **フレームあたりの処理時間**: {performance.get('time_per_frame_seconds', 0.0):.2f} 秒（目標: ≤ {targets.get('time_per_frame', 2.0):.1f} 秒）"
        )
        report_lines.append("")
        report_lines.append(f"- **メモリピーク**: {performance.get('memory_peak_mb', 0.0):.0f} MB")
        report_lines.append(
            f"- **メモリ増加**: {performance.get('memory_increase_mb', 0.0):.0f} MB（目標: ≤ {targets.get('memory_mb', 12288):.0f} MB）"
        )
        report_lines.append("")

        # フェーズ別処理時間
        phase_times = performance.get("phase_times", {})
        if phase_times:
            report_lines.append("### フェーズ別処理時間")
            report_lines.append("")
            report_lines.append("| フェーズ | 処理時間 (秒) | 割合 (%) |")
            report_lines.append("|---------|--------------|----------|")
            total_time = phase_times.get("total", 0.0)
            for phase, time_sec in phase_times.items():
                if phase != "total":
                    percentage = (time_sec / total_time * 100) if total_time > 0 else 0.0
                    report_lines.append(f"| {phase} | {time_sec:.2f} | {percentage:.1f} |")
            report_lines.append("")
    else:
        report_lines.append("**データなし**: パフォーマンス評価に失敗しました。")
        report_lines.append("")

    # 達成状況サマリー
    report_lines.append("## 達成状況サマリー")
    report_lines.append("")
    achieved = metrics.get("achieved", {})
    all_achieved = all(achieved.values())
    report_lines.append(f"**全体**: {'✅ すべて達成' if all_achieved else '❌ 一部未達成'}")
    report_lines.append("")
    report_lines.append("| 項目 | 達成状況 |")
    report_lines.append("|------|----------|")
    for key, value in achieved.items():
        report_lines.append(f"| {key} | {format_status(value)} |")
    report_lines.append("")

    # 推奨アクション
    if not all_achieved:
        report_lines.append("## 推奨アクション")
        report_lines.append("")
        report_lines.append("以下の項目が目標値を達成していません:")
        report_lines.append("")
        for key, value in achieved.items():
            if not value:
                report_lines.append(f"- **{key}**: 目標値を達成していません。")
                if key == "MOTA" or key == "IDF1":
                    report_lines.append(
                        "  - 追跡パラメータ（appearance_weight, motion_weight, iou_threshold等）の調整を検討してください。"
                    )
                elif key == "mean_error" or key == "max_error":
                    report_lines.append("  - ホモグラフィ行列の再キャリブレーションを検討してください。")
                elif key == "time_per_frame":
                    report_lines.append("  - バッチサイズの調整や処理の最適化を検討してください。")
                elif key == "memory":
                    report_lines.append("  - メモリ使用量の最適化を検討してください。")
        report_lines.append("")

    # ファイル情報
    report_lines.append("## 関連ファイル")
    report_lines.append("")
    report_lines.append(f"- **評価結果JSON**: `{metrics_path.relative_to(output_dir)}`")
    if mot_metrics.get("output_path"):
        report_lines.append(f"- **MOTメトリクス**: `{Path(mot_metrics['output_path']).relative_to(output_dir)}`")
    if reprojection_error.get("output_path"):
        report_lines.append(f"- **再投影誤差**: `{Path(reprojection_error['output_path']).relative_to(output_dir)}`")
    if performance.get("output_path"):
        report_lines.append(f"- **パフォーマンス**: `{Path(performance['output_path']).relative_to(output_dir)}`")
    report_lines.append("")

    report_content = "\n".join(report_lines)

    # ファイルに保存
    if output_path is None:
        output_path = session_dir / "baseline_report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"ベースラインレポートを保存しました: {output_path}")

    return report_content


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ベースラインレポート生成スクリプト")
    parser.add_argument("--session", type=str, required=True, help="セッションID")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--output", type=str, help="出力ファイルのパス（オプション）")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    try:
        output_path = Path(args.output) if args.output else None
        generate_report(args.session, args.config, output_path)
        logger.info("=" * 80)
        logger.info("ベースラインレポート生成が完了しました")
        logger.info("=" * 80)
        return 0
    except Exception as e:
        logger.error(f"ベースラインレポート生成中にエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
