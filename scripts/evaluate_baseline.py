#!/usr/bin/env python3
"""ベースライン評価統合スクリプト

指定セッションに対してMOTメトリクス、再投影誤差、パフォーマンスを評価し、結果を統合します。
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def evaluate_mot_metrics(session_dir: Path, gt_tracks_path: Path | None, config_path: str) -> dict | None:
    """MOTメトリクスを評価

    Args:
        session_dir: セッションディレクトリ
        gt_tracks_path: Ground Truthトラックファイルのパス（オプション）
        config_path: 設定ファイルのパス

    Returns:
        評価結果の辞書（データがない場合はNone）
    """
    if not gt_tracks_path or not gt_tracks_path.exists():
        logger.info("Ground Truthトラックファイルが指定されていないか存在しません。MOT評価をスキップします。")
        return None

    tracks_path = session_dir / "phase2.5_tracking" / "tracks.json"
    if not tracks_path.exists():
        logger.warning("トラックファイルが見つかりません。MOT評価をスキップします。")
        return None

    logger.info("MOTメトリクスを評価中...")

    # summary.jsonからフレーム数を取得
    summary_path = session_dir / "summary.json"
    frame_count = 100  # デフォルト値
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
            frame_count = summary.get("num_frames", frame_count)

    output_path = session_dir / "mot_metrics.json"

    cmd = [
        sys.executable,
        "scripts/evaluate_mot_metrics.py",
        "--gt",
        str(gt_tracks_path),
        "--tracks",
        str(tracks_path),
        "--frames",
        str(frame_count),
        "--output",
        str(output_path),
        "--config",
        config_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("MOTメトリクス評価が完了しました")

        # 結果を読み込み
        if output_path.exists():
            with output_path.open(encoding="utf-8") as f:
                mot_result = json.load(f)
                return {
                    "MOTA": mot_result.get("metrics", {}).get("MOTA", 0.0),
                    "IDF1": mot_result.get("metrics", {}).get("IDF1", 0.0),
                    "ID_Switches": mot_result.get("metrics", {}).get("ID_Switches", 0),
                    "available": True,
                    "output_path": str(output_path),
                }
    except subprocess.CalledProcessError as e:
        logger.error(f"MOTメトリクス評価に失敗しました: {e}")
        logger.error(f"標準エラー: {e.stderr}")
        return None


def evaluate_reprojection_error(session_dir: Path, points_path: Path | None, config_path: str) -> dict | None:
    """再投影誤差を評価

    Args:
        session_dir: セッションディレクトリ
        points_path: 対応点ファイルのパス（オプション）
        config_path: 設定ファイルのパス

    Returns:
        評価結果の辞書（データがない場合はNone）
    """
    if not points_path or not points_path.exists():
        logger.info("対応点ファイルが指定されていないか存在しません。再投影誤差評価をスキップします。")
        return None

    logger.info("再投影誤差を評価中...")

    output_path = session_dir / "reprojection_error.json"
    error_map_path = session_dir / "reprojection_error_map.png"

    # 設定から画像サイズを取得
    config = ConfigManager(config_path)
    floormap_config = config.get("floormap", {})
    image_height = floormap_config.get("image_height", 1369)
    image_width = floormap_config.get("image_width", 1878)

    cmd = [
        sys.executable,
        "scripts/evaluate_reprojection_error.py",
        "--points",
        str(points_path),
        "--config",
        config_path,
        "--output",
        str(output_path),
        "--error-map",
        str(error_map_path),
        "--image-shape",
        str(image_height),
        str(image_width),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("再投影誤差評価が完了しました")

        # 結果を読み込み
        if output_path.exists():
            with output_path.open(encoding="utf-8") as f:
                error_result = json.load(f)
                error_metrics = error_result.get("error_metrics", {})
                return {
                    "mean_error": error_metrics.get("mean_error", 0.0),
                    "max_error": error_metrics.get("max_error", 0.0),
                    "min_error": error_metrics.get("min_error", 0.0),
                    "std_error": error_metrics.get("std_error", 0.0),
                    "available": True,
                    "output_path": str(output_path),
                    "error_map_path": str(error_map_path),
                }
    except subprocess.CalledProcessError as e:
        logger.error(f"再投影誤差評価に失敗しました: {e}")
        logger.error(f"標準エラー: {e.stderr}")
        return None


def evaluate_performance(session_dir: Path, video_path: str, config_path: str) -> dict | None:
    """パフォーマンスを評価

    Args:
        session_dir: セッションディレクトリ
        video_path: 動画ファイルのパス
        config_path: 設定ファイルのパス

    Returns:
        評価結果の辞書
    """
    logger.info("パフォーマンスを評価中...")

    output_path = session_dir / "performance_metrics.json"

    cmd = [
        sys.executable,
        "scripts/measure_performance.py",
        "--video",
        video_path,
        "--config",
        config_path,
        "--output",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("パフォーマンス評価が完了しました")

        # 結果を読み込み
        if output_path.exists():
            with output_path.open(encoding="utf-8") as f:
                perf_result = json.load(f)
                phase_times = perf_result.get("phase_times", {})
                num_frames = perf_result.get("num_frames", 1)
                total_time = phase_times.get("total", 0.0)
                time_per_frame = total_time / num_frames if num_frames > 0 else 0.0

                memory = perf_result.get("memory", {})
                memory_increase = memory.get("increase_mb", 0.0)

                return {
                    "time_per_frame_seconds": time_per_frame,
                    "total_time_seconds": total_time,
                    "memory_peak_mb": memory.get("final_mb", 0.0),
                    "memory_increase_mb": memory_increase,
                    "num_frames": num_frames,
                    "phase_times": phase_times,
                    "available": True,
                    "output_path": str(output_path),
                }
    except subprocess.CalledProcessError as e:
        logger.error(f"パフォーマンス評価に失敗しました: {e}")
        logger.error(f"標準エラー: {e.stderr}")
        return None


def evaluate_baseline(
    session_id: str,
    config_path: str,
    gt_tracks_path: Path | None = None,
    points_path: Path | None = None,
) -> dict:
    """ベースライン評価を統合実行

    Args:
        session_id: セッションID
        config_path: 設定ファイルのパス
        gt_tracks_path: Ground Truthトラックファイルのパス（オプション）
        points_path: 対応点ファイルのパス（オプション）

    Returns:
        統合評価結果の辞書
    """
    logger.info("=" * 80)
    logger.info("ベースライン評価を開始")
    logger.info(f"セッションID: {session_id}")
    logger.info("=" * 80)

    # 設定ファイルから情報を取得
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output"))
    video_path = config.get("video.input_path", "")

    session_dir = output_dir / "sessions" / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"セッションディレクトリが見つかりません: {session_dir}")

    # 各評価を実行
    mot_metrics = evaluate_mot_metrics(session_dir, gt_tracks_path, config_path)
    reprojection_error = evaluate_reprojection_error(session_dir, points_path, config_path)
    performance = evaluate_performance(session_dir, video_path, config_path)

    # 目標値を設定
    targets = {
        "MOTA": 0.7,
        "IDF1": 0.8,
        "mean_error": 2.0,
        "max_error": 4.0,
        "time_per_frame": 2.0,
        "memory_mb": 12288,  # 12GB
    }

    # 達成状況を判定
    achieved = {
        "MOTA": mot_metrics is not None and mot_metrics.get("MOTA", 0.0) >= targets["MOTA"],
        "IDF1": mot_metrics is not None and mot_metrics.get("IDF1", 0.0) >= targets["IDF1"],
        "mean_error": reprojection_error is not None
        and reprojection_error.get("mean_error", float("inf")) <= targets["mean_error"],
        "max_error": reprojection_error is not None
        and reprojection_error.get("max_error", float("inf")) <= targets["max_error"],
        "time_per_frame": performance is not None
        and performance.get("time_per_frame_seconds", float("inf")) <= targets["time_per_frame"],
        "memory": performance is not None
        and performance.get("memory_increase_mb", float("inf")) <= targets["memory_mb"],
    }

    # summary.jsonからパイプライン情報を取得
    pipeline_info = {}
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
            pipeline_info = {
                "num_frames": summary.get("num_frames", 0),
                "total_time_seconds": summary.get("total_time_seconds", 0.0),
            }

    # 統合結果を構築
    result = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline": pipeline_info,
        "mot_metrics": mot_metrics or {"available": False},
        "reprojection_error": reprojection_error or {"available": False},
        "performance": performance or {"available": False},
        "targets": targets,
        "achieved": achieved,
    }

    # 結果を保存
    output_path = session_dir / "baseline_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"ベースライン評価結果を保存しました: {output_path}")

    return result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ベースライン評価統合スクリプト")
    parser.add_argument("--session", type=str, required=True, help="セッションID")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--gt", type=str, help="Ground Truthトラックファイルのパス（オプション）")
    parser.add_argument("--points", type=str, help="対応点ファイルのパス（オプション）")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    try:
        gt_path = Path(args.gt) if args.gt else None
        points_path = Path(args.points) if args.points else None

        result = evaluate_baseline(args.session, args.config, gt_path, points_path)

        logger.info("=" * 80)
        logger.info("ベースライン評価が完了しました")
        logger.info("=" * 80)

        # 結果サマリーを表示
        logger.info("評価結果サマリー:")
        if result["mot_metrics"].get("available"):
            logger.info(f"  MOTA: {result['mot_metrics']['MOTA']:.3f} (目標: {result['targets']['MOTA']:.1f})")
            logger.info(f"  IDF1: {result['mot_metrics']['IDF1']:.3f} (目標: {result['targets']['IDF1']:.1f})")
        else:
            logger.info("  MOTメトリクス: データなし")

        if result["reprojection_error"].get("available"):
            logger.info(
                f"  再投影平均誤差: {result['reprojection_error']['mean_error']:.2f} px (目標: {result['targets']['mean_error']:.1f} px)"
            )
            logger.info(
                f"  再投影最大誤差: {result['reprojection_error']['max_error']:.2f} px (目標: {result['targets']['max_error']:.1f} px)"
            )
        else:
            logger.info("  再投影誤差: データなし")

        if result["performance"].get("available"):
            logger.info(
                f"  処理時間/フレーム: {result['performance']['time_per_frame_seconds']:.2f} 秒 (目標: {result['targets']['time_per_frame']:.1f} 秒)"
            )
            logger.info(
                f"  メモリ増加: {result['performance']['memory_increase_mb']:.0f} MB (目標: {result['targets']['memory_mb']:.0f} MB)"
            )
        else:
            logger.info("  パフォーマンス: データなし")

        return 0
    except Exception as e:
        logger.error(f"ベースライン評価中にエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
