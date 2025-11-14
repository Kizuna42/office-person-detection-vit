#!/usr/bin/env python3
"""ベースライン統合実行スクリプト

パイプライン実行、評価、レポート生成を一括で実行します。
"""

import argparse
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
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


def run_full_baseline(
    config_path: str,
    tag: str | None = None,
    gt_tracks_path: Path | None = None,
    points_path: Path | None = None,
    skip_pipeline: bool = False,
    session_id: str | None = None,
) -> int:
    """ベースライン評価を統合実行

    Args:
        config_path: 設定ファイルのパス
        tag: セッションタグ（オプション）
        gt_tracks_path: Ground Truthトラックファイルのパス（オプション）
        points_path: 対応点ファイルのパス（オプション）
        skip_pipeline: パイプライン実行をスキップするか（既存セッションを使用）
        session_id: 既存セッションID（skip_pipeline=Trueの場合に必須）

    Returns:
        終了コード（0: 成功、1: 失敗）
    """
    logger.info("=" * 80)
    logger.info("ベースライン統合実行を開始")
    logger.info("=" * 80)

    # 設定ファイルの読み込み
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output"))

    # ステップ1: パイプライン実行（スキップ可能）
    if not skip_pipeline:
        logger.info("=" * 80)
        logger.info("ステップ1: パイプライン実行")
        logger.info("=" * 80)

        cmd = [sys.executable, "scripts/run_baseline.py", "--config", config_path]
        if tag:
            cmd.extend(["--tag", tag])

        try:
            subprocess.run(cmd, check=True)
            logger.info("パイプライン実行が完了しました")
        except subprocess.CalledProcessError as e:
            logger.error(f"パイプライン実行に失敗しました: {e}")
            return 1

        # セッションIDを取得
        output_manager = OutputManager(output_dir)
        latest_session = output_manager.get_latest_session()
        if not latest_session:
            logger.error("最新セッションが見つかりませんでした")
            return 1
        session_id = latest_session.name
        logger.info(f"セッションIDを取得しました: {session_id}")
    else:
        if not session_id:
            logger.error("skip_pipeline=Trueの場合、session_idを指定してください")
            return 1
        logger.info(f"パイプライン実行をスキップします。既存セッションを使用: {session_id}")

    # ステップ2: 評価実行
    logger.info("=" * 80)
    logger.info("ステップ2: 評価実行")
    logger.info("=" * 80)

    cmd = [sys.executable, "scripts/evaluate_baseline.py", "--session", session_id, "--config", config_path]
    if gt_tracks_path:
        cmd.extend(["--gt", str(gt_tracks_path)])
    if points_path:
        cmd.extend(["--points", str(points_path)])

    try:
        subprocess.run(cmd, check=True)
        logger.info("評価実行が完了しました")
    except subprocess.CalledProcessError as e:
        logger.error(f"評価実行に失敗しました: {e}")
        return 1

    # ステップ3: レポート生成
    logger.info("=" * 80)
    logger.info("ステップ3: レポート生成")
    logger.info("=" * 80)

    cmd = [sys.executable, "scripts/generate_baseline_report.py", "--session", session_id, "--config", config_path]

    try:
        subprocess.run(cmd, check=True)
        logger.info("レポート生成が完了しました")
    except subprocess.CalledProcessError as e:
        logger.error(f"レポート生成に失敗しました: {e}")
        return 1

    # 結果サマリー
    logger.info("=" * 80)
    logger.info("ベースライン統合実行が完了しました")
    logger.info("=" * 80)
    logger.info(f"セッションID: {session_id}")

    session_dir = output_dir / "sessions" / session_id
    logger.info("生成されたファイル:")
    logger.info(f"  - 評価結果: {session_dir / 'baseline_metrics.json'}")
    logger.info(f"  - レポート: {session_dir / 'baseline_report.md'}")

    # 評価結果を読み込んでサマリーを表示
    metrics_path = session_dir / "baseline_metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)

        logger.info("")
        logger.info("評価結果サマリー:")
        achieved = metrics.get("achieved", {})
        all_achieved = all(achieved.values())

        if all_achieved:
            logger.info("  ✅ すべての目標値を達成しました")
        else:
            logger.info("  ❌ 一部の目標値を達成していません:")
            for key, value in achieved.items():
                if not value:
                    logger.info(f"    - {key}: 未達成")

    return 0


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ベースライン統合実行スクリプト")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--tag", type=str, help="セッションタグ（オプション）")
    parser.add_argument("--gt", type=str, help="Ground Truthトラックファイルのパス（オプション）")
    parser.add_argument("--points", type=str, help="対応点ファイルのパス（オプション）")
    parser.add_argument(
        "--skip-pipeline", action="store_true", help="パイプライン実行をスキップ（既存セッションを使用）"
    )
    parser.add_argument("--session", type=str, help="既存セッションID（--skip-pipeline使用時）")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    try:
        gt_path = Path(args.gt) if args.gt else None
        points_path = Path(args.points) if args.points else None

        return run_full_baseline(
            args.config,
            args.tag,
            gt_path,
            points_path,
            args.skip_pipeline,
            args.session,
        )
    except Exception as e:
        logger.error(f"ベースライン統合実行中にエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
