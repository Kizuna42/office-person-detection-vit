#!/usr/bin/env python3
"""ベースライン実行スクリプト

実セッション動画に対してパイプラインを実行し、セッションIDを取得・記録します。
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
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str, tag: str | None = None) -> dict:
    """パイプラインを実行し、セッション情報を取得

    Args:
        config_path: 設定ファイルのパス
        tag: セッションタグ（オプション）

    Returns:
        実行結果の辞書（session_id, timestamp, num_frames等を含む）
    """
    logger.info("=" * 80)
    logger.info("ベースライン実行を開始")
    logger.info("=" * 80)

    # 設定ファイルの読み込み
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output"))

    # セッション管理が有効でない場合は警告
    use_session_management = config.get("output.use_session_management", False)
    if not use_session_management:
        logger.warning("セッション管理が無効です。有効化することを推奨します。")
        logger.warning("config.yaml の output.use_session_management を true に設定してください。")

    # main.pyを実行
    logger.info(f"パイプラインを実行中: python main.py --config {config_path}")
    cmd = [sys.executable, "main.py", "--config", config_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("パイプライン実行が完了しました")
    except subprocess.CalledProcessError as e:
        logger.error(f"パイプライン実行に失敗しました: {e}")
        logger.error(f"標準出力: {e.stdout}")
        logger.error(f"標準エラー: {e.stderr}")
        raise

    # セッションIDを取得
    session_id = None
    if use_session_management:
        output_manager = OutputManager(output_dir)
        latest_session = output_manager.get_latest_session()
        if latest_session:
            session_id = latest_session.name
            logger.info(f"セッションIDを取得しました: {session_id}")
        else:
            logger.warning("最新セッションが見つかりませんでした")
    else:
        logger.warning("セッション管理が無効のため、セッションIDを取得できませんでした")

    # 実行結果を記録
    result_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "tag": tag,
        "pipeline_executed": True,
        "session_management_enabled": use_session_management,
    }

    # セッション情報を保存
    if session_id:
        session_dir = output_dir / "sessions" / session_id
        baseline_info_path = session_dir / "baseline_info.json"
        baseline_info_path.parent.mkdir(parents=True, exist_ok=True)

        with baseline_info_path.open("w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"ベースライン情報を保存しました: {baseline_info_path}")

    return result_data


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ベースライン実行スクリプト")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--tag", type=str, help="セッションタグ（オプション）")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    try:
        result = run_pipeline(args.config, args.tag)
        logger.info("=" * 80)
        logger.info("ベースライン実行が完了しました")
        logger.info(f"セッションID: {result.get('session_id', 'N/A')}")
        logger.info("=" * 80)
        return 0
    except Exception as e:
        logger.error(f"ベースライン実行中にエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
