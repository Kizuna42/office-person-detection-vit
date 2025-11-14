#!/usr/bin/env python3
"""Ground Truthトラック手動編集ツール

自動生成されたGround Truthトラックを手動で編集するためのインタラクティブツールです。
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.utils.logging_utils import setup_logging
from tools.gt_editor.editor import GTracksEditor

logger = logging.getLogger(__name__)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Ground Truthトラック手動編集ツール")
    parser.add_argument(
        "--tracks",
        type=str,
        default="data/gt_tracks_auto.json",
        help="Ground Truthトラックファイルのパス（デフォルト: data/gt_tracks_auto.json）",
    )
    parser.add_argument(
        "--floormap",
        type=str,
        help="フロアマップ画像のパス（設定ファイルから取得する場合は省略）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルパス (default: config.yaml)",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="セッションディレクトリのパス（カメラ画像表示用、オプション）",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    # ファイルパスの確認
    tracks_path = Path(args.tracks)
    config_path = Path(args.config)

    # トラックファイルが存在しない場合は新規作成
    if not tracks_path.exists():
        logger.info(f"トラックファイルが存在しません。新規作成します: {tracks_path}")
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tracks_path, "w", encoding="utf-8") as f:
            json.dump({"tracks": [], "metadata": {}}, f, indent=2, ensure_ascii=False)

    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    # フロアマップパスの取得
    if args.floormap:
        floormap_path = Path(args.floormap)
    else:
        config = ConfigManager(str(config_path))
        floormap_path = Path(config.get("floormap.image_path"))

    if not floormap_path.exists():
        logger.error(f"フロアマップ画像が見つかりません: {floormap_path}")
        return 1

    try:
        # セッションディレクトリの取得
        session_dir = None
        if args.session:
            session_dir = Path(args.session)
            if not session_dir.exists():
                logger.warning(f"セッションディレクトリが見つかりません: {session_dir}")
                session_dir = None
        else:
            # デフォルトでoutput/latestを試す
            default_session = Path("output/latest")
            if default_session.exists():
                session_dir = default_session
                logger.info(f"デフォルトセッションディレクトリを使用: {session_dir}")

        # エディタを起動
        editor = GTracksEditor(tracks_path, floormap_path, config_path, session_dir)
        editor.run()

        logger.info("編集ツールを終了しました")
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
