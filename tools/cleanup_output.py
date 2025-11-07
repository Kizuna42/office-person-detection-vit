#!/usr/bin/env python
"""Output cleanup and archive management tool."""

import argparse
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager  # noqa: E402
from src.utils import OutputManager  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.output_manager import format_file_size  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="出力ファイルのクリーンアップとアーカイブ管理")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="出力ディレクトリ（設定ファイルから取得する場合は省略）",
    )
    parser.add_argument(
        "--archive-days",
        type=int,
        default=30,
        help="アーカイブ対象の日数（デフォルト: 30日）",
    )
    parser.add_argument(
        "--delete-archive-days",
        type=int,
        default=90,
        help="アーカイブ削除対象の日数（デフォルト: 90日）",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="古いセッションをアーカイブ",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="古いアーカイブを削除",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="特定のセッションID（アーカイブまたは削除）",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="削除モード（--sessionと組み合わせて使用）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="セッション一覧を表示",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="統計情報を表示",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 出力ディレクトリの決定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config = ConfigManager(args.config)
        output_dir = Path(config.get("output.directory", "output"))

    # OutputManagerを初期化
    manager = OutputManager(output_dir)

    # セッション一覧表示
    if args.list:
        sessions = manager.find_sessions()
        if not sessions:
            logger.info("セッションが見つかりませんでした")
            return 0

        logger.info(f"セッション一覧 ({len(sessions)}件):")
        for session_dir in sessions:
            size = manager.get_session_size(session_dir)
            logger.info(f"  {session_dir.name} - {format_file_size(size)}")

        return 0

    # 統計情報表示
    if args.stats:
        sessions = manager.find_sessions()
        total_size = sum(manager.get_session_size(s) for s in sessions)

        logger.info("統計情報:")
        logger.info(f"  セッション数: {len(sessions)}")
        logger.info(f"  合計サイズ: {format_file_size(total_size)}")

        if sessions:
            latest = sessions[0]
            latest_size = manager.get_session_size(latest)
            logger.info(f"  最新セッション: {latest.name} ({format_file_size(latest_size)})")

        return 0

    # 特定セッションの操作
    if args.session:
        session_dir = manager.sessions_dir / args.session
        if not session_dir.exists():
            logger.error(f"セッションが見つかりません: {args.session}")
            return 1

        if args.delete:
            import shutil

            shutil.rmtree(session_dir)
            logger.info(f"セッションを削除しました: {args.session}")
        else:
            # アーカイブに移動
            archive_path = manager.archive_dir / args.session
            if archive_path.exists():
                logger.error(f"アーカイブが既に存在します: {args.session}")
                return 1

            import shutil

            shutil.move(str(session_dir), str(archive_path))
            logger.info(f"セッションをアーカイブしました: {args.session}")

        return 0

    # アーカイブ処理
    if args.archive:
        archived_count = manager.archive_old_sessions(args.archive_days)
        logger.info(f"{archived_count}件のセッションをアーカイブしました")
        return 0

    # アーカイブ削除
    if args.delete_archives:
        deleted_count = manager.cleanup_old_archives(args.delete_archive_days)
        logger.info(f"{deleted_count}件のアーカイブを削除しました")
        return 0

    # デフォルト: 統計情報を表示
    logger.info("使用方法:")
    logger.info("  --list: セッション一覧を表示")
    logger.info("  --stats: 統計情報を表示")
    logger.info("  --archive: 古いセッションをアーカイブ")
    logger.info("  --delete-archives: 古いアーカイブを削除")
    logger.info("  --session <ID> [--delete]: 特定セッションの操作")

    return 0


if __name__ == "__main__":
    sys.exit(main())
