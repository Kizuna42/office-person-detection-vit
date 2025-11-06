#!/usr/bin/env python
"""既存の古い出力ファイルとディレクトリを整理するツール"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def find_old_output_files(output_dir: Path) -> dict:
    """古い出力ファイルとディレクトリを検索

    Args:
        output_dir: 出力ディレクトリ

    Returns:
        検出されたファイルとディレクトリの辞書
    """
    old_files = {
        "root_files": [],
        "empty_dirs": [],
        "old_dirs": [],
    }

    # ルートディレクトリの古いファイル
    old_patterns = [
        "coordinate_transformations.json",
        "detection_statistics.json",
        "zone_counts.csv",
        "timestamp_extraction_*.csv",
    ]

    for pattern in old_patterns:
        if "*" in pattern:
            # ワイルドカードパターン
            for file_path in output_dir.glob(pattern):
                if file_path.is_file():
                    old_files["root_files"].append(file_path)
        else:
            file_path = output_dir / pattern
            if file_path.is_file():
                old_files["root_files"].append(file_path)

    # 空のディレクトリ（セッション管理が有効な場合、不要）
    old_dirs = ["detections", "floormaps", "graphs"]
    for dir_name in old_dirs:
        dir_path = output_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # ディレクトリが空か、manual/などのサブディレクトリのみかチェック
            has_content = False
            for item in dir_path.rglob("*"):
                if item.is_file():
                    has_content = True
                    break

            if not has_content:
                old_files["empty_dirs"].append(dir_path)
            else:
                old_files["old_dirs"].append(dir_path)

    return old_files


def cleanup_old_outputs(output_dir: Path, dry_run: bool = True) -> None:
    """古い出力ファイルとディレクトリを整理

    Args:
        output_dir: 出力ディレクトリ
        dry_run: Trueの場合は削除せずに表示のみ
    """
    old_files = find_old_output_files(output_dir)

    logger.info("=" * 80)
    logger.info("古い出力ファイルとディレクトリの検出結果")
    logger.info("=" * 80)

    # ルートファイル
    if old_files["root_files"]:
        logger.info(f"\nルートディレクトリの古いファイル ({len(old_files['root_files'])}件):")
        for file_path in old_files["root_files"]:
            size = file_path.stat().st_size if file_path.exists() else 0
            logger.info(f"  - {file_path.name} ({size:,} bytes)")
            if not dry_run:
                file_path.unlink()
                logger.info(f"    削除しました: {file_path}")

    # 空のディレクトリ
    if old_files["empty_dirs"]:
        logger.info(f"\n空のディレクトリ ({len(old_files['empty_dirs'])}件):")
        for dir_path in old_files["empty_dirs"]:
            logger.info(f"  - {dir_path.name}/")
            if not dry_run:
                try:
                    dir_path.rmdir()
                    logger.info(f"    削除しました: {dir_path}")
                except OSError as e:
                    logger.warning(f"    削除に失敗しました: {e}")

    # 内容がある古いディレクトリ
    if old_files["old_dirs"]:
        logger.info(f"\n内容がある古いディレクトリ ({len(old_files['old_dirs'])}件):")
        logger.info("  注意: これらのディレクトリにはファイルが含まれています")
        for dir_path in old_files["old_dirs"]:
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            logger.info(f"  - {dir_path.name}/ ({file_count}ファイル)")
            logger.info("    手動で確認してから削除してください（必要に応じてバックアップ）")

    if not old_files["root_files"] and not old_files["empty_dirs"]:
        logger.info("\n整理が必要なファイルやディレクトリは見つかりませんでした")

    if dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUNモード: 実際の削除は実行されませんでした")
        logger.info("実際に削除する場合は --execute オプションを指定してください")
        logger.info("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="既存の古い出力ファイルとディレクトリを整理")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="出力ディレクトリ（デフォルト: output）",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="実際に削除を実行（デフォルト: DRY RUNモード）",
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

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"出力ディレクトリが見つかりません: {output_dir}")
        return 1

    # 整理実行
    cleanup_old_outputs(output_dir, dry_run=not args.execute)

    return 0


if __name__ == "__main__":
    sys.exit(main())
