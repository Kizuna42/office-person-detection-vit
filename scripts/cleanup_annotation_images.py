#!/usr/bin/env python3
"""annotation_imagesフォルダ内の画像をresult_fixed.jsonと照合し、対応していない画像を削除するスクリプト"""

import json
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logging_utils import setup_logging
except ImportError:
    # フォールバック: 基本的なログ設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def setup_logging() -> None:
        """フォールバック用の空のsetup_logging関数"""


logger = logging.getLogger(__name__)


def load_json_filenames(json_path: Path) -> set[str]:
    """JSONファイルからfile_nameのセットを取得

    Args:
        json_path: JSONファイルのパス

    Returns:
        file_nameのセット
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    filenames = set()
    for image in data.get("images", []):
        filename = image.get("file_name")
        if filename:
            filenames.add(filename)

    logger.info(f"JSONファイルから {len(filenames)} 個のfile_nameを取得しました")
    return filenames


def cleanup_annotation_images(json_path: Path, annotation_images_dir: Path, dry_run: bool = False) -> None:
    """annotation_imagesフォルダ内の画像をJSONと照合し、対応していない画像を削除

    Args:
        json_path: result_fixed.jsonのパス
        annotation_images_dir: annotation_imagesフォルダのパス
        dry_run: Trueの場合、実際には削除せずに削除対象を表示するだけ
    """
    # JSONファイルからfile_nameを取得
    json_filenames = load_json_filenames(json_path)

    # annotation_imagesフォルダ内の画像ファイルを取得
    image_files = list(annotation_images_dir.glob("*.jpg"))
    logger.info(f"annotation_imagesフォルダ内に {len(image_files)} 個の画像ファイルが見つかりました")

    # 削除対象の画像を特定
    files_to_delete = []
    for image_file in image_files:
        if image_file.name not in json_filenames:
            files_to_delete.append(image_file)

    if not files_to_delete:
        logger.info("削除対象の画像はありませんでした")
        return

    logger.info(f"削除対象: {len(files_to_delete)} 個の画像ファイル")

    if dry_run:
        logger.info("【DRY RUN】以下のファイルが削除対象です:")
        for file in sorted(files_to_delete):
            logger.info(f"  - {file.name}")
    else:
        # 実際に削除
        deleted_count = 0
        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
                logger.info(f"削除しました: {file.name}")
            except Exception as e:
                logger.error(f"削除に失敗しました: {file.name} - {e}")

        logger.info(f"合計 {deleted_count} 個の画像ファイルを削除しました")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="annotation_imagesフォルダ内の画像をresult_fixed.jsonと照合し、対応していない画像を削除"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("output/labels/result_fixed.json"),
        help="result_fixed.jsonのパス（デフォルト: output/labels/result_fixed.json）",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/annotation_images"),
        help="annotation_imagesフォルダのパス（デフォルト: data/annotation_images）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には削除せず、削除対象を表示するだけ",
    )

    args = parser.parse_args()

    setup_logging()

    if not args.json.exists():
        logger.error(f"JSONファイルが見つかりません: {args.json}")
        return

    if not args.images_dir.exists():
        logger.error(f"annotation_imagesフォルダが見つかりません: {args.images_dir}")
        return

    cleanup_annotation_images(args.json, args.images_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
