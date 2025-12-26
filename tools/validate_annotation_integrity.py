#!/usr/bin/env python3
"""
アノテーションデータ整合性検証スクリプト。

画像ファイルとアノテーションデータの整合性をチェックし、レポートを出力する。

使用例:
    python tools/validate_annotation_integrity.py \\
        --images data/annotation_images \\
        --annotation output/labels/result_fixed.json \\
        --format coco
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Literal

from lxml import etree
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """検証問題を表すデータクラス。"""

    level: Literal["ERROR", "WARNING", "INFO"]
    code: str
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """検証レポートを表すデータクラス。"""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, code: str, message: str, details: list[str] | None = None) -> None:
        """エラーを追加。"""
        self.issues.append(ValidationIssue("ERROR", code, message, details or []))

    def add_warning(self, code: str, message: str, details: list[str] | None = None) -> None:
        """警告を追加。"""
        self.issues.append(ValidationIssue("WARNING", code, message, details or []))

    def add_info(self, code: str, message: str, details: list[str] | None = None) -> None:
        """情報を追加。"""
        self.issues.append(ValidationIssue("INFO", code, message, details or []))

    @property
    def has_errors(self) -> bool:
        """エラーがあるかどうか。"""
        return any(i.level == "ERROR" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """警告があるかどうか。"""
        return any(i.level == "WARNING" for i in self.issues)

    def print_summary(self) -> None:
        """サマリーを出力。"""
        errors = [i for i in self.issues if i.level == "ERROR"]
        warnings = [i for i in self.issues if i.level == "WARNING"]
        infos = [i for i in self.issues if i.level == "INFO"]

        print("\n" + "=" * 60)
        print("Validation Report")
        print("=" * 60)

        if not self.issues:
            print("✅ No issues found. All checks passed.")
        else:
            print(f"❌ Errors: {len(errors)}")
            print(f"⚠️  Warnings: {len(warnings)}")
            print(f"ℹ️  Info: {len(infos)}")

            for issue in errors:
                print(f"\n[ERROR] {issue.code}: {issue.message}")
                for detail in issue.details[:5]:
                    print(f"  - {detail}")
                if len(issue.details) > 5:
                    print(f"  ... and {len(issue.details) - 5} more")

            for issue in warnings:
                print(f"\n[WARNING] {issue.code}: {issue.message}")
                for detail in issue.details[:5]:
                    print(f"  - {detail}")

        print("=" * 60 + "\n")


def extract_frame_number(file_name: str) -> int | None:
    """ファイル名からフレーム番号を抽出。"""
    match = re.search(r"frame_(\d+)", file_name)
    if match:
        return int(match.group(1))
    return None


def validate_coco(images_dir: Path, annotation_path: Path) -> ValidationReport:
    """COCO形式アノテーションを検証。"""
    report = ValidationReport()

    # 画像ファイル一覧
    image_files = sorted(images_dir.glob("*.jpg"))
    image_names = {f.name for f in image_files}

    if not image_files:
        report.add_error("NO_IMAGES", f"画像ファイルが見つかりません: {images_dir}")
        return report

    report.add_info("IMAGE_COUNT", f"{len(image_files)} 画像ファイルを検出")

    # アノテーション読込
    with annotation_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    report.add_info("ANNOTATION_COUNT", f"{len(annotations)} アノテーション, {len(images)} 画像エントリ")

    # チェック1: 画像存在確認
    missing_images = []
    for img in images:
        file_name = img.get("file_name", "")
        if file_name not in image_names:
            missing_images.append(file_name)

    if missing_images:
        report.add_error(
            "MISSING_IMAGES",
            f"{len(missing_images)} 個の参照画像が見つかりません",
            missing_images[:10],
        )

    # チェック2: 画像サイズの欠損
    null_dimensions = []
    for img in images:
        if img.get("width") is None or img.get("height") is None:
            null_dimensions.append(img.get("file_name", "unknown"))

    if null_dimensions:
        report.add_warning(
            "NULL_DIMENSIONS",
            f"{len(null_dimensions)} 個の画像でwidth/heightがnull",
            null_dimensions[:10],
        )

    # チェック3: BBox座標範囲
    invalid_bboxes = []
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            invalid_bboxes.append(f"ann_id={ann.get('id')}: bbox長が不正")
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            invalid_bboxes.append(f"ann_id={ann.get('id')}: 幅または高さが0以下")

    if invalid_bboxes:
        report.add_error("INVALID_BBOX", f"{len(invalid_bboxes)} 個の不正なBBox", invalid_bboxes[:10])

    # チェック4: 空フレーム検出
    annotated_image_ids = {ann.get("image_id") for ann in annotations}
    empty_frames = [img.get("file_name") for img in images if img.get("id") not in annotated_image_ids]

    if empty_frames:
        report.add_warning(
            "EMPTY_FRAMES",
            f"{len(empty_frames)} 個のフレームにアノテーションなし",
            empty_frames[:10],
        )

    return report


def validate_cvat(images_dir: Path, annotation_path: Path) -> ValidationReport:
    """CVAT XML形式アノテーションを検証。"""
    report = ValidationReport()

    tree = etree.parse(str(annotation_path))
    root = tree.getroot()

    tracks = root.findall(".//track")
    report.add_info("TRACK_COUNT", f"{len(tracks)} トラックを検出")

    # Track ID一意性
    track_ids = [int(t.get("id", "0")) for t in tracks]
    if len(track_ids) != len(set(track_ids)):
        report.add_error("DUPLICATE_TRACK_ID", "重複するTrack IDが存在します")

    # BBox検証
    total_boxes = 0
    invalid_boxes = []

    for track in tracks:
        track_id = track.get("id")
        for box in track.findall("box"):
            total_boxes += 1
            xtl = float(box.get("xtl", "0"))
            ytl = float(box.get("ytl", "0"))
            xbr = float(box.get("xbr", "0"))
            ybr = float(box.get("ybr", "0"))

            if xbr <= xtl or ybr <= ytl:
                invalid_boxes.append(f"track={track_id}, frame={box.get('frame')}")

    report.add_info("BOX_COUNT", f"{total_boxes} ボックスを検出")

    if invalid_boxes:
        report.add_error("INVALID_BBOX", f"{len(invalid_boxes)} 個の不正なBBox", invalid_boxes[:10])

    return report


def validate_mot(images_dir: Path, annotation_path: Path) -> ValidationReport:
    """MOT形式CSVアノテーションを検証。"""
    report = ValidationReport()

    # ヘッダーなし形式で読込
    try:
        df = pd.read_csv(
            annotation_path,
            header=None,
            names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
        )
    except Exception as e:
        report.add_error("PARSE_ERROR", f"CSVパースエラー: {e}")
        return report

    report.add_info("ANNOTATION_COUNT", f"{len(df)} アノテーション")
    report.add_info("TRACK_COUNT", f"{df['id'].nunique()} ユニークトラック")
    report.add_info("FRAME_RANGE", f"フレーム {df['frame'].min()} - {df['frame'].max()}")

    # BBox検証
    invalid_size = df[(df["bb_width"] <= 0) | (df["bb_height"] <= 0)]
    if not invalid_size.empty:
        report.add_error(
            "INVALID_BBOX",
            f"{len(invalid_size)} 個の不正なBBox（幅/高さ <= 0）",
        )

    return report


def validate_integrity(
    images_dir: Path,
    annotation_path: Path,
    format_type: Literal["coco", "cvat", "mot"],
) -> ValidationReport:
    """アノテーション整合性検証のメインエントリ。"""
    if format_type == "coco":
        return validate_coco(images_dir, annotation_path)
    if format_type == "cvat":
        return validate_cvat(images_dir, annotation_path)
    if format_type == "mot":
        return validate_mot(images_dir, annotation_path)
    report = ValidationReport()
    report.add_error("UNKNOWN_FORMAT", f"未対応の形式: {format_type}")
    return report


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="アノテーションデータ整合性検証",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="画像ディレクトリパス",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        type=Path,
        help="アノテーションファイルパス",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["coco", "cvat", "mot"],
        help="アノテーション形式",
    )
    return parser.parse_args()


def main() -> int:
    """エントリーポイント。"""
    args = parse_args()

    if not args.images.is_dir():
        logger.error("画像ディレクトリが見つかりません: %s", args.images)
        return 1

    if not args.annotation.exists():
        logger.error("アノテーションファイルが見つかりません: %s", args.annotation)
        return 1

    report = validate_integrity(args.images, args.annotation, args.format)
    report.print_summary()

    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
