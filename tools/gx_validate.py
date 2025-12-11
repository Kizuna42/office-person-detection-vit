"""
Great Expectations 互換の軽量バリデーションスクリプト。

目的:
 - `output/summary.json` と `data/gt_tracks_auto.json` を簡易チェック。
 - GE 未導入でも標準ライブラリで同等チェックを実行。

使い方:
  python tools/gx_validate.py \
    --summary output/sessions/.../summary.json \
    --gt data/gt_tracks_auto.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import great_expectations as ge  # type: ignore

    _HAS_GX = True
except Exception:  # pragma: no cover - 任意依存
    _HAS_GX = False


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_summary(summary_path: Path) -> list[str]:
    """summary.json の必須キーと範囲チェック。"""
    data = _load_json(summary_path)
    errors: list[str] = []
    stats = data.get("statistics", {})
    perf = data.get("performance", {})

    required = ["frames_extracted", "total_detections", "frames_processed"]
    for k in required:
        if k not in stats:
            errors.append(f"statistics.{k} が存在しません")
        elif stats[k] < 0:
            errors.append(f"statistics.{k} が負の値です: {stats[k]}")

    if "floormaps_generated" in stats and stats["floormaps_generated"] < 0:
        errors.append("floormaps_generated が負の値です")

    if perf and "phases" in perf:
        for phase, value in perf["phases"].items():
            if value is None:
                errors.append(f"performance.{phase} が null です")

    return errors


def validate_gt_tracks(gt_path: Path) -> list[str]:
    """GT JSON の null/空チェック。"""
    data = _load_json(gt_path)
    errors: list[str] = []
    tracks: list[dict[str, Any]] = data.get("tracks") or data.get("data") or []
    if len(tracks) == 0:
        errors.append("tracks が空です")
    for idx, trk in enumerate(tracks[:50]):  # すべてチェックすると重いので先頭のみ
        if trk.get("track_id") is None:
            errors.append(f"track[{idx}] track_id がありません")
        if not trk.get("trajectory"):
            errors.append(f"track[{idx}] trajectory が空です")
    return errors


def run_gx_if_available(summary_path: Path, gt_path: Path) -> None:
    """Great Expectations を使った場合の最小例。"""
    if not _HAS_GX:
        print("[INFO] Great Expectations が未インストールのため、標準チェックのみ実行しました。")
        return

    ge.get_context(mode="ephemeral")
    ds = ge.dataset.PandasDataset.from_json(summary_path)  # type: ignore[attr-defined]
    ds.expect_column_values_to_not_be_null("statistics.frames_extracted")
    ds.expect_column_values_to_be_between("statistics.frames_extracted", 0, None)
    print("[GX] summary.json の簡易チェックを実行しました（詳細なスイートは未定義）")

    gt_ds = ge.dataset.PandasDataset.from_json(gt_path)  # type: ignore[attr-defined]
    gt_ds.expect_column_values_to_not_be_null("tracks")
    print("[GX] gt_tracks_auto.json の簡易チェックを実行しました")


def main() -> None:
    parser = argparse.ArgumentParser(description="Great Expectations style validator")
    parser.add_argument("--summary", type=Path, required=True, help="summary.json のパス")
    parser.add_argument("--gt", type=Path, required=True, help="gt_tracks_auto.json のパス")
    args = parser.parse_args()

    summary_errors = validate_summary(args.summary)
    gt_errors = validate_gt_tracks(args.gt)

    if summary_errors:
        print("[NG] summary.json:")
        for e in summary_errors:
            print(f"  - {e}")
    else:
        print("[OK] summary.json の基本チェックを通過")

    if gt_errors:
        print("[NG] gt_tracks_auto.json:")
        for e in gt_errors:
            print(f"  - {e}")
    else:
        print("[OK] gt_tracks_auto.json の基本チェックを通過")

    run_gx_if_available(args.summary, args.gt)


if __name__ == "__main__":
    main()
