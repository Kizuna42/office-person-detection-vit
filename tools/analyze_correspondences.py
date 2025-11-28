#!/usr/bin/env python3
"""対応点データの品質分析と外れ値除去ツール。

対応点の空間分布、外れ値、一貫性を分析します。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_correspondences(correspondences: list[tuple[tuple[float, float], tuple[float, float]]]):
    """対応点の統計を分析。"""
    src_pts = np.array([c[0] for c in correspondences])
    dst_pts = np.array([c[1] for c in correspondences])

    print("\n=== 対応点統計 ===")
    print(f"総数: {len(correspondences)}")
    print("\nカメラ座標 (src):")
    print(f"  X: {src_pts[:, 0].min():.1f} - {src_pts[:, 0].max():.1f} (mean={src_pts[:, 0].mean():.1f})")
    print(f"  Y: {src_pts[:, 1].min():.1f} - {src_pts[:, 1].max():.1f} (mean={src_pts[:, 1].mean():.1f})")
    print("\nフロアマップ座標 (dst):")
    print(f"  X: {dst_pts[:, 0].min():.1f} - {dst_pts[:, 0].max():.1f} (mean={dst_pts[:, 0].mean():.1f})")
    print(f"  Y: {dst_pts[:, 1].min():.1f} - {dst_pts[:, 1].max():.1f} (mean={dst_pts[:, 1].mean():.1f})")

    return src_pts, dst_pts


def detect_outliers_ransac(
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]], threshold: float = 10.0
):
    """RANSACで外れ値を検出。"""
    src_pts = np.array([c[0] for c in correspondences], dtype=np.float32)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float32)

    # ホモグラフィを推定
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

    if H is None:
        print("ホモグラフィの推定に失敗")
        return None, None, None

    inlier_mask = mask.ravel().astype(bool)
    outlier_mask = ~inlier_mask

    # 変換誤差を計算
    src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    transformed = (H @ src_pts_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    errors = np.linalg.norm(transformed - dst_pts, axis=1)

    print(f"\n=== RANSAC分析 (閾値={threshold}px) ===")
    print(f"インライア: {np.sum(inlier_mask)}")
    print(f"外れ値: {np.sum(outlier_mask)}")
    print(
        f"\nインライア誤差: min={errors[inlier_mask].min():.2f}, max={errors[inlier_mask].max():.2f}, mean={errors[inlier_mask].mean():.2f}"
    )
    if np.any(outlier_mask):
        print(
            f"外れ値誤差: min={errors[outlier_mask].min():.2f}, max={errors[outlier_mask].max():.2f}, mean={errors[outlier_mask].mean():.2f}"
        )

    print("\n外れ値のインデックス:")
    for i, (is_outlier, err) in enumerate(zip(outlier_mask, errors, strict=False)):
        if is_outlier:
            print(f"  [{i}] src={correspondences[i][0]}, dst={correspondences[i][1]}, error={err:.1f}px")

    return H, inlier_mask, errors


def find_best_homography(correspondences: list[tuple[tuple[float, float], tuple[float, float]]]):
    """最良のホモグラフィを見つける（閾値を変えながら）。"""
    src_pts = np.array([c[0] for c in correspondences], dtype=np.float32)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float32)

    print("\n=== 最適閾値探索 ===")

    best_rmse = float("inf")
    best_threshold = 0
    best_H = None
    best_mask = None

    for threshold in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        if H is None:
            continue

        inlier_mask = mask.ravel().astype(bool)
        if np.sum(inlier_mask) < 4:
            continue

        # インライアのみでRMSEを計算
        src_inliers = src_pts[inlier_mask]
        dst_inliers = dst_pts[inlier_mask]

        src_h = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])
        transformed = (H @ src_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        errors = np.linalg.norm(transformed - dst_inliers, axis=1)
        rmse = np.sqrt(np.mean(errors**2))

        print(f"  閾値={threshold:3d}: インライア={np.sum(inlier_mask):2d}, RMSE={rmse:.2f}px")

        if rmse < best_rmse:
            best_rmse = rmse
            best_threshold = threshold
            best_H = H
            best_mask = inlier_mask

    print(f"\n最良: 閾値={best_threshold}, RMSE={best_rmse:.2f}px, インライア={np.sum(best_mask)}")

    return best_H, best_mask, best_threshold


def filter_correspondences(
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
    mask: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """マスクに基づいて対応点をフィルタリング。"""
    return [c for c, m in zip(correspondences, mask, strict=False) if m]


def save_filtered_correspondences(
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
    output_path: Path,
):
    """フィルタリング済み対応点を保存。"""
    data = {
        "camera_id": "cam01",
        "description": f"フィルタリング済み対応点（{len(correspondences)}点）",
        "metadata": {
            "image_size": {"width": 1280, "height": 720},
            "floormap_size": {"width": 1878, "height": 1369},
            "num_line_segment_correspondences": 0,
            "num_point_correspondences": len(correspondences),
        },
        "line_segment_correspondences": [],
        "point_correspondences": [{"src_point": list(c[0]), "dst_point": list(c[1])} for c in correspondences],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"フィルタリング済み対応点を保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="対応点品質分析")
    parser.add_argument("--config", default="config.yaml", help="設定ファイル")
    parser.add_argument(
        "--filter-output",
        default="output/calibration/correspondence_points_filtered.json",
        help="フィルタリング済み出力",
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    if not correspondence_file or not Path(correspondence_file).exists():
        logger.error(f"対応点ファイルが見つかりません: {correspondence_file}")
        sys.exit(1)

    data = load_correspondence_file(correspondence_file)
    correspondences = [(pc.src_point, pc.dst_point) for pc in data.point_pairs]

    print("=" * 70)
    print("対応点品質分析")
    print("=" * 70)

    # 統計分析
    _src_pts, _dst_pts = analyze_correspondences(correspondences)

    # 最適閾値探索
    best_H, best_mask, _best_threshold = find_best_homography(correspondences)

    if best_H is not None and best_mask is not None:
        # フィルタリング
        filtered = filter_correspondences(correspondences, best_mask)
        print(f"\nフィルタリング後: {len(filtered)}/{len(correspondences)} 点")

        # 保存
        output_path = Path(args.filter_output)
        save_filtered_correspondences(filtered, output_path)

        # フィルタリング済みでホモグラフィを再計算
        src_filtered = np.array([c[0] for c in filtered], dtype=np.float32)
        dst_filtered = np.array([c[1] for c in filtered], dtype=np.float32)

        H_refined, _ = cv2.findHomography(src_filtered, dst_filtered, 0)  # 全点使用

        src_h = np.hstack([src_filtered, np.ones((len(src_filtered), 1))])
        transformed = (H_refined @ src_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        errors = np.linalg.norm(transformed - dst_filtered, axis=1)
        rmse = np.sqrt(np.mean(errors**2))

        print("\n=== フィルタリング後のホモグラフィ精度 ===")
        print(f"RMSE: {rmse:.2f} px")
        print(f"最大誤差: {errors.max():.2f} px")
        print(f"平均誤差: {errors.mean():.2f} px")

        print("\nホモグラフィ行列:")
        print(H_refined)

        # ホモグラフィをconfig用にフォーマット
        print("\nconfig.yamlに追加するホモグラフィ:")
        print("homography:")
        print("  matrix:")
        for row in H_refined:
            print(f"    - [{row[0]:.10f}, {row[1]:.10f}, {row[2]:.10f}]")


if __name__ == "__main__":
    main()
