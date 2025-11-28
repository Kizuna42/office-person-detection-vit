#!/usr/bin/env python3
"""対応点のカバレッジ分析と精度診断ツール。

カメラ画像のどの領域に対応点が集中しているか、
どの領域で変換精度が低いかを分析します。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file


def analyze_correspondence_coverage(src_pts: np.ndarray, image_size: tuple[int, int]):
    """対応点のカバレッジを分析。"""
    width, height = image_size

    # 画像を4x4グリッドに分割
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = height // grid_rows, width // grid_cols

    print("\n" + "=" * 70)
    print("対応点カバレッジ分析（カメラ画像を4x4グリッドに分割）")
    print("=" * 70)

    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for pt in src_pts:
        col = min(int(pt[0] // cell_w), grid_cols - 1)
        row = min(int(pt[1] // cell_h), grid_rows - 1)
        grid_counts[row, col] += 1

    print("\nグリッド別対応点数:")
    print("     ", end="")
    for c in range(grid_cols):
        print(f"  {c * cell_w:4d}-{(c + 1) * cell_w:4d}", end="")
    print()

    for r in range(grid_rows):
        print(f"{r * cell_h:3d}-{(r + 1) * cell_h:3d}:", end="")
        for c in range(grid_cols):
            count = grid_counts[r, c]
            if count == 0:
                print("     ⚠️ 0   ", end="")
            elif count < 3:
                print(f"     △ {count}   ", end="")
            else:
                print(f"     ✓ {count}   ", end="")
        print()

    # 問題のある領域を特定
    print("\n問題のある領域:")
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid_counts[r, c] < 2:
                y_range = f"{r * cell_h}-{(r + 1) * cell_h}"
                x_range = f"{c * cell_w}-{(c + 1) * cell_w}"
                print(f"  - Y={y_range}, X={x_range}: 対応点 {grid_counts[r, c]}個 → 追加が必要")

    return grid_counts


def analyze_detection_distribution(detections_path: str, image_size: tuple[int, int]):
    """検出結果の分布を分析。"""
    with open(detections_path, encoding="utf-8") as f:
        data = json.load(f)

    foot_points = []
    for frame in data:
        for det in frame.get("detections", []):
            bbox = det.get("bbox", {})
            x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
            foot_x, foot_y = x + w / 2, y + h
            foot_points.append((foot_x, foot_y))

    foot_points = np.array(foot_points)

    width, height = image_size
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = height // grid_rows, width // grid_cols

    print("\n" + "=" * 70)
    print("検出結果の分布（カメラ画像を4x4グリッドに分割）")
    print("=" * 70)

    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for pt in foot_points:
        col = min(int(pt[0] // cell_w), grid_cols - 1)
        row = min(int(pt[1] // cell_h), grid_rows - 1)
        grid_counts[row, col] += 1

    print("\nグリッド別検出数:")
    for r in range(grid_rows):
        print(f"{r * cell_h:3d}-{(r + 1) * cell_h:3d}:", end="")
        for c in range(grid_cols):
            print(f"     {grid_counts[r, c]:3d}  ", end="")
        print()

    return grid_counts, foot_points


def visualize_coverage(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    foot_points: np.ndarray,
    frame_path: str,
    floormap_path: str,
    output_path: str,
):
    """カバレッジを可視化。"""
    frame = cv2.imread(frame_path)
    floormap = cv2.imread(floormap_path)

    if frame is None or floormap is None:
        print("画像を読み込めません")
        return

    # フレーム上に対応点を描画
    for i, pt in enumerate(src_pts):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (int(pt[0]) + 10, int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 検出された足元点を描画（赤）
    for pt in foot_points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

    # グリッドを描画
    height, width = frame.shape[:2]
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = height // grid_rows, width // grid_cols

    for r in range(1, grid_rows):
        cv2.line(frame, (0, r * cell_h), (width, r * cell_h), (255, 255, 0), 1)
    for c in range(1, grid_cols):
        cv2.line(frame, (c * cell_w, 0), (c * cell_w, height), (255, 255, 0), 1)

    # 凡例
    cv2.putText(frame, "Green: Correspondences", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Red: Detection foot points", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # フロアマップ上に対応点を描画
    for pt in dst_pts:
        cv2.circle(floormap, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)

    # リサイズして結合
    target_height = 700
    frame_scale = target_height / frame.shape[0]
    floormap_scale = target_height / floormap.shape[0]

    frame_resized = cv2.resize(frame, None, fx=frame_scale, fy=frame_scale)
    floormap_resized = cv2.resize(floormap, None, fx=floormap_scale, fy=floormap_scale)

    combined = np.hstack([frame_resized, floormap_resized])
    cv2.imwrite(output_path, combined)
    print(f"\n可視化画像を保存: {output_path}")


def suggest_additional_correspondences(
    src_pts: np.ndarray,
    foot_points: np.ndarray,
    image_size: tuple[int, int],
):
    """追加すべき対応点の領域を提案。"""
    width, height = image_size
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = height // grid_rows, width // grid_cols

    # 対応点のグリッド
    corr_grid = np.zeros((grid_rows, grid_cols), dtype=int)
    for pt in src_pts:
        col = min(int(pt[0] // cell_w), grid_cols - 1)
        row = min(int(pt[1] // cell_h), grid_rows - 1)
        corr_grid[row, col] += 1

    # 検出点のグリッド
    det_grid = np.zeros((grid_rows, grid_cols), dtype=int)
    for pt in foot_points:
        col = min(int(pt[0] // cell_w), grid_cols - 1)
        row = min(int(pt[1] // cell_h), grid_rows - 1)
        det_grid[row, col] += 1

    print("\n" + "=" * 70)
    print("追加すべき対応点の推奨領域")
    print("=" * 70)

    priorities = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            corr_count = corr_grid[r, c]
            det_count = det_grid[r, c]

            # 検出があるのに対応点が少ない領域を優先
            if det_count > 0 and corr_count < 3:
                priority = det_count / (corr_count + 1)
                y_start, y_end = r * cell_h, (r + 1) * cell_h
                x_start, x_end = c * cell_w, (c + 1) * cell_w
                priorities.append(
                    {
                        "row": r,
                        "col": c,
                        "y_range": (y_start, y_end),
                        "x_range": (x_start, x_end),
                        "corr_count": corr_count,
                        "det_count": det_count,
                        "priority": priority,
                    }
                )

    priorities.sort(key=lambda x: x["priority"], reverse=True)

    for i, p in enumerate(priorities[:5]):
        print(f"\n優先度 {i + 1}:")
        print(f"  領域: X={p['x_range'][0]}-{p['x_range'][1]}, Y={p['y_range'][0]}-{p['y_range'][1]}")
        print(f"  現在の対応点: {p['corr_count']}個")
        print(f"  検出数: {p['det_count']}件")
        print(f"  → この領域に {max(3 - p['corr_count'], 2)}個以上の対応点を追加してください")

    return priorities


def main():
    config = ConfigManager("config.yaml")

    # 対応点を読み込み
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    data = load_correspondence_file(correspondence_file)
    correspondences = [(pc.src_point, pc.dst_point) for pc in data.point_pairs]
    src_pts = np.array([c[0] for c in correspondences], dtype=np.float64)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float64)

    image_size = (1280, 720)

    # カバレッジ分析
    analyze_correspondence_coverage(src_pts, image_size)

    # 検出結果の分布
    detections_path = "output/latest/phase3_transform/coordinate_transformations.json"
    if Path(detections_path).exists():
        _det_grid, foot_points = analyze_detection_distribution(detections_path, image_size)

        # 追加推奨
        suggest_additional_correspondences(src_pts, foot_points, image_size)

        # 可視化
        frames_dir = Path("output/latest/phase1_extraction/frames")
        if frames_dir.exists():
            frames = list(frames_dir.glob("*.jpg"))
            if frames:
                frame_path = str(sorted(frames)[0])
                floormap_path = config.get("floormap.image_path", "data/floormap.png")
                visualize_coverage(
                    src_pts, dst_pts, foot_points, frame_path, floormap_path, "output/calibration/coverage_analysis.png"
                )


if __name__ == "__main__":
    main()
