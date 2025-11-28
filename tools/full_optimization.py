#!/usr/bin/env python3
"""全対応点を使用した完全最適化ツール。

50点すべての対応点を使用して、最適なホモグラフィ行列を推定します。
外れ値はないという前提で、全点を使用した最適化を行います。

使用方法:
    python tools/full_optimization.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_correspondences(config: ConfigManager) -> tuple[np.ndarray, np.ndarray]:
    """対応点を読み込む。"""
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    if not correspondence_file or not Path(correspondence_file).exists():
        raise FileNotFoundError(f"対応点ファイルが見つかりません: {correspondence_file}")

    data = load_correspondence_file(correspondence_file)
    correspondences = [(pc.src_point, pc.dst_point) for pc in data.point_pairs]

    src_pts = np.array([c[0] for c in correspondences], dtype=np.float64)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float64)

    logger.info(f"対応点数: {len(correspondences)}")
    return src_pts, dst_pts


def compute_homography_rmse(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, np.ndarray]:
    """ホモグラフィによるRMSEを計算。"""
    ones = np.ones((len(src_pts), 1))
    src_h = np.hstack([src_pts, ones])
    transformed = (H @ src_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    errors = np.linalg.norm(transformed - dst_pts, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def optimize_homography_least_squares(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """最小二乗法でホモグラフィを最適化（全点使用）。"""
    # 初期ホモグラフィを計算（全点使用、RANSACなし）
    H_init, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), 0)

    if H_init is None:
        raise ValueError("初期ホモグラフィの計算に失敗")

    # H[2,2] = 1 として8パラメータを最適化
    def objective(params):
        H = np.array(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]]
        )
        rmse, _ = compute_homography_rmse(H, src_pts, dst_pts)
        return rmse

    x0 = [
        H_init[0, 0],
        H_init[0, 1],
        H_init[0, 2],
        H_init[1, 0],
        H_init[1, 1],
        H_init[1, 2],
        H_init[2, 0],
        H_init[2, 1],
    ]

    result = minimize(objective, x0, method="Powell", options={"maxiter": 10000, "disp": True})

    H_opt = np.array(
        [
            [result.x[0], result.x[1], result.x[2]],
            [result.x[3], result.x[4], result.x[5]],
            [result.x[6], result.x[7], 1.0],
        ]
    )

    return H_opt


def optimize_homography_global(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Differential Evolutionでホモグラフィを最適化。"""
    # 初期ホモグラフィを計算して境界を設定
    H_init, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), 0)

    if H_init is None:
        raise ValueError("初期ホモグラフィの計算に失敗")

    def objective(params):
        H = np.array(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]]
        )
        rmse, _ = compute_homography_rmse(H, src_pts, dst_pts)
        return rmse

    # 初期値から±大きめの範囲で探索
    bounds = []
    for i in range(8):
        if i < 6:  # H[0:2, :] の要素
            val = H_init.flatten()[i]
            bounds.append((val - abs(val) * 2 - 100, val + abs(val) * 2 + 100))
        else:  # H[2, 0:2] の要素（小さい値）
            val = H_init.flatten()[i]
            bounds.append((val - 0.01, val + 0.01))

    print(f"初期ホモグラフィ:\n{H_init}")
    print(f"初期RMSE: {objective(H_init.flatten()[:8]):.2f} px")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=1000,
        tol=0.001,
        seed=42,
        workers=1,
        polish=True,
        disp=True,
    )

    H_opt = np.array(
        [
            [result.x[0], result.x[1], result.x[2]],
            [result.x[3], result.x[4], result.x[5]],
            [result.x[6], result.x[7], 1.0],
        ]
    )

    return H_opt


def analyze_correspondences(src_pts: np.ndarray, dst_pts: np.ndarray):
    """対応点を詳細に分析。"""
    print("\n=== 対応点の詳細分析 ===")
    print(f"対応点数: {len(src_pts)}")

    print("\nカメラ座標 (src):")
    print(f"  X: min={src_pts[:, 0].min():.1f}, max={src_pts[:, 0].max():.1f}, mean={src_pts[:, 0].mean():.1f}")
    print(f"  Y: min={src_pts[:, 1].min():.1f}, max={src_pts[:, 1].max():.1f}, mean={src_pts[:, 1].mean():.1f}")

    print("\nフロアマップ座標 (dst):")
    print(f"  X: min={dst_pts[:, 0].min():.1f}, max={dst_pts[:, 0].max():.1f}, mean={dst_pts[:, 0].mean():.1f}")
    print(f"  Y: min={dst_pts[:, 1].min():.1f}, max={dst_pts[:, 1].max():.1f}, mean={dst_pts[:, 1].mean():.1f}")

    # 変換ベクトルの分析
    print("\n変位ベクトル分析:")
    dx = dst_pts[:, 0] - src_pts[:, 0]
    dy = dst_pts[:, 1] - src_pts[:, 1]
    print(f"  dX: min={dx.min():.1f}, max={dx.max():.1f}, mean={dx.mean():.1f}, std={dx.std():.1f}")
    print(f"  dY: min={dy.min():.1f}, max={dy.max():.1f}, mean={dy.mean():.1f}, std={dy.std():.1f}")


def visualize_transformation(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
    output_path: Path,
    floormap_path: str,
    frame_path: str | None = None,
):
    """変換結果を可視化。"""
    # フロアマップを読み込み
    floormap = cv2.imread(floormap_path)
    if floormap is None:
        logger.warning(f"フロアマップが読み込めません: {floormap_path}")
        return

    # 変換を適用
    ones = np.ones((len(src_pts), 1))
    src_h = np.hstack([src_pts, ones])
    transformed = (H @ src_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    # 描画
    for i, (expected, actual) in enumerate(zip(dst_pts, transformed, strict=False)):
        exp_pt = tuple(map(int, expected))
        act_pt = tuple(map(int, actual))

        # 期待位置（緑）
        cv2.circle(floormap, exp_pt, 8, (0, 255, 0), 2)
        # 実際の位置（赤）
        cv2.circle(floormap, act_pt, 6, (0, 0, 255), -1)
        # 誤差ベクトル
        cv2.arrowedLine(floormap, exp_pt, act_pt, (255, 0, 255), 2, tipLength=0.3)
        # インデックス
        cv2.putText(floormap, str(i), (exp_pt[0] + 10, exp_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 統計を計算
    errors = np.linalg.norm(transformed - dst_pts, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    # 統計情報を描画
    cv2.putText(floormap, f"RMSE: {rmse:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(floormap, f"Max Error: {errors.max():.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(floormap, f"Points: {len(src_pts)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(str(output_path), floormap)
    logger.info(f"可視化画像を保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="全対応点を使用した完全最適化")
    parser.add_argument("--config", default="config.yaml", help="設定ファイル")
    parser.add_argument("--output", default="output/calibration/full_optimization_result.json", help="出力ファイル")
    args = parser.parse_args()

    config = ConfigManager(args.config)

    print("=" * 70)
    print("全対応点を使用した完全最適化")
    print("=" * 70)

    # 対応点を読み込み
    src_pts, dst_pts = load_correspondences(config)

    # 詳細分析
    analyze_correspondences(src_pts, dst_pts)

    # === 方法1: 標準的なホモグラフィ（全点使用） ===
    print("\n" + "=" * 70)
    print("[方法1] 標準的なホモグラフィ（全点使用、RANSACなし）")
    print("=" * 70)

    H_standard, _ = cv2.findHomography(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32),
        0,  # RANSACなし
    )

    if H_standard is not None:
        rmse_standard, errors_standard = compute_homography_rmse(H_standard, src_pts, dst_pts)
        print(f"RMSE: {rmse_standard:.2f} px")
        print(f"最大誤差: {errors_standard.max():.2f} px")
        print(f"平均誤差: {errors_standard.mean():.2f} px")
        print(f"ホモグラフィ行列:\n{H_standard}")
    else:
        print("ホモグラフィの計算に失敗")
        rmse_standard = float("inf")

    # === 方法2: Powell法による最適化 ===
    print("\n" + "=" * 70)
    print("[方法2] Powell法による最適化")
    print("=" * 70)

    try:
        H_powell = optimize_homography_least_squares(src_pts, dst_pts)
        rmse_powell, errors_powell = compute_homography_rmse(H_powell, src_pts, dst_pts)
        print(f"\nRMSE: {rmse_powell:.2f} px")
        print(f"最大誤差: {errors_powell.max():.2f} px")
        print(f"ホモグラフィ行列:\n{H_powell}")
    except Exception as e:
        print(f"エラー: {e}")
        rmse_powell = float("inf")
        H_powell = None

    # === 方法3: Differential Evolution ===
    print("\n" + "=" * 70)
    print("[方法3] Differential Evolution（グローバル最適化）")
    print("=" * 70)

    try:
        H_de = optimize_homography_global(src_pts, dst_pts)
        rmse_de, errors_de = compute_homography_rmse(H_de, src_pts, dst_pts)
        print(f"\nRMSE: {rmse_de:.2f} px")
        print(f"最大誤差: {errors_de.max():.2f} px")
        print(f"ホモグラフィ行列:\n{H_de}")
    except Exception as e:
        print(f"エラー: {e}")
        rmse_de = float("inf")
        H_de = None

    # === 最良の結果を選択 ===
    print("\n" + "=" * 70)
    print("最良の結果")
    print("=" * 70)

    results = [
        ("標準ホモグラフィ", rmse_standard, H_standard),
        ("Powell最適化", rmse_powell, H_powell),
        ("Differential Evolution", rmse_de, H_de),
    ]

    best_name, best_rmse, best_H = min(results, key=lambda x: x[1] if x[2] is not None else float("inf"))

    if best_H is not None:
        print(f"\n最良の方法: {best_name}")
        print(f"RMSE: {best_rmse:.2f} px")

        _rmse, errors = compute_homography_rmse(best_H, src_pts, dst_pts)
        print("\n各点の誤差:")
        for i, err in enumerate(errors):
            status = "OK" if err < 30 else "要確認"
            print(
                f"  [{i:2d}] src=({src_pts[i, 0]:7.1f}, {src_pts[i, 1]:6.1f}) -> "
                f"dst=({dst_pts[i, 0]:7.1f}, {dst_pts[i, 1]:7.1f}) -> error={err:6.1f} px ({status})"
            )

        print("\nconfig.yamlに追加するホモグラフィ:")
        print("homography:")
        print("  matrix:")
        for row in best_H:
            print(f"    - [{row[0]:.10f}, {row[1]:.10f}, {row[2]:.10f}]")

        # 結果を保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            "method": best_name,
            "rmse": best_rmse,
            "max_error": float(errors.max()),
            "mean_error": float(errors.mean()),
            "num_points": len(src_pts),
            "homography": best_H.tolist(),
            "per_point_errors": [{"index": i, "error": float(e)} for i, e in enumerate(errors)],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n結果を保存: {output_path}")

        # 可視化
        floormap_path = config.get("floormap.image_path", "data/floormap.png")
        vis_path = output_path.parent / "full_optimization_visualization.png"
        visualize_transformation(src_pts, dst_pts, best_H, vis_path, floormap_path)


if __name__ == "__main__":
    main()
