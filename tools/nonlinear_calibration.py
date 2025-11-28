#!/usr/bin/env python3
"""非線形変換による高精度キャリブレーション。

ホモグラフィ（射影変換）では対応できない歪みを考慮した変換。

手法:
1. 薄板スプライン (Thin Plate Spline)
2. 多項式変換 (Polynomial Transform)
3. Piecewise Affine変換

使用方法:
    python tools/nonlinear_calibration.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

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


class RBFTransformer:
    """RBF (Radial Basis Function) による非線形変換。"""

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, kernel: str = "thin_plate_spline"):
        """初期化。

        Args:
            src_pts: 入力座標 (N, 2)
            dst_pts: 出力座標 (N, 2)
            kernel: RBFカーネル
        """
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.kernel = kernel

        # X座標とY座標を別々に補間
        self.interp_x = RBFInterpolator(src_pts, dst_pts[:, 0], kernel=kernel, smoothing=0.0)
        self.interp_y = RBFInterpolator(src_pts, dst_pts[:, 1], kernel=kernel, smoothing=0.0)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """座標変換。

        Args:
            points: 入力座標 (N, 2)

        Returns:
            変換後の座標 (N, 2)
        """
        x_out = self.interp_x(points)
        y_out = self.interp_y(points)
        return np.column_stack([x_out, y_out])

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        """訓練データでのRMSEを計算。"""
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


class PolynomialTransformer:
    """多項式変換。"""

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, degree: int = 3):
        """初期化。

        Args:
            src_pts: 入力座標 (N, 2)
            dst_pts: 出力座標 (N, 2)
            degree: 多項式の次数
        """
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.degree = degree

        # 特徴量行列を構築
        self.features = self._build_features(src_pts, degree)

        # 係数を最小二乗法で推定
        self.coef_x, _, _, _ = np.linalg.lstsq(self.features, dst_pts[:, 0], rcond=None)
        self.coef_y, _, _, _ = np.linalg.lstsq(self.features, dst_pts[:, 1], rcond=None)

    def _build_features(self, pts: np.ndarray, degree: int) -> np.ndarray:
        """多項式特徴量を構築。"""
        x, y = pts[:, 0], pts[:, 1]
        features = [np.ones(len(pts))]

        for d in range(1, degree + 1):
            for i in range(d + 1):
                features.append((x ** (d - i)) * (y**i))

        return np.column_stack(features)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """座標変換。"""
        features = self._build_features(points, self.degree)
        x_out = features @ self.coef_x
        y_out = features @ self.coef_y
        return np.column_stack([x_out, y_out])

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        """訓練データでのRMSEを計算。"""
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


class PiecewiseAffineTransformer:
    """Piecewise Affine 変換。"""

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray):
        """初期化。"""
        self.src_pts = src_pts
        self.dst_pts = dst_pts

        # Delaunay三角形分割
        self.tri = Delaunay(src_pts)

        # 各三角形のアフィン変換を計算
        self.transforms = []
        for simplex in self.tri.simplices:
            src_tri = src_pts[simplex]
            dst_tri = dst_pts[simplex]

            # アフィン変換行列を計算
            A = self._compute_affine(src_tri, dst_tri)
            self.transforms.append(A)

    def _compute_affine(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """3点からアフィン変換行列を計算。"""
        src_h = np.column_stack([src, np.ones(3)])
        A = np.linalg.lstsq(src_h, dst, rcond=None)[0]
        return A.T

    def transform(self, points: np.ndarray) -> np.ndarray:
        """座標変換。"""
        # 各点がどの三角形に属するか
        simplex_indices = self.tri.find_simplex(points)

        result = np.zeros_like(points)
        for i, (pt, idx) in enumerate(zip(points, simplex_indices, strict=False)):
            if idx == -1:
                # 三角形外の点は最近傍の三角形を使用
                distances = np.linalg.norm(self.tri.points - pt, axis=1)
                nearest = np.argmin(distances)
                # 最近傍点を含む三角形を見つける
                for j, simplex in enumerate(self.tri.simplices):
                    if nearest in simplex:
                        idx = j
                        break

            if idx == -1:
                idx = 0  # フォールバック

            A = self.transforms[idx]
            pt_h = np.array([pt[0], pt[1], 1])
            result[i] = A @ pt_h

        return result

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        """訓練データでのRMSEを計算。"""
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


def cross_validate(src_pts: np.ndarray, dst_pts: np.ndarray, transformer_class, **kwargs):
    """Leave-One-Out交差検証でRMSEを計算。"""
    n = len(src_pts)
    errors = []

    for i in range(n):
        # i番目の点を除外
        train_src = np.delete(src_pts, i, axis=0)
        train_dst = np.delete(dst_pts, i, axis=0)
        test_src = src_pts[i : i + 1]
        test_dst = dst_pts[i]

        # 学習
        transformer = transformer_class(train_src, train_dst, **kwargs)

        # 予測
        pred = transformer.transform(test_src)[0]
        error = np.linalg.norm(pred - test_dst)
        errors.append(error)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def visualize_transformation(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transformer,
    output_path: Path,
    floormap_path: str,
    title: str = "",
):
    """変換結果を可視化。"""
    floormap = cv2.imread(floormap_path)
    if floormap is None:
        return

    transformed = transformer.transform(src_pts)

    for i, (expected, actual) in enumerate(zip(dst_pts, transformed, strict=False)):
        exp_pt = tuple(map(int, expected))
        act_pt = tuple(map(int, actual))

        cv2.circle(floormap, exp_pt, 8, (0, 255, 0), 2)
        cv2.circle(floormap, act_pt, 6, (0, 0, 255), -1)
        cv2.arrowedLine(floormap, exp_pt, act_pt, (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(floormap, str(i), (exp_pt[0] + 10, exp_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    rmse, errors = transformer.compute_rmse()
    cv2.putText(floormap, f"{title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(floormap, f"RMSE: {rmse:.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(floormap, f"Max: {errors.max():.2f} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), floormap)
    logger.info(f"保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="非線形変換による高精度キャリブレーション")
    parser.add_argument("--config", default="config.yaml", help="設定ファイル")
    parser.add_argument("--output-dir", default="output/calibration", help="出力ディレクトリ")
    args = parser.parse_args()

    config = ConfigManager(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("非線形変換による高精度キャリブレーション")
    print("=" * 70)

    src_pts, dst_pts = load_correspondences(config)
    floormap_path = config.get("floormap.image_path", "data/floormap.png")

    results = []

    # === 方法1: 薄板スプライン (Thin Plate Spline) ===
    print("\n" + "=" * 70)
    print("[方法1] 薄板スプライン (Thin Plate Spline)")
    print("=" * 70)

    tps = RBFTransformer(src_pts, dst_pts, kernel="thin_plate_spline")
    rmse_tps, errors_tps = tps.compute_rmse()
    print(f"訓練RMSE: {rmse_tps:.4f} px (ほぼ0に近いはず)")
    print(f"最大誤差: {errors_tps.max():.4f} px")

    # 交差検証
    rmse_cv_tps, errors_cv_tps = cross_validate(src_pts, dst_pts, RBFTransformer, kernel="thin_plate_spline")
    print(f"交差検証RMSE: {rmse_cv_tps:.2f} px")
    print(f"交差検証最大誤差: {errors_cv_tps.max():.2f} px")

    results.append(("TPS", rmse_cv_tps, tps))
    visualize_transformation(src_pts, dst_pts, tps, output_dir / "tps_visualization.png", floormap_path, "TPS")

    # === 方法2: 多項式変換 ===
    for degree in [2, 3, 4, 5]:
        print("\n" + "=" * 70)
        print(f"[方法2-{degree}] 多項式変換 (degree={degree})")
        print("=" * 70)

        poly = PolynomialTransformer(src_pts, dst_pts, degree=degree)
        rmse_poly, errors_poly = poly.compute_rmse()
        print(f"訓練RMSE: {rmse_poly:.2f} px")
        print(f"最大誤差: {errors_poly.max():.2f} px")

        # 交差検証
        rmse_cv_poly, errors_cv_poly = cross_validate(src_pts, dst_pts, PolynomialTransformer, degree=degree)
        print(f"交差検証RMSE: {rmse_cv_poly:.2f} px")
        print(f"交差検証最大誤差: {errors_cv_poly.max():.2f} px")

        results.append((f"Poly-{degree}", rmse_cv_poly, poly))

        if degree == 3:  # 最も一般的な次数で可視化
            visualize_transformation(
                src_pts, dst_pts, poly, output_dir / "poly3_visualization.png", floormap_path, "Polynomial-3"
            )

    # === 方法3: Piecewise Affine ===
    print("\n" + "=" * 70)
    print("[方法3] Piecewise Affine 変換")
    print("=" * 70)

    pwa = PiecewiseAffineTransformer(src_pts, dst_pts)
    rmse_pwa, errors_pwa = pwa.compute_rmse()
    print(f"訓練RMSE: {rmse_pwa:.4f} px (ほぼ0に近いはず)")
    print(f"最大誤差: {errors_pwa.max():.4f} px")

    results.append(("PWA", rmse_pwa, pwa))  # PWAは補間なので交差検証不要
    visualize_transformation(
        src_pts, dst_pts, pwa, output_dir / "pwa_visualization.png", floormap_path, "Piecewise Affine"
    )

    # === 結果サマリー ===
    print("\n" + "=" * 70)
    print("結果サマリー")
    print("=" * 70)
    print(f"{'方法':<15} {'交差検証RMSE':<15}")
    print("-" * 30)
    for name, rmse, _ in sorted(results, key=lambda x: x[1]):
        print(f"{name:<15} {rmse:<15.2f}")

    # 最良の結果
    best_name, best_rmse, best_transformer = min(results, key=lambda x: x[1])
    print(f"\n最良の方法: {best_name} (RMSE: {best_rmse:.2f} px)")

    # 最良のモデルを保存
    model_path = output_dir / f"best_transformer_{best_name.lower().replace('-', '_')}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_transformer, f)
    print(f"\nモデルを保存: {model_path}")

    # 結果をJSON出力
    result_data = {
        "best_method": best_name,
        "best_rmse": best_rmse,
        "results": [{"method": n, "rmse": r} for n, r, _ in results],
        "model_path": str(model_path),
    }

    with open(output_dir / "nonlinear_calibration_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
