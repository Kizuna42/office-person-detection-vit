#!/usr/bin/env python3
"""高度な座標変換ツール。

レンズ歪み補正と適応的な補間を組み合わせた高精度変換。

機能:
1. レンズ歪み係数の推定と補正
2. TPS (Thin Plate Spline) with smoothing
3. 複数手法のアンサンブル
4. 局所重み付き変換

使用方法:
    python tools/advanced_transform.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file


@dataclass
class DistortionParams:
    """レンズ歪み係数。"""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    fx: float = 1250.0
    fy: float = 1250.0
    cx: float = 640.0
    cy: float = 360.0


def undistort_points(points: np.ndarray, params: DistortionParams) -> np.ndarray:
    """歪み補正を適用。"""
    # 正規化座標に変換
    x = (points[:, 0] - params.cx) / params.fx
    y = (points[:, 1] - params.cy) / params.fy

    # 歪み補正
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3

    # Radial distortion
    radial = 1 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6

    # Tangential distortion
    x_corrected = x * radial + 2 * params.p1 * x * y + params.p2 * (r2 + 2 * x**2)
    y_corrected = y * radial + params.p1 * (r2 + 2 * y**2) + 2 * params.p2 * x * y

    # ピクセル座標に戻す
    x_out = x_corrected * params.fx + params.cx
    y_out = y_corrected * params.fy + params.cy

    return np.column_stack([x_out, y_out])


def estimate_distortion(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> DistortionParams:
    """ホモグラフィからの残差を使って歪み係数を推定。"""
    print("\n" + "=" * 70)
    print("レンズ歪み係数の推定")
    print("=" * 70)

    # 初期パラメータ
    DistortionParams()

    def objective(p):
        params = DistortionParams(k1=p[0], k2=p[1], k3=0.0, p1=p[2], p2=p[3])
        undistorted = undistort_points(src_pts, params)

        # ホモグラフィで変換
        ones = np.ones((len(undistorted), 1))
        pts_h = np.hstack([undistorted, ones])
        transformed = (H @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        errors = np.linalg.norm(transformed - dst_pts, axis=1)
        return np.sqrt(np.mean(errors**2))

    # 最適化
    result = minimize(objective, [0.0, 0.0, 0.0, 0.0], method="Powell", options={"maxiter": 1000})

    best_params = DistortionParams(k1=result.x[0], k2=result.x[1], k3=0.0, p1=result.x[2], p2=result.x[3])

    print("推定された歪み係数:")
    print(f"  k1 = {best_params.k1:.6f}")
    print(f"  k2 = {best_params.k2:.6f}")
    print(f"  p1 = {best_params.p1:.6f}")
    print(f"  p2 = {best_params.p2:.6f}")
    print(f"最適化後RMSE: {result.fun:.2f} px")

    return best_params


class SmoothTPSTransformer:
    """スムージング付きTPS変換。"""

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, smoothing: float = 0.0):
        """初期化。

        Args:
            src_pts: 入力座標 (N, 2)
            dst_pts: 出力座標 (N, 2)
            smoothing: スムージング係数 (0.0=補間, >0=スムージング)
        """
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.smoothing = smoothing

        self.interp_x = RBFInterpolator(src_pts, dst_pts[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
        self.interp_y = RBFInterpolator(src_pts, dst_pts[:, 1], kernel="thin_plate_spline", smoothing=smoothing)

    def transform(self, points: np.ndarray) -> np.ndarray:
        x_out = self.interp_x(points)
        y_out = self.interp_y(points)
        return np.column_stack([x_out, y_out])

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


class LocalWeightedTransformer:
    """局所重み付き変換。

    各点の変換に対して、近傍の対応点を重視した局所的なアフィン変換を適用。
    """

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, k: int = 6):
        """初期化。

        Args:
            src_pts: 入力座標 (N, 2)
            dst_pts: 出力座標 (N, 2)
            k: 近傍点の数
        """
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.k = min(k, len(src_pts))

    def transform(self, points: np.ndarray) -> np.ndarray:
        results = np.zeros_like(points)

        for i, pt in enumerate(points):
            # 距離を計算
            distances = np.linalg.norm(self.src_pts - pt, axis=1)

            # k近傍を取得
            indices = np.argsort(distances)[: self.k]

            # 重み（距離の逆数）
            d = distances[indices]
            d = np.maximum(d, 1e-6)  # ゼロ除算防止
            weights = 1.0 / d
            weights /= weights.sum()

            # 近傍点で局所アフィン変換を計算
            src_local = self.src_pts[indices]
            dst_local = self.dst_pts[indices]

            # 重み付き最小二乗でアフィン変換を推定
            W = np.diag(weights)
            src_h = np.column_stack([src_local, np.ones(self.k)])

            try:
                A = np.linalg.lstsq(W @ src_h, W @ dst_local, rcond=None)[0]
                pt_h = np.array([pt[0], pt[1], 1.0])
                results[i] = pt_h @ A
            except Exception:
                # フォールバック: 重み付き平均
                shift = dst_local - src_local
                avg_shift = (shift.T @ weights).T
                results[i] = pt + avg_shift

        return results

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


class EnsembleTransformer:
    """複数の変換手法のアンサンブル。"""

    def __init__(self, transformers: list, weights: list[float] | None = None):
        """初期化。

        Args:
            transformers: 変換器のリスト
            weights: 各変換器の重み
        """
        self.transformers = transformers
        if weights is None:
            weights = [1.0 / len(transformers)] * len(transformers)
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def transform(self, points: np.ndarray) -> np.ndarray:
        results = np.zeros((len(points), 2))

        for transformer, weight in zip(self.transformers, self.weights, strict=False):
            results += weight * transformer.transform(points)

        return results


class HybridTransformer:
    """ハイブリッド変換器。

    歪み補正 → TPS（スムージング付き）→ 局所補正
    """

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        distortion_params: DistortionParams | None = None,
        smoothing: float = 1.0,
    ):
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.distortion_params = distortion_params
        self.smoothing = smoothing

        # 歪み補正を適用した座標を使用
        src_undistorted = undistort_points(src_pts, distortion_params) if distortion_params else src_pts.copy()

        # TPS変換器を構築
        self.tps = SmoothTPSTransformer(src_undistorted, dst_pts, smoothing=smoothing)

        # 残差を学習する局所変換器
        tps_result = self.tps.transform(src_undistorted)
        residuals = dst_pts - tps_result

        # 残差が大きい点を補正する局所変換器
        self.local_corrector = RBFInterpolator(
            src_undistorted, residuals, kernel="thin_plate_spline", smoothing=smoothing * 2
        )

        self.src_undistorted = src_undistorted

    def transform(self, points: np.ndarray) -> np.ndarray:
        # 歪み補正
        points_undist = undistort_points(points, self.distortion_params) if self.distortion_params else points.copy()

        # TPS変換
        result = self.tps.transform(points_undist)

        # 局所補正
        correction = self.local_corrector(points_undist)
        result += correction

        return result

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


def cross_validate(src_pts: np.ndarray, dst_pts: np.ndarray, transformer_class, **kwargs):
    """Leave-One-Out交差検証。"""
    n = len(src_pts)
    errors = []

    for i in range(n):
        train_src = np.delete(src_pts, i, axis=0)
        train_dst = np.delete(dst_pts, i, axis=0)
        test_src = src_pts[i : i + 1]
        test_dst = dst_pts[i]

        transformer = transformer_class(train_src, train_dst, **kwargs)
        pred = transformer.transform(test_src)[0]
        error = np.linalg.norm(pred - test_dst)
        errors.append(error)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def save_hybrid_model(model: HybridTransformer, path: str):
    """ハイブリッドモデルを保存。"""
    data = {
        "src_pts": model.src_pts,
        "dst_pts": model.dst_pts,
        "distortion_params": model.distortion_params,
        "smoothing": model.smoothing,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_hybrid_model(path: str) -> HybridTransformer:
    """ハイブリッドモデルを読み込み。"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    return HybridTransformer(
        src_pts=data["src_pts"],
        dst_pts=data["dst_pts"],
        distortion_params=data["distortion_params"],
        smoothing=data["smoothing"],
    )


def main():
    config = ConfigManager("config.yaml")

    # 対応点を読み込み
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    data = load_correspondence_file(correspondence_file)
    correspondences = [(pc.src_point, pc.dst_point) for pc in data.point_pairs]
    src_pts = np.array([c[0] for c in correspondences], dtype=np.float64)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float64)

    print("=" * 70)
    print("高度な座標変換の比較")
    print("=" * 70)
    print(f"対応点数: {len(correspondences)}")

    # ホモグラフィを計算（ベースライン）
    H, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), 0)

    # 歪み係数を推定
    distortion = estimate_distortion(src_pts, dst_pts, H)

    results = []

    # 1. TPS (スムージングなし)
    print("\n" + "=" * 70)
    print("[1] TPS (スムージングなし)")
    print("=" * 70)
    tps_0 = SmoothTPSTransformer(src_pts, dst_pts, smoothing=0.0)
    rmse_train, _ = tps_0.compute_rmse()
    rmse_cv, _ = cross_validate(src_pts, dst_pts, SmoothTPSTransformer, smoothing=0.0)
    print(f"訓練RMSE: {rmse_train:.2f} px")
    print(f"交差検証RMSE: {rmse_cv:.2f} px")
    results.append(("TPS-0", rmse_cv, tps_0))

    # 2. TPS (スムージングあり)
    for smoothing in [0.5, 1.0, 2.0, 5.0]:
        print(f"\n[2-{smoothing}] TPS (smoothing={smoothing})")
        tps = SmoothTPSTransformer(src_pts, dst_pts, smoothing=smoothing)
        rmse_train, _ = tps.compute_rmse()
        rmse_cv, _ = cross_validate(src_pts, dst_pts, SmoothTPSTransformer, smoothing=smoothing)
        print(f"訓練RMSE: {rmse_train:.2f} px, 交差検証RMSE: {rmse_cv:.2f} px")
        results.append((f"TPS-{smoothing}", rmse_cv, tps))

    # 3. 局所重み付き変換
    print("\n" + "=" * 70)
    print("[3] 局所重み付き変換")
    print("=" * 70)
    for k in [4, 6, 8]:
        lwt = LocalWeightedTransformer(src_pts, dst_pts, k=k)
        rmse_train, _ = lwt.compute_rmse()
        rmse_cv, _ = cross_validate(src_pts, dst_pts, LocalWeightedTransformer, k=k)
        print(f"k={k}: 訓練RMSE={rmse_train:.2f} px, 交差検証RMSE={rmse_cv:.2f} px")
        results.append((f"LWT-{k}", rmse_cv, lwt))

    # 4. ハイブリッド変換
    print("\n" + "=" * 70)
    print("[4] ハイブリッド変換（歪み補正 + TPS + 局所補正）")
    print("=" * 70)
    for smoothing in [0.5, 1.0, 2.0]:
        hybrid = HybridTransformer(src_pts, dst_pts, distortion, smoothing=smoothing)
        rmse_train, errors = hybrid.compute_rmse()
        print(f"smoothing={smoothing}: 訓練RMSE={rmse_train:.2f} px, 最大誤差={errors.max():.2f} px")
        results.append((f"Hybrid-{smoothing}", rmse_train, hybrid))

    # 最良の結果を選択
    print("\n" + "=" * 70)
    print("結果サマリー")
    print("=" * 70)
    results.sort(key=lambda x: x[1])

    for name, rmse, _ in results[:5]:
        print(f"{name:15s}: RMSE = {rmse:.2f} px")

    best_name, best_rmse, best_transformer = results[0]
    print(f"\n最良の方法: {best_name} (RMSE: {best_rmse:.2f} px)")

    # モデルを保存
    output_dir = Path("output/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "advanced_transformer.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "transformer": best_transformer,
                "name": best_name,
                "rmse": best_rmse,
            },
            f,
        )
    print(f"\nモデルを保存: {model_path}")

    # 推奨事項
    print("\n" + "=" * 70)
    print("精度向上のための推奨事項")
    print("=" * 70)
    print("""
1. 対応点の追加（最優先）:
   - Y=540-720 の領域（カメラ手前）に 5点以上追加
   - X=0-320 の領域（画像左端）に 3点以上追加
   - 床の角、机の脚など明確な特徴点を使用

2. 対応点収集ツールの起動:
   python tools/correspondence_collector.py

3. 追加後に再学習:
   python tools/advanced_transform.py

4. パイプラインに適用:
   config.yamlのtransform.model_pathを更新
""")


if __name__ == "__main__":
    main()
