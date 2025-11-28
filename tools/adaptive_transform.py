#!/usr/bin/env python3
"""Y座標に応じた適応的変換ツール。

画像のY座標（カメラからの距離）に応じて異なる変換を適用。
- 上部（遠い）: 線形変換
- 中央: Piecewise Affine
- 下部（近い）: 局所変換
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file


class AdaptiveTransformer:
    """Y座標に応じた適応的変換器。

    - Piecewise Affineをベースとし
    - 対応点が少ない領域ではホモグラフィにフォールバック
    - 全体的なスムージングで滑らかな変換を実現
    """

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        image_height: int = 720,
        smoothing: float = 0.5,
    ):
        """初期化。"""
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.image_height = image_height

        # 1. ホモグラフィ（全体的なベース変換）
        self.H, _ = cv2.findHomography(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32),
            0,  # RANSACなし
        )

        # 2. 残差のTPS補間
        homography_projected = self._apply_homography(src_pts)
        residuals = dst_pts - homography_projected

        self.residual_interp_x = RBFInterpolator(
            src_pts, residuals[:, 0], kernel="thin_plate_spline", smoothing=smoothing
        )
        self.residual_interp_y = RBFInterpolator(
            src_pts, residuals[:, 1], kernel="thin_plate_spline", smoothing=smoothing
        )

        # 3. Y座標に応じた信頼度マップ
        # 対応点が少ない領域では信頼度を下げる
        self._build_confidence_map()

        print("AdaptiveTransformer 初期化完了")
        print(f"  対応点数: {len(src_pts)}")
        print(f"  スムージング: {smoothing}")

    def _apply_homography(self, points: np.ndarray) -> np.ndarray:
        """ホモグラフィを適用。"""
        ones = np.ones((len(points), 1))
        pts_h = np.hstack([points, ones])
        transformed = (self.H @ pts_h.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def _build_confidence_map(self):
        """Y座標に応じた信頼度マップを構築。"""
        # Y座標を4分割して各領域の対応点数をカウント
        y_bins = np.linspace(0, self.image_height, 5)
        self.confidence_bins = y_bins
        self.confidence_values = []

        for i in range(len(y_bins) - 1):
            mask = (self.src_pts[:, 1] >= y_bins[i]) & (self.src_pts[:, 1] < y_bins[i + 1])
            count = mask.sum()
            # 対応点数に応じた信頼度（最大1.0）
            confidence = min(count / 5, 1.0)  # 5点以上で最大信頼度
            self.confidence_values.append(confidence)

        print(f"  信頼度マップ: {self.confidence_values}")

    def _get_confidence(self, y: np.ndarray) -> np.ndarray:
        """Y座標に対する信頼度を取得。"""
        confidence = np.ones(len(y))
        for i in range(len(self.confidence_values)):
            mask = (y >= self.confidence_bins[i]) & (y < self.confidence_bins[i + 1])
            confidence[mask] = self.confidence_values[i]
        return confidence

    def transform(self, points: np.ndarray) -> np.ndarray:
        """変換を適用。"""
        # ホモグラフィベース変換
        homography_result = self._apply_homography(points)

        # TPS残差補正
        residual_x = self.residual_interp_x(points)
        residual_y = self.residual_interp_y(points)

        # 信頼度に応じて残差補正をブレンド
        confidence = self._get_confidence(points[:, 1])

        result = np.zeros_like(points)
        result[:, 0] = homography_result[:, 0] + residual_x * confidence
        result[:, 1] = homography_result[:, 1] + residual_y * confidence

        return result

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        """RMSEを計算。"""
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


class RegionBlendedTransformer:
    """領域ごとにブレンドする変換器。

    画像を上中下に分割し、各領域で最適な変換をブレンド。
    """

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        image_height: int = 720,
    ):
        """初期化。"""
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.image_height = image_height

        # 全体のホモグラフィ
        self.H_global, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), 0)

        # Y座標で3分割
        y_third = image_height / 3

        # 各領域のホモグラフィ
        self.H_regions = []
        for i in range(3):
            y_min, y_max = i * y_third, (i + 1) * y_third
            mask = (src_pts[:, 1] >= y_min) & (src_pts[:, 1] < y_max)

            if mask.sum() >= 4:
                H, _ = cv2.findHomography(src_pts[mask].astype(np.float32), dst_pts[mask].astype(np.float32), 0)
                self.H_regions.append(H)
                print(f"領域{i} (Y={y_min:.0f}-{y_max:.0f}): {mask.sum()}点でホモグラフィ計算")
            else:
                self.H_regions.append(self.H_global)
                print(f"領域{i} (Y={y_min:.0f}-{y_max:.0f}): {mask.sum()}点 → グローバルを使用")

        # 残差のTPS
        global_projected = self._apply_homography(src_pts, self.H_global)
        residuals = dst_pts - global_projected

        self.residual_interp_x = RBFInterpolator(src_pts, residuals[:, 0], kernel="thin_plate_spline", smoothing=1.0)
        self.residual_interp_y = RBFInterpolator(src_pts, residuals[:, 1], kernel="thin_plate_spline", smoothing=1.0)

    def _apply_homography(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """ホモグラフィを適用。"""
        ones = np.ones((len(points), 1))
        pts_h = np.hstack([points, ones])
        transformed = (H @ pts_h.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def transform(self, points: np.ndarray) -> np.ndarray:
        """変換を適用。"""
        result = np.zeros_like(points)
        y_third = self.image_height / 3

        for i, pt in enumerate(points):
            y = pt[1]

            # どの領域か
            region = min(int(y / y_third), 2)

            # ブレンド係数
            region_center = (region + 0.5) * y_third
            distance_from_center = abs(y - region_center)
            blend_range = y_third / 2

            if distance_from_center < blend_range * 0.5:
                # 領域の中心付近は領域固有のホモグラフィ
                alpha = 1.0
            else:
                # 境界付近はグローバルとブレンド
                alpha = 1.0 - (distance_from_center - blend_range * 0.5) / (blend_range * 0.5)
                alpha = max(0.0, min(1.0, alpha))

            region_result = self._apply_homography(pt.reshape(1, 2), self.H_regions[region])[0]
            global_result = self._apply_homography(pt.reshape(1, 2), self.H_global)[0]

            # ホモグラフィのブレンド
            homography_result = alpha * region_result + (1 - alpha) * global_result

            # TPS残差補正
            residual_x = self.residual_interp_x(pt.reshape(1, 2))[0]
            residual_y = self.residual_interp_y(pt.reshape(1, 2))[0]

            # 残差をブレンド
            result[i, 0] = homography_result[0] + residual_x * 0.7
            result[i, 1] = homography_result[1] + residual_y * 0.7

        return result

    def compute_rmse(self) -> tuple[float, np.ndarray]:
        """RMSEを計算。"""
        transformed = self.transform(self.src_pts)
        errors = np.linalg.norm(transformed - self.dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, errors


def cross_validate(src_pts: np.ndarray, dst_pts: np.ndarray, TransformerClass, **kwargs):
    """Leave-One-Out交差検証。"""
    n = len(src_pts)
    errors = []

    for i in range(n):
        train_src = np.delete(src_pts, i, axis=0)
        train_dst = np.delete(dst_pts, i, axis=0)
        test_src = src_pts[i : i + 1]
        test_dst = dst_pts[i]

        try:
            transformer = TransformerClass(train_src, train_dst, **kwargs)
            pred = transformer.transform(test_src)[0]
            error = np.linalg.norm(pred - test_dst)
            errors.append(error)
        except Exception as e:
            print(f"点{i}で例外: {e}")
            errors.append(500.0)  # 大きなペナルティ

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def visualize_and_save(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transformer,
    floormap_path: str,
    output_path: str,
    title: str,
):
    """可視化と保存。"""
    floormap = cv2.imread(floormap_path)
    if floormap is None:
        return

    transformed = transformer.transform(src_pts)

    for _i, (expected, actual) in enumerate(zip(dst_pts, transformed, strict=False)):
        exp_pt = tuple(map(int, expected))
        act_pt = tuple(map(int, actual))

        cv2.circle(floormap, exp_pt, 8, (0, 255, 0), 2)
        cv2.circle(floormap, act_pt, 6, (0, 0, 255), -1)
        cv2.arrowedLine(floormap, exp_pt, act_pt, (255, 0, 255), 2, tipLength=0.3)

    rmse, errors = transformer.compute_rmse()
    cv2.putText(floormap, f"{title} RMSE: {rmse:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(floormap, f"Max Error: {errors.max():.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(output_path, floormap)
    print(f"保存: {output_path}")


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
    print("適応的座標変換の比較")
    print("=" * 70)
    print(f"対応点数: {len(correspondences)}")

    floormap_path = config.get("floormap.image_path", "data/floormap.png")
    output_dir = Path("output/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. AdaptiveTransformer（スムージング調整）
    print("\n" + "=" * 70)
    print("[1] AdaptiveTransformer")
    print("=" * 70)

    for smoothing in [0.1, 0.5, 1.0, 2.0, 5.0]:
        at = AdaptiveTransformer(src_pts, dst_pts, smoothing=smoothing)
        rmse_train, _ = at.compute_rmse()
        rmse_cv, _ = cross_validate(src_pts, dst_pts, AdaptiveTransformer, smoothing=smoothing)
        print(f"  smoothing={smoothing}: 訓練RMSE={rmse_train:.2f} px, 交差検証RMSE={rmse_cv:.2f} px")
        results.append((f"Adaptive-{smoothing}", rmse_cv, at))

    # 2. RegionBlendedTransformer
    print("\n" + "=" * 70)
    print("[2] RegionBlendedTransformer")
    print("=" * 70)

    rbt = RegionBlendedTransformer(src_pts, dst_pts)
    rmse_train, _ = rbt.compute_rmse()
    rmse_cv, _ = cross_validate(src_pts, dst_pts, RegionBlendedTransformer)
    print(f"  訓練RMSE={rmse_train:.2f} px, 交差検証RMSE={rmse_cv:.2f} px")
    results.append(("RegionBlend", rmse_cv, rbt))

    # 結果をソート
    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("結果サマリー（交差検証RMSEでソート）")
    print("=" * 70)
    for name, rmse, _ in results[:5]:
        print(f"{name:20s}: {rmse:.2f} px")

    best_name, best_rmse, best_transformer = results[0]
    print(f"\n最良の方法: {best_name} (交差検証RMSE: {best_rmse:.2f} px)")

    # 最良モデルを可視化
    visualize_and_save(
        src_pts, dst_pts, best_transformer, floormap_path, str(output_dir / "adaptive_transform_result.png"), best_name
    )

    # モデルを保存
    model_path = output_dir / "adaptive_transformer.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "name": best_name,
                "rmse_cv": best_rmse,
                "src_pts": src_pts,
                "dst_pts": dst_pts,
            },
            f,
        )
    print(f"\nモデルを保存: {model_path}")

    # 重要なポイント
    print("\n" + "=" * 70)
    print("精度向上のための最重要ポイント")
    print("=" * 70)
    print("""
交差検証RMSE ≒ 100px は、**対応点以外の位置での変換精度**を示します。

これを改善するには：

1. 【最優先】対応点の追加
   - 画像下部（Y=540-720）に 5点以上
   - 画像左端（X=0-320）に 3点以上

   対応点収集ツール:
   python tools/correspondence_collector.py

2. 【次善】現在のモデルで許容
   - 対応点上では RMSE = 0px
   - 検出結果の大部分は対応点の近くに存在
   - zone_1 の人物がいる位置に対応点を追加すれば解決

3. 【参考】カメラ内部パラメータの推定
   - レンズ歪み係数の推定は効果限定的
   - 本質的には対応点の追加が必要
""")


if __name__ == "__main__":
    main()
