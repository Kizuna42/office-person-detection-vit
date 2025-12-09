"""PWA変換のテストスクリプト

PWA変換とホモグラフィ変換の精度を比較します。
"""

import json
from pathlib import Path
import sys

import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import TransformEvaluator
from src.transform import (
    FloorMapConfig,
    HomographyTransformer,
    PiecewiseAffineTransformer,
    ThinPlateSplineTransformer,
)


def load_config():
    """設定を読み込み"""
    import yaml

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("PWA変換テスト")
    print("=" * 60)

    # 設定読み込み
    config = load_config()

    # フロアマップ設定
    fm_config = FloorMapConfig(
        width_px=config["floormap"]["image_width"],
        height_px=config["floormap"]["image_height"],
        scale_x_mm_per_px=config["floormap"]["image_x_mm_per_pixel"],
        scale_y_mm_per_px=config["floormap"]["image_y_mm_per_pixel"],
    )

    # 対応点ファイル
    correspondence_file = PROJECT_ROOT / config["calibration"]["correspondence_file"]
    print(f"\n対応点ファイル: {correspondence_file}")

    # 対応点を読み込み
    with open(correspondence_file, encoding="utf-8") as f:
        data = json.load(f)
    points = data["point_correspondences"]
    print(f"対応点数: {len(points)}")

    # 評価器を作成
    evaluator = TransformEvaluator(correspondence_file=correspondence_file)

    # ==== 1. ホモグラフィ変換 ====
    print("\n" + "-" * 40)
    print("1. ホモグラフィ変換")
    print("-" * 40)

    H = np.array(config["homography"]["matrix"], dtype=np.float64)
    homography = HomographyTransformer(H, fm_config)

    h_metrics = evaluator.evaluate(homography)
    print(f"  RMSE: {h_metrics.rmse:.2f} px")
    print(f"  MAE: {h_metrics.mae:.2f} px")
    print(f"  Max Error: {h_metrics.max_error:.2f} px")
    print(f"  90th Percentile: {h_metrics.percentile_90:.2f} px")

    # ==== 2. PWA変換 ====
    print("\n" + "-" * 40)
    print("2. PWA変換 (Piecewise Affine)")
    print("-" * 40)

    pwa = PiecewiseAffineTransformer.from_correspondence_file(
        correspondence_file,
        fm_config,
    )

    print(f"  三角形数: {len(pwa.delaunay.simplices)}")

    pwa_metrics = evaluator.evaluate(pwa)
    print(f"  RMSE: {pwa_metrics.rmse:.2f} px")
    print(f"  MAE: {pwa_metrics.mae:.2f} px")
    print(f"  Max Error: {pwa_metrics.max_error:.2f} px")
    print(f"  90th Percentile: {pwa_metrics.percentile_90:.2f} px")

    # 訓練誤差（補間精度の確認）
    training_error = pwa.evaluate_training_error()
    print(f"  訓練RMSE: {training_error['rmse']:.4f} px (補間精度)")

    # ==== 3. TPS変換 ====
    print("\n" + "-" * 40)
    print("3. TPS変換 (Thin-Plate Spline)")
    print("-" * 40)

    tps = ThinPlateSplineTransformer.from_correspondence_file(
        correspondence_file,
        fm_config,
    )

    tps_metrics = evaluator.evaluate(tps)
    print(f"  RMSE: {tps_metrics.rmse:.2f} px")
    print(f"  MAE: {tps_metrics.mae:.2f} px")
    print(f"  Max Error: {tps_metrics.max_error:.2f} px")
    print(f"  90th Percentile: {tps_metrics.percentile_90:.2f} px")

    training_error_tps = tps.evaluate_training_error()
    print(f"  訓練RMSE: {training_error_tps['rmse']:.4f} px (補間精度)")

    # ==== 比較サマリー ====
    print("\n" + "=" * 60)
    print("比較サマリー")
    print("=" * 60)

    print("\n| 手法 | RMSE (px) | MAE (px) | Max (px) | 90%ile (px) |")
    print("|------|-----------|----------|----------|-------------|")
    print(
        f"| Homography | {h_metrics.rmse:.1f} | {h_metrics.mae:.1f} | {h_metrics.max_error:.1f} | {h_metrics.percentile_90:.1f} |"
    )
    print(
        f"| PWA | {pwa_metrics.rmse:.1f} | {pwa_metrics.mae:.1f} | {pwa_metrics.max_error:.1f} | {pwa_metrics.percentile_90:.1f} |"
    )
    print(
        f"| TPS | {tps_metrics.rmse:.1f} | {tps_metrics.mae:.1f} | {tps_metrics.max_error:.1f} | {tps_metrics.percentile_90:.1f} |"
    )

    # 改善率
    if h_metrics.rmse > 0:
        pwa_improvement = (1 - pwa_metrics.rmse / h_metrics.rmse) * 100
        tps_improvement = (1 - tps_metrics.rmse / h_metrics.rmse) * 100
        print(f"\nPWA改善率: {pwa_improvement:.1f}%")
        print(f"TPS改善率: {tps_improvement:.1f}%")

    # ==== PWAモデルを保存 ====
    output_dir = PROJECT_ROOT / "output/calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    pwa_model_path = output_dir / "pwa_model.pkl"
    pwa.save(pwa_model_path)
    print(f"\nPWAモデルを保存: {pwa_model_path}")

    # ==== 可視化 ====
    print("\n三角形分割を可視化中...")
    reference_image = PROJECT_ROOT / "output/latest/phase1_extraction/frames/frame_20250826_160500_idx4.jpg"

    vis_path = output_dir / "pwa_triangulation.png"
    if reference_image.exists():
        pwa.visualize_triangulation(
            image=reference_image,
            output_path=vis_path,
        )
    else:
        pwa.visualize_triangulation(
            image_size=(1280, 720),
            output_path=vis_path,
        )
    print(f"  保存先: {vis_path}")

    # 誤差可視化
    floormap_path = PROJECT_ROOT / "data/floormap.png"
    error_vis_path = output_dir / "pwa_error_visualization.png"
    evaluator.visualize_errors(
        pwa_metrics,
        floormap_path,
        error_vis_path,
    )
    print(f"  誤差可視化: {error_vis_path}")

    print("\n" + "=" * 60)
    print("テスト完了！")
    print("=" * 60)

    # 推奨
    print("\n推奨:")
    best = min(
        [
            ("Homography", h_metrics.rmse),
            ("PWA", pwa_metrics.rmse),
            ("TPS", tps_metrics.rmse),
        ],
        key=lambda x: x[1],
    )
    print(f"  最良の手法: {best[0]} (RMSE: {best[1]:.2f}px)")

    if best[0] == "PWA":
        print("  → config.yaml の transform.method を 'piecewise_affine' に設定してください")
    elif best[0] == "TPS":
        print("  → config.yaml の transform.method を 'thin_plate_spline' に設定してください")

    return h_metrics, pwa_metrics, tps_metrics


if __name__ == "__main__":
    main()
