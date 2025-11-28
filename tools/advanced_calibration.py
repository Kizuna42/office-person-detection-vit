#!/usr/bin/env python3
"""高度なカメラキャリブレーションツール。

対応点データから最適なカメラパラメータを推定します。

アプローチ:
1. ホモグラフィ推定（2D→2D直接変換）
2. 非線形最適化（全パラメータ同時最適化）
3. カメラ位置も含めた探索

使用方法:
    python tools/advanced_calibration.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import differential_evolution

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file
from src.transform.floormap_transformer import FloorMapConfig, FloorMapTransformer
from src.transform.projection.pinhole_model import CameraExtrinsics, CameraIntrinsics
from src.transform.projection.ray_caster import RayCaster
from src.transform.unified_transformer import UnifiedTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """キャリブレーション結果。"""

    method: str
    rmse: float
    max_error: float
    num_inliers: int
    params: dict
    homography: np.ndarray | None = None


def load_correspondences(config: ConfigManager) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """対応点を読み込む。"""
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    if not correspondence_file or not Path(correspondence_file).exists():
        raise FileNotFoundError(f"対応点ファイルが見つかりません: {correspondence_file}")

    data = load_correspondence_file(correspondence_file)

    # 点-点対応を優先して使用（重複を避ける）
    correspondences = []
    if data.point_pairs:
        correspondences = [(pc.src_point, pc.dst_point) for pc in data.point_pairs]
    else:
        correspondences = data.get_foot_points()

    logger.info(f"対応点数: {len(correspondences)}")
    return correspondences


def compute_homography(
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[np.ndarray, float, float, int]:
    """ホモグラフィ行列を推定。

    Returns:
        (H, rmse, max_error, num_inliers)
    """
    src_pts = np.array([c[0] for c in correspondences], dtype=np.float32)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float32)

    # RANSACでホモグラフィを推定
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("ホモグラフィの推定に失敗しました")

    # 変換して誤差を計算
    src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    transformed = (H @ src_pts_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    errors = np.linalg.norm(transformed - dst_pts, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    num_inliers = int(np.sum(mask))

    return H, rmse, max_error, num_inliers


def compute_rmse_pinhole(
    params: np.ndarray,
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
    config: ConfigManager,
) -> float:
    """ピンホールモデルでRMSEを計算。

    params: [pitch_deg, yaw_deg, roll_deg, height_m, fx, fy, cx, cy, cam_x_px, cam_y_px]
    """
    pitch_deg, yaw_deg, roll_deg, height_m, fx, fy, cx, cy, cam_x_px, cam_y_px = params

    try:
        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_width=1280,
            image_height=720,
        )

        extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=height_m,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
        )

        floormap_config_dict = config.get("floormap", {})
        floormap_config = FloorMapConfig(
            width_px=int(floormap_config_dict.get("image_width", 1878)),
            height_px=int(floormap_config_dict.get("image_height", 1369)),
            scale_x_mm_per_px=float(floormap_config_dict.get("image_x_mm_per_pixel", 28.1926)),
            scale_y_mm_per_px=float(floormap_config_dict.get("image_y_mm_per_pixel", 28.2414)),
        )

        ray_caster = RayCaster(intrinsics, extrinsics)
        floormap_transformer = FloorMapTransformer(floormap_config, (cam_x_px, cam_y_px))
        transformer = UnifiedTransformer(ray_caster, floormap_transformer)

        errors = []
        for image_pt, expected_fm in correspondences:
            result = transformer.transform_pixel(image_pt)
            if result.is_valid and result.floor_coords_px:
                actual = result.floor_coords_px
                error = np.linalg.norm(np.array(actual) - np.array(expected_fm))
                errors.append(error)
            else:
                errors.append(1000.0)  # ペナルティ

        if not errors:
            return float("inf")

        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        return rmse

    except Exception:
        return float("inf")


def optimize_pinhole_params(
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
    config: ConfigManager,
    initial_guess: dict | None = None,
) -> CalibrationResult:
    """ピンホールモデルのパラメータを最適化。"""

    # 初期値
    if initial_guess is None:
        camera_params = config.get("camera_params", {})
        initial_guess = {
            "pitch_deg": camera_params.get("pitch_deg", 15.0),
            "yaw_deg": camera_params.get("yaw_deg", 30.0),
            "roll_deg": camera_params.get("roll_deg", 0.0),
            "height_m": camera_params.get("height_m", 2.2),
            "fx": camera_params.get("focal_length_x", 1250.0),
            "fy": camera_params.get("focal_length_y", 1250.0),
            "cx": camera_params.get("center_x", 640.0),
            "cy": camera_params.get("center_y", 360.0),
            "cam_x_px": camera_params.get("position_x_px", 859.0),
            "cam_y_px": camera_params.get("position_y_px", 1040.0),
        }

    np.array(
        [
            initial_guess["pitch_deg"],
            initial_guess["yaw_deg"],
            initial_guess["roll_deg"],
            initial_guess["height_m"],
            initial_guess["fx"],
            initial_guess["fy"],
            initial_guess["cx"],
            initial_guess["cy"],
            initial_guess["cam_x_px"],
            initial_guess["cam_y_px"],
        ]
    )

    # 境界
    bounds = [
        (0, 60),  # pitch_deg
        (-90, 90),  # yaw_deg
        (-30, 30),  # roll_deg
        (1.0, 5.0),  # height_m
        (500, 3000),  # fx
        (500, 3000),  # fy
        (400, 900),  # cx
        (200, 550),  # cy
        (400, 1400),  # cam_x_px
        (700, 1300),  # cam_y_px
    ]

    logger.info("Differential Evolution による最適化を開始...")

    def objective(p):
        return compute_rmse_pinhole(p, correspondences, config)

    # Differential Evolution (グローバル最適化)
    result = differential_evolution(
        objective,
        bounds,
        maxiter=300,
        tol=0.01,
        seed=42,
        workers=1,  # シングルスレッドで実行
        polish=True,
        disp=True,
    )

    best_params = result.x
    best_rmse = result.fun

    # 最終的な誤差を計算
    pitch_deg, yaw_deg, roll_deg, height_m, fx, fy, cx, cy, cam_x_px, cam_y_px = best_params

    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, image_width=1280, image_height=720)
    extrinsics = CameraExtrinsics.from_pose(
        camera_height_m=height_m, pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg
    )

    floormap_config_dict = config.get("floormap", {})
    floormap_config = FloorMapConfig(
        width_px=int(floormap_config_dict.get("image_width", 1878)),
        height_px=int(floormap_config_dict.get("image_height", 1369)),
        scale_x_mm_per_px=float(floormap_config_dict.get("image_x_mm_per_pixel", 28.1926)),
        scale_y_mm_per_px=float(floormap_config_dict.get("image_y_mm_per_pixel", 28.2414)),
    )

    ray_caster = RayCaster(intrinsics, extrinsics)
    floormap_transformer = FloorMapTransformer(floormap_config, (cam_x_px, cam_y_px))
    transformer = UnifiedTransformer(ray_caster, floormap_transformer)

    errors = []
    for image_pt, expected_fm in correspondences:
        result_t = transformer.transform_pixel(image_pt)
        if result_t.is_valid and result_t.floor_coords_px:
            actual = result_t.floor_coords_px
            error = np.linalg.norm(np.array(actual) - np.array(expected_fm))
            errors.append(error)

    max_error = float(np.max(errors)) if errors else 0.0

    return CalibrationResult(
        method="differential_evolution",
        rmse=best_rmse,
        max_error=max_error,
        num_inliers=len(errors),
        params={
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "roll_deg": float(roll_deg),
            "height_m": float(height_m),
            "focal_length_x": float(fx),
            "focal_length_y": float(fy),
            "center_x": float(cx),
            "center_y": float(cy),
            "position_x_px": float(cam_x_px),
            "position_y_px": float(cam_y_px),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="高度なカメラキャリブレーション")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument("--output", default="output/calibration/advanced_calibration_result.json", help="出力ファイル")
    args = parser.parse_args()

    config = ConfigManager(args.config)

    print("=" * 70)
    print("高度なカメラキャリブレーション")
    print("=" * 70)

    # 対応点を読み込み
    correspondences = load_correspondences(config)

    # === 方法1: ホモグラフィ推定 ===
    print("\n[方法1] ホモグラフィ推定 (2D→2D直接変換)")
    print("-" * 50)

    try:
        H, rmse, max_error, num_inliers = compute_homography(correspondences)
        print(f"RMSE: {rmse:.2f} px")
        print(f"最大誤差: {max_error:.2f} px")
        print(f"インライア数: {num_inliers}/{len(correspondences)}")
        print(f"ホモグラフィ行列:\n{H}")

        homography_result = CalibrationResult(
            method="homography",
            rmse=rmse,
            max_error=max_error,
            num_inliers=num_inliers,
            params={},
            homography=H,
        )
    except Exception as e:
        print(f"エラー: {e}")
        homography_result = None

    # === 方法2: ピンホールモデル最適化 ===
    print("\n[方法2] ピンホールモデル最適化 (全パラメータ同時最適化)")
    print("-" * 50)

    pinhole_result = optimize_pinhole_params(correspondences, config)

    print("\n最適化完了!")
    print(f"RMSE: {pinhole_result.rmse:.2f} px")
    print(f"最大誤差: {pinhole_result.max_error:.2f} px")
    print(f"有効点数: {pinhole_result.num_inliers}/{len(correspondences)}")
    print("\n最適パラメータ:")
    for key, value in pinhole_result.params.items():
        print(f"  {key}: {value:.4f}")

    # === 結果を保存 ===
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "correspondences_count": len(correspondences),
        "homography": {
            "rmse": homography_result.rmse if homography_result else None,
            "max_error": homography_result.max_error if homography_result else None,
            "num_inliers": homography_result.num_inliers if homography_result else None,
            "matrix": homography_result.homography.tolist()
            if homography_result and homography_result.homography is not None
            else None,
        },
        "pinhole": {
            "rmse": pinhole_result.rmse,
            "max_error": pinhole_result.max_error,
            "num_inliers": pinhole_result.num_inliers,
            "params": pinhole_result.params,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存しました: {output_path}")

    # === 推奨事項 ===
    print("\n" + "=" * 70)
    print("推奨事項")
    print("=" * 70)

    if homography_result and homography_result.rmse < pinhole_result.rmse:
        print(f"\nホモグラフィ (RMSE: {homography_result.rmse:.2f} px) の方が精度が高いです。")
        print("ホモグラフィ行列を使用することを推奨します。")
        print("\nconfig.yaml に以下を追加してください:")
        print("homography:")
        print("  matrix:")
        for row in homography_result.homography:
            print(f"    - [{row[0]:.10f}, {row[1]:.10f}, {row[2]:.10f}]")
    else:
        print(f"\nピンホールモデル (RMSE: {pinhole_result.rmse:.2f} px) を使用します。")
        print("\nconfig.yaml の camera_params を以下に更新してください:")
        for key, value in pinhole_result.params.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
