#!/usr/bin/env python3
"""カメラモデルベースの最適化ツール。

ピンホールカメラモデル + ホモグラフィのハイブリッドで
物理的に妥当な座標変換を実現します。

利点:
- 対応点がない領域でも幾何学的に妥当な変換
- 外挿（対応点外の領域）に強い
- カメラパラメータを調整可能
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import differential_evolution

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import load_correspondence_file


@dataclass
class CameraModel:
    """カメラモデルパラメータ。"""

    # 内部パラメータ
    fx: float = 1200.0
    fy: float = 1200.0
    cx: float = 640.0
    cy: float = 360.0

    # 外部パラメータ
    pitch_deg: float = 15.0
    yaw_deg: float = 20.0
    roll_deg: float = 0.0
    height_m: float = 2.5

    # フロアマップ変換
    cam_x_px: float = 550.0  # カメラ位置（フロアマップ上）
    cam_y_px: float = 850.0
    scale: float = 30.0  # mm/px
    rotation_deg: float = 0.0  # フロアマップの回転


def project_to_floor(
    image_pts: np.ndarray,
    model: CameraModel,
) -> np.ndarray:
    """画像座標を床面（フロアマップ座標）に投影。"""
    # 正規化座標
    x = (image_pts[:, 0] - model.cx) / model.fx
    y = (image_pts[:, 1] - model.cy) / model.fy

    # カメラ座標系での光線方向
    rays = np.column_stack([x, y, np.ones(len(x))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # 回転行列
    pitch = np.radians(model.pitch_deg)
    yaw = np.radians(model.yaw_deg)
    roll = np.radians(model.roll_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx

    # ワールド座標系での光線方向
    rays_world = (R @ rays.T).T

    # 床面との交点（Z = 0）
    # camera_pos = (0, 0, height_m)
    # intersection: camera_pos + t * ray = (x, y, 0)
    # t = -height_m / ray_z

    t = -model.height_m / rays_world[:, 2]

    # 無効な点（カメラより後ろ、または地平線）をマーク
    invalid = (t < 0) | (np.abs(rays_world[:, 2]) < 0.01)

    # ワールド座標
    world_x = t * rays_world[:, 0]
    world_y = t * rays_world[:, 1]

    # フロアマップ座標に変換
    rot = np.radians(model.rotation_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)

    # ワールド座標（メートル）からピクセルへ
    floor_x = model.cam_x_px + (world_x * cos_r - world_y * sin_r) * 1000 / model.scale
    floor_y = model.cam_y_px + (world_x * sin_r + world_y * cos_r) * 1000 / model.scale

    # 無効な点はNaN
    floor_x[invalid] = np.nan
    floor_y[invalid] = np.nan

    return np.column_stack([floor_x, floor_y])


def optimize_camera_model(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    initial_model: CameraModel,
) -> tuple[CameraModel, float]:
    """カメラモデルを最適化。"""
    print("\n" + "=" * 70)
    print("カメラモデルの最適化（Differential Evolution）")
    print("=" * 70)

    def objective(params):
        model = CameraModel(
            fx=params[0],
            fy=params[0],  # fx = fy を仮定
            cx=params[1],
            cy=params[2],
            pitch_deg=params[3],
            yaw_deg=params[4],
            roll_deg=params[5],
            height_m=params[6],
            cam_x_px=params[7],
            cam_y_px=params[8],
            scale=params[9],
            rotation_deg=params[10],
        )

        projected = project_to_floor(src_pts, model)

        # NaNを含む点は大きなペナルティ
        valid = ~np.isnan(projected[:, 0])
        if valid.sum() < len(src_pts) * 0.8:
            return 10000.0

        errors = np.linalg.norm(projected[valid] - dst_pts[valid], axis=1)
        rmse = np.sqrt(np.mean(errors**2))

        return rmse

    # パラメータ境界
    bounds = [
        (800, 2000),  # fx
        (500, 800),  # cx
        (250, 450),  # cy
        (5, 40),  # pitch_deg
        (-30, 50),  # yaw_deg
        (-10, 10),  # roll_deg
        (1.5, 4.0),  # height_m
        (400, 700),  # cam_x_px
        (700, 1000),  # cam_y_px
        (20, 40),  # scale
        (-30, 30),  # rotation_deg
    ]

    # 初期値
    x0 = [
        initial_model.fx,
        initial_model.cx,
        initial_model.cy,
        initial_model.pitch_deg,
        initial_model.yaw_deg,
        initial_model.roll_deg,
        initial_model.height_m,
        initial_model.cam_x_px,
        initial_model.cam_y_px,
        initial_model.scale,
        initial_model.rotation_deg,
    ]

    print(f"初期RMSE: {objective(x0):.2f} px")

    # Differential Evolution (シングルスレッドで実行 - pickleエラー回避)
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500,
        workers=1,  # シングルスレッド
        polish=True,
        disp=True,
    )

    best_model = CameraModel(
        fx=result.x[0],
        fy=result.x[0],
        cx=result.x[1],
        cy=result.x[2],
        pitch_deg=result.x[3],
        yaw_deg=result.x[4],
        roll_deg=result.x[5],
        height_m=result.x[6],
        cam_x_px=result.x[7],
        cam_y_px=result.x[8],
        scale=result.x[9],
        rotation_deg=result.x[10],
    )

    print(f"\n最適化後RMSE: {result.fun:.2f} px")
    print("\n最適パラメータ:")
    print(f"  fx = fy = {best_model.fx:.1f}")
    print(f"  cx = {best_model.cx:.1f}")
    print(f"  cy = {best_model.cy:.1f}")
    print(f"  pitch = {best_model.pitch_deg:.1f}°")
    print(f"  yaw = {best_model.yaw_deg:.1f}°")
    print(f"  roll = {best_model.roll_deg:.1f}°")
    print(f"  height = {best_model.height_m:.2f} m")
    print(f"  cam_pos = ({best_model.cam_x_px:.1f}, {best_model.cam_y_px:.1f}) px")
    print(f"  scale = {best_model.scale:.2f} mm/px")
    print(f"  rotation = {best_model.rotation_deg:.1f}°")

    return best_model, result.fun


def refine_with_homography_correction(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    model: CameraModel,
) -> tuple[np.ndarray, float]:
    """カメラモデル投影後に残差をホモグラフィで補正。"""
    print("\n" + "=" * 70)
    print("ホモグラフィによる残差補正")
    print("=" * 70)

    # カメラモデルで投影
    projected = project_to_floor(src_pts, model)

    valid = ~np.isnan(projected[:, 0])
    if valid.sum() < 4:
        print("有効な点が不足しています")
        return None, float("inf")

    # 残差を計算
    residuals = dst_pts[valid] - projected[valid]
    print(f"残差の平均: ({residuals[:, 0].mean():.1f}, {residuals[:, 1].mean():.1f}) px")
    print(f"残差の標準偏差: ({residuals[:, 0].std():.1f}, {residuals[:, 1].std():.1f}) px")

    # ホモグラフィで残差を補正
    H, _ = cv2.findHomography(
        projected[valid].astype(np.float32),
        dst_pts[valid].astype(np.float32),
        0,  # RANSACなし
    )

    if H is None:
        print("ホモグラフィの計算に失敗")
        return None, float("inf")

    # 補正後の誤差
    ones = np.ones((valid.sum(), 1))
    pts_h = np.hstack([projected[valid], ones])
    corrected = (H @ pts_h.T).T
    corrected = corrected[:, :2] / corrected[:, 2:3]

    errors = np.linalg.norm(corrected - dst_pts[valid], axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"補正後RMSE: {rmse:.2f} px")
    print(f"最大誤差: {errors.max():.2f} px")

    return H, rmse


class CameraModelTransformer:
    """カメラモデルベースの変換器。"""

    def __init__(self, model: CameraModel, correction_H: np.ndarray | None = None):
        self.model = model
        self.correction_H = correction_H

    def transform(self, points: np.ndarray) -> np.ndarray:
        # カメラモデルで投影
        projected = project_to_floor(points, self.model)

        # ホモグラフィ補正
        if self.correction_H is not None:
            valid = ~np.isnan(projected[:, 0])
            if valid.any():
                ones = np.ones((valid.sum(), 1))
                pts_h = np.hstack([projected[valid], ones])
                corrected = (self.correction_H @ pts_h.T).T
                projected[valid] = corrected[:, :2] / corrected[:, 2:3]

        return projected


def visualize_result(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transformer: CameraModelTransformer,
    floormap_path: str,
    output_path: str,
):
    """結果を可視化。"""
    floormap = cv2.imread(floormap_path)
    if floormap is None:
        print(f"フロアマップを読み込めません: {floormap_path}")
        return

    transformed = transformer.transform(src_pts)

    for _i, (expected, actual) in enumerate(zip(dst_pts, transformed, strict=False)):
        if np.isnan(actual[0]):
            continue

        exp_pt = tuple(map(int, expected))
        act_pt = tuple(map(int, actual))

        cv2.circle(floormap, exp_pt, 8, (0, 255, 0), 2)
        cv2.circle(floormap, act_pt, 6, (0, 0, 255), -1)
        cv2.arrowedLine(floormap, exp_pt, act_pt, (255, 0, 255), 2, tipLength=0.3)

    valid = ~np.isnan(transformed[:, 0])
    errors = np.linalg.norm(transformed[valid] - dst_pts[valid], axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    cv2.putText(floormap, f"Camera Model RMSE: {rmse:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(floormap, f"Max Error: {errors.max():.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(output_path, floormap)
    print(f"\n可視化画像を保存: {output_path}")


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
    print("カメラモデルベースの座標変換最適化")
    print("=" * 70)
    print(f"対応点数: {len(correspondences)}")

    # 初期モデル
    initial_model = CameraModel()

    # 最適化
    best_model, _rmse = optimize_camera_model(src_pts, dst_pts, initial_model)

    # ホモグラフィ補正
    correction_H, final_rmse = refine_with_homography_correction(src_pts, dst_pts, best_model)

    # 変換器を作成
    transformer = CameraModelTransformer(best_model, correction_H)

    # 結果を可視化
    floormap_path = config.get("floormap.image_path", "data/floormap.png")
    output_dir = Path("output/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_result(src_pts, dst_pts, transformer, floormap_path, str(output_dir / "camera_model_result.png"))

    # モデルを保存
    model_path = output_dir / "camera_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "correction_H": correction_H,
                "rmse": final_rmse,
            },
            f,
        )
    print(f"モデルを保存: {model_path}")

    # config.yaml用の出力
    print("\n" + "=" * 70)
    print("config.yaml 用のパラメータ")
    print("=" * 70)
    print(f"""
camera_params:
  intrinsics:
    focal_length_x: {best_model.fx:.1f}
    focal_length_y: {best_model.fy:.1f}
    center_x: {best_model.cx:.1f}
    center_y: {best_model.cy:.1f}
  extrinsics:
    pitch_deg: {best_model.pitch_deg:.2f}
    yaw_deg: {best_model.yaw_deg:.2f}
    roll_deg: {best_model.roll_deg:.2f}
    height_m: {best_model.height_m:.3f}
    position_x_px: {best_model.cam_x_px:.1f}
    position_y_px: {best_model.cam_y_px:.1f}

floormap:
  image_x_mm_per_pixel: {best_model.scale:.4f}
  image_y_mm_per_pixel: {best_model.scale:.4f}
""")


if __name__ == "__main__":
    main()
