"""カメラパラメータ最適化ツール

Grid SearchとLevenberg-Marquardt法を使用してカメラパラメータを最適化します。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CameraParams:
    """カメラパラメータ"""

    height_m: float = 2.2
    pitch_deg: float = 45.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    position_x_px: float = 859.0
    position_y_px: float = 1040.0
    focal_length_x: float = 1250.0
    focal_length_y: float = 1250.0
    center_x: float = 640.0
    center_y: float = 360.0

    def to_array(self) -> np.ndarray:
        """最適化対象パラメータを配列に変換"""
        return np.array(
            [
                self.height_m,
                self.pitch_deg,
                self.yaw_deg,
                self.roll_deg,
                self.position_x_px,
                self.position_y_px,
            ]
        )

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        focal_length_x: float = 1250.0,
        focal_length_y: float = 1250.0,
        center_x: float = 640.0,
        center_y: float = 360.0,
    ) -> CameraParams:
        """配列からパラメータを作成"""
        return cls(
            height_m=arr[0],
            pitch_deg=arr[1],
            yaw_deg=arr[2],
            roll_deg=arr[3],
            position_x_px=arr[4],
            position_y_px=arr[5],
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            center_x=center_x,
            center_y=center_y,
        )

    def to_dict(self) -> dict[str, float]:
        """辞書に変換"""
        return {
            "height_m": self.height_m,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "roll_deg": self.roll_deg,
            "position_x_px": self.position_x_px,
            "position_y_px": self.position_y_px,
            "focal_length_x": self.focal_length_x,
            "focal_length_y": self.focal_length_y,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


class PinholeCameraModel:
    """ピンホールカメラモデル

    画像座標からワールド座標（床面）への変換を行います。
    """

    def __init__(self, params: CameraParams, scale_mm_per_px: tuple[float, float]):
        """初期化

        Args:
            params: カメラパラメータ
            scale_mm_per_px: (scale_x, scale_y) mm/pixel
        """
        self.params = params
        self.scale_x_mm_per_px = scale_mm_per_px[0]
        self.scale_y_mm_per_px = scale_mm_per_px[1]

        # 内部パラメータ行列
        self.K = np.array(
            [
                [params.focal_length_x, 0, params.center_x],
                [0, params.focal_length_y, params.center_y],
                [0, 0, 1],
            ]
        )

        # 回転行列を計算
        self.R = self._compute_rotation_matrix()

    def _compute_rotation_matrix(self) -> np.ndarray:
        """回転行列を計算"""
        pitch = np.radians(self.params.pitch_deg)
        yaw = np.radians(self.params.yaw_deg)
        roll = np.radians(self.params.roll_deg)

        # 回転行列（ZYX順序）
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )

        return Rz @ Ry @ Rx

    def image_to_floor(self, image_point: tuple[float, float]) -> tuple[float, float] | None:
        """画像座標からフロアマップ座標に変換

        Args:
            image_point: 画像座標 (u, v)

        Returns:
            フロアマップ座標 (px, py) または None
        """
        u, v = image_point

        # 正規化画像座標
        K_inv = np.linalg.inv(self.K)
        normalized = K_inv @ np.array([u, v, 1.0])

        # カメラ座標系でのレイ方向
        ray_camera = normalized / np.linalg.norm(normalized)

        # ワールド座標系でのレイ方向
        R_inv = self.R.T
        ray_world = R_inv @ ray_camera

        # 床面（Z=0）との交点を計算
        if abs(ray_world[2]) < 1e-10:
            return None  # 床面と平行

        # カメラ位置は原点上空 height_m
        camera_z = self.params.height_m

        # レイが床面と交差するパラメータ t
        # P = (0, 0, camera_z) + t * ray_world
        # P_z = 0 → t = -camera_z / ray_world[2]
        t = -camera_z / ray_world[2]

        if t < 0:
            return None  # カメラの後ろ

        # ワールド座標での交点
        world_x = t * ray_world[0]
        world_y = t * ray_world[1]

        # フロアマップ座標に変換
        # スケール変換（m → px）
        scale_x_px_per_m = 1000.0 / self.scale_x_mm_per_px
        scale_y_px_per_m = 1000.0 / self.scale_y_mm_per_px

        floor_x = self.params.position_x_px + world_x * scale_x_px_per_m
        floor_y = self.params.position_y_px - world_y * scale_y_px_per_m  # Y軸反転

        return (floor_x, floor_y)


class CameraParamOptimizer:
    """カメラパラメータ最適化器"""

    def __init__(
        self,
        correspondence_points: list[dict],
        scale_mm_per_px: tuple[float, float],
        initial_params: CameraParams | None = None,
    ):
        """初期化

        Args:
            correspondence_points: 対応点リスト
            scale_mm_per_px: (scale_x, scale_y)
            initial_params: 初期パラメータ
        """
        self.src_points = np.array([p["src_point"] for p in correspondence_points])
        self.dst_points = np.array([p["dst_point"] for p in correspondence_points])
        self.scale_mm_per_px = scale_mm_per_px
        self.initial_params = initial_params or CameraParams()

    def _compute_residuals(self, param_array: np.ndarray) -> np.ndarray:
        """残差を計算"""
        params = CameraParams.from_array(
            param_array,
            self.initial_params.focal_length_x,
            self.initial_params.focal_length_y,
            self.initial_params.center_x,
            self.initial_params.center_y,
        )

        model = PinholeCameraModel(params, self.scale_mm_per_px)
        residuals = []

        for src, dst in zip(self.src_points, self.dst_points, strict=False):
            result = model.image_to_floor((src[0], src[1]))
            if result is None:
                residuals.extend([1000.0, 1000.0])  # ペナルティ
            else:
                residuals.append(result[0] - dst[0])
                residuals.append(result[1] - dst[1])

        return np.array(residuals)

    def _compute_rmse(self, param_array: np.ndarray) -> float:
        """RMSEを計算"""
        residuals = self._compute_residuals(param_array)
        # 2次元の残差なので2つずつペアにして距離を計算
        errors = np.sqrt(residuals[0::2] ** 2 + residuals[1::2] ** 2)
        return float(np.sqrt(np.mean(errors**2)))

    def grid_search(
        self,
        height_range: tuple[float, float, float] = (1.5, 3.5, 0.5),
        pitch_range: tuple[float, float, float] = (10, 60, 10),
        yaw_range: tuple[float, float, float] = (-45, 45, 15),
        roll_fixed: float = 0.0,
        position_x_range: tuple[float, float, float] | None = None,
        position_y_range: tuple[float, float, float] | None = None,
    ) -> tuple[CameraParams, float, list[dict]]:
        """Grid Searchで最適パラメータを探索

        Args:
            height_range: (min, max, step)
            pitch_range: (min, max, step)
            yaw_range: (min, max, step)
            roll_fixed: roll角（固定）
            position_x_range: X位置の範囲（Noneなら固定）
            position_y_range: Y位置の範囲（Noneなら固定）

        Returns:
            (最適パラメータ, 最小RMSE, 全結果リスト)
        """
        heights = np.arange(height_range[0], height_range[1] + 0.01, height_range[2])
        pitches = np.arange(pitch_range[0], pitch_range[1] + 0.01, pitch_range[2])
        yaws = np.arange(yaw_range[0], yaw_range[1] + 0.01, yaw_range[2])

        if position_x_range:
            pos_xs = np.arange(position_x_range[0], position_x_range[1] + 0.01, position_x_range[2])
        else:
            pos_xs = [self.initial_params.position_x_px]

        if position_y_range:
            pos_ys = np.arange(position_y_range[0], position_y_range[1] + 0.01, position_y_range[2])
        else:
            pos_ys = [self.initial_params.position_y_px]

        results = []
        best_rmse = float("inf")
        best_params = None

        total = len(heights) * len(pitches) * len(yaws) * len(pos_xs) * len(pos_ys)
        print(f"Grid Search: {total} combinations")

        for h in heights:
            for p in pitches:
                for y in yaws:
                    for px in pos_xs:
                        for py in pos_ys:
                            params = CameraParams(
                                height_m=h,
                                pitch_deg=p,
                                yaw_deg=y,
                                roll_deg=roll_fixed,
                                position_x_px=px,
                                position_y_px=py,
                                focal_length_x=self.initial_params.focal_length_x,
                                focal_length_y=self.initial_params.focal_length_y,
                                center_x=self.initial_params.center_x,
                                center_y=self.initial_params.center_y,
                            )

                            rmse = self._compute_rmse(params.to_array())

                            results.append(
                                {
                                    "params": params.to_dict(),
                                    "rmse": rmse,
                                }
                            )

                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params = params

        print(f"Best RMSE: {best_rmse:.2f}px")

        return best_params, best_rmse, results

    def refine_lm(
        self,
        initial_params: CameraParams,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[CameraParams, float]:
        """Levenberg-Marquardt法で精密最適化

        Args:
            initial_params: 初期パラメータ
            bounds: パラメータの上下限

        Returns:
            (最適パラメータ, 最小RMSE)
        """
        x0 = initial_params.to_array()

        # デフォルト境界
        if bounds is None:
            bounds = {
                "height_m": (0.5, 5.0),
                "pitch_deg": (-90, 90),
                "yaw_deg": (-90, 90),
                "roll_deg": (-15, 15),
                "position_x_px": (0, 2000),
                "position_y_px": (0, 2000),
            }

        lower = np.array(
            [
                bounds["height_m"][0],
                bounds["pitch_deg"][0],
                bounds["yaw_deg"][0],
                bounds["roll_deg"][0],
                bounds["position_x_px"][0],
                bounds["position_y_px"][0],
            ]
        )

        upper = np.array(
            [
                bounds["height_m"][1],
                bounds["pitch_deg"][1],
                bounds["yaw_deg"][1],
                bounds["roll_deg"][1],
                bounds["position_x_px"][1],
                bounds["position_y_px"][1],
            ]
        )

        result = least_squares(
            self._compute_residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=1000,
            ftol=1e-8,
            verbose=0,
        )

        opt_params = CameraParams.from_array(
            result.x,
            initial_params.focal_length_x,
            initial_params.focal_length_y,
            initial_params.center_x,
            initial_params.center_y,
        )

        rmse = self._compute_rmse(result.x)

        return opt_params, rmse

    def optimize(
        self,
        use_grid_search: bool = True,
        use_refinement: bool = True,
    ) -> tuple[CameraParams, float, dict]:
        """最適化を実行

        Args:
            use_grid_search: Grid Searchを使用するか
            use_refinement: LM法で精密化するか

        Returns:
            (最適パラメータ, RMSE, 詳細情報)
        """
        info: dict[str, Any] = {}

        if use_grid_search:
            print("Step 1: Grid Search...")
            best_params, best_rmse, grid_results = self.grid_search()
            info["grid_search"] = {
                "rmse": best_rmse,
                "params": best_params.to_dict(),
                "num_combinations": len(grid_results),
            }
        else:
            best_params = self.initial_params
            best_rmse = self._compute_rmse(best_params.to_array())

        if use_refinement:
            print("Step 2: LM Refinement...")
            refined_params, refined_rmse = self.refine_lm(best_params)
            info["refinement"] = {
                "initial_rmse": best_rmse,
                "final_rmse": refined_rmse,
                "params": refined_params.to_dict(),
            }

            if refined_rmse < best_rmse:
                best_params = refined_params
                best_rmse = refined_rmse

        info["final"] = {
            "rmse": best_rmse,
            "params": best_params.to_dict(),
        }

        return best_params, best_rmse, info

    def evaluate(self, params: CameraParams) -> dict:
        """パラメータを評価

        Args:
            params: カメラパラメータ

        Returns:
            評価結果
        """
        model = PinholeCameraModel(params, self.scale_mm_per_px)
        errors = []
        per_point = []

        for i, (src, dst) in enumerate(zip(self.src_points, self.dst_points, strict=False)):
            result = model.image_to_floor((src[0], src[1]))
            if result is None:
                per_point.append(
                    {
                        "index": i,
                        "src": src.tolist(),
                        "dst_expected": dst.tolist(),
                        "dst_actual": None,
                        "error": None,
                        "is_valid": False,
                    }
                )
                continue

            error = np.sqrt((result[0] - dst[0]) ** 2 + (result[1] - dst[1]) ** 2)
            errors.append(error)

            per_point.append(
                {
                    "index": i,
                    "src": src.tolist(),
                    "dst_expected": dst.tolist(),
                    "dst_actual": list(result),
                    "error": float(error),
                    "is_valid": True,
                }
            )

        if not errors:
            return {"rmse": float("inf"), "valid_ratio": 0.0}

        errors_array = np.array(errors)

        return {
            "rmse": float(np.sqrt(np.mean(errors_array**2))),
            "mae": float(np.mean(errors_array)),
            "max_error": float(np.max(errors_array)),
            "min_error": float(np.min(errors_array)),
            "std_error": float(np.std(errors_array)),
            "num_valid": len(errors),
            "num_total": len(self.src_points),
            "valid_ratio": len(errors) / len(self.src_points),
            "per_point_errors": per_point,
        }


def load_correspondence_points(file_path: Path) -> list[dict]:
    """対応点を読み込み"""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("point_correspondences", [])


def main():
    import yaml

    print("=" * 60)
    print("カメラパラメータ最適化ツール")
    print("=" * 60)

    # 設定読み込み
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 対応点読み込み
    correspondence_file = PROJECT_ROOT / config["calibration"]["correspondence_file"]
    print(f"\n対応点ファイル: {correspondence_file}")
    points = load_correspondence_points(correspondence_file)
    print(f"対応点数: {len(points)}")

    # スケール
    scale_mm_per_px = (
        config["floormap"]["image_x_mm_per_pixel"],
        config["floormap"]["image_y_mm_per_pixel"],
    )

    # 初期パラメータ
    cam_params = config.get("camera_params", {})
    initial = CameraParams(
        height_m=cam_params.get("height_m", 2.2),
        pitch_deg=cam_params.get("pitch_deg", 45.0),
        yaw_deg=cam_params.get("yaw_deg", 0.0),
        roll_deg=cam_params.get("roll_deg", 0.0),
        position_x_px=cam_params.get("position_x_px", 859.0),
        position_y_px=cam_params.get("position_y_px", 1040.0),
        focal_length_x=cam_params.get("focal_length_x", 1250.0),
        focal_length_y=cam_params.get("focal_length_y", 1250.0),
        center_x=cam_params.get("center_x", 640.0),
        center_y=cam_params.get("center_y", 360.0),
    )

    print("\n初期パラメータ:")
    for k, v in initial.to_dict().items():
        print(f"  {k}: {v}")

    # 最適化
    optimizer = CameraParamOptimizer(points, scale_mm_per_px, initial)

    # 初期誤差
    initial_eval = optimizer.evaluate(initial)
    print(f"\n初期RMSE: {initial_eval['rmse']:.2f}px")

    # 最適化実行
    print("\n" + "-" * 40)
    best_params, best_rmse, info = optimizer.optimize(
        use_grid_search=True,
        use_refinement=True,
    )
    print("-" * 40)

    print("\n最適化結果:")
    print(f"  RMSE: {best_rmse:.2f}px")
    print("\n最適パラメータ:")
    for k, v in best_params.to_dict().items():
        print(f"  {k}: {v}")

    # 詳細評価
    final_eval = optimizer.evaluate(best_params)
    print("\n詳細評価:")
    print(f"  RMSE: {final_eval['rmse']:.2f}px")
    print(f"  MAE: {final_eval['mae']:.2f}px")
    print(f"  Max Error: {final_eval['max_error']:.2f}px")
    print(f"  Valid Ratio: {final_eval['valid_ratio']:.1%}")

    # 結果を保存
    output_dir = PROJECT_ROOT / "output/calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "initial_params": initial.to_dict(),
        "optimized_params": best_params.to_dict(),
        "initial_rmse": initial_eval["rmse"],
        "optimized_rmse": best_rmse,
        "evaluation": {
            "rmse": final_eval["rmse"],
            "mae": final_eval["mae"],
            "max_error": final_eval["max_error"],
            "valid_ratio": final_eval["valid_ratio"],
        },
        "optimization_info": info,
    }

    output_path = output_dir / "camera_param_optimization_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {output_path}")

    # config.yaml用のYAMLを出力
    print("\n" + "=" * 60)
    print("config.yaml用のパラメータ:")
    print("=" * 60)
    print(
        """
camera_params:
  height_m: {height_m}
  pitch_deg: {pitch_deg}
  yaw_deg: {yaw_deg}
  roll_deg: {roll_deg}
  position_x_px: {position_x_px}
  position_y_px: {position_y_px}
  focal_length_x: {focal_length_x}
  focal_length_y: {focal_length_y}
  center_x: {center_x}
  center_y: {center_y}
""".format(**best_params.to_dict())
    )

    return best_params, best_rmse


if __name__ == "__main__":
    main()
