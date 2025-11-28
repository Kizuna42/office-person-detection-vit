#!/usr/bin/env python3
"""カメラキャリブレーション調整ツール。

対話的にカメラパラメータを調整し、座標変換精度を確認するためのツール。

使用方法:
    python tools/calibration_tool.py [--config CONFIG_PATH]

機能:
    - パラメータの対話的調整
    - 対応点からの自動キャリブレーション
    - 再投影誤差のリアルタイム表示
    - 結果の保存
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.transform import (
    CameraExtrinsics,
    CameraIntrinsics,
    FloorMapConfig,
    FloorMapTransformer,
    RayCaster,
    UnifiedTransformer,
)
from src.transform.calibration import (
    CorrespondenceCalibrator,
    InteractiveCalibrator,
    load_correspondence_file,
)

if TYPE_CHECKING:
    from src.transform.calibration.correspondence import CorrespondenceData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationTool:
    """キャリブレーション調整ツール。

    対話的にカメラパラメータを調整し、精度を確認する。
    """

    def __init__(self, config_path: str = "config.yaml"):
        """初期化。

        Args:
            config_path: 設定ファイルパス
        """
        self.config = ConfigManager(config_path)
        self.correspondence_data: CorrespondenceData | None = None

        # カメラパラメータを読み込み
        camera_params = self.config.get("camera_params", {})
        floormap_config = self.config.get("floormap", {})

        # Intrinsics
        self.intrinsics = CameraIntrinsics(
            fx=float(camera_params.get("focal_length_x", 1250.0)),
            fy=float(camera_params.get("focal_length_y", 1250.0)),
            cx=float(camera_params.get("center_x", 640.0)),
            cy=float(camera_params.get("center_y", 360.0)),
            image_width=int(camera_params.get("image_width", 1280)),
            image_height=int(camera_params.get("image_height", 720)),
            dist_coeffs=np.array(camera_params.get("dist_coeffs", [0.0] * 5), dtype=np.float64),
        )

        # Extrinsics
        self.extrinsics = CameraExtrinsics.from_pose(
            camera_height_m=float(camera_params.get("height_m", 2.2)),
            pitch_deg=float(camera_params.get("pitch_deg", 45.0)),
            yaw_deg=float(camera_params.get("yaw_deg", 0.0)),
            roll_deg=float(camera_params.get("roll_deg", 0.0)),
        )

        # カメラ位置（フロアマップ座標）
        self.camera_position_px = (
            float(camera_params.get("position_x_px", 1200.0)),
            float(camera_params.get("position_y_px", 800.0)),
        )

        # FloorMap
        self.floormap_config = FloorMapConfig(
            width_px=int(floormap_config.get("image_width", 1878)),
            height_px=int(floormap_config.get("image_height", 1369)),
            scale_x_mm_per_px=float(floormap_config.get("image_x_mm_per_pixel", 28.1926)),
            scale_y_mm_per_px=float(floormap_config.get("image_y_mm_per_pixel", 28.2414)),
        )

        # InteractiveCalibrator
        self.interactive_calibrator = InteractiveCalibrator(
            self.intrinsics,
            self.extrinsics,
            self.floormap_config.scale_x_mm_per_px,
            self.floormap_config.scale_y_mm_per_px,
        )

        # 対応点ファイルを読み込み
        self._load_correspondences()

    def _load_correspondences(self) -> None:
        """対応点ファイルを読み込む。"""
        calibration_config = self.config.get("calibration", {})
        correspondence_file = calibration_config.get("correspondence_file", "")

        if correspondence_file and Path(correspondence_file).exists():
            try:
                self.correspondence_data = load_correspondence_file(correspondence_file)
                logger.info(
                    f"Loaded {self.correspondence_data.num_correspondences} correspondences from {correspondence_file}"
                )
            except Exception as e:
                logger.warning(f"Failed to load correspondence file: {e}")
                self.correspondence_data = None
        else:
            logger.info("No correspondence file found")

    def show_current_params(self) -> None:
        """現在のパラメータを表示。"""
        params = self.interactive_calibrator.get_current_params()

        print("\n" + "=" * 60)
        print("現在のカメラパラメータ")
        print("=" * 60)
        print(f"  高さ (height_m):     {params['height_m']:.3f} m")
        print(f"  俯角 (pitch_deg):    {params['pitch_deg']:.2f} deg")
        print(f"  方位角 (yaw_deg):    {params['yaw_deg']:.2f} deg")
        print(f"  回転角 (roll_deg):   {params['roll_deg']:.2f} deg")
        print(f"  カメラX (cam_x):     {params['camera_x_m']:.3f} m")
        print(f"  カメラY (cam_y):     {params['camera_y_m']:.3f} m")
        print(f"  フロアマップ位置:   ({self.camera_position_px[0]:.1f}, {self.camera_position_px[1]:.1f}) px")
        print("=" * 60)

        # 再投影誤差を計算
        if self.correspondence_data:
            error = self._compute_current_error()
            print(f"  再投影誤差 (RMSE):   {error:.2f} pixels")
            print("=" * 60)

    def _compute_current_error(self) -> float:
        """現在の設定での再投影誤差を計算。"""
        if not self.correspondence_data:
            return float("inf")

        point_pairs = self.correspondence_data.get_foot_points()
        return self.interactive_calibrator.get_current_error(point_pairs, self.camera_position_px)

    def adjust_parameter(self, param: str, delta: float) -> None:
        """パラメータを調整。

        Args:
            param: パラメータ名 (height, pitch, yaw, roll, cam_x, cam_y)
            delta: 増減量
        """
        self.interactive_calibrator.adjust_parameter(param, delta)
        self.extrinsics = self.interactive_calibrator.current_extrinsics

        print(f"\n{param} を {delta:+.3f} 調整しました")
        self.show_current_params()

    def set_parameter(self, param: str, value: float) -> None:
        """パラメータを直接設定。

        Args:
            param: パラメータ名
            value: 新しい値
        """
        self.interactive_calibrator.set_params(**{param: value})
        self.extrinsics = self.interactive_calibrator.current_extrinsics

        print(f"\n{param} を {value:.3f} に設定しました")
        self.show_current_params()

    def auto_calibrate(self) -> None:
        """対応点から自動キャリブレーション。"""
        if not self.correspondence_data:
            print("\nエラー: 対応点データがありません")
            return

        print("\n自動キャリブレーションを実行中...")

        # 現在のパラメータを表示
        print("  初期パラメータ:")
        print(f"    高さ: {self.extrinsics.camera_position_world[2]:.2f} m")
        pose = self.extrinsics.to_pose_params()
        print(f"    俯角: {pose['pitch_deg']:.1f} deg")
        print(f"    方位角: {pose['yaw_deg']:.1f} deg")
        print(f"    カメラ位置: {self.camera_position_px}")

        calibrator = CorrespondenceCalibrator(
            self.intrinsics,
            self.floormap_config.scale_x_mm_per_px,
            self.floormap_config.scale_y_mm_per_px,
        )

        try:
            result = calibrator.calibrate_from_correspondences(
                self.correspondence_data,
                initial_guess=self.extrinsics,
                camera_position_px=self.camera_position_px,
            )

            self.extrinsics = result.extrinsics
            self.interactive_calibrator = InteractiveCalibrator(
                self.intrinsics,
                self.extrinsics,
                self.floormap_config.scale_x_mm_per_px,
                self.floormap_config.scale_y_mm_per_px,
            )

            print("\n" + "=" * 60)
            print("自動キャリブレーション結果")
            print("=" * 60)
            print(f"  収束: {result.optimization_converged}")
            print(f"  反復回数: {result.iterations}")
            print(f"  再投影誤差 (RMSE): {result.reprojection_error:.2f} pixels")
            print(f"  インライアー比率: {result.inlier_ratio:.1%}")
            print("=" * 60)

            self.show_current_params()

        except ValueError as e:
            print(f"\nエラー: {e}")
            print("\n対応点データを確認してください:")
            foot_pts = self.correspondence_data.get_foot_points()
            print(f"  対応点数: {len(foot_pts)}")
            if foot_pts:
                for i, (img_pt, fm_pt) in enumerate(foot_pts[:3]):
                    print(f"  [{i}] 画像: {img_pt}, フロアマップ: {fm_pt}")
        except Exception as e:
            print(f"\nエラー: 自動キャリブレーションに失敗しました: {e}")
            import traceback

            traceback.print_exc()

    def test_transform(self, u: float, v: float) -> None:
        """座標変換をテスト。

        Args:
            u: 画像X座標
            v: 画像Y座標
        """
        ray_caster = RayCaster(self.intrinsics, self.extrinsics)
        floormap_transformer = FloorMapTransformer(self.floormap_config, self.camera_position_px)
        transformer = UnifiedTransformer(ray_caster, floormap_transformer)

        result = transformer.transform_pixel((u, v))

        print("\n" + "=" * 60)
        print(f"座標変換テスト: 画像座標 ({u:.1f}, {v:.1f})")
        print("=" * 60)

        if result.is_valid:
            print(f"  World座標:      ({result.world_coords_m[0]:.3f}, {result.world_coords_m[1]:.3f}) m")
            print(f"  フロアマップ座標: ({result.floor_coords_px[0]:.1f}, {result.floor_coords_px[1]:.1f}) px")
            print(f"  範囲内:         {'Yes' if result.is_within_bounds else 'No'}")
        else:
            print(f"  エラー: {result.error_reason}")

        print("=" * 60)

    def save_params(self, output_path: str | None = None) -> None:
        """現在のパラメータを保存。

        Args:
            output_path: 出力パス（None の場合は標準出力）
        """
        params = self.interactive_calibrator.get_current_params()

        output = {
            "camera_params": {
                "height_m": params["height_m"],
                "pitch_deg": params["pitch_deg"],
                "yaw_deg": params["yaw_deg"],
                "roll_deg": params["roll_deg"],
                "camera_x_m": params["camera_x_m"],
                "camera_y_m": params["camera_y_m"],
                "position_x_px": self.camera_position_px[0],
                "position_y_px": self.camera_position_px[1],
                "focal_length_x": self.intrinsics.fx,
                "focal_length_y": self.intrinsics.fy,
                "center_x": self.intrinsics.cx,
                "center_y": self.intrinsics.cy,
            }
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nパラメータを {output_path} に保存しました")
        else:
            print("\n" + json.dumps(output, indent=2, ensure_ascii=False))

    def run_interactive(self) -> None:
        """対話モードを実行。"""
        print("\n" + "=" * 60)
        print("カメラキャリブレーション調整ツール")
        print("=" * 60)
        print("コマンド:")
        print("  show            - 現在のパラメータを表示")
        print("  adj <param> <delta> - パラメータを調整")
        print("                     param: height, pitch, yaw, roll, cam_x, cam_y")
        print("  set <param> <value> - パラメータを設定")
        print("  auto            - 自動キャリブレーション")
        print("  test <u> <v>    - 座標変換テスト")
        print("  save [path]     - パラメータを保存")
        print("  quit            - 終了")
        print("=" * 60)

        self.show_current_params()

        while True:
            try:
                cmd = input("\n> ").strip()
                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0].lower()

                if command == "quit" or command == "exit" or command == "q":
                    break

                if command == "show":
                    self.show_current_params()

                elif command == "adj" and len(parts) >= 3:
                    param = parts[1]
                    delta = float(parts[2])
                    self.adjust_parameter(param, delta)

                elif command == "set" and len(parts) >= 3:
                    param = parts[1]
                    value = float(parts[2])
                    self.set_parameter(param, value)

                elif command == "auto":
                    self.auto_calibrate()

                elif command == "test" and len(parts) >= 3:
                    u = float(parts[1])
                    v = float(parts[2])
                    self.test_transform(u, v)

                elif command == "save":
                    path = parts[1] if len(parts) > 1 else None
                    self.save_params(path)

                else:
                    print("不明なコマンドです。'show' でヘルプを確認してください。")

            except KeyboardInterrupt:
                print("\n終了します")
                break
            except Exception as e:
                print(f"エラー: {e}")


def main():
    """メイン関数。"""
    parser = argparse.ArgumentParser(description="カメラキャリブレーション調整ツール")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自動キャリブレーションを実行して終了",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="結果の出力パス",
    )

    args = parser.parse_args()

    tool = CalibrationTool(args.config)

    if args.auto:
        tool.auto_calibrate()
        if args.output:
            tool.save_params(args.output)
    else:
        tool.run_interactive()


if __name__ == "__main__":
    main()
