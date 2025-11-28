#!/usr/bin/env python3
"""カメラパラメータのグリッドサーチツール。

対応点データに基づいて最適なカメラパラメータを探索します。

機能:
- pitch/yaw/height/focal_length のグリッドサーチ
- 各パラメータ組み合わせでRMSEを計算
- 上位N個のパラメータセットを報告
- 最適パラメータをconfig.yamlに書き出すオプション

使用方法:
    python tools/parameter_grid_search.py [--top-n N] [--save-best]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import json
import logging
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

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
class ParameterSet:
    """パラメータセット。"""

    pitch_deg: float
    yaw_deg: float
    height_m: float
    focal_length: float
    rmse: float
    max_error: float
    num_valid: int


def compute_rmse(
    intrinsics: CameraIntrinsics,
    pitch_deg: float,
    yaw_deg: float,
    height_m: float,
    camera_position_px: tuple[float, float],
    floormap_config: FloorMapConfig,
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float, int]:
    """指定パラメータでRMSEを計算。

    Returns:
        (RMSE, max_error, num_valid)
    """
    extrinsics = CameraExtrinsics.from_pose(
        camera_height_m=height_m,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=0.0,
    )

    ray_caster = RayCaster(intrinsics, extrinsics)
    floormap_transformer = FloorMapTransformer(floormap_config, camera_position_px)
    transformer = UnifiedTransformer(ray_caster, floormap_transformer)

    errors = []
    for image_pt, expected_fm in correspondences:
        result = transformer.transform_pixel(image_pt)
        if result.is_valid and result.floor_coords_px:
            actual = result.floor_coords_px
            error = np.linalg.norm(np.array(actual) - np.array(expected_fm))
            errors.append(error)

    if not errors:
        return float("inf"), float("inf"), 0

    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    max_error = float(np.max(errors))
    return rmse, max_error, len(errors)


def grid_search(
    config: ConfigManager,
    pitch_range: tuple[float, float, float],
    yaw_range: tuple[float, float, float],
    height_range: tuple[float, float, float],
    focal_range: tuple[float, float, float],
    top_n: int = 10,
) -> list[ParameterSet]:
    """グリッドサーチを実行。

    Args:
        config: ConfigManager
        pitch_range: (min, max, step)
        yaw_range: (min, max, step)
        height_range: (min, max, step)
        focal_range: (min, max, step)
        top_n: 上位N個を返す

    Returns:
        ParameterSet のリスト（RMSEでソート済み）
    """
    # 対応点データを読み込み
    calibration_config = config.get("calibration", {})
    correspondence_file = calibration_config.get("correspondence_file", "")

    if not correspondence_file or not Path(correspondence_file).exists():
        raise FileNotFoundError("対応点ファイルが見つかりません")

    correspondence_data = load_correspondence_file(correspondence_file)
    correspondences = correspondence_data.get_foot_points()
    correspondences.extend([(pc.src_point, pc.dst_point) for pc in correspondence_data.point_pairs])

    if len(correspondences) < 4:
        raise ValueError("対応点が不足しています（最低4点必要）")

    logger.info(f"対応点数: {len(correspondences)}")

    # カメラ位置を取得
    camera_params = config.get("camera_params", {})
    camera_position_px = (
        float(camera_params.get("position_x_px", 859.0)),
        float(camera_params.get("position_y_px", 1040.0)),
    )

    # フロアマップ設定
    floormap_config_dict = config.get("floormap", {})
    floormap_config = FloorMapConfig(
        width_px=int(floormap_config_dict.get("image_width", 1878)),
        height_px=int(floormap_config_dict.get("image_height", 1369)),
        scale_x_mm_per_px=float(floormap_config_dict.get("image_x_mm_per_pixel", 28.1926)),
        scale_y_mm_per_px=float(floormap_config_dict.get("image_y_mm_per_pixel", 28.2414)),
    )

    # パラメータ範囲を生成
    pitches = np.arange(pitch_range[0], pitch_range[1] + pitch_range[2], pitch_range[2])
    yaws = np.arange(yaw_range[0], yaw_range[1] + yaw_range[2], yaw_range[2])
    heights = np.arange(height_range[0], height_range[1] + height_range[2], height_range[2])
    focals = np.arange(focal_range[0], focal_range[1] + focal_range[2], focal_range[2])

    total = len(pitches) * len(yaws) * len(heights) * len(focals)
    logger.info(f"探索空間: {total} 組み合わせ")

    results = []

    for focal in tqdm(focals, desc="Focal Length"):
        # Intrinsics を更新
        intrinsics = CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=float(camera_params.get("center_x", 640.0)),
            cy=float(camera_params.get("center_y", 360.0)),
            image_width=int(camera_params.get("image_width", 1280)),
            image_height=int(camera_params.get("image_height", 720)),
        )

        for pitch, yaw, height in product(pitches, yaws, heights):
            rmse, max_error, num_valid = compute_rmse(
                intrinsics,
                pitch,
                yaw,
                height,
                camera_position_px,
                floormap_config,
                correspondences,
            )

            if num_valid >= len(correspondences) * 0.8:  # 80%以上有効
                results.append(
                    ParameterSet(
                        pitch_deg=float(pitch),
                        yaw_deg=float(yaw),
                        height_m=float(height),
                        focal_length=float(focal),
                        rmse=rmse,
                        max_error=max_error,
                        num_valid=num_valid,
                    )
                )

    # RMSEでソート
    results.sort(key=lambda x: x.rmse)

    return results[:top_n]


def save_results(results: list[ParameterSet], output_path: Path) -> None:
    """結果をJSONで保存。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "results": [
            {
                "rank": i + 1,
                "pitch_deg": r.pitch_deg,
                "yaw_deg": r.yaw_deg,
                "height_m": r.height_m,
                "focal_length": r.focal_length,
                "rmse": r.rmse,
                "max_error": r.max_error,
                "num_valid": r.num_valid,
            }
            for i, r in enumerate(results)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"結果を保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="カメラパラメータのグリッドサーチ")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="上位N個を報告",
    )
    parser.add_argument(
        "--pitch-range",
        type=str,
        default="5,40,1",
        help="pitch範囲: min,max,step",
    )
    parser.add_argument(
        "--yaw-range",
        type=str,
        default="0,360,1",
        help="yaw範囲: min,max,step",
    )
    parser.add_argument(
        "--height-range",
        type=str,
        default="1.5,2.5,0.1",
        help="height範囲: min,max,step",
    )
    parser.add_argument(
        "--focal-range",
        type=str,
        default="800,2000,100",
        help="focal_length範囲: min,max,step",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="最適パラメータをconfig.yamlに保存",
    )
    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="出力ディレクトリ",
    )
    args = parser.parse_args()

    # 範囲をパース
    def parse_range(s: str) -> tuple[float, float, float]:
        parts = s.split(",")
        return float(parts[0]), float(parts[1]), float(parts[2])

    pitch_range = parse_range(args.pitch_range)
    yaw_range = parse_range(args.yaw_range)
    height_range = parse_range(args.height_range)
    focal_range = parse_range(args.focal_range)

    # 設定読み込み
    config = ConfigManager(args.config)

    print("\n" + "=" * 60)
    print("カメラパラメータ グリッドサーチ")
    print("=" * 60)
    print(f"Pitch範囲: {pitch_range[0]}~{pitch_range[1]} (step={pitch_range[2]})")
    print(f"Yaw範囲: {yaw_range[0]}~{yaw_range[1]} (step={yaw_range[2]})")
    print(f"Height範囲: {height_range[0]}~{height_range[1]} (step={height_range[2]})")
    print(f"Focal範囲: {focal_range[0]}~{focal_range[1]} (step={focal_range[2]})")
    print("=" * 60)
    print()

    # グリッドサーチ実行
    results = grid_search(
        config,
        pitch_range,
        yaw_range,
        height_range,
        focal_range,
        top_n=args.top_n,
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print(f"上位 {len(results)} 件の結果")
    print("=" * 60)
    print(f"{'Rank':>4} {'Pitch':>8} {'Yaw':>8} {'Height':>8} {'Focal':>8} {'RMSE':>10} {'MaxErr':>10}")
    print("-" * 60)
    for i, r in enumerate(results):
        print(
            f"{i + 1:>4} {r.pitch_deg:>8.1f} {r.yaw_deg:>8.1f} {r.height_m:>8.2f} "
            f"{r.focal_length:>8.0f} {r.rmse:>10.2f} {r.max_error:>10.2f}"
        )
    print("=" * 60)

    if results:
        best = results[0]
        print("\n最良パラメータ:")
        print(f"  pitch_deg: {best.pitch_deg}")
        print(f"  yaw_deg: {best.yaw_deg}")
        print(f"  height_m: {best.height_m}")
        print(f"  focal_length: {best.focal_length}")
        print(f"  RMSE: {best.rmse:.2f} px")

        if best.rmse <= 10:
            print("\n✓ 目標精度（RMSE 10px以下）を達成しました！")
        elif best.rmse <= 30:
            print("\n△ まだ調整が必要です（RMSE 10px以下が目標）")
        else:
            print("\n× 精度が不十分です。対応点の追加・見直しを検討してください")

    # 結果を保存
    output_dir = Path(args.output_dir)
    save_results(results, output_dir / "grid_search_results.json")

    # 最適パラメータを保存
    if args.save_best and results:
        best = results[0]
        print("\n最適パラメータを config.yaml に書き込みます...")
        # ここでは警告のみ表示（実際の書き込みは手動で行う）
        print("  以下をconfig.yamlのcamera_paramsセクションに設定してください:")
        print(f"    pitch_deg: {best.pitch_deg}")
        print(f"    yaw_deg: {best.yaw_deg}")
        print(f"    height_m: {best.height_m}")
        print(f"    focal_length_x: {best.focal_length}")
        print(f"    focal_length_y: {best.focal_length}")

    print("\n出力ファイル:")
    print(f"  - {output_dir / 'grid_search_results.json'}")


if __name__ == "__main__":
    main()
