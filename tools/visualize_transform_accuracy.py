#!/usr/bin/env python3
"""座標変換精度の診断・可視化ツール。

現在のカメラパラメータで対応点を変換し、期待値との誤差を可視化します。

機能:
- カメラフレーム上の対応点（線分）を表示
- フロアマップ上の期待位置と実際の変換結果を表示
- 誤差ベクトル（矢印）を描画
- RMSE/最大誤差を計算・表示
- 精度レポートをJSONで出力

使用方法:
    python tools/visualize_transform_accuracy.py [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

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
class CorrespondenceError:
    """対応点ごとの誤差情報。"""

    index: int
    image_point: tuple[float, float]
    expected_floormap: tuple[float, float]
    actual_floormap: tuple[float, float] | None
    error_distance: float | None
    error_vector: tuple[float, float] | None
    is_valid: bool


@dataclass
class AccuracyReport:
    """精度レポート。"""

    timestamp: str
    num_correspondences: int
    num_valid: int
    rmse: float | None
    max_error: float | None
    min_error: float | None
    mean_error: float | None
    std_error: float | None
    errors: list[CorrespondenceError]
    camera_params: dict


class TransformAccuracyVisualizer:
    """変換精度可視化クラス。"""

    def __init__(self, config: ConfigManager):
        """初期化。

        Args:
            config: ConfigManager インスタンス
        """
        self.config = config

        # カメラパラメータを読み込み
        camera_params = config.get("camera_params", {})
        floormap_config = config.get("floormap", {})

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

        # 変換器を作成
        ray_caster = RayCaster(self.intrinsics, self.extrinsics)
        floormap_transformer = FloorMapTransformer(self.floormap_config, self.camera_position_px)
        self.transformer = UnifiedTransformer(ray_caster, floormap_transformer)

        # 対応点データを読み込み
        calibration_config = config.get("calibration", {})
        correspondence_file = calibration_config.get("correspondence_file", "")

        if correspondence_file and Path(correspondence_file).exists():
            self.correspondence_data = load_correspondence_file(correspondence_file)
            logger.info(f"対応点データを読み込みました: {self.correspondence_data.num_correspondences} 点")
        else:
            self.correspondence_data = None
            logger.warning("対応点データが見つかりません")

        # フロアマップ画像を読み込み
        floormap_path = floormap_config.get("image_path", "data/floormap.png")
        if Path(floormap_path).exists():
            self.floormap_image = cv2.imread(str(floormap_path))
            logger.info(f"フロアマップ画像を読み込みました: {floormap_path}")
        else:
            self.floormap_image = None
            logger.warning(f"フロアマップ画像が見つかりません: {floormap_path}")

        # リファレンス画像を読み込み
        if self.correspondence_data and self.correspondence_data.metadata:
            ref_image_path = self.correspondence_data.metadata.get("reference_image", "")
            if ref_image_path and Path(ref_image_path).exists():
                self.reference_image = cv2.imread(str(ref_image_path))
                logger.info(f"リファレンス画像を読み込みました: {ref_image_path}")
            else:
                self.reference_image = None
        else:
            self.reference_image = None

    def compute_errors(self) -> list[CorrespondenceError]:
        """全対応点の誤差を計算。

        Returns:
            CorrespondenceError のリスト
        """
        if not self.correspondence_data:
            return []

        errors = []
        foot_points = self.correspondence_data.get_foot_points()

        for i, (image_pt, expected_fm) in enumerate(foot_points):
            result = self.transformer.transform_pixel(image_pt)

            if result.is_valid and result.floor_coords_px:
                actual_fm = result.floor_coords_px
                error_vec = (
                    actual_fm[0] - expected_fm[0],
                    actual_fm[1] - expected_fm[1],
                )
                error_dist = float(np.linalg.norm(error_vec))

                errors.append(
                    CorrespondenceError(
                        index=i,
                        image_point=image_pt,
                        expected_floormap=expected_fm,
                        actual_floormap=actual_fm,
                        error_distance=error_dist,
                        error_vector=error_vec,
                        is_valid=True,
                    )
                )
            else:
                errors.append(
                    CorrespondenceError(
                        index=i,
                        image_point=image_pt,
                        expected_floormap=expected_fm,
                        actual_floormap=None,
                        error_distance=None,
                        error_vector=None,
                        is_valid=False,
                    )
                )

        return errors

    def generate_report(self, errors: list[CorrespondenceError]) -> AccuracyReport:
        """精度レポートを生成。

        Args:
            errors: CorrespondenceError のリスト

        Returns:
            AccuracyReport インスタンス
        """
        valid_errors = [e for e in errors if e.is_valid and e.error_distance is not None]
        distances = [e.error_distance for e in valid_errors]

        if distances:
            rmse = float(np.sqrt(np.mean(np.array(distances) ** 2)))
            max_error = float(np.max(distances))
            min_error = float(np.min(distances))
            mean_error = float(np.mean(distances))
            std_error = float(np.std(distances))
        else:
            rmse = max_error = min_error = mean_error = std_error = None

        # カメラパラメータを取得
        pose = self.extrinsics.to_pose_params()
        camera_params = {
            "height_m": float(self.extrinsics.camera_position_world[2]),
            "pitch_deg": pose["pitch_deg"],
            "yaw_deg": pose["yaw_deg"],
            "roll_deg": pose["roll_deg"],
            "position_x_px": self.camera_position_px[0],
            "position_y_px": self.camera_position_px[1],
            "focal_length_x": self.intrinsics.fx,
            "focal_length_y": self.intrinsics.fy,
        }

        return AccuracyReport(
            timestamp=datetime.now().isoformat(),
            num_correspondences=len(errors),
            num_valid=len(valid_errors),
            rmse=rmse,
            max_error=max_error,
            min_error=min_error,
            mean_error=mean_error,
            std_error=std_error,
            errors=errors,
            camera_params=camera_params,
        )

    def visualize(self, errors: list[CorrespondenceError], report: AccuracyReport) -> np.ndarray:
        """可視化画像を生成。

        Args:
            errors: CorrespondenceError のリスト
            report: AccuracyReport インスタンス

        Returns:
            可視化画像（numpy配列）
        """
        # 左: カメラフレーム、右: フロアマップ
        if self.reference_image is not None:
            camera_img = self.reference_image.copy()
        else:
            camera_img = np.zeros(
                (self.intrinsics.image_height, self.intrinsics.image_width, 3),
                dtype=np.uint8,
            )
            camera_img[:] = (50, 50, 50)

        if self.floormap_image is not None:
            floormap_img = self.floormap_image.copy()
        else:
            floormap_img = np.zeros(
                (self.floormap_config.height_px, self.floormap_config.width_px, 3),
                dtype=np.uint8,
            )
            floormap_img[:] = (50, 50, 50)

        # カメラフレーム上に対応点（線分）を描画
        if self.correspondence_data:
            for lpc in self.correspondence_data.line_point_pairs:
                p1 = tuple(map(int, lpc.src_line[0]))
                p2 = tuple(map(int, lpc.src_line[1]))
                cv2.line(camera_img, p1, p2, (0, 255, 0), 2)
                # 足元点（下端）を強調
                foot = tuple(map(int, lpc.line_bottom))
                cv2.circle(camera_img, foot, 5, (0, 255, 255), -1)

        # フロアマップ上に期待位置と実際の位置を描画
        for err in errors:
            expected = tuple(map(int, err.expected_floormap))

            # 期待位置（緑）
            cv2.circle(floormap_img, expected, 8, (0, 255, 0), 2)
            cv2.putText(
                floormap_img,
                str(err.index),
                (expected[0] + 10, expected[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            if err.is_valid and err.actual_floormap:
                actual = tuple(map(int, err.actual_floormap))

                # 実際の位置（赤）
                cv2.circle(floormap_img, actual, 6, (0, 0, 255), -1)

                # 誤差ベクトル（矢印）
                cv2.arrowedLine(
                    floormap_img,
                    expected,
                    actual,
                    (255, 0, 255),
                    2,
                    tipLength=0.3,
                )

                # 誤差表示
                if err.error_distance:
                    cv2.putText(
                        floormap_img,
                        f"{err.error_distance:.1f}px",
                        (actual[0] + 10, actual[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

        # カメラ位置を描画
        cam_pos = tuple(map(int, self.camera_position_px))
        cv2.drawMarker(
            floormap_img,
            cam_pos,
            (255, 0, 0),
            cv2.MARKER_TRIANGLE_UP,
            20,
            2,
        )
        cv2.putText(
            floormap_img,
            "Camera",
            (cam_pos[0] + 15, cam_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # ゾーンを描画
        zones = self.config.get("zones", [])
        for zone in zones:
            polygon = np.array(zone.get("polygon", []), dtype=np.int32)
            if len(polygon) > 2:
                cv2.polylines(floormap_img, [polygon], True, (255, 255, 0), 2)
                # ゾーン名
                center = polygon.mean(axis=0).astype(int)
                cv2.putText(
                    floormap_img,
                    zone.get("name", zone.get("id", "")),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

        # 画像をリサイズして結合
        target_height = 800
        camera_scale = target_height / camera_img.shape[0]
        floormap_scale = target_height / floormap_img.shape[0]

        camera_resized = cv2.resize(camera_img, None, fx=camera_scale, fy=camera_scale)
        floormap_resized = cv2.resize(floormap_img, None, fx=floormap_scale, fy=floormap_scale)

        # 統計情報パネルを作成
        stats_height = target_height
        stats_width = 300
        stats_panel = np.zeros((stats_height, stats_width, 3), dtype=np.uint8)
        stats_panel[:] = (30, 30, 30)

        y = 30
        cv2.putText(
            stats_panel,
            "Transform Accuracy",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y += 40
        cv2.putText(
            stats_panel,
            f"Correspondences: {report.num_correspondences}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 25
        cv2.putText(
            stats_panel,
            f"Valid: {report.num_valid}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 40
        if report.rmse is not None:
            color = (0, 255, 0) if report.rmse <= 10 else (0, 165, 255) if report.rmse <= 30 else (0, 0, 255)
            cv2.putText(
                stats_panel,
                f"RMSE: {report.rmse:.2f} px",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        y += 30
        if report.max_error is not None:
            color = (0, 255, 0) if report.max_error <= 30 else (0, 0, 255)
            cv2.putText(
                stats_panel,
                f"Max Error: {report.max_error:.2f} px",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        y += 25
        if report.mean_error is not None:
            cv2.putText(
                stats_panel,
                f"Mean Error: {report.mean_error:.2f} px",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        y += 25
        if report.std_error is not None:
            cv2.putText(
                stats_panel,
                f"Std Error: {report.std_error:.2f} px",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        y += 50
        cv2.putText(
            stats_panel,
            "Camera Params:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        params = report.camera_params
        y += 25
        cv2.putText(
            stats_panel,
            f"Height: {params['height_m']:.2f} m",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 25
        cv2.putText(
            stats_panel,
            f"Pitch: {params['pitch_deg']:.1f} deg",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 25
        cv2.putText(
            stats_panel,
            f"Yaw: {params['yaw_deg']:.1f} deg",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 25
        cv2.putText(
            stats_panel,
            f"Focal: {params['focal_length_x']:.0f} px",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 50
        cv2.putText(
            stats_panel,
            "Legend:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        y += 25
        cv2.circle(stats_panel, (20, y), 5, (0, 255, 0), 2)
        cv2.putText(
            stats_panel,
            "Expected",
            (35, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        y += 20
        cv2.circle(stats_panel, (20, y), 5, (0, 0, 255), -1)
        cv2.putText(
            stats_panel,
            "Actual",
            (35, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        y += 20
        cv2.arrowedLine(stats_panel, (10, y), (30, y), (255, 0, 255), 2)
        cv2.putText(
            stats_panel,
            "Error Vector",
            (35, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
        )

        # 画像を結合
        combined = np.hstack([camera_resized, floormap_resized, stats_panel])

        return combined

    def save_report(self, report: AccuracyReport, output_path: Path) -> None:
        """精度レポートをJSONで保存。

        Args:
            report: AccuracyReport インスタンス
            output_path: 出力パス
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": report.timestamp,
            "num_correspondences": report.num_correspondences,
            "num_valid": report.num_valid,
            "statistics": {
                "rmse": report.rmse,
                "max_error": report.max_error,
                "min_error": report.min_error,
                "mean_error": report.mean_error,
                "std_error": report.std_error,
            },
            "camera_params": report.camera_params,
            "errors": [
                {
                    "index": e.index,
                    "image_point": list(e.image_point),
                    "expected_floormap": list(e.expected_floormap),
                    "actual_floormap": list(e.actual_floormap) if e.actual_floormap else None,
                    "error_distance": e.error_distance,
                    "error_vector": list(e.error_vector) if e.error_vector else None,
                    "is_valid": e.is_valid,
                }
                for e in report.errors
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"精度レポートを保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="座標変換精度の診断・可視化")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="出力ディレクトリ",
    )
    args = parser.parse_args()

    # 設定読み込み
    config = ConfigManager(args.config)

    # 可視化ツールを初期化
    visualizer = TransformAccuracyVisualizer(config)

    if not visualizer.correspondence_data:
        logger.error("対応点データがありません。終了します。")
        sys.exit(1)

    # 誤差を計算
    errors = visualizer.compute_errors()

    # レポートを生成
    report = visualizer.generate_report(errors)

    # コンソールに結果を表示
    print("\n" + "=" * 60)
    print("座標変換精度レポート")
    print("=" * 60)
    print(f"対応点数: {report.num_correspondences}")
    print(f"有効な変換: {report.num_valid}")
    print()
    if report.rmse is not None:
        status = "OK" if report.rmse <= 10 else "要調整" if report.rmse <= 30 else "精度不足"
        print(f"RMSE: {report.rmse:.2f} px ({status})")
        print(f"最大誤差: {report.max_error:.2f} px")
        print(f"最小誤差: {report.min_error:.2f} px")
        print(f"平均誤差: {report.mean_error:.2f} px")
        print(f"標準偏差: {report.std_error:.2f} px")
    print()
    print("各対応点の誤差:")
    for e in errors:
        if e.is_valid:
            print(f"  [{e.index}] 画像{e.image_point} → 誤差: {e.error_distance:.1f} px")
        else:
            print(f"  [{e.index}] 画像{e.image_point} → 変換失敗")
    print("=" * 60)

    # 出力
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # レポートを保存
    report_path = output_dir / "transform_accuracy_report.json"
    visualizer.save_report(report, report_path)

    # 可視化画像を生成・保存
    vis_image = visualizer.visualize(errors, report)
    vis_path = output_dir / "transform_accuracy_visualization.png"
    cv2.imwrite(str(vis_path), vis_image)
    logger.info(f"可視化画像を保存しました: {vis_path}")

    print("\n出力ファイル:")
    print(f"  - レポート: {report_path}")
    print(f"  - 可視化画像: {vis_path}")
    print()
    print("ユーザータスク:")
    print("  1. 可視化画像を確認してください")
    print("  2. 誤差の傾向（どの方向にずれているか）を分析してください")
    print("  3. RMSEが10px以下になるようパラメータ調整が必要です")


if __name__ == "__main__":
    main()
