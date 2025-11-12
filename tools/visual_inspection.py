"""Visual inspection tool for calibration, tracking, and reprojection error visualization."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CameraCalibrator, ReprojectionErrorEvaluator
from src.config.config_manager import ConfigManager
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class VisualInspectionTool:
    """目視確認用ツール

    キャリブレーション結果、追跡結果、座標変換精度を可視化します。
    """

    def __init__(self, session_dir: str | Path):
        """VisualInspectionToolを初期化

        Args:
            session_dir: セッションディレクトリのパス
        """
        self.session_dir = Path(session_dir)
        if not self.session_dir.exists():
            raise FileNotFoundError(f"セッションディレクトリが見つかりません: {session_dir}")

        logger.info(f"VisualInspectionTool initialized: {self.session_dir}")

    def visualize_calibration(
        self,
        chessboard_images: list[str | Path],
        output_path: str | Path | None = None,
    ) -> None:
        """キャリブレーション結果を可視化

        Args:
            chessboard_images: チェスボード画像のパスリスト
            output_path: 出力パス（オプション）
        """
        calibrator = CameraCalibrator()
        calibrator.calibrate_from_images(chessboard_images)

        # 検出されたコーナーを可視化
        for img_path in chessboard_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, calibrator.chessboard_size, None)

            if ret:
                # コーナーを描画
                cv2.drawChessboardCorners(img, calibrator.chessboard_size, corners, ret)

                # 保存
                if output_path:
                    output_file = Path(output_path) / f"calibration_{Path(img_path).stem}.jpg"
                    cv2.imwrite(str(output_file), img)
                    logger.info(f"Calibration visualization saved: {output_file}")

        logger.info("Calibration visualization completed")

    def visualize_tracking(
        self,
        tracks_data: list[dict],
        floormap_path: str | Path,
        output_path: str | Path | None = None,
    ) -> None:
        """追跡結果を可視化

        Args:
            tracks_data: トラックデータのリスト
            floormap_path: フロアマップ画像のパス
            output_path: 出力パス（オプション）
        """
        floormap = cv2.imread(str(floormap_path))
        if floormap is None:
            raise ValueError(f"フロアマップ画像を読み込めません: {floormap_path}")

        # 各トラックの軌跡を描画
        for track_data in tracks_data:
            track_id = track_data.get("track_id", 0)
            trajectory = track_data.get("trajectory", [])

            if len(trajectory) < 2:
                continue

            # 色を生成
            hue = (track_id * 137) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

            # 軌跡線を描画
            for i in range(len(trajectory) - 1):
                pt1 = trajectory[i]
                pt2 = trajectory[i + 1]
                x1, y1 = int(pt1["x"]), int(pt1["y"])
                x2, y2 = int(pt2["x"]), int(pt2["y"])
                cv2.line(floormap, (x1, y1), (x2, y2), color, 2)

            # 現在位置を描画
            if trajectory:
                last_pt = trajectory[-1]
                x, y = int(last_pt["x"]), int(last_pt["y"])
                cv2.circle(floormap, (x, y), 5, color, -1)
                cv2.putText(
                    floormap,
                    f"ID:{track_id}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # 保存
        if output_path:
            output_file = Path(output_path) / "tracking_visualization.jpg"
            cv2.imwrite(str(output_file), floormap)
            logger.info(f"Tracking visualization saved: {output_file}")

    def visualize_reprojection_error(
        self,
        src_points: list[tuple[float, float]],
        dst_points: list[tuple[float, float]],
        homography_matrix: np.ndarray,
        image_shape: tuple[int, int],
        output_path: str | Path | None = None,
    ) -> None:
        """再投影誤差を可視化

        Args:
            src_points: 変換元の点のリスト
            dst_points: 変換先の点のリスト
            homography_matrix: ホモグラフィ変換行列
            image_shape: 画像形状 (height, width)
            output_path: 出力パス（オプション）
        """
        evaluator = ReprojectionErrorEvaluator()
        error_map = evaluator.create_error_map(src_points, dst_points, homography_matrix, image_shape)

        # 誤差マップを可視化
        error_map_normalized = (error_map / (error_map.max() + 1e-8) * 255).astype(np.uint8)
        error_colored = cv2.applyColorMap(error_map_normalized, cv2.COLORMAP_JET)

        # 保存
        if output_path:
            output_file = Path(output_path) / "reprojection_error_map.jpg"
            cv2.imwrite(str(output_file), error_colored)
            logger.info(f"Reprojection error map saved: {output_file}")

        # 評価結果を表示
        result = evaluator.evaluate_homography(src_points, dst_points, homography_matrix)
        logger.info(f"Reprojection error - Mean: {result['mean_error']:.2f}px, Max: {result['max_error']:.2f}px")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Visual inspection tool for calibration and tracking")
    parser.add_argument("--mode", choices=["calibration", "tracking", "reprojection"], required=True)
    parser.add_argument("--session", type=str, required=True, help="Session directory path")
    parser.add_argument("--output", type=str, help="Output directory path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    # ログ設定
    setup_logging(debug_mode=False)

    # セッションディレクトリを確認
    session_dir = Path(args.session)
    if not session_dir.exists():
        logger.error(f"セッションディレクトリが見つかりません: {session_dir}")
        return

    tool = VisualInspectionTool(session_dir)

    # モードに応じて処理
    if args.mode == "calibration":
        # キャリブレーション画像を探す
        chessboard_images = list(session_dir.glob("calibration_*.jpg"))
        if not chessboard_images:
            logger.warning("キャリブレーション画像が見つかりません")
            return

        output_dir = Path(args.output) if args.output else session_dir / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        tool.visualize_calibration(chessboard_images, output_dir)

    elif args.mode == "tracking":
        # トラックデータを読み込む
        tracks_file = session_dir / "tracks.json"
        if not tracks_file.exists():
            logger.warning(f"トラックデータファイルが見つかりません: {tracks_file}")
            return

        with open(tracks_file, encoding="utf-8") as f:
            tracks_data = json.load(f).get("tracks", [])

        # フロアマップを読み込む
        config = ConfigManager(args.config)
        floormap_path = config.get("floormap.image_path")

        output_dir = Path(args.output) if args.output else session_dir / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        tool.visualize_tracking(tracks_data, floormap_path, output_dir)

    elif args.mode == "reprojection":
        # 対応点データを読み込む
        points_file = session_dir / "correspondence_points.json"
        if not points_file.exists():
            logger.warning(f"対応点データファイルが見つかりません: {points_file}")
            return

        with open(points_file, encoding="utf-8") as f:
            data = json.load(f)

        src_points = [tuple(pt) for pt in data["src_points"]]
        dst_points = [tuple(pt) for pt in data["dst_points"]]
        homography_matrix = np.array(data["homography_matrix"])

        config = ConfigManager(args.config)
        image_shape = (
            config.get("floormap.image_height"),
            config.get("floormap.image_width"),
        )

        output_dir = Path(args.output) if args.output else session_dir / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        tool.visualize_reprojection_error(src_points, dst_points, homography_matrix, image_shape, output_dir)


if __name__ == "__main__":
    main()
