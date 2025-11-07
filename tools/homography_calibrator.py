"""Interactive homography calibration tool.

This utility helps collect correspondence points between a camera reference
frame and the floor map, compute a homography matrix, and optionally update
`config.yaml`. Point pairs and computed matrices are saved under
`output/calibration/` for traceability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager  # noqa: E402
from src.utils import setup_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass
class PointCollector:
    """Stores and renders points for a single image."""

    image: np.ndarray
    window_name: str
    color: tuple[int, int, int]
    points: list[tuple[int, int]]

    def __init__(self, image: np.ndarray, window_name: str, color: tuple[int, int, int]):
        self.image = image
        self.window_name = window_name
        self.color = color
        self.points = []

    def add_point(self, x: int, y: int) -> None:
        self.points.append((x, y))

    def pop_last(self) -> None:
        if self.points:
            self.points.pop()

    def clear(self) -> None:
        self.points.clear()

    def render(self, status_text: str) -> np.ndarray:
        canvas = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, (x, y) in enumerate(self.points, start=1):
            cv2.circle(canvas, (x, y), 6, self.color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(idx),
                (x + 8, y - 8),
                font,
                0.45,
                self.color,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, status_text, (10, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラ画像とフロアマップの対応点からホモグラフィを計算します。")

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルのパス (default: config.yaml)",
    )

    parser.add_argument(
        "--reference-image",
        required=True,
        help="カメラ参照画像のパス (例: tools/export_reference_frame.py の出力)",
    )

    parser.add_argument(
        "--floormap-image",
        help="フロアマップ画像のパス。指定しない場合は config から取得します。",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="ホモグラフィ計算に必要な最小対応点数 (default: 4)",
    )

    parser.add_argument(
        "--method",
        choices=["default", "ransac", "lmeds"],
        default="ransac",
        help="cv2.findHomography の推定手法 (default: ransac)",
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="RANSAC の再投影誤差閾値 (default: 3.0)",
    )

    parser.add_argument(
        "--load-points",
        help="既存の対応点JSONを読み込んで計算のみ行います。",
    )

    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="校正データの出力ディレクトリ (default: output/calibration)",
    )

    parser.add_argument(
        "--update-config",
        action="store_true",
        help="計算した行列で config.yaml の homography.matrix を上書きします。",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを表示します。",
    )

    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return image


def collect_points_interactively(
    camera_image: np.ndarray,
    floormap_image: np.ndarray,
    min_points: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    LOGGER.info(
        "クリック操作で対応点を収集します。左クリックで点を追加、'u' で取り消し、'c' で全消去、's' または Enter で確定、'q' で終了します。"
    )

    camera_collector = PointCollector(camera_image, "Camera", (0, 180, 255))
    floormap_collector = PointCollector(floormap_image, "Floormap", (0, 255, 0))

    state = {"mode": "camera", "message": "Camera: 点を選択"}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if state["mode"] == "camera":
            camera_collector.add_point(x, y)
            state["mode"] = "floormap"
            state["message"] = "Floormap: 対応点を選択"
        else:
            floormap_collector.add_point(x, y)
            state["mode"] = "camera"
            state["message"] = "Camera: 点を選択"

    cv2.namedWindow(camera_collector.window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(floormap_collector.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(camera_collector.window_name, on_mouse)
    cv2.setMouseCallback(floormap_collector.window_name, on_mouse)

    try:
        while True:
            cam_status = f"{state['message']} | ペア数: {len(floormap_collector.points)}"
            map_status = f"{state['message']} | ペア数: {len(floormap_collector.points)}"

            cv2.imshow(
                camera_collector.window_name,
                camera_collector.render(cam_status),
            )
            cv2.imshow(
                floormap_collector.window_name,
                floormap_collector.render(map_status),
            )

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                raise KeyboardInterrupt("ユーザーにより中断されました")
            if key in (ord("s"), 13):
                if len(floormap_collector.points) >= min_points and len(camera_collector.points) == len(
                    floormap_collector.points
                ):
                    LOGGER.info("%d 組の対応点が確定しました。", len(floormap_collector.points))
                    break
                LOGGER.warning("対応点が不足しています (必要: %d)", min_points)
            if key == ord("u"):
                if state["mode"] == "floormap" and camera_collector.points:
                    camera_collector.pop_last()
                    state["mode"] = "camera"
                    state["message"] = "Camera: 点を選択"
                else:
                    camera_collector.pop_last()
                    floormap_collector.pop_last()
                    state["mode"] = "camera"
                    state["message"] = "Camera: 点を選択"
            if key == ord("c"):
                camera_collector.clear()
                floormap_collector.clear()
                state["mode"] = "camera"
                state["message"] = "Camera: 点を選択"
    finally:
        cv2.destroyWindow(camera_collector.window_name)
        cv2.destroyWindow(floormap_collector.window_name)

    if len(camera_collector.points) != len(floormap_collector.points):
        raise RuntimeError("対応点の数が一致しません。操作をやり直してください。")

    return camera_collector.points, floormap_collector.points


def compute_homography(
    camera_points: list[tuple[int, int]],
    floormap_points: list[tuple[int, int]],
    method: str,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(camera_points) < 4:
        raise ValueError("ホモグラフィ計算には4組以上の対応点が必要です。")

    src = np.array(camera_points, dtype=np.float32)
    dst = np.array(floormap_points, dtype=np.float32)

    method_flag = 0
    if method == "ransac":
        method_flag = cv2.RANSAC
    elif method == "lmeds":
        method_flag = cv2.LMEDS

    LOGGER.info("cv2.findHomography を実行します (method=%s)", method)
    H, mask = cv2.findHomography(src, dst, method_flag, ransac_threshold)
    if H is None or mask is None:
        raise RuntimeError("ホモグラフィの推定に失敗しました")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    inliers = int(mask.sum())

    metrics = {
        "rmse": rmse,
        "max_error": max_error,
        "inliers": inliers,
        "total_points": int(len(camera_points)),
    }
    LOGGER.info(
        "再投影RMSE=%.3f px, 最大誤差=%.3f px, インライア=%d/%d",
        rmse,
        max_error,
        inliers,
        len(camera_points),
    )

    return H, mask, metrics


def save_points_json(
    output_dir: Path,
    camera_points: list[tuple[int, int]],
    floormap_points: list[tuple[int, int]],
    reference_image: Path,
    floormap_image: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"points_{timestamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(),
        "reference_image": str(reference_image),
        "floormap_image": str(floormap_image),
        "camera_points": camera_points,
        "floormap_points": floormap_points,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("対応点JSONを保存しました: %s", path)
    return path


def save_homography_yaml(
    output_dir: Path,
    H: np.ndarray,
    metrics: dict,
    points_json: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"homography_{timestamp}.yaml"

    matrix_list = [[float(value) for value in row] for row in H.tolist()]
    payload = {
        "homography": {
            "matrix": matrix_list,
            "rmse": metrics["rmse"],
            "max_error": metrics["max_error"],
            "inliers": metrics["inliers"],
            "total_points": metrics["total_points"],
            "source_points_file": str(points_json),
        }
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    LOGGER.info("ホモグラフィ行列を保存しました: %s", path)
    return path


def update_config_homography(config_path: Path, matrix: np.ndarray, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"homography_{timestamp}.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        original = f.readlines()

    with backup_path.open("w", encoding="utf-8") as f:
        f.writelines(original)

    LOGGER.info("config.yaml をバックアップしました: %s", backup_path)

    matrix_line_index = None
    for idx, line in enumerate(original):
        if line.strip().startswith("matrix:") and line.lstrip().startswith("matrix:"):
            matrix_line_index = idx
            break

    if matrix_line_index is None:
        raise RuntimeError("config.yaml に homography.matrix の定義が見つかりません")

    insert_index = matrix_line_index + 1
    indent = ""
    if insert_index < len(original):
        indent = original[insert_index][: len(original[insert_index]) - len(original[insert_index].lstrip())]
    if not indent:
        indent = "    "

    end_index = insert_index
    while end_index < len(original) and original[end_index].lstrip().startswith("- ["):
        end_index += 1

    formatted_rows = []
    for row in matrix.tolist():
        values = ", ".join(f"{value:.10f}" for value in row)
        formatted_rows.append(f"{indent}- [{values}]\n")

    updated = original[:insert_index] + formatted_rows + original[end_index:]

    with config_path.open("w", encoding="utf-8") as f:
        f.writelines(updated)

    LOGGER.info("config.yaml の homography.matrix を更新しました。")


def load_points_from_json(
    path: Path,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_points = [tuple(map(int, p)) for p in data.get("camera_points", [])]
    floormap_points = [tuple(map(int, p)) for p in data.get("floormap_points", [])]

    if len(camera_points) != len(floormap_points):
        raise ValueError("JSON 内の対応点数が一致しません")

    return camera_points, floormap_points


def main() -> None:
    args = parse_args()
    setup_logging(debug_mode=args.verbose)

    config = ConfigManager(args.config)
    output_dir = Path(args.output_dir)
    reference_path = Path(args.reference_image)
    floormap_path = Path(args.floormap_image) if args.floormap_image else Path(config.get("floormap.image_path"))

    if not reference_path.exists():
        raise FileNotFoundError(f"参照画像が存在しません: {reference_path}")
    if not floormap_path.exists():
        raise FileNotFoundError(f"フロアマップ画像が存在しません: {floormap_path}")

    if args.load_points:
        camera_points, floormap_points = load_points_from_json(Path(args.load_points))
    else:
        camera_image = load_image(reference_path)
        floormap_image = load_image(floormap_path)
        camera_points, floormap_points = collect_points_interactively(
            camera_image,
            floormap_image,
            args.min_points,
        )

    points_json = save_points_json(output_dir, camera_points, floormap_points, reference_path, floormap_path)
    H, mask, metrics = compute_homography(camera_points, floormap_points, args.method, args.ransac_threshold)
    homography_yaml = save_homography_yaml(output_dir, H, metrics, points_json)

    if args.update_config:
        update_config_homography(Path(args.config), H, Path("config.backup"))

    LOGGER.info("校正が完了しました。")
    LOGGER.info("対応点: %s", points_json)
    LOGGER.info("ホモグラフィ: %s", homography_yaml)


if __name__ == "__main__":
    main()
