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

import cv2
import numpy as np
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.utils import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class LineSegment:
    """線分対応データクラス"""

    start_point: tuple[int, int]
    end_point: tuple[int, int]
    floormap_point: tuple[int, int]


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


@dataclass
class LineSegmentCollector:
    """線分を収集・可視化するクラス"""

    image: np.ndarray
    window_name: str
    line_color: tuple[int, int, int]
    start_color: tuple[int, int, int]
    end_color: tuple[int, int, int]
    line_segments: list[LineSegment]
    current_start: tuple[int, int] | None
    current_end: tuple[int, int] | None

    def __init__(
        self,
        image: np.ndarray,
        window_name: str,
        line_color: tuple[int, int, int] = (255, 0, 255),
        start_color: tuple[int, int, int] = (0, 255, 255),
        end_color: tuple[int, int, int] = (255, 255, 0),
    ):
        self.image = image
        self.window_name = window_name
        self.line_color = line_color
        self.start_color = start_color
        self.end_color = end_color
        self.line_segments = []
        self.current_start = None
        self.current_end = None

    def add_start_point(self, x: int, y: int) -> None:
        """線分の始点を設定"""
        self.current_start = (x, y)
        self.current_end = None

    def add_end_point(self, x: int, y: int) -> None:
        """線分の終点を設定"""
        if self.current_start is None:
            return
        self.current_end = (x, y)

    def complete_line_segment(self, floormap_point: tuple[int, int]) -> None:
        """線分を完成させる（フロアマップ上の対応点を設定）"""
        if self.current_start is not None and self.current_end is not None:
            self.line_segments.append(LineSegment(self.current_start, self.current_end, floormap_point))
            self.current_start = None
            self.current_end = None

    def pop_last(self) -> None:
        """最後の線分を削除"""
        if self.line_segments:
            self.line_segments.pop()
        elif self.current_end is not None:
            self.current_end = None
        elif self.current_start is not None:
            self.current_start = None

    def clear(self) -> None:
        """全ての線分をクリア"""
        self.line_segments.clear()
        self.current_start = None
        self.current_end = None

    def render(self, status_text: str) -> np.ndarray:
        """線分を可視化"""
        canvas = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 完成した線分を描画
        for idx, seg in enumerate(self.line_segments, start=1):
            # 線分を描画
            cv2.line(canvas, seg.start_point, seg.end_point, self.line_color, 3, cv2.LINE_AA)
            # 始点を描画
            cv2.circle(canvas, seg.start_point, 8, self.start_color, -1, cv2.LINE_AA)
            # 終点を描画
            cv2.circle(canvas, seg.end_point, 8, self.end_color, -1, cv2.LINE_AA)
            # 線分IDを表示
            mid_x = (seg.start_point[0] + seg.end_point[0]) // 2
            mid_y = (seg.start_point[1] + seg.end_point[1]) // 2
            cv2.putText(
                canvas,
                f"L{idx}",
                (mid_x + 10, mid_y),
                font,
                0.5,
                self.line_color,
                2,
                cv2.LINE_AA,
            )

        # 編集中の線分を描画
        if self.current_start is not None:
            cv2.circle(canvas, self.current_start, 8, self.start_color, -1, cv2.LINE_AA)
            if self.current_end is not None:
                cv2.line(canvas, self.current_start, self.current_end, self.line_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, self.current_end, 8, self.end_color, -1, cv2.LINE_AA)

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

    parser.add_argument(
        "--camera-id",
        default="cam01",
        help="カメラID (default: cam01)",
    )

    parser.add_argument(
        "--output-format",
        choices=["template", "legacy"],
        default="template",
        help="出力形式: 'template' (src_points/dst_points) または 'legacy' (camera_points/floormap_points) (default: template)",
    )

    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return image


def sample_points_from_line_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    num_samples: int = 3,
) -> list[tuple[int, int]]:
    """線分から等間隔で点をサンプリング

    Args:
        start: 線分の始点
        end: 線分の終点
        num_samples: サンプリング点数（デフォルト: 3 = 始点・中点・終点）

    Returns:
        サンプリングされた点のリスト
    """
    if num_samples < 2:
        num_samples = 2

    points = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.0
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        points.append((x, y))

    return points


def collect_correspondences_interactively(
    camera_image: np.ndarray,
    floormap_image: np.ndarray,
    min_points: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[LineSegment]]:
    """対応点と線分を対話的に収集

    Args:
        camera_image: カメラ画像
        floormap_image: フロアマップ画像
        min_points: 最小対応点数（点対応 + 線分対応×3）

    Returns:
        (camera_points, floormap_points, line_segments) のタプル
    """
    LOGGER.info(
        "クリック操作で対応点を収集します。\n"
        "操作:\n"
        "  'p': 点モードに切り替え\n"
        "  'l': 線分モードに切り替え\n"
        "  左クリック: 点/線分を追加\n"
        "  'u': 取り消し\n"
        "  'c': 全消去\n"
        "  's' または Enter: 確定\n"
        "  'q': 終了"
    )

    camera_point_collector = PointCollector(camera_image, "Camera", (0, 180, 255))
    floormap_point_collector = PointCollector(floormap_image, "Floormap", (0, 255, 0))
    camera_line_collector = LineSegmentCollector(camera_image, "Camera")
    PointCollector(floormap_image, "Floormap", (255, 255, 0))  # シアン色

    collection_mode = "point"  # "point" または "line"
    state = {"mode": "camera", "message": "Camera: 点を選択 [モード: 点]"}

    def on_mouse_camera(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if collection_mode == "point":
            if state["mode"] == "camera":
                camera_point_collector.add_point(x, y)
                state["mode"] = "floormap"
                state["message"] = "Floormap: 対応点を選択 [モード: 点]"
        else:  # line mode
            if state["mode"] == "camera_start":
                camera_line_collector.add_start_point(x, y)
                state["mode"] = "camera_end"
                state["message"] = "Camera: 線分の終点を選択 [モード: 線分]"
            elif state["mode"] == "camera_end":
                camera_line_collector.add_end_point(x, y)
                state["mode"] = "floormap_line"
                state["message"] = "Floormap: 対応点を選択 [モード: 線分]"

    def on_mouse_floormap(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if collection_mode == "point":
            if state["mode"] == "floormap":
                floormap_point_collector.add_point(x, y)
                state["mode"] = "camera"
                state["message"] = "Camera: 点を選択 [モード: 点]"
        else:  # line mode
            if state["mode"] == "floormap_line":
                floormap_point = (x, y)
                camera_line_collector.complete_line_segment(floormap_point)
                state["mode"] = "camera_start"
                state["message"] = "Camera: 線分の始点を選択 [モード: 線分]"

    cv2.namedWindow(camera_point_collector.window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(floormap_point_collector.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(camera_point_collector.window_name, on_mouse_camera)
    cv2.setMouseCallback(floormap_point_collector.window_name, on_mouse_floormap)

    try:
        while True:
            # 点モードの表示
            if collection_mode == "point":
                cam_status = f"{state['message']} | 点ペア数: {len(floormap_point_collector.points)}"
                map_status = f"{state['message']} | 点ペア数: {len(floormap_point_collector.points)}"

                cam_canvas = camera_point_collector.render(cam_status)
                map_canvas = floormap_point_collector.render(map_status)
            else:  # line mode
                cam_status = f"{state['message']} | 線分数: {len(camera_line_collector.line_segments)}"
                map_status = f"{state['message']} | 線分数: {len(camera_line_collector.line_segments)}"

                cam_canvas = camera_line_collector.render(cam_status)
                # フロアマップ側は線分に対応する点を表示
                map_canvas = floormap_point_collector.image.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                for idx, seg in enumerate(camera_line_collector.line_segments, start=1):
                    cv2.circle(map_canvas, seg.floormap_point, 8, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(
                        map_canvas,
                        f"L{idx}",
                        (seg.floormap_point[0] + 10, seg.floormap_point[1]),
                        font,
                        0.5,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(map_canvas, (0, 0), (map_canvas.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(map_canvas, map_status, (10, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(camera_point_collector.window_name, cam_canvas)
            cv2.imshow(floormap_point_collector.window_name, map_canvas)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                raise KeyboardInterrupt("ユーザーにより中断されました")
            if key in (ord("s"), 13):
                # 最小対応点数の計算（点対応 + 線分対応×3）
                point_count = len(floormap_point_collector.points)
                line_count = len(camera_line_collector.line_segments)
                total_samples = point_count + line_count * 3

                if total_samples >= min_points:
                    LOGGER.info(
                        "対応点が確定しました: 点対応=%d, 線分対応=%d, 総サンプル点数=%d",
                        point_count,
                        line_count,
                        total_samples,
                    )
                    break

                LOGGER.warning(
                    "対応点が不足しています: 点対応=%d, 線分対応=%d, 総サンプル点数=%d (必要: %d)",
                    point_count,
                    line_count,
                    total_samples,
                    min_points,
                )
            if key == ord("p"):
                # 点モードに切り替え
                if collection_mode != "point" and camera_line_collector.current_start is not None:
                    LOGGER.warning("編集中の線分があります。'u'で取り消してから切り替えてください。")
                elif collection_mode != "point":
                    collection_mode = "point"
                    state["mode"] = "camera"
                    state["message"] = "Camera: 点を選択 [モード: 点]"
            if key == ord("l") and collection_mode != "line":
                # 線分モードに切り替え
                collection_mode = "line"
                state["mode"] = "camera_start"
                state["message"] = "Camera: 線分の始点を選択 [モード: 線分]"
            if key == ord("u"):
                # 取り消し
                if collection_mode == "point":
                    if state["mode"] == "floormap" and camera_point_collector.points:
                        camera_point_collector.pop_last()
                        state["mode"] = "camera"
                        state["message"] = "Camera: 点を選択 [モード: 点]"
                    else:
                        camera_point_collector.pop_last()
                        floormap_point_collector.pop_last()
                        state["mode"] = "camera"
                        state["message"] = "Camera: 点を選択 [モード: 点]"
                else:  # line mode
                    camera_line_collector.pop_last()
                    if camera_line_collector.current_start is None:
                        state["mode"] = "camera_start"
                        state["message"] = "Camera: 線分の始点を選択 [モード: 線分]"
            if key == ord("c"):
                # 全消去
                camera_point_collector.clear()
                floormap_point_collector.clear()
                camera_line_collector.clear()
                if collection_mode == "point":
                    state["mode"] = "camera"
                    state["message"] = "Camera: 点を選択 [モード: 点]"
                else:
                    state["mode"] = "camera_start"
                    state["message"] = "Camera: 線分の始点を選択 [モード: 線分]"
    finally:
        cv2.destroyWindow(camera_point_collector.window_name)
        cv2.destroyWindow(floormap_point_collector.window_name)

    return (
        camera_point_collector.points,
        floormap_point_collector.points,
        camera_line_collector.line_segments,
    )


def collect_points_interactively(
    camera_image: np.ndarray,
    floormap_image: np.ndarray,
    min_points: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """後方互換性のためのラッパー関数"""
    camera_points, floormap_points, _ = collect_correspondences_interactively(camera_image, floormap_image, min_points)
    return camera_points, floormap_points


def compute_homography(
    camera_points: list[tuple[int, int]] | None = None,
    floormap_points: list[tuple[int, int]] | None = None,
    line_segments: list[LineSegment] | None = None,
    method: str = "ransac",
    ransac_threshold: float = 3.0,
    line_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """ホモグラフィ行列を計算

    Args:
        camera_points: カメラ画像上の点のリスト（点対応）
        floormap_points: フロアマップ上の点のリスト（点対応）
        line_segments: 線分対応のリスト
        method: 推定手法 ("default", "ransac", "lmeds")
        ransac_threshold: RANSACの再投影誤差閾値
        line_samples: 線分からサンプリングする点数（デフォルト: 3）

    Returns:
        (ホモグラフィ行列, マスク, メトリクス) のタプル
    """
    # 点対応を準備
    src_points = list(camera_points) if camera_points else []
    dst_points = list(floormap_points) if floormap_points else []

    # 線分対応からサンプリング点を生成
    if line_segments:
        for seg in line_segments:
            sampled = sample_points_from_line_segment(seg.start_point, seg.end_point, line_samples)
            src_points.extend(sampled)
            # 各サンプリング点は同じフロアマップ上の点に対応
            dst_points.extend([seg.floormap_point] * len(sampled))

    total_points = len(src_points)
    if total_points < 4:
        raise ValueError(f"ホモグラフィ計算には4組以上の対応点が必要です（現在: {total_points}点）。")

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    method_flag = 0
    if method == "ransac":
        method_flag = cv2.RANSAC
    elif method == "lmeds":
        method_flag = cv2.LMEDS

    point_count = len(camera_points) if camera_points else 0
    line_count = len(line_segments) if line_segments else 0
    LOGGER.info(
        "cv2.findHomography を実行します (method=%s, 点対応=%d, 線分対応=%d, 総サンプル点数=%d)",
        method,
        point_count,
        line_count,
        total_points,
    )
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
        "total_points": total_points,
        "point_correspondences": point_count,
        "line_segment_correspondences": line_count,
    }
    LOGGER.info(
        "再投影RMSE=%.3f px, 最大誤差=%.3f px, インライア=%d/%d",
        rmse,
        max_error,
        inliers,
        total_points,
    )

    return H, mask, metrics


def save_correspondences_json(
    output_dir: Path,
    camera_points: list[tuple[int, int]] | None = None,
    floormap_points: list[tuple[int, int]] | None = None,
    line_segments: list[LineSegment] | None = None,
    reference_image: Path | None = None,
    floormap_image: Path | None = None,
    output_format: str = "template",
    camera_id: str = "cam01",
) -> Path:
    """対応点と線分をJSONファイルに保存

    Args:
        output_dir: 出力ディレクトリ
        camera_points: カメラ画像上の座標リスト（点対応）
        floormap_points: フロアマップ上の座標リスト（点対応）
        line_segments: 線分対応のリスト
        reference_image: 参照画像のパス
        floormap_image: フロアマップ画像のパス
        output_format: 出力形式 ("template" または "legacy")
        camera_id: カメラID

    Returns:
        保存されたファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    camera_points = camera_points or []
    floormap_points = floormap_points or []
    line_segments = line_segments or []

    if output_format == "template":
        # テンプレート形式で保存
        path = output_dir / f"correspondence_points_{camera_id}.json"

        # 画像サイズを取得
        ref_img = cv2.imread(str(reference_image)) if reference_image else None
        floor_img = cv2.imread(str(floormap_image)) if floormap_image else None
        ref_size = (ref_img.shape[1], ref_img.shape[0]) if ref_img is not None else (1280, 720)
        floor_size = (floor_img.shape[1], floor_img.shape[0]) if floor_img is not None else (1878, 1369)

        # 点対応データ
        point_correspondences = None
        if camera_points and floormap_points:
            point_correspondences = {
                "src_points": [[float(x), float(y)] for x, y in camera_points],
                "dst_points": [[float(x), float(y)] for x, y in floormap_points],
            }

        # 線分対応データ
        line_segment_correspondences = None
        if line_segments:
            line_segment_correspondences = [
                {
                    "src_line": [
                        [float(seg.start_point[0]), float(seg.start_point[1])],
                        [float(seg.end_point[0]), float(seg.end_point[1])],
                    ],
                    "dst_point": [float(seg.floormap_point[0]), float(seg.floormap_point[1])],
                }
                for seg in line_segments
            ]

        payload = {
            "camera_id": camera_id,
            "description": f"対応点データ（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}に作成）",
            "metadata": {
                "image_size": {"width": ref_size[0], "height": ref_size[1]},
                "floormap_size": {"width": floor_size[0], "height": floor_size[1]},
                "num_point_correspondences": len(camera_points),
                "num_line_segment_correspondences": len(line_segments),
                "has_line_segments": len(line_segments) > 0,
                "coordinate_system_src": "camera_pixels",
                "coordinate_system_dst": "floormap_pixels",
                "origin_offset_applied": False,
                "reference_image": str(reference_image) if reference_image else None,
                "floormap_image": str(floormap_image) if floormap_image else None,
                "created_at": datetime.now().isoformat(),
            },
        }

        if point_correspondences:
            payload["point_correspondences"] = point_correspondences
        if line_segment_correspondences:
            payload["line_segment_correspondences"] = line_segment_correspondences

        # 後方互換性のため、点のみの場合は従来の形式も含める
        if point_correspondences and not line_segment_correspondences:
            payload["src_points"] = point_correspondences["src_points"]
            payload["dst_points"] = point_correspondences["dst_points"]
    else:
        # レガシー形式（camera_points, floormap_points）で保存
        path = output_dir / f"points_{timestamp}.json"
        payload = {
            "created_at": datetime.now().isoformat(),
            "reference_image": str(reference_image) if reference_image else None,
            "floormap_image": str(floormap_image) if floormap_image else None,
            "camera_points": camera_points,
            "floormap_points": floormap_points,
        }
        if line_segments:
            payload["line_segments"] = [
                {
                    "start_point": list(seg.start_point),
                    "end_point": list(seg.end_point),
                    "floormap_point": list(seg.floormap_point),
                }
                for seg in line_segments
            ]

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    LOGGER.info("対応点JSONを保存しました: %s", path)
    return path


def save_points_json(
    output_dir: Path,
    camera_points: list[tuple[int, int]],
    floormap_points: list[tuple[int, int]],
    reference_image: Path,
    floormap_image: Path,
    output_format: str = "template",
    camera_id: str = "cam01",
) -> Path:
    """後方互換性のためのラッパー関数"""
    return save_correspondences_json(
        output_dir,
        camera_points,
        floormap_points,
        None,
        reference_image,
        floormap_image,
        output_format,
        camera_id,
    )


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


def load_correspondences_from_json(
    path: Path,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[LineSegment]]:
    """対応点と線分をJSONファイルから読み込む

    新しい形式（point_correspondences/line_segment_correspondences）と
    後方互換形式（src_points/dst_points または camera_points/floormap_points）の両方に対応。

    Args:
        path: JSONファイルのパス

    Returns:
        (camera_points, floormap_points, line_segments) のタプル
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_points: list[tuple[int, int]] = []
    floormap_points: list[tuple[int, int]] = []
    line_segments: list[LineSegment] = []

    # 新しい形式を優先
    if "point_correspondences" in data:
        pc = data["point_correspondences"]
        camera_points = [tuple(map(int, p)) for p in pc.get("src_points", [])]
        floormap_points = [tuple(map(int, p)) for p in pc.get("dst_points", [])]

    if "line_segment_correspondences" in data:
        for ls_data in data["line_segment_correspondences"]:
            src_line = ls_data["src_line"]
            dst_point = ls_data["dst_point"]
            line_segments.append(
                LineSegment(
                    start_point=tuple(map(int, src_line[0])),
                    end_point=tuple(map(int, src_line[1])),
                    floormap_point=tuple(map(int, dst_point)),
                )
            )

    # 後方互換性: 従来の形式（src_points/dst_points）
    if not camera_points and "src_points" in data and "dst_points" in data:
        camera_points = [tuple(map(int, p)) for p in data.get("src_points", [])]
        floormap_points = [tuple(map(int, p)) for p in data.get("dst_points", [])]

    # 後方互換性: レガシー形式（camera_points/floormap_points）
    if not camera_points and "camera_points" in data and "floormap_points" in data:
        camera_points = [tuple(map(int, p)) for p in data.get("camera_points", [])]
        floormap_points = [tuple(map(int, p)) for p in data.get("floormap_points", [])]

    # レガシー形式の線分データ
    if not line_segments and "line_segments" in data:
        for ls_data in data["line_segments"]:
            line_segments.append(
                LineSegment(
                    start_point=tuple(map(int, ls_data["start_point"])),
                    end_point=tuple(map(int, ls_data["end_point"])),
                    floormap_point=tuple(map(int, ls_data["floormap_point"])),
                )
            )

    if camera_points and len(camera_points) != len(floormap_points):
        raise ValueError("JSON 内の対応点数が一致しません")

    return camera_points, floormap_points, line_segments


def load_points_from_json(
    path: Path,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """後方互換性のためのラッパー関数"""
    camera_points, floormap_points, _ = load_correspondences_from_json(path)
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
        camera_points, floormap_points, line_segments = load_correspondences_from_json(Path(args.load_points))
    else:
        camera_image = load_image(reference_path)
        floormap_image = load_image(floormap_path)
        camera_points, floormap_points, line_segments = collect_correspondences_interactively(
            camera_image,
            floormap_image,
            args.min_points,
        )

    points_json = save_correspondences_json(
        output_dir,
        camera_points,
        floormap_points,
        line_segments,
        reference_path,
        floormap_path,
        output_format=args.output_format,
        camera_id=args.camera_id,
    )
    H, _mask, metrics = compute_homography(
        camera_points if camera_points else None,
        floormap_points if floormap_points else None,
        line_segments if line_segments else None,
        args.method,
        args.ransac_threshold,
    )
    homography_yaml = save_homography_yaml(output_dir, H, metrics, points_json)

    if args.update_config:
        update_config_homography(Path(args.config), H, Path("config.backup"))

    LOGGER.info("校正が完了しました。")
    LOGGER.info("対応点: %s", points_json)
    LOGGER.info("ホモグラフィ: %s", homography_yaml)


if __name__ == "__main__":
    main()
