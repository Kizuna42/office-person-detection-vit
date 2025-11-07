"""Export utilities for trajectory data."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.models.data_models import Detection, FrameResult
    from src.tracking.track import Track

logger = logging.getLogger(__name__)


class TrajectoryExporter:
    """軌跡データのエクスポートクラス

    追跡結果を様々な形式でエクスポートします。
    """

    def __init__(self, output_dir: str | Path):
        """TrajectoryExporterを初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TrajectoryExporter initialized: {self.output_dir}")

    def export_csv(
        self,
        tracks: list[Track],
        filename: str = "trajectories.csv",
        include_features: bool = False,
    ) -> Path:
        """軌跡データをCSV形式でエクスポート

        Args:
            tracks: トラックのリスト
            filename: 出力ファイル名
            include_features: 特徴量を含めるか（デフォルト: False）

        Returns:
            出力ファイルのパス
        """
        output_path = self.output_dir / filename

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # ヘッダー
            header = ["track_id", "frame_index", "timestamp", "x", "y", "zone_ids", "confidence"]
            if include_features:
                header.append("features")
            writer.writerow(header)

            # データ行
            for track in tracks:
                for i, (x, y) in enumerate(track.trajectory):
                    row = [
                        track.track_id,
                        i,
                        "",  # timestampは後で追加可能
                        f"{x:.2f}",
                        f"{y:.2f}",
                        ",".join(track.detection.zone_ids) if track.detection.zone_ids else "",
                        f"{track.detection.confidence:.3f}",
                    ]
                    if include_features and track.detection.features is not None:
                        features_str = ",".join([f"{f:.6f}" for f in track.detection.features])
                        row.append(features_str)
                    writer.writerow(row)

        logger.info(f"CSV exported: {output_path}")
        return output_path

    def export_json(
        self,
        tracks: list[Track],
        filename: str = "trajectories.json",
        include_features: bool = False,
    ) -> Path:
        """軌跡データをJSON形式でエクスポート

        Args:
            tracks: トラックのリスト
            filename: 出力ファイル名
            include_features: 特徴量を含めるか（デフォルト: False）

        Returns:
            出力ファイルのパス
        """
        output_path = self.output_dir / filename

        data = {
            "tracks": [],
            "metadata": {
                "num_tracks": len(tracks),
                "total_points": sum(len(track.trajectory) for track in tracks),
            },
        }

        for track in tracks:
            track_data: dict[str, Any] = {
                "track_id": track.track_id,
                "age": track.age,
                "hits": track.hits,
                "trajectory": [{"x": float(x), "y": float(y)} for x, y in track.trajectory],
                "detections": [],
            }

            if include_features and track.detection.features is not None:
                track_data["features"] = track.detection.features.tolist()

            data["tracks"].append(track_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON exported: {output_path}")
        return output_path

    def export_image_sequence(
        self,
        tracks: list[Track],
        floormap_image: np.ndarray,
        output_prefix: str = "trajectory_frame",
        draw_trajectories: bool = True,
        draw_ids: bool = True,
    ) -> list[Path]:
        """軌跡を画像シーケンスとしてエクスポート

        Args:
            tracks: トラックのリスト
            floormap_image: フロアマップ画像
            output_prefix: 出力ファイル名のプレフィックス
            draw_trajectories: 軌跡線を描画するか
            draw_ids: IDを表示するか

        Returns:
            出力ファイルのパスのリスト
        """
        output_paths = []

        # 最大フレーム数を取得
        max_frames = max((len(track.trajectory) for track in tracks), default=0)

        for frame_idx in range(max_frames):
            image = floormap_image.copy()

            # 各トラックの現在位置を描画
            for track in tracks:
                if frame_idx >= len(track.trajectory):
                    continue

                x, y = track.trajectory[frame_idx]
                x, y = int(x), int(y)

                # 軌跡線を描画
                if draw_trajectories and frame_idx > 0:
                    prev_x, prev_y = track.trajectory[frame_idx - 1]
                    prev_x, prev_y = int(prev_x), int(prev_y)
                    cv2.line(image, (prev_x, prev_y), (x, y), (0, 255, 0), 2)

                # 現在位置を描画
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

                # IDを表示
                if draw_ids:
                    cv2.putText(
                        image,
                        f"ID:{track.track_id}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

            # 保存
            filename = f"{output_prefix}_{frame_idx:04d}.png"
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), image)
            output_paths.append(output_path)

        logger.info(f"Image sequence exported: {len(output_paths)} frames")
        return output_paths

    def export_video(
        self,
        tracks: list[Track],
        floormap_image: np.ndarray,
        filename: str = "trajectories.mp4",
        fps: float = 2.0,
        draw_trajectories: bool = True,
        draw_ids: bool = True,
    ) -> Path:
        """軌跡を動画としてエクスポート

        Args:
            tracks: トラックのリスト
            floormap_image: フロアマップ画像
            filename: 出力ファイル名
            fps: フレームレート
            draw_trajectories: 軌跡線を描画するか
            draw_ids: IDを表示するか

        Returns:
            出力ファイルのパス
        """
        output_path = self.output_dir / filename

        # 最大フレーム数を取得
        max_frames = max((len(track.trajectory) for track in tracks), default=0)

        if max_frames == 0:
            logger.warning("No trajectories to export")
            return output_path

        # 動画ライターを初期化
        h, w = floormap_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        try:
            for frame_idx in range(max_frames):
                image = floormap_image.copy()

                # 各トラックの現在位置を描画
                for track in tracks:
                    if frame_idx >= len(track.trajectory):
                        continue

                    x, y = track.trajectory[frame_idx]
                    x, y = int(x), int(y)

                    # 軌跡線を描画（過去の軌跡も含む）
                    if draw_trajectories:
                        for i in range(1, min(frame_idx + 1, len(track.trajectory))):
                            prev_x, prev_y = track.trajectory[i - 1]
                            curr_x, curr_y = track.trajectory[i]
                            prev_x, prev_y = int(prev_x), int(prev_y)
                            curr_x, curr_y = int(curr_x), int(curr_y)
                            cv2.line(image, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 2)

                    # 現在位置を描画
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

                    # IDを表示
                    if draw_ids:
                        cv2.putText(
                            image,
                            f"ID:{track.track_id}",
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

                writer.write(image)

        finally:
            writer.release()

        logger.info(f"Video exported: {output_path}")
        return output_path


class SideBySideVideoExporter:
    """検出画像とフロアマップ画像を左右に並べて動画として出力するクラス"""

    def __init__(self, output_dir: str | Path):
        """SideBySideVideoExporterを初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SideBySideVideoExporter initialized: {self.output_dir}")

    def _normalize_timestamp(self, timestamp: str) -> str:
        """タイムスタンプを正規化（ファイル名用）

        Args:
            timestamp: タイムスタンプ文字列（例: "2025/08/26 16:04:56"）

        Returns:
            正規化されたタイムスタンプ（例: "2025_08_26_160456"）
        """
        # / と : と スペースを _ に置換
        normalized = timestamp.replace("/", "_").replace(":", "").replace(" ", "_")
        return normalized

    def _find_detection_image(self, detection_images_dir: Path, timestamp: str) -> Path | None:
        """検出画像のパスを検索

        Args:
            detection_images_dir: 検出画像ディレクトリ
            timestamp: タイムスタンプ文字列

        Returns:
            検出画像のパス（見つからない場合はNone）
        """
        normalized_ts = self._normalize_timestamp(timestamp)
        # 検出画像のファイル名パターン: detection_2025_08_26_160456.jpg
        pattern = f"detection_{normalized_ts}.jpg"
        detection_path = detection_images_dir / pattern

        if detection_path.exists():
            return detection_path

        # パターンマッチングで検索（タイムスタンプの形式が異なる場合）
        for img_path in detection_images_dir.glob("detection_*.jpg"):
            if normalized_ts in img_path.stem:
                return img_path

        return None

    def _find_floormap_image(self, floormap_images_dir: Path, timestamp: str) -> Path | None:
        """フロアマップ画像のパスを検索

        Args:
            floormap_images_dir: フロアマップ画像ディレクトリ
            timestamp: タイムスタンプ文字列（例: "2025/08/26 16:04:56"）

        Returns:
            フロアマップ画像のパス（見つからない場合はNone）
        """
        # フロアマップ画像のパスパターン: floormaps/floormap_2025/08/26 160456.png
        # タイムスタンプからファイル名を生成
        # timestamp: "2025/08/26 16:04:56" -> "2025/08/26 160456"
        timestamp_no_colon = timestamp.replace(":", "")

        # パターン1: floormap_2025/08/26 160456.png（階層構造）
        pattern1 = f"floormap_{timestamp_no_colon}.png"
        floormap_path1 = floormap_images_dir / pattern1
        if floormap_path1.exists():
            return floormap_path1

        # パターン2: 再帰的に検索（階層構造の場合）
        normalized_ts = self._normalize_timestamp(timestamp)
        for img_path in floormap_images_dir.rglob("floormap_*.png"):
            # タイムスタンプがファイル名またはパスに含まれているか確認
            path_str = str(img_path)
            if normalized_ts in path_str or timestamp_no_colon in path_str:
                return img_path

        return None

    def add_track_ids_to_detection_image(self, detection_image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """検出画像にtrack_idを描画

        Args:
            detection_image: 検出画像
            detections: 検出結果のリスト（track_idが含まれる）

        Returns:
            track_idを描画した画像
        """
        result_image = detection_image.copy()

        for detection in detections:
            if detection.track_id is None:
                continue

            x, y, w, h = detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # track_idに基づいて色を生成（HSV色空間、黄金角を使用）
            hue = (detection.track_id * 137) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

            # バウンディングボックスを描画（track_id色）
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)

            # track_idと信頼度を表示
            label = f"ID:{detection.track_id} {detection.confidence:.2f}"
            cv2.putText(
                result_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # 足元座標を描画
            foot_x, foot_y = detection.camera_coords
            cv2.circle(result_image, (int(foot_x), int(foot_y)), 5, color, -1)

        return result_image

    def crop_and_zoom_floormap(
        self,
        floormap_image: np.ndarray,
        detections: list[Detection],
        zoom_margin: float = 0.8,
        min_zoom_ratio: float = 0.5,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """フロアマップ画像を検出座標を中心に拡大表示

        Args:
            floormap_image: フロアマップ画像
            detections: 検出結果のリスト（floor_coordsが含まれる）
            zoom_margin: 拡大時のマージン（0.0-1.0、デフォルト: 0.8 = 80%）
            min_zoom_ratio: 最小表示サイズの比率（0.0-1.0、デフォルト: 0.5 = 50%）
                            これより小さくクロップしない（拡大しすぎない）
            target_size: リサイズ後のサイズ（幅, 高さ）。Noneの場合は元のサイズ

        Returns:
            拡大表示されたフロアマップ画像
        """
        h, w = floormap_image.shape[:2]

        # 検出座標を収集
        floor_coords_list = []
        for detection in detections:
            if detection.floor_coords is not None:
                x, y = detection.floor_coords
                # 座標が範囲内か確認
                if 0 <= x < w and 0 <= y < h:
                    floor_coords_list.append((x, y))

        # 検出座標がない場合は元の画像を返す
        if not floor_coords_list:
            return floormap_image

        # バウンディングボックスを計算
        x_coords = [coord[0] for coord in floor_coords_list]
        y_coords = [coord[1] for coord in floor_coords_list]

        min_x = max(0, min(x_coords))
        max_x = min(w, max(x_coords))
        min_y = max(0, min(y_coords))
        max_y = min(h, max(y_coords))

        # マージンを追加
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        margin_x = bbox_width * zoom_margin
        margin_y = bbox_height * zoom_margin

        crop_x1 = max(0, int(min_x - margin_x))
        crop_y1 = max(0, int(min_y - margin_y))
        crop_x2 = min(w, int(max_x + margin_x))
        crop_y2 = min(h, int(max_y + margin_y))

        # 最小表示サイズを確保（拡大しすぎない）
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        min_width = int(w * min_zoom_ratio)
        min_height = int(h * min_zoom_ratio)

        if crop_width < min_width:
            # 幅が最小サイズより小さい場合、中心を維持して拡張
            center_x = (crop_x1 + crop_x2) // 2
            crop_x1 = max(0, center_x - min_width // 2)
            crop_x2 = min(w, crop_x1 + min_width)
            if crop_x2 == w:
                crop_x1 = max(0, w - min_width)

        if crop_height < min_height:
            # 高さが最小サイズより小さい場合、中心を維持して拡張
            center_y = (crop_y1 + crop_y2) // 2
            crop_y1 = max(0, center_y - min_height // 2)
            crop_y2 = min(h, crop_y1 + min_height)
            if crop_y2 == h:
                crop_y1 = max(0, h - min_height)

        # クロップ
        cropped = floormap_image[crop_y1:crop_y2, crop_x1:crop_x2]

        # リサイズ（target_sizeが指定されている場合）
        if target_size is not None:
            target_w, target_h = target_size
            cropped = cv2.resize(cropped, (target_w, target_h))

        return cropped

    def combine_images_side_by_side(
        self, left_image: np.ndarray, right_image: np.ndarray, add_divider: bool = True
    ) -> np.ndarray:
        """2つの画像を左右に結合

        Args:
            left_image: 左側の画像（検出画像）
            right_image: 右側の画像（フロアマップ画像）
            add_divider: 中央に区切り線を追加するか

        Returns:
            結合された画像
        """
        # 高さを揃える（大きい方に合わせてリサイズ）
        h1, w1 = left_image.shape[:2]
        h2, w2 = right_image.shape[:2]

        target_height = max(h1, h2)

        # 左側の画像をリサイズ
        if h1 != target_height:
            scale = target_height / h1
            new_w1 = int(w1 * scale)
            left_image = cv2.resize(left_image, (new_w1, target_height))
            w1 = new_w1

        # 右側の画像をリサイズ
        if h2 != target_height:
            scale = target_height / h2
            new_w2 = int(w2 * scale)
            right_image = cv2.resize(right_image, (new_w2, target_height))
            w2 = new_w2

        # 左右に結合
        combined = np.hstack([left_image, right_image])

        # 中央に区切り線を描画
        if add_divider:
            divider_x = w1
            cv2.line(combined, (divider_x, 0), (divider_x, target_height), (255, 255, 255), 2)

        return combined

    def export_side_by_side_video(
        self,
        frame_results: list[FrameResult],
        detection_images_dir: Path,
        floormap_images_dir: Path,
        filename: str = "side_by_side_tracking.mp4",
        fps: float = 1.0,
    ) -> Path:
        """検出画像とフロアマップ画像を左右に並べて動画として出力

        Args:
            frame_results: FrameResultのリスト（track_id情報を含む）
            detection_images_dir: 検出画像ディレクトリ
            floormap_images_dir: フロアマップ画像ディレクトリ
            filename: 出力ファイル名
            fps: フレームレート（デフォルト: 1.0）

        Returns:
            出力ファイルのパス
        """
        output_path = self.output_dir / filename

        if not frame_results:
            logger.warning("FrameResultが空です。動画を生成できません。")
            return output_path

        # 最初のフレームで画像サイズを取得（拡大表示を考慮）
        first_frame_result = frame_results[0]
        detection_path = self._find_detection_image(detection_images_dir, first_frame_result.timestamp)
        floormap_path = self._find_floormap_image(floormap_images_dir, first_frame_result.timestamp)

        if detection_path is None or floormap_path is None:
            logger.warning(
                f"最初のフレームの画像が見つかりません。検出画像: {detection_path}, フロアマップ: {floormap_path}"
            )
            return output_path

        # サンプル画像でサイズを取得
        sample_detection = cv2.imread(str(detection_path))
        sample_floormap = cv2.imread(str(floormap_path))

        if sample_detection is None or sample_floormap is None:
            logger.error("サンプル画像の読み込みに失敗しました")
            return output_path

        # フロアマップ画像を拡大表示（最初のフレームでサイズ計算用）
        zoom_margin = 0.8  # 80%のマージンでより俯瞰的に
        min_zoom_ratio = 0.5  # 最小50%は表示（拡大しすぎない）
        sample_floormap_zoomed = self.crop_and_zoom_floormap(
            sample_floormap, first_frame_result.detections, zoom_margin=zoom_margin, min_zoom_ratio=min_zoom_ratio
        )

        # 結合後のサイズを計算
        h1, w1 = sample_detection.shape[:2]
        h2, w2 = sample_floormap_zoomed.shape[:2]
        target_height = max(h1, h2)

        if h1 != target_height:
            scale = target_height / h1
            w1 = int(w1 * scale)
        if h2 != target_height:
            scale = target_height / h2
            w2 = int(w2 * scale)

        video_width = w1 + w2
        video_height = target_height

        # 動画ライターを初期化
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, video_height))

        if not writer.isOpened():
            logger.error(f"動画ライターの初期化に失敗しました: {output_path}")
            return output_path

        try:
            for frame_result in frame_results:
                # 検出画像を読み込み
                detection_path = self._find_detection_image(detection_images_dir, frame_result.timestamp)
                if detection_path is None:
                    logger.warning(f"検出画像が見つかりません: {frame_result.timestamp}")
                    continue

                detection_image = cv2.imread(str(detection_path))
                if detection_image is None:
                    logger.warning(f"検出画像の読み込みに失敗: {detection_path}")
                    continue

                # track_idを描画
                detection_image_with_tracks = self.add_track_ids_to_detection_image(
                    detection_image, frame_result.detections
                )

                # フロアマップ画像を読み込み
                floormap_path = self._find_floormap_image(floormap_images_dir, frame_result.timestamp)
                if floormap_path is None:
                    logger.warning(f"フロアマップ画像が見つかりません: {frame_result.timestamp}")
                    continue

                floormap_image = cv2.imread(str(floormap_path))
                if floormap_image is None:
                    logger.warning(f"フロアマップ画像の読み込みに失敗: {floormap_path}")
                    continue

                # フロアマップ画像を検出座標を中心に拡大表示
                zoom_margin = 0.8  # 80%のマージンでより俯瞰的に
                min_zoom_ratio = 0.5  # 最小50%は表示（拡大しすぎない）
                floormap_zoomed = self.crop_and_zoom_floormap(
                    floormap_image, frame_result.detections, zoom_margin=zoom_margin, min_zoom_ratio=min_zoom_ratio
                )

                # 左右に結合
                combined_image = self.combine_images_side_by_side(
                    detection_image_with_tracks, floormap_zoomed, add_divider=True
                )

                # タイムスタンプを上部に表示
                cv2.putText(
                    combined_image,
                    frame_result.timestamp,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                # 動画サイズに合わせてリサイズ（動画ライターのサイズと一致させる）
                combined_h, combined_w = combined_image.shape[:2]
                if combined_w != video_width or combined_h != video_height:
                    combined_image = cv2.resize(combined_image, (video_width, video_height))

                # 動画に書き込み
                writer.write(combined_image)

        finally:
            writer.release()

        logger.info(f"Side-by-side video exported: {output_path}")
        return output_path
