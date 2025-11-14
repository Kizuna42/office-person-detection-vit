"""描画モジュール: フロアマップとカメラ画像の描画"""

import logging

import cv2
import numpy as np

from tools.gt_editor.utils import clip_coordinates, generate_track_color, is_within_bounds

logger = logging.getLogger(__name__)


class FloormapRenderer:
    """フロアマップ上のトラック描画"""

    def __init__(
        self,
        floormap_image: np.ndarray,
        tracks_data: list[dict],
        session_tracks: dict[int, dict],
    ):
        """初期化

        Args:
            floormap_image: フロアマップ画像
            tracks_data: Ground Truthトラックデータ
            session_tracks: セッショントラックデータ（情報表示用）
        """
        self.floormap_image = floormap_image
        self.tracks_data = tracks_data
        self.session_tracks = session_tracks
        self.image_height, self.image_width = floormap_image.shape[:2]

    def render(
        self,
        frame: int,
        max_frame: int,
        selected_track_id: int | None = None,
        id_input_mode: bool = False,
        id_input_buffer: str = "",
    ) -> np.ndarray:
        """トラックを描画

        Args:
            frame: 現在のフレーム
            max_frame: 最大フレーム数
            selected_track_id: 選択されたトラックID
            id_input_mode: ID入力モード中か
            id_input_buffer: ID入力バッファ

        Returns:
            描画された画像
        """
        image = self.floormap_image.copy()

        # トラックを描画
        for track in self.tracks_data:
            track_id = track.get("track_id")
            if track_id is None:
                continue

            self._draw_track(image, track, frame, selected_track_id)

        # 情報テキストを表示
        self._draw_info_text(image, frame, max_frame, selected_track_id, id_input_mode, id_input_buffer)

        # 操作説明を表示
        self._draw_help_text(image)

        return image

    def _draw_track(
        self,
        image: np.ndarray,
        track: dict,
        frame: int,
        selected_track_id: int | None,
    ) -> None:
        """1つのトラックを描画

        Args:
            image: 描画先画像
            track: トラックデータ
            frame: 現在のフレーム
            selected_track_id: 選択されたトラックID
        """
        track_id = track.get("track_id")
        trajectory = track.get("trajectory", [])

        # 現在フレームのポイントを探す
        current_point = None
        for point in trajectory:
            if point.get("frame") == frame:
                current_point = point
                break

        if current_point is None:
            return

        px, py = int(current_point.get("x", 0)), int(current_point.get("y", 0))
        is_selected = selected_track_id == track_id
        within_bounds = is_within_bounds(px, py, self.image_width, self.image_height)

        # 色を生成
        color = generate_track_color(track_id)

        # 軌跡線を描画
        self._draw_trajectory(image, trajectory, frame, color)

        # ポイントを描画
        self._draw_point(image, px, py, within_bounds, is_selected, color, track_id)

    def _draw_trajectory(
        self,
        image: np.ndarray,
        trajectory: list[dict],
        frame: int,
        color: tuple[int, int, int],
    ) -> None:
        """軌跡線を描画

        Args:
            image: 描画先画像
            trajectory: 軌跡データ
            frame: 現在のフレーム
            color: 色
        """
        trajectory_to_draw = [p for p in trajectory if p.get("frame", 0) <= frame]
        if len(trajectory_to_draw) <= 1:
            return

        for i in range(len(trajectory_to_draw) - 1):
            pt1 = trajectory_to_draw[i]
            pt2 = trajectory_to_draw[i + 1]
            x1, y1 = int(pt1.get("x", 0)), int(pt1.get("y", 0))
            x2, y2 = int(pt2.get("x", 0)), int(pt2.get("y", 0))

            # 両方の点が範囲内の場合のみ線を描画
            if is_within_bounds(x1, y1, self.image_width, self.image_height) and is_within_bounds(
                x2, y2, self.image_width, self.image_height
            ):
                cv2.line(image, (x1, y1), (x2, y2), color, 2)

    def _draw_point(
        self,
        image: np.ndarray,
        px: int,
        py: int,
        within_bounds: bool,
        is_selected: bool,
        color: tuple[int, int, int],
        track_id: int,
    ) -> None:
        """ポイントを描画

        Args:
            image: 描画先画像
            px: X座標
            py: Y座標
            within_bounds: 範囲内か
            is_selected: 選択されているか
            color: 色
            track_id: トラックID
        """
        if within_bounds:
            # 範囲内の点：通常の描画
            if is_selected:
                cv2.circle(image, (px, py), 12, (0, 255, 255), -1)  # 黄色
                cv2.circle(image, (px, py), 14, (0, 0, 0), 2)  # 黒い枠
            else:
                cv2.circle(image, (px, py), 8, color, -1)
                cv2.circle(image, (px, py), 10, (0, 0, 0), 2)  # 黒い枠
        else:
            # 範囲外の点：クリップ位置に表示
            clipped_x, clipped_y = clip_coordinates(px, py, self.image_width, self.image_height)

            if is_selected:
                cv2.circle(image, (clipped_x, clipped_y), 12, (0, 255, 255), 2)  # 黄色（点線）
                cv2.circle(image, (clipped_x, clipped_y), 14, (0, 0, 0), 2)  # 黒い枠
            else:
                cv2.circle(image, (clipped_x, clipped_y), 8, color, 2)  # 点線
                cv2.circle(image, (clipped_x, clipped_y), 10, (0, 0, 0), 2)  # 黒い枠

            # 範囲外であることを示す矢印を描画
            self._draw_out_of_bounds_arrow(image, px, py, clipped_x, clipped_y, color)

        # IDを表示
        self._draw_track_id(image, px, py, within_bounds, track_id)

    def _draw_out_of_bounds_arrow(
        self,
        image: np.ndarray,
        px: int,
        py: int,
        clipped_x: int,
        clipped_y: int,
        color: tuple[int, int, int],
    ) -> None:
        """範囲外を示す矢印を描画

        Args:
            image: 描画先画像
            px: 実際のX座標
            py: 実際のY座標
            clipped_x: クリップされたX座標
            clipped_y: クリップされたY座標
            color: 色
        """
        if px < 0:
            cv2.arrowedLine(image, (10, clipped_y), (0, clipped_y), color, 2, tipLength=0.3)
        elif px >= self.image_width:
            cv2.arrowedLine(
                image,
                (self.image_width - 10, clipped_y),
                (self.image_width - 1, clipped_y),
                color,
                2,
                tipLength=0.3,
            )
        if py < 0:
            cv2.arrowedLine(image, (clipped_x, 10), (clipped_x, 0), color, 2, tipLength=0.3)
        elif py >= self.image_height:
            cv2.arrowedLine(
                image,
                (clipped_x, self.image_height - 10),
                (clipped_x, self.image_height - 1),
                color,
                2,
                tipLength=0.3,
            )

    def _draw_track_id(
        self,
        image: np.ndarray,
        px: int,
        py: int,
        within_bounds: bool,
        track_id: int,
    ) -> None:
        """トラックIDを表示

        Args:
            image: 描画先画像
            px: X座標
            py: Y座標
            within_bounds: 範囲内か
            track_id: トラックID
        """
        id_text = f"ID:{track_id} (GT)"

        if within_bounds:
            label_x = px + 12
            label_y = py - 12
        else:
            label_x = max(10, min(px + 12, self.image_width - 100))
            label_y = max(25, min(py - 12, self.image_height - 10))

        # 白い縁取り
        cv2.putText(
            image,
            id_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        # 黒い文字
        cv2.putText(
            image,
            id_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

    def _draw_info_text(
        self,
        image: np.ndarray,
        frame: int,
        max_frame: int,
        selected_track_id: int | None,
        id_input_mode: bool,
        id_input_buffer: str,
    ) -> None:
        """情報テキストを表示

        Args:
            image: 描画先画像
            frame: 現在のフレーム
            max_frame: 最大フレーム数
            selected_track_id: 選択されたトラックID
            id_input_mode: ID入力モード中か
            id_input_buffer: ID入力バッファ
        """
        info_text = (
            f"Frame: {frame}/{max_frame} | "
            f"Session Tracks: {len(self.session_tracks)} | "
            f"GT Tracks: {len(self.tracks_data)}"
        )
        if selected_track_id is not None:
            info_text += f" | Selected: ID{selected_track_id}"
        if id_input_mode:
            info_text += f" | ID入力: {id_input_buffer}_"

        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_help_text(self, image: np.ndarray) -> None:
        """操作説明を表示

        Args:
            image: 描画先画像
        """
        help_text = [
            "Controls:",
            "Left/Right Arrow: Move frame",
            "Click: Select track",
            "Drag: Move point",
            "1-9: Change track ID (quick)",
            "i: ID input mode (1-30)",
            "m: Match ID from camera",
            "d: Delete point",
            "a: Add new track",
            "s: Save",
            "q: Quit",
        ]

        y_offset = image.shape[0] - len(help_text) * 25 - 10
        cv2.rectangle(image, (0, y_offset), (300, image.shape[0]), (0, 0, 0), -1)
        for i, text in enumerate(help_text):
            cv2.putText(
                image,
                text,
                (10, y_offset + (i + 1) * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )


class CameraRenderer:
    """カメラ画像の描画"""

    def __init__(
        self,
        frame_images: dict[int, np.ndarray],
        detection_images: dict[int, np.ndarray],
        detection_results: dict[int, list[dict]],
        session_tracks: dict[int, dict],
        gt_frame_to_frame_index: dict[int, int],
    ):
        """初期化

        Args:
            frame_images: フレーム画像 {frame_index: image}
            detection_images: 検出結果画像 {frame_index: image}
            detection_results: 検出結果JSON {frame_index: detections}
            session_tracks: セッショントラックデータ
            gt_frame_to_frame_index: GTフレーム→フレームインデックス対応
        """
        self.frame_images = frame_images
        self.detection_images = detection_images
        self.detection_results = detection_results
        self.session_tracks = session_tracks
        self.gt_frame_to_frame_index = gt_frame_to_frame_index

    def render(self, gt_frame: int, max_frame: int) -> np.ndarray | None:
        """カメラ画像を描画

        Args:
            gt_frame: Ground Truthフレーム番号
            max_frame: 最大フレーム数

        Returns:
            描画された画像、画像がない場合はNone
        """
        frame_idx = self.gt_frame_to_frame_index.get(gt_frame, -1)
        if frame_idx < 0:
            return None

        # 画像を取得
        image = None
        if frame_idx in self.detection_images:
            image = self.detection_images[frame_idx].copy()
        elif frame_idx in self.frame_images:
            image = self.frame_images[frame_idx].copy()

        if image is None:
            return None

        # 検出結果を描画
        if frame_idx in self.detection_results:
            self._draw_detections(image, frame_idx)

        # 情報テキストを表示
        self._draw_info_text(image, gt_frame, frame_idx, max_frame)

        return image

    def _draw_detections(self, image: np.ndarray, frame_idx: int) -> None:
        """検出結果を描画

        Args:
            image: 描画先画像
            frame_idx: フレームインデックス
        """
        detections = self.detection_results[frame_idx]
        for det in detections:
            bbox_data = det.get("bbox", {})
            x = int(bbox_data.get("x", 0))
            y = int(bbox_data.get("y", 0))
            w = int(bbox_data.get("width", 0))
            h = int(bbox_data.get("height", 0))
            track_id = det.get("track_id")

            if track_id is not None:
                color = generate_track_color(track_id)

                # バウンディングボックスを描画
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

                # IDを表示
                label = f"ID:{track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # ラベル背景を描画
                cv2.rectangle(
                    image,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    color,
                    -1,
                )

                # ラベルテキストを描画
                cv2.putText(
                    image,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # 足元座標を描画
                camera_coords = det.get("camera_coords", {})
                foot_x = int(camera_coords.get("x", x + w // 2))
                foot_y = int(camera_coords.get("y", y + h))
                cv2.circle(image, (foot_x, foot_y), 5, color, -1)

    def _draw_info_text(self, image: np.ndarray, gt_frame: int, frame_idx: int, max_frame: int) -> None:
        """情報テキストを表示

        Args:
            image: 描画先画像
            gt_frame: GTフレーム番号
            frame_idx: フレームインデックス
            max_frame: 最大フレーム数
        """
        # 現在フレームに存在するトラックIDを取得
        active_tracks = []
        for track_id, track_data in self.session_tracks.items():
            trajectory = track_data.get("trajectory", [])
            if gt_frame < len(trajectory):
                active_tracks.append(track_id)

        info_text = f"Frame: {gt_frame}/{max_frame} (Index: {frame_idx}) | Active IDs: {sorted(active_tracks)}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
