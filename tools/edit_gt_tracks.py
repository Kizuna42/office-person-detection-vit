#!/usr/bin/env python3
"""Ground Truthトラック手動編集ツール

自動生成されたGround Truthトラックを手動で編集するためのインタラクティブツールです。
"""

import argparse
import csv
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
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class GTracksEditor:
    """Ground Truthトラック編集クラス"""

    def __init__(
        self,
        tracks_path: Path,
        floormap_path: Path,
        config_path: Path,
        session_dir: Path | None = None,
    ):
        """初期化

        Args:
            tracks_path: Ground Truthトラックファイルのパス
            floormap_path: フロアマップ画像のパス
            config_path: 設定ファイルのパス
            session_dir: セッションディレクトリ（カメラ画像表示用、オプション）
        """
        self.tracks_path = tracks_path
        self.floormap_path = floormap_path
        self.config_path = config_path
        self.session_dir = session_dir

        # データの読み込み
        self.tracks_data = self._load_tracks()
        self.floormap_image = self._load_floormap()
        self.config = ConfigManager(str(config_path))

        # フレーム画像と検出結果のマッピング
        self.frame_images: dict[int, np.ndarray] = {}
        self.detection_images: dict[int, np.ndarray] = {}
        if session_dir:
            self._load_frame_mapping()

        # 編集状態
        self.current_frame = 0
        self.selected_track_id: int | None = None
        self.selected_point_idx: int | None = None
        self.dragging = False
        self.max_frame = self._get_max_frame()

        # ウィンドウ名
        self.window_name_floormap = "Ground Truth Tracks Editor - Floormap"
        self.window_name_camera = "Ground Truth Tracks Editor - Camera"

        logger.info(f"トラック数: {len(self.tracks_data)}")
        logger.info(f"最大フレーム数: {self.max_frame}")
        if session_dir:
            logger.info(f"カメラ画像: {len(self.frame_images)}フレーム読み込み済み")

    def _load_tracks(self) -> list[dict]:
        """トラックデータを読み込む"""
        with open(self.tracks_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("tracks", [])

    def _load_floormap(self) -> np.ndarray:
        """フロアマップ画像を読み込む"""
        image = cv2.imread(str(self.floormap_path))
        if image is None:
            raise FileNotFoundError(f"フロアマップ画像を読み込めません: {self.floormap_path}")
        return image

    def _get_max_frame(self) -> int:
        """最大フレーム数を取得"""
        max_frame = 0
        for track in self.tracks_data:
            trajectory = track.get("trajectory", [])
            for point in trajectory:
                frame = point.get("frame", 0)
                max_frame = max(max_frame, frame)
        return max_frame

    def _load_frame_mapping(self) -> None:
        """フレームインデックスと画像ファイルのマッピングを読み込む"""
        if not self.session_dir:
            return

        # extraction_results.csvからフレームインデックスとタイムスタンプの対応を取得
        frame_index_map: dict[str, int] = {}  # {timestamp_str: frame_idx}
        csv_path = self.session_dir / "phase1_extraction" / "extraction_results.csv"
        if csv_path.exists():
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_idx = int(row.get("frame_index", -1))
                    timestamp = row.get("extracted_timestamp", "")
                    if frame_idx >= 0 and timestamp:
                        # タイムスタンプを正規化（YYYY/MM/DD HH:MM:SS形式）
                        frame_index_map[timestamp] = frame_idx

        # phase1_extraction/frames/ からフレーム画像を読み込む
        frames_dir = self.session_dir / "phase1_extraction" / "frames"
        if frames_dir.exists():
            for frame_file in sorted(frames_dir.glob("frame_*_idx*.jpg")):
                # ファイル名からフレームインデックスを抽出: frame_YYYYMMDD_HHMMSS_idx{N}.jpg
                parts = frame_file.stem.split("_idx")
                if len(parts) == 2:
                    try:
                        frame_idx = int(parts[1])
                        image = cv2.imread(str(frame_file))
                        if image is not None:
                            self.frame_images[frame_idx] = image
                    except ValueError:
                        continue

        # phase2_detection/images/ から検出結果画像を読み込む
        detection_dir = self.session_dir / "phase2_detection" / "images"
        if detection_dir.exists():
            for det_file in sorted(detection_dir.glob("detection_*.jpg")):
                # ファイル名からタイムスタンプを抽出: detection_YYYY_MM_DD_HHMMSS.jpg
                # → YYYY/MM/DD HH:MM:SS 形式に変換
                parts = det_file.stem.split("_")
                if len(parts) >= 5:
                    try:
                        year = parts[1]
                        month = parts[2]
                        day = parts[3]
                        time_str = parts[4]  # HHMMSS
                        if len(time_str) == 6:
                            hour = time_str[:2]
                            minute = time_str[2:4]
                            second = time_str[4:6]
                            timestamp_str = f"{year}/{month}/{day} {hour}:{minute}:{second}"

                            # フレームインデックスを取得
                            frame_idx = frame_index_map.get(timestamp_str, -1)
                            if frame_idx >= 0:
                                image = cv2.imread(str(det_file))
                                if image is not None:
                                    self.detection_images[frame_idx] = image
                    except (ValueError, IndexError):
                        continue

        logger.info(f"フレーム画像: {len(self.frame_images)}枚, 検出結果画像: {len(self.detection_images)}枚")

    def _get_track_by_id(self, track_id: int) -> dict | None:
        """IDでトラックを取得"""
        for track in self.tracks_data:
            if track.get("track_id") == track_id:
                return track
        return None

    def _get_point_at_frame(self, track_id: int, frame: int) -> dict | None:
        """指定フレームのトラックポイントを取得"""
        track = self._get_track_by_id(track_id)
        if track is None:
            return None

        trajectory = track.get("trajectory", [])
        for point in trajectory:
            if point.get("frame") == frame:
                return point
        return None

    def _find_nearest_point(self, x: int, y: int, frame: int, threshold: float = 20.0) -> tuple[int, int] | None:
        """指定座標に最も近いトラックポイントを検索

        Args:
            x: マウスX座標
            y: マウスY座標
            frame: 現在のフレーム
            threshold: 選択閾値（ピクセル）

        Returns:
            (track_id, point_idx) のタプル、見つからない場合はNone
        """
        min_distance = float("inf")
        nearest = None

        for track in self.tracks_data:
            track_id = track.get("track_id")
            trajectory = track.get("trajectory", [])

            for idx, point in enumerate(trajectory):
                if point.get("frame") != frame:
                    continue

                px = point.get("x", 0)
                py = point.get("y", 0)

                distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)

                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    nearest = (track_id, idx)

        return nearest

    def _draw_tracks(self, frame: int) -> np.ndarray:
        """トラックを描画

        Args:
            frame: 現在のフレーム

        Returns:
            描画された画像
        """
        image = self.floormap_image.copy()

        # トラックを描画
        for track in self.tracks_data:
            track_id = track.get("track_id")
            trajectory = track.get("trajectory", [])

            # 色を生成（HSV色空間で均等に分散）
            hue = (track_id * 137) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

            # 軌跡線を描画（現在フレームまでの軌跡）
            trajectory_to_draw = [p for p in trajectory if p.get("frame", 0) <= frame]
            if len(trajectory_to_draw) > 1:
                for i in range(len(trajectory_to_draw) - 1):
                    pt1 = trajectory_to_draw[i]
                    pt2 = trajectory_to_draw[i + 1]
                    x1, y1 = int(pt1.get("x", 0)), int(pt1.get("y", 0))
                    x2, y2 = int(pt2.get("x", 0)), int(pt2.get("y", 0))
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)

            # 現在フレームのポイントを描画
            point = self._get_point_at_frame(track_id, frame)
            if point is not None:
                px, py = int(point.get("x", 0)), int(point.get("y", 0))

                # 選択されている場合は大きく表示
                if self.selected_track_id == track_id:
                    cv2.circle(image, (px, py), 10, (0, 255, 255), -1)  # 黄色
                    cv2.circle(image, (px, py), 12, (0, 0, 0), 2)  # 黒い枠
                else:
                    cv2.circle(image, (px, py), 6, color, -1)
                    cv2.circle(image, (px, py), 8, (0, 0, 0), 1)  # 黒い枠

                # IDを表示
                cv2.putText(
                    image,
                    f"ID:{track_id}",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    f"ID:{track_id}",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        # 情報テキストを表示
        info_text = f"Frame: {frame}/{self.max_frame} | Tracks: {len(self.tracks_data)}"
        if self.selected_track_id is not None:
            info_text += f" | Selected: ID{self.selected_track_id}"

        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 操作説明
        help_text = [
            "Controls:",
            "← → : Move frame",
            "Click: Select track",
            "Drag: Move point",
            "1-9: Change track ID",
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

        return image

    def _draw_camera_image(self, frame: int) -> np.ndarray | None:
        """カメラ画像を描画（検出結果付き）

        Args:
            frame: 現在のフレーム

        Returns:
            描画された画像、画像がない場合はNone
        """
        # フレームインデックスに対応する画像を取得
        # まず検出結果画像を探す
        if frame in self.detection_images:
            image = self.detection_images[frame].copy()
        elif frame in self.frame_images:
            image = self.frame_images[frame].copy()
        else:
            return None

        # 情報テキストを表示
        info_text = f"Frame: {frame}/{self.max_frame} | Camera View"
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image

    def _on_mouse(self, event, x, y, flags, param):
        """マウスイベントハンドラ"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 最も近いポイントを検索
            nearest = self._find_nearest_point(x, y, self.current_frame)
            if nearest is not None:
                self.selected_track_id, self.selected_point_idx = nearest
                self.dragging = True
                logger.info(f"トラックID {self.selected_track_id} を選択しました")
            else:
                self.selected_track_id = None
                self.selected_point_idx = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # ドラッグ中：ポイントを移動
            if self.selected_track_id is not None and self.selected_point_idx is not None:
                track = self._get_track_by_id(self.selected_track_id)
                if track is not None:
                    trajectory = track.get("trajectory", [])
                    if self.selected_point_idx < len(trajectory):
                        point = trajectory[self.selected_point_idx]
                        if point.get("frame") == self.current_frame:
                            point["x"] = float(x)
                            point["y"] = float(y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _change_track_id(self, new_id: int):
        """トラックIDを変更"""
        if self.selected_track_id is None:
            logger.warning("トラックが選択されていません")
            return

        # 既存のIDと重複していないか確認
        for track in self.tracks_data:
            if track.get("track_id") == new_id:
                logger.warning(f"トラックID {new_id} は既に使用されています")
                return

        # IDを変更
        track = self._get_track_by_id(self.selected_track_id)
        if track is not None:
            track["track_id"] = new_id
            self.selected_track_id = new_id
            logger.info(f"トラックIDを {self.selected_track_id} から {new_id} に変更しました")

    def _delete_point(self):
        """選択されたポイントを削除"""
        if self.selected_track_id is None or self.selected_point_idx is None:
            logger.warning("ポイントが選択されていません")
            return

        track = self._get_track_by_id(self.selected_track_id)
        if track is not None:
            trajectory = track.get("trajectory", [])
            if self.selected_point_idx < len(trajectory):
                point = trajectory[self.selected_point_idx]
                if point.get("frame") == self.current_frame:
                    trajectory.pop(self.selected_point_idx)
                    logger.info(f"フレーム {self.current_frame} のポイントを削除しました")

                    # トラックが空になった場合は削除
                    if len(trajectory) == 0:
                        self.tracks_data.remove(track)
                        logger.info(f"トラックID {self.selected_track_id} を削除しました")
                        self.selected_track_id = None
                        self.selected_point_idx = None

    def _add_new_track(self):
        """新しいトラックを追加"""
        # 新しいIDを生成
        max_id = 0
        for track in self.tracks_data:
            max_id = max(max_id, track.get("track_id", 0))
        new_id = max_id + 1

        # 新しいトラックを作成
        new_track = {
            "track_id": new_id,
            "trajectory": [
                {
                    "x": self.floormap_image.shape[1] // 2,
                    "y": self.floormap_image.shape[0] // 2,
                    "frame": self.current_frame,
                }
            ],
        }

        self.tracks_data.append(new_track)
        self.selected_track_id = new_id
        self.selected_point_idx = 0
        logger.info(f"新しいトラックID {new_id} を追加しました")

    def _save_tracks(self):
        """トラックデータを保存"""
        # 元のファイルをバックアップ
        backup_path = self.tracks_path.with_suffix(".json.bak")
        if self.tracks_path.exists():
            import shutil

            shutil.copy(self.tracks_path, backup_path)
            logger.info(f"バックアップを作成しました: {backup_path}")

        # メタデータを読み込み（存在する場合）
        with open(self.tracks_path, encoding="utf-8") as f:
            original_data = json.load(f)
            metadata = original_data.get("metadata", {})

        # 更新されたデータを保存
        data = {
            "tracks": self.tracks_data,
            "metadata": {
                **metadata,
                "num_tracks": len(self.tracks_data),
                "num_frames": self.max_frame + 1,
                "note": "手動編集されたデータ",
            },
        }

        with open(self.tracks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"トラックデータを保存しました: {self.tracks_path}")

    def run(self):
        """編集ツールを実行"""
        cv2.namedWindow(self.window_name_floormap, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name_floormap, self._on_mouse)

        # カメラ画像ウィンドウも作成（セッションディレクトリが指定されている場合）
        if self.session_dir:
            cv2.namedWindow(self.window_name_camera, cv2.WINDOW_NORMAL)

        logger.info("=" * 80)
        logger.info("Ground Truthトラック編集ツール")
        logger.info("=" * 80)
        logger.info("操作説明:")
        logger.info("  ← → : フレーム移動")
        logger.info("  クリック: トラック選択")
        logger.info("  ドラッグ: ポイント移動")
        logger.info("  1-9: トラックID変更")
        logger.info("  d: ポイント削除")
        logger.info("  a: 新しいトラック追加")
        logger.info("  s: 保存")
        logger.info("  q: 終了")
        logger.info("=" * 80)

        try:
            while True:
                # フロアマップを描画
                floormap_image = self._draw_tracks(self.current_frame)
                cv2.imshow(self.window_name_floormap, floormap_image)

                # カメラ画像を描画（利用可能な場合）
                if self.session_dir:
                    camera_image = self._draw_camera_image(self.current_frame)
                    if camera_image is not None:
                        cv2.imshow(self.window_name_camera, camera_image)

                # キー入力待ち
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q") or key == 27:  # 'q' または ESC
                    break
                if key == ord("s"):  # 保存
                    self._save_tracks()
                elif key == ord("a"):  # 新しいトラック追加
                    self._add_new_track()
                elif key == ord("d"):  # ポイント削除
                    self._delete_point()
                elif key == 81 or key == 2:  # 左矢印（Linux/Mac）
                    self.current_frame = max(0, self.current_frame - 1)
                elif key == 83 or key == 3:  # 右矢印（Linux/Mac）
                    self.current_frame = min(self.max_frame, self.current_frame + 1)
                elif ord("1") <= key <= ord("9"):  # 数字キー（1-9）
                    new_id = key - ord("0")
                    self._change_track_id(new_id)

        except KeyboardInterrupt:
            logger.info("ユーザーにより中断されました")
        finally:
            cv2.destroyAllWindows()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Ground Truthトラック手動編集ツール")
    parser.add_argument("--tracks", type=str, required=True, help="Ground Truthトラックファイルのパス")
    parser.add_argument("--floormap", type=str, help="フロアマップ画像のパス（設定ファイルから取得する場合は省略）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルパス (default: config.yaml)")
    parser.add_argument(
        "--session",
        type=str,
        help="セッションディレクトリのパス（カメラ画像表示用、オプション）",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    # ファイルパスの確認
    tracks_path = Path(args.tracks)
    config_path = Path(args.config)

    if not tracks_path.exists():
        logger.error(f"トラックファイルが見つかりません: {tracks_path}")
        return 1

    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    # フロアマップパスの取得
    if args.floormap:
        floormap_path = Path(args.floormap)
    else:
        config = ConfigManager(str(config_path))
        floormap_path = Path(config.get("floormap.image_path"))

    if not floormap_path.exists():
        logger.error(f"フロアマップ画像が見つかりません: {floormap_path}")
        return 1

    try:
        # セッションディレクトリの取得
        session_dir = None
        if args.session:
            session_dir = Path(args.session)
            if not session_dir.exists():
                logger.warning(f"セッションディレクトリが見つかりません: {session_dir}")
                session_dir = None
        else:
            # デフォルトでoutput/latestを試す
            default_session = Path("output/latest")
            if default_session.exists():
                session_dir = default_session
                logger.info(f"デフォルトセッションディレクトリを使用: {session_dir}")

        # エディタを起動
        editor = GTracksEditor(tracks_path, floormap_path, config_path, session_dir)
        editor.run()

        logger.info("編集ツールを終了しました")
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
