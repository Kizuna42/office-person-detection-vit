"""データローダーモジュール: トラック、画像、フレームマッピングの読み込み"""

from __future__ import annotations

import csv
import json
import logging
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from src.transform.coordinate_transformer import CoordinateTransformer


logger = logging.getLogger(__name__)


class TrackDataLoader:
    """Ground Truthトラックデータの読み込み"""

    def __init__(self, tracks_path: Path):
        """初期化

        Args:
            tracks_path: トラックファイルのパス
        """
        self.tracks_path = tracks_path
        self.tracks_data: list[dict] = []

    def load(self) -> list[dict]:
        """トラックデータを読み込む

        Returns:
            トラックデータのリスト
        """
        if not self.tracks_path.exists():
            logger.info(f"トラックファイルが存在しません。新規作成します: {self.tracks_path}")
            self.tracks_data = []
            return []

        with open(self.tracks_path, encoding="utf-8") as f:
            data = json.load(f)
            self.tracks_data = data.get("tracks", [])
            logger.info(
                f"既存のトラックファイルを読み込みました: {self.tracks_path} ({len(self.tracks_data)}件のトラック)"
            )
            return self.tracks_data

    def save(self, metadata: dict | None = None) -> None:
        """トラックデータを保存

        Args:
            metadata: メタデータ（オプション）
        """
        # バックアップを作成
        backup_path = self.tracks_path.with_suffix(".json.bak")
        if self.tracks_path.exists():
            import shutil

            shutil.copy(self.tracks_path, backup_path)
            logger.info(f"バックアップを作成しました: {backup_path}")

        # 既存のメタデータを読み込み
        original_metadata = {}
        if self.tracks_path.exists():
            with open(self.tracks_path, encoding="utf-8") as f:
                original_data = json.load(f)
                original_metadata = original_data.get("metadata", {})

        # メタデータをマージ
        merged_metadata = {**original_metadata}
        if metadata:
            merged_metadata.update(metadata)

        # 保存
        data = {
            "tracks": self.tracks_data,
            "metadata": {
                **merged_metadata,
                "num_tracks": len(self.tracks_data),
                "note": "手動編集されたデータ",
            },
        }

        with open(self.tracks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"トラックデータを保存しました: {self.tracks_path}")


class SessionTrackLoader:
    """セッショントラックデータの読み込み"""

    def __init__(self, session_dir: Path):
        """初期化

        Args:
            session_dir: セッションディレクトリのパス
        """
        self.session_dir = session_dir
        self.session_tracks: dict[int, dict] = {}

    def load(self) -> dict[int, dict]:
        """セッショントラックを読み込む

        Returns:
            {track_id: track_data} の辞書
        """
        tracks_file = self.session_dir / "phase2.5_tracking" / "tracks.json"
        if not tracks_file.exists():
            tracks_file = self.session_dir / "tracks.json"

        if tracks_file.exists():
            with open(tracks_file, encoding="utf-8") as f:
                data = json.load(f)
                tracks = data.get("tracks", [])
                for track in tracks:
                    track_id = track.get("track_id")
                    if track_id is not None:
                        self.session_tracks[track_id] = track
                logger.info(f"セッショントラック: {len(self.session_tracks)}件読み込み")
        else:
            logger.warning(f"トラックファイルが見つかりません: {tracks_file}")

        return self.session_tracks


class FrameImageLoader:
    """フレーム画像と検出結果の読み込み"""

    def __init__(self, session_dir: Path):
        """初期化

        Args:
            session_dir: セッションディレクトリのパス
        """
        self.session_dir = session_dir
        self.frame_images: dict[int, np.ndarray] = {}
        self.detection_images: dict[int, np.ndarray] = {}
        self.detection_results: dict[int, list[dict]] = {}
        self.frame_index_to_gt_frame: dict[int, int] = {}
        self.gt_frame_to_frame_index: dict[int, int] = {}

    def load(self) -> None:
        """フレーム画像とマッピングを読み込む"""
        if not self.session_dir:
            return

        # extraction_results.csvからフレームインデックスとタイムスタンプの対応を取得
        frame_index_map: dict[str, int] = {}
        frame_indices: list[int] = []
        csv_path = self.session_dir / "phase1_extraction" / "extraction_results.csv"
        if csv_path.exists():
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_idx = int(row.get("frame_index", -1))
                    timestamp = row.get("extracted_timestamp", "")
                    if frame_idx >= 0 and timestamp:
                        frame_index_map[timestamp] = frame_idx
                        frame_indices.append(frame_idx)

        # Ground Truthフレーム番号とフレームインデックスの対応を作成
        sorted_frame_indices = sorted(frame_indices)
        for gt_frame, frame_idx in enumerate(sorted_frame_indices):
            self.frame_index_to_gt_frame[frame_idx] = gt_frame
            self.gt_frame_to_frame_index[gt_frame] = frame_idx

        # フレーム画像を読み込む
        self._load_frame_images()
        # 検出結果画像を読み込む
        self._load_detection_images(frame_index_map)
        # 検出結果JSONを読み込む
        self._load_detection_results(frame_index_map)

        logger.info(f"フレーム画像: {len(self.frame_images)}枚, 検出結果画像: {len(self.detection_images)}枚")
        logger.info(f"検出結果JSON: {len(self.detection_results)}フレーム")

    def _load_frame_images(self) -> None:
        """フレーム画像を読み込む"""
        frames_dir = self.session_dir / "phase1_extraction" / "frames"
        if not frames_dir.exists():
            return

        for frame_file in sorted(frames_dir.glob("frame_*_idx*.jpg")):
            parts = frame_file.stem.split("_idx")
            if len(parts) == 2:
                try:
                    frame_idx = int(parts[1])
                    image = cv2.imread(str(frame_file))
                    if image is not None:
                        self.frame_images[frame_idx] = image
                except ValueError:
                    continue

    def _load_detection_images(self, frame_index_map: dict[str, int]) -> None:
        """検出結果画像を読み込む"""
        tracking_dir = self.session_dir / "phase2.5_tracking" / "images"
        if not tracking_dir.exists():
            return

        for tracking_file in sorted(tracking_dir.glob("tracking_*.jpg")):
            parts = tracking_file.stem.split("_")
            if len(parts) >= 5:
                try:
                    year = parts[1]
                    month = parts[2]
                    day = parts[3]
                    time_str = parts[4]
                    if len(time_str) == 6:
                        hour = time_str[:2]
                        minute = time_str[2:4]
                        second = time_str[4:6]
                        timestamp_str = f"{year}/{month}/{day} {hour}:{minute}:{second}"

                        frame_idx = frame_index_map.get(timestamp_str, -1)
                        if frame_idx >= 0:
                            image = cv2.imread(str(tracking_file))
                            if image is not None:
                                self.detection_images[frame_idx] = image
                except (ValueError, IndexError):
                    continue

    def _load_detection_results(self, frame_index_map: dict[str, int]) -> None:
        """検出結果JSONを読み込む"""
        detection_json_path = self.session_dir / "phase2_detection" / "detection_results.json"
        if not detection_json_path.exists():
            return

        with open(detection_json_path, encoding="utf-8") as f:
            detection_data = json.load(f)
            for frame_data in detection_data:
                timestamp = frame_data.get("timestamp", "")
                detections = frame_data.get("detections", [])

                if timestamp and timestamp in frame_index_map:
                    frame_idx = frame_index_map[timestamp]
                    self.detection_results[frame_idx] = detections


class TrackGenerator:
    """セッショントラックからGround Truthトラックを生成"""

    def __init__(
        self,
        session_tracks: dict[int, dict],
        gt_frame_to_frame_index: dict[int, int],
        coordinate_transformer: CoordinateTransformer,
    ):
        """初期化

        Args:
            session_tracks: セッショントラックデータ
            gt_frame_to_frame_index: GTフレーム→フレームインデックス対応
            coordinate_transformer: 座標変換器
        """
        self.session_tracks = session_tracks
        self.gt_frame_to_frame_index = gt_frame_to_frame_index
        self.coordinate_transformer = coordinate_transformer

    def generate(self) -> list[dict]:
        """Ground Truthトラックを生成

        Returns:
            生成されたトラックデータのリスト
        """
        if not self.session_tracks or not self.gt_frame_to_frame_index:
            logger.warning("セッショントラックまたはフレームマッピングがありません")
            return []

        if not self.coordinate_transformer:
            logger.warning("座標変換器が初期化されていません")
            return []

        tracks_data = []

        for track_id, track_data in self.session_tracks.items():
            trajectory = track_data.get("trajectory", [])
            if not trajectory:
                logger.debug(f"トラックID {track_id}: 軌跡が空です")
                continue

            gt_trajectory = []
            for gt_frame, _frame_idx in sorted(self.gt_frame_to_frame_index.items()):
                if gt_frame < len(trajectory):
                    point = trajectory[gt_frame]
                    camera_x = point.get("x", 0)
                    camera_y = point.get("y", 0)

                    try:
                        floor_coords = self.coordinate_transformer.transform(
                            (camera_x, camera_y),
                            apply_origin_offset=True,
                        )
                        gt_trajectory.append(
                            {
                                "x": float(floor_coords[0]),
                                "y": float(floor_coords[1]),
                                "frame": gt_frame,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"トラックID {track_id}, フレーム {gt_frame}: 座標変換エラー - {e}")

            if gt_trajectory:
                tracks_data.append(
                    {
                        "track_id": track_id,
                        "trajectory": gt_trajectory,
                    }
                )

        logger.info(f"セッショントラックから {len(tracks_data)} 件のGround Truthトラックを生成しました")
        return tracks_data
