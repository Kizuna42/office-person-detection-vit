"""トラック管理モジュール: トラックデータの操作と検索"""

import logging

logger = logging.getLogger(__name__)


class TrackManager:
    """トラックデータの管理と操作"""

    def __init__(self, tracks_data: list[dict], session_tracks: dict[int, dict]):
        """初期化

        Args:
            tracks_data: Ground Truthトラックデータ
            session_tracks: セッショントラックデータ
        """
        self.tracks_data = tracks_data
        self.session_tracks = session_tracks

    def get_track_by_id(self, track_id: int) -> dict | None:
        """IDでトラックを取得

        Args:
            track_id: トラックID

        Returns:
            トラックデータ、見つからない場合はNone
        """
        for track in self.tracks_data:
            if track.get("track_id") == track_id:
                return track
        return None

    def get_point_at_frame(self, track_id: int, frame: int) -> dict | None:
        """指定フレームのトラックポイントを取得（Ground Truth優先）

        Args:
            track_id: トラックID
            frame: フレーム番号

        Returns:
            ポイントデータ、見つからない場合はNone
        """
        # Ground Truthトラックから取得
        track = self.get_track_by_id(track_id)
        if track is not None:
            trajectory = track.get("trajectory", [])
            for point in trajectory:
                if point.get("frame") == frame:
                    return point

        # セッショントラックから取得
        if track_id in self.session_tracks:
            track_data = self.session_tracks[track_id]
            trajectory = track_data.get("trajectory", [])
            if frame < len(trajectory):
                return trajectory[frame]

        return None

    def find_nearest_point(
        self,
        x: int,
        y: int,
        frame: int,
        image_width: int,
        image_height: int,
        threshold: float = 30.0,
    ) -> tuple[int, int] | None:
        """指定座標に最も近いトラックポイントを検索

        Args:
            x: マウスX座標
            y: マウスY座標
            frame: 現在のフレーム
            image_width: 画像幅
            image_height: 画像高さ
            threshold: 選択閾値（ピクセル）

        Returns:
            (track_id, point_idx) のタプル、見つからない場合はNone
        """
        from tools.gt_editor.utils import calculate_distance, clip_coordinates

        min_distance = float("inf")
        nearest = None

        # Ground Truthトラックから検索（優先）
        for track in self.tracks_data:
            track_id = track.get("track_id")
            if track_id is None:
                continue

            trajectory = track.get("trajectory", [])
            for point_idx, point in enumerate(trajectory):
                if point.get("frame") != frame:
                    continue

                px = point.get("x", 0)
                py = point.get("y", 0)

                # 範囲外の点も検索対象にする（クリップ位置で検索）
                search_x, search_y = clip_coordinates(px, py, image_width, image_height)
                distance = calculate_distance(search_x, search_y, x, y)

                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    nearest = (track_id, point_idx)

        # セッショントラックから検索
        if nearest is None:
            for track_id, track_data in self.session_tracks.items():
                trajectory = track_data.get("trajectory", [])
                if frame >= len(trajectory):
                    continue

                point = trajectory[frame]
                px = point.get("x", 0)
                py = point.get("y", 0)

                distance = calculate_distance(px, py, x, y)

                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    nearest = (track_id, 0)

        return nearest

    def update_point(self, track_id: int, frame: int, x: float, y: float) -> None:
        """ポイントを更新または作成

        Args:
            track_id: トラックID
            frame: フレーム番号
            x: X座標
            y: Y座標
        """
        track = self.get_track_by_id(track_id)
        if track is None:
            # 新規作成
            track = {
                "track_id": track_id,
                "trajectory": [],
            }
            self.tracks_data.append(track)

        trajectory = track.get("trajectory", [])

        # 既存のポイントを探す
        point_found = False
        for point in trajectory:
            if point.get("frame") == frame:
                point["x"] = float(x)
                point["y"] = float(y)
                point_found = True
                break

        # 新規作成
        if not point_found:
            trajectory.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "frame": frame,
                }
            )

    def delete_point(self, track_id: int, point_idx: int, frame: int) -> bool:
        """ポイントを削除

        Args:
            track_id: トラックID
            point_idx: ポイントインデックス
            frame: フレーム番号

        Returns:
            トラックが空になった場合True
        """
        track = self.get_track_by_id(track_id)
        if track is None:
            return False

        trajectory = track.get("trajectory", [])
        if point_idx < len(trajectory):
            point = trajectory[point_idx]
            if point.get("frame") == frame:
                trajectory.pop(point_idx)

                # トラックが空になった場合は削除
                if len(trajectory) == 0:
                    self.tracks_data.remove(track)
                    return True

        return False

    def change_track_id(self, old_id: int, new_id: int) -> bool:
        """トラックIDを変更

        Args:
            old_id: 現在のID
            new_id: 新しいID

        Returns:
            変更成功した場合True
        """
        # 重複チェック
        if self.get_track_by_id(new_id) is not None:
            logger.warning(f"トラックID {new_id} は既に使用されています")
            return False

        track = self.get_track_by_id(old_id)
        if track is not None:
            track["track_id"] = new_id
            logger.info(f"トラックIDを {old_id} から {new_id} に変更しました")
            return True

        return False

    def add_new_track(self, frame: int, x: int, y: int) -> int:
        """新しいトラックを追加

        Args:
            frame: フレーム番号
            x: X座標
            y: Y座標

        Returns:
            新しいトラックID
        """
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
                    "x": float(x),
                    "y": float(y),
                    "frame": frame,
                }
            ],
        }

        self.tracks_data.append(new_track)
        logger.info(f"新しいトラックID {new_id} を追加しました")
        return new_id

    def get_max_frame(self) -> int:
        """最大フレーム数を取得

        Returns:
            最大フレーム数
        """
        max_frame = 0

        # セッショントラックから取得
        if self.session_tracks:
            for track in self.session_tracks.values():
                trajectory = track.get("trajectory", [])
                max_frame = max(max_frame, len(trajectory) - 1)
        else:
            # Ground Truthトラックから取得
            for track in self.tracks_data:
                trajectory = track.get("trajectory", [])
                for point in trajectory:
                    frame = point.get("frame", 0)
                    max_frame = max(max_frame, frame)

        return max_frame
