"""Object tracking module using DeepSORT-like algorithm.

This module provides tracking functionality to assign consistent IDs
to detected objects across frames.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np

from src.models.data_models import Detection
from src.tracking.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track状態の列挙型"""

    TENTATIVE = "tentative"  # 仮確定（検証中）
    CONFIRMED = "confirmed"  # 確定
    DELETED = "deleted"  # 削除済み


@dataclass
class Track:
    """追跡対象を表すクラス

    Attributes:
        track_id: 追跡ID
        bbox: 最新のバウンディングボックス (x, y, width, height)
        floor_coords: 最新のフロアマップ座標 (x, y)
        features: 最新の外観特徴量
        state: Track状態
        hits: マッチング成功回数
        time_since_update: 最後の更新からのフレーム数
        age: Trackの存在期間（フレーム数）
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    floor_coords: tuple[float, float] | None
    features: np.ndarray | None
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    time_since_update: int = 0
    age: int = 1

    # 軌跡情報（可視化用）
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        """初期化後の処理"""
        if self.floor_coords is not None:
            self.trajectory.append(self.floor_coords)

    def update(self, detection: Detection) -> None:
        """Trackを更新

        Args:
            detection: マッチした検出結果
        """
        self.bbox = detection.bbox
        self.floor_coords = detection.floor_coords
        self.features = detection.features
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

        # 軌跡に追加
        if self.floor_coords is not None:
            self.trajectory.append(self.floor_coords)

    def mark_missed(self) -> None:
        """マッチしなかったことを記録"""
        self.time_since_update += 1

    def predict(self) -> None:
        """次の位置を予測（簡略版：現在位置を維持）"""
        # 簡略化：Kalman Filterの代わりに現在位置を維持
        # 実際の実装では、速度を考慮した予測を行う


class Tracker:
    """DeepSORTベースのオブジェクト追跡クラス

    フレーム間で検出結果に一貫したIDを割り当てます。
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.7,
        motion_weight: float = 0.3,
    ):
        """Trackerを初期化

        Args:
            max_age: Trackが削除されるまでの最大フレーム数（更新されない場合）
            min_hits: Trackが確定状態になるのに必要な最小マッチング回数
            iou_threshold: IoU距離の閾値
            appearance_weight: 外観特徴量の重み（0.0-1.0）
            motion_weight: 位置情報の重み（0.0-1.0）
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: list[Track] = []

        self.similarity_calculator = SimilarityCalculator(
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
            iou_threshold=iou_threshold,
        )

        logger.info(
            f"Tracker initialized: max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}, appearance_weight={appearance_weight}, "
            f"motion_weight={motion_weight}"
        )

    def update(self, detections: list[Detection]) -> list[Detection]:
        """検出結果を更新して追跡IDを割り当て

        Args:
            detections: 現在フレームの検出結果

        Returns:
            track_idが割り当てられた検出結果のリスト
        """
        # 既存のTrackを予測
        for track in self.tracks:
            track.predict()

        # マッチング
        matches, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(detections)

        # マッチしたTrackを更新
        for det_idx, track_idx in matches:
            track = self.tracks[track_idx]
            track.update(detections[det_idx])
            detections[det_idx].track_id = track.track_id

            # 確定状態に移行
            if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED

        # マッチしなかったTrackをマーク
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()

            # 削除条件をチェック
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED

        # マッチしなかった検出結果で新しいTrackを作成
        for det_idx in unmatched_dets:
            track = self._create_track(detections[det_idx])
            detections[det_idx].track_id = track.track_id

        # 削除されたTrackを除去
        self.tracks = [track for track in self.tracks if track.state != TrackState.DELETED]

        logger.debug(
            f"Tracking update: {len(matches)} matches, "
            f"{len(unmatched_dets)} new tracks, "
            f"{len(unmatched_tracks)} unmatched tracks, "
            f"{len(detections)} total detections"
        )

        return detections

    def _associate_detections_to_tracks(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """検出結果とTrackをマッチング

        Args:
            detections: 検出結果のリスト

        Returns:
            (マッチしたペアのリスト, マッチしなかった検出結果のインデックス, マッチしなかったTrackのインデックス)
        """
        if len(self.tracks) == 0:
            return ([], list(range(len(detections))), [])

        if len(detections) == 0:
            return ([], [], list(range(len(self.tracks))))

        # 確定状態のTrackのみをマッチング対象とする
        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.state == TrackState.CONFIRMED]
        tentative_tracks = [i for i, track in enumerate(self.tracks) if track.state == TrackState.TENTATIVE]

        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = []

        # 確定Trackとのマッチング
        if confirmed_tracks:
            confirmed_matches, unmatched_dets, unmatched_confirmed = self._match_tracks(
                detections, confirmed_tracks, unmatched_dets
            )
            matches.extend(confirmed_matches)
            unmatched_tracks.extend(unmatched_confirmed)

        # 仮確定Trackとのマッチング（確定Trackとマッチしなかった検出結果のみ）
        if tentative_tracks and unmatched_dets:
            tentative_matches, unmatched_dets, unmatched_tentative = self._match_tracks(
                detections, tentative_tracks, unmatched_dets
            )
            matches.extend(tentative_matches)
            unmatched_tracks.extend(unmatched_tentative)

        return (matches, unmatched_dets, unmatched_tracks)

    def _match_tracks(
        self, detections: list[Detection], track_indices: list[int], det_indices: list[int]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Trackと検出結果をマッチング（Hungarian Algorithmの簡略版）

        Args:
            detections: 検出結果のリスト
            track_indices: マッチング対象のTrackインデックス
            det_indices: マッチング対象の検出結果インデックス

        Returns:
            (マッチしたペアのリスト, マッチしなかった検出結果のインデックス, マッチしなかったTrackのインデックス)
        """
        if not track_indices or not det_indices:
            return ([], det_indices, track_indices)

        # 類似度行列を計算
        similarity_matrix = np.zeros((len(det_indices), len(track_indices)), dtype=np.float32)

        for i, det_idx in enumerate(det_indices):
            detection = detections[det_idx]
            for j, track_idx in enumerate(track_indices):
                track = self.tracks[track_idx]

                # Trackの最新情報でDetectionオブジェクトを作成
                track_detection = Detection(
                    bbox=track.bbox,
                    confidence=0.0,  # 使用しない
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    camera_coords=detection.camera_coords,  # 使用しない
                    floor_coords=track.floor_coords,
                    features=track.features,
                )

                # 類似度を計算
                combined_dist, _, _ = self.similarity_calculator.compute_similarity(detection, track_detection)
                similarity_matrix[i, j] = combined_dist

        # 簡略版Hungarian Algorithm（貪欲法）
        # 実際の実装では、scipy.optimize.linear_sum_assignmentを使用
        matches = []
        unmatched_dets = list(det_indices)
        unmatched_tracks = list(track_indices)

        # 距離が小さい順にソート
        match_candidates = []
        for i, det_idx in enumerate(det_indices):
            for j, track_idx in enumerate(track_indices):
                distance = similarity_matrix[i, j]
                match_candidates.append((distance, det_idx, track_idx))

        match_candidates.sort(key=lambda x: x[0])

        # 貪欲にマッチング
        used_dets = set()
        used_tracks = set()

        for distance, det_idx, track_idx in match_candidates:
            if det_idx in used_dets or track_idx in used_tracks:
                continue

            # 距離が閾値以下の場合のみマッチ
            if distance < 0.5:  # 閾値（調整可能）
                matches.append((det_idx, track_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)

        return (matches, unmatched_dets, unmatched_tracks)

    def _create_track(self, detection: Detection) -> Track:
        """新しいTrackを作成

        Args:
            detection: 検出結果

        Returns:
            作成されたTrack
        """
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            floor_coords=detection.floor_coords,
            features=detection.features,
            state=TrackState.TENTATIVE,
        )
        self.tracks.append(track)
        self.next_id += 1

        logger.debug(f"Created new track: ID={track.track_id}")
        return track

    def get_tracks(self) -> list[Track]:
        """現在のTrackリストを取得

        Returns:
            Trackのリスト
        """
        return [track for track in self.tracks if track.state != TrackState.DELETED]

    def get_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """全Trackの軌跡を取得

        Returns:
            {track_id: 軌跡座標のリスト}の辞書
        """
        trajectories = {}
        for track in self.tracks:
            if track.state != TrackState.DELETED and track.trajectory:
                trajectories[track.track_id] = track.trajectory
        return trajectories

