"""DeepSORT-based tracker implementation."""

from __future__ import annotations

import logging

import numpy as np

from src.models.data_models import Detection
from src.tracking.hungarian import HungarianAlgorithm
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.similarity import SimilarityCalculator
from src.tracking.track import Track

logger = logging.getLogger(__name__)


class Tracker:
    """DeepSORTベースのトラッカー

    検出結果に対して一貫したIDを割り当て、フレーム間で追跡します。
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
            max_age: IDが消失してから削除するまでの最大フレーム数
            min_hits: 追跡確立に必要な最小検出回数
            iou_threshold: IoU閾値（マッチング用）
            appearance_weight: 外観特徴量の重み（0.0-1.0）
            motion_weight: 位置情報の重み（0.0-1.0）
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # 類似度計算器
        self.similarity_calculator = SimilarityCalculator(
            appearance_weight=appearance_weight, motion_weight=motion_weight
        )

        # ハンガリアンアルゴリズム
        self.hungarian = HungarianAlgorithm()

        # トラック管理
        self.tracks: list[Track] = []
        self.next_id = 1  # 次のトラックID

        logger.info(f"Tracker initialized: max_age={max_age}, min_hits={min_hits}, " f"iou_threshold={iou_threshold}")

    def update(self, detections: list[Detection]) -> list[Detection]:
        """検出結果でトラッカーを更新

        Args:
            detections: 現在フレームの検出結果

        Returns:
            track_idが割り当てられた検出結果のリスト
        """
        # 既存トラックの予測
        for track in self.tracks:
            track.predict()

        # マッチング
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(detections)

        # マッチしたトラックを更新
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
            detections[det_idx].track_id = self.tracks[track_idx].track_id

        # 未マッチの検出結果で新しいトラックを作成
        for det_idx in unmatched_dets:
            track = self._create_track(detections[det_idx])
            # 検出結果にtrack_idを割り当て
            detections[det_idx].track_id = track.track_id

        # 未マッチのトラックを削除（max_ageを超えた場合）
        self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]

        # track_idが割り当てられた検出結果のみを返す
        tracked_detections = [det for det in detections if det.track_id is not None]

        logger.debug(
            f"Tracker update: {len(matched)} matched, {len(unmatched_dets)} new tracks, "
            f"{len(unmatched_tracks)} unmatched tracks, {len(self.tracks)} total tracks"
        )

        return tracked_detections

    def _associate_detections_to_tracks(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """検出結果とトラックを関連付け

        Args:
            detections: 検出結果のリスト

        Returns:
            (マッチしたペアのリスト, 未マッチの検出インデックス, 未マッチのトラックインデックス)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # 確立されたトラックと仮のトラックを分離
        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.is_confirmed(self.min_hits)]
        tentative_tracks = [i for i, track in enumerate(self.tracks) if not track.is_confirmed(self.min_hits)]

        # 確立されたトラックと検出結果をマッチング
        matches_a, unmatched_dets_a, unmatched_tracks_a = self._match(confirmed_tracks, detections, self.iou_threshold)

        # 仮のトラックと未マッチの検出結果をマッチング（より厳しい閾値）
        iou_threshold_tentative = 0.5  # 仮のトラックにはより厳しい閾値
        matches_b, unmatched_dets_b, unmatched_tracks_b = self._match(
            tentative_tracks, [detections[i] for i in unmatched_dets_a], iou_threshold_tentative
        )

        # インデックスを調整
        matches_b = [(tentative_tracks[m[0]], unmatched_dets_a[m[1]]) for m in matches_b]

        # 結果を統合
        matches = matches_a + matches_b
        unmatched_dets = [unmatched_dets_a[i] for i in unmatched_dets_b]
        unmatched_tracks = [confirmed_tracks[i] for i in unmatched_tracks_a] + [
            tentative_tracks[i] for i in unmatched_tracks_b
        ]

        return matches, unmatched_dets, unmatched_tracks

    def _match(
        self, track_indices: list[int], detections: list[Detection], iou_threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """トラックと検出結果をマッチング

        Args:
            track_indices: トラックのインデックスリスト
            detections: 検出結果のリスト
            iou_threshold: IoU閾値

        Returns:
            (マッチしたペアのリスト, 未マッチの検出インデックス, 未マッチのトラックインデックス)
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(track_indices)))

        # コスト行列を計算
        cost_matrix = self._compute_cost_matrix(track_indices, detections)

        # ハンガリアンアルゴリズムで最適割り当て
        assignment, _ = self.hungarian.solve(cost_matrix)

        # マッチング結果を処理
        matches = []
        unmatched_dets = []
        unmatched_tracks = []

        for track_idx_in_list, det_idx in enumerate(assignment):
            track_idx = track_indices[track_idx_in_list]

            if det_idx >= 0 and cost_matrix[track_idx_in_list, det_idx] < (1.0 - iou_threshold):
                # マッチ成功
                matches.append((track_idx, det_idx))
            else:
                # 未マッチ
                unmatched_tracks.append(track_idx)

        # 未マッチの検出結果を特定
        matched_det_indices = {det_idx for _, det_idx in matches}
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]

        return matches, unmatched_dets, unmatched_tracks

    def _compute_cost_matrix(self, track_indices: list[int], detections: list[Detection]) -> np.ndarray:
        """コスト行列を計算

        Args:
            track_indices: トラックのインデックスリスト
            detections: 検出結果のリスト

        Returns:
            コスト行列 (len(track_indices), len(detections))
            コストは距離（0が最も類似、1が最も異なる）
        """
        n_tracks = len(track_indices)
        n_dets = len(detections)

        cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_detection = track.detection

            for j, detection in enumerate(detections):
                # 統合距離を計算
                distance = self.similarity_calculator.compute_distance(
                    track_detection, detection, use_appearance=True, use_motion=True
                )
                cost_matrix[i, j] = distance

        return cost_matrix

    def _create_track(self, detection: Detection) -> Track:
        """新しいトラックを作成

        Args:
            detection: 検出結果

        Returns:
            作成されたTrackオブジェクト
        """
        kalman_filter = KalmanFilter()
        track = Track(
            track_id=self.next_id,
            detection=detection,
            kalman_filter=kalman_filter,
        )
        self.tracks.append(track)
        self.next_id += 1

        logger.debug(f"Created new track {track.track_id}")

        return track

    def get_tracks(self) -> list[Track]:
        """全てのトラックを取得

        Returns:
            トラックのリスト
        """
        return self.tracks.copy()

    def get_confirmed_tracks(self) -> list[Track]:
        """確立されたトラックのみを取得

        Returns:
            確立されたトラックのリスト
        """
        return [track for track in self.tracks if track.is_confirmed(self.min_hits)]

    def reset(self) -> None:
        """トラッカーをリセット"""
        self.tracks = []
        self.next_id = 1
        logger.info("Tracker reset")
