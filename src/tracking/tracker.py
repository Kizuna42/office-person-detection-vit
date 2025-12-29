"""DeepSORT-based tracker implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.tracking.hungarian import HungarianAlgorithm
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.similarity import SimilarityCalculator
from src.tracking.track import Track

if TYPE_CHECKING:
    from src.models.data_models import Detection

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
        max_position_distance: float = 150.0,
        high_conf_threshold: float = 0.5,
    ):
        """Trackerを初期化

        Args:
            max_age: IDが消失してから削除するまでの最大フレーム数
            min_hits: 追跡確立に必要な最小検出回数
            iou_threshold: IoU閾値（マッチング用）
            appearance_weight: 外観特徴量の重み（0.0-1.0）
            motion_weight: 位置情報の重み（0.0-1.0）
            max_position_distance: 予測位置と検出の最大許容距離（ピクセル）
            high_conf_threshold: ByteTrack用High/Low-conf分離閾値
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_position_distance = max_position_distance
        self.high_conf_threshold = high_conf_threshold

        # 類似度計算器
        self.similarity_calculator = SimilarityCalculator(
            appearance_weight=appearance_weight, motion_weight=motion_weight
        )

        # ハンガリアンアルゴリズム
        self.hungarian = HungarianAlgorithm()

        # トラック管理
        self.tracks: list[Track] = []
        self.next_id = 1  # 次のトラックID

        logger.info(
            f"Tracker initialized: max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}, high_conf_threshold={high_conf_threshold} (ByteTrack 2-Stage enabled)"
        )

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
        """検出結果とトラックを関連付け（ByteTrack 2-Stage + カスケードマッチング）

        ByteTrack方式:
        - High-conf検出でまずマッチング
        - 未マッチトラックをLow-conf検出で救済（部分遮蔽対策）

        Args:
            detections: 検出結果のリスト

        Returns:
            (マッチしたペアのリスト, 未マッチの検出インデックス, 未マッチのトラックインデックス)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # === ByteTrack: High-conf / Low-conf 分離 ===
        high_conf_indices = [i for i, det in enumerate(detections) if det.confidence >= self.high_conf_threshold]
        low_conf_indices = [i for i, det in enumerate(detections) if det.confidence < self.high_conf_threshold]

        logger.debug(f"ByteTrack split: {len(high_conf_indices)} high-conf, {len(low_conf_indices)} low-conf")

        # 確立されたトラックと仮のトラックを分離
        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.is_confirmed(self.min_hits)]
        tentative_tracks = [i for i, track in enumerate(self.tracks) if not track.is_confirmed(self.min_hits)]

        all_matches = []
        remaining_high_conf = list(high_conf_indices)
        remaining_track_indices = list(confirmed_tracks)

        # === Stage 1: High-conf検出で外観優先マッチング ===
        if remaining_track_indices and remaining_high_conf:
            stage1_dets = [detections[i] for i in remaining_high_conf]
            matches_1, _unmatched_dets_1, unmatched_tracks_1 = self._match_by_appearance(
                remaining_track_indices, stage1_dets, appearance_threshold=0.3
            )
            matches_1 = [(t, remaining_high_conf[d]) for t, d in matches_1]
            all_matches.extend(matches_1)
            matched_det_set_1 = {d for _, d in matches_1}
            remaining_high_conf = [i for i in remaining_high_conf if i not in matched_det_set_1]
            remaining_track_indices = unmatched_tracks_1

        # === Stage 2: High-conf検出で外観+IoU併用マッチング ===
        if remaining_track_indices and remaining_high_conf:
            stage2_dets = [detections[i] for i in remaining_high_conf]
            matches_2, _unmatched_dets_2, unmatched_tracks_2 = self._match(
                remaining_track_indices, stage2_dets, iou_threshold=0.5
            )
            matches_2 = [(t, remaining_high_conf[d]) for t, d in matches_2]
            all_matches.extend(matches_2)
            matched_det_set_2 = {d for _, d in matches_2}
            remaining_high_conf = [i for i in remaining_high_conf if i not in matched_det_set_2]
            remaining_track_indices = unmatched_tracks_2

        # === Stage 3: High-conf検出でIoUのみマッチング（フォールバック）===
        if remaining_track_indices and remaining_high_conf:
            stage3_dets = [detections[i] for i in remaining_high_conf]
            matches_3, _unmatched_dets_3, unmatched_tracks_3 = self._match_by_iou(
                remaining_track_indices, stage3_dets, iou_threshold=0.4
            )
            matches_3 = [(t, remaining_high_conf[d]) for t, d in matches_3]
            all_matches.extend(matches_3)
            matched_det_set_3 = {d for _, d in matches_3}
            remaining_high_conf = [i for i in remaining_high_conf if i not in matched_det_set_3]
            remaining_track_indices = unmatched_tracks_3

        # === ByteTrack Stage 4: 未マッチトラックをLOW-CONF検出で救済 ===
        # 部分遮蔽時の低信頼度検出を活用してトラック継続
        if remaining_track_indices and low_conf_indices:
            low_conf_dets = [detections[i] for i in low_conf_indices]
            # Low-confはIoUのみでマッチ（外観特徴は不安定なため）
            matches_low, _unmatched_dets_low, unmatched_tracks_low = self._match_by_iou(
                remaining_track_indices, low_conf_dets, iou_threshold=0.5
            )
            matches_low = [(t, low_conf_indices[d]) for t, d in matches_low]
            all_matches.extend(matches_low)

            logger.debug(f"ByteTrack low-conf rescue: {len(matches_low)} tracks saved")

            matched_low_det_set = {d for _, d in matches_low}  # noqa: F841 - 将来の拡張用
            remaining_track_indices = unmatched_tracks_low
        # else: Low-conf検出は新規トラック作成に使用しないため残りは無視

        # === 仮トラックのマッチング（High-confの残りのみ）===
        remaining_det_indices = remaining_high_conf  # Low-confは新規トラック作成しない
        if tentative_tracks and remaining_det_indices:
            tentative_dets = [detections[i] for i in remaining_det_indices]
            matches_tent, _unmatched_dets_tent, unmatched_tracks_tent = self._match(
                tentative_tracks, tentative_dets, iou_threshold=0.5
            )
            matches_tent = [(t, remaining_det_indices[d]) for t, d in matches_tent]
            all_matches.extend(matches_tent)
            matched_det_set_tent = {d for _, d in matches_tent}
            remaining_det_indices = [i for i in remaining_det_indices if i not in matched_det_set_tent]
            remaining_track_indices.extend(unmatched_tracks_tent)

        # 未マッチ検出 = High-confの残りのみ（Low-confは新規トラック作成しない）
        return all_matches, remaining_det_indices, remaining_track_indices

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
            return [], list(range(len(detections))), list(track_indices)

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
            track_feature = track.get_smoothed_feature()
            # 予測位置（Kalman）を使用して距離ゲートをかける
            predicted_pos = None
            try:
                predicted_pos = track.kalman_filter.get_position()
            except Exception:
                predicted_pos = None

            for j, detection in enumerate(detections):
                # 位置距離でゲート（外れ値を強制的に高コストにする）
                if predicted_pos is not None and detection.camera_coords is not None:
                    pos_dist = np.linalg.norm(predicted_pos - np.array(detection.camera_coords, dtype=np.float32))
                    if self.max_position_distance > 0 and pos_dist > self.max_position_distance:
                        cost_matrix[i, j] = 1.0
                        continue

                # 外観距離
                appearance_distance = None
                if track_feature is not None and detection.features is not None:
                    appearance_distance = self.similarity_calculator.cosine_distance(track_feature, detection.features)

                # モーション距離（IoUベース）
                motion_distance = self.similarity_calculator.iou_distance(track_detection.bbox, detection.bbox)

                app_w = self.similarity_calculator.appearance_weight if appearance_distance is not None else 0.0
                mot_w = self.similarity_calculator.motion_weight
                total_w = app_w + mot_w

                if total_w == 0:
                    cost_matrix[i, j] = 1.0
                else:
                    # 重み付き平均（利用可能な重みのみ正規化）
                    distance = 0.0
                    if appearance_distance is not None:
                        distance += app_w * appearance_distance
                    distance += mot_w * motion_distance
                    cost_matrix[i, j] = distance / total_w

        return cost_matrix

    def _match_by_appearance(
        self, track_indices: list[int], detections: list[Detection], appearance_threshold: float = 0.3
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """外観特徴量のみでマッチング（Stage 1用）

        Args:
            track_indices: トラックのインデックスリスト
            detections: 検出結果のリスト
            appearance_threshold: 外観距離の閾値（これ以下でマッチ）

        Returns:
            (マッチしたペアのリスト, 未マッチの検出インデックス, 未マッチのトラックインデックス)
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(track_indices)

        n_tracks = len(track_indices)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_feature = track.get_smoothed_feature()
            if track_feature is None:
                continue

            for j, detection in enumerate(detections):
                if detection.features is None:
                    continue
                # 外観距離のみ使用
                appearance_distance = self.similarity_calculator.cosine_distance(track_feature, detection.features)
                cost_matrix[i, j] = appearance_distance

        # ハンガリアンアルゴリズムで最適割り当て
        assignment, _ = self.hungarian.solve(cost_matrix)

        matches = []
        unmatched_tracks = []

        for track_idx_in_list, det_idx in enumerate(assignment):
            track_idx = track_indices[track_idx_in_list]
            if det_idx >= 0 and cost_matrix[track_idx_in_list, det_idx] <= appearance_threshold:
                matches.append((track_idx, det_idx))
            else:
                unmatched_tracks.append(track_idx)

        matched_det_indices = {det_idx for _, det_idx in matches}
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]

        return matches, unmatched_dets, unmatched_tracks

    def _match_by_iou(
        self, track_indices: list[int], detections: list[Detection], iou_threshold: float = 0.7
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """IoUのみでマッチング（Stage 3用）

        Args:
            track_indices: トラックのインデックスリスト
            detections: 検出結果のリスト
            iou_threshold: IoU閾値（1.0 - iou_distanceがこれ以上でマッチ）

        Returns:
            (マッチしたペアのリスト, 未マッチの検出インデックス, 未マッチのトラックインデックス)
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(track_indices)

        n_tracks = len(track_indices)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_bbox = track.detection.bbox

            for j, detection in enumerate(detections):
                # IoU距離のみ使用
                iou_distance = self.similarity_calculator.iou_distance(track_bbox, detection.bbox)
                cost_matrix[i, j] = iou_distance

        # ハンガリアンアルゴリズムで最適割り当て
        assignment, _ = self.hungarian.solve(cost_matrix)

        matches = []
        unmatched_tracks = []

        for track_idx_in_list, det_idx in enumerate(assignment):
            track_idx = track_indices[track_idx_in_list]
            # IoU = 1.0 - iou_distance なので、cost < (1.0 - threshold) でマッチ
            if det_idx >= 0 and cost_matrix[track_idx_in_list, det_idx] < (1.0 - iou_threshold):
                matches.append((track_idx, det_idx))
            else:
                unmatched_tracks.append(track_idx)

        matched_det_indices = {det_idx for _, det_idx in matches}
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]

        return matches, unmatched_dets, unmatched_tracks

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
