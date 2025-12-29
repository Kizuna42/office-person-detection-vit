"""Lightweight tracker implementation for interpolation between detection frames.

This module provides a ByteTrack-style tracker combined with Optical Flow
for efficient tracking between heavy detection frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.tracking.kalman_filter import KalmanFilter

if TYPE_CHECKING:
    from src.models.data_models import Detection

logger = logging.getLogger(__name__)


@dataclass
class LightweightTrack:
    """軽量トラック: Optical Flow追跡用の簡易トラック構造"""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    center: np.ndarray  # (x, y) center point
    kalman: KalmanFilter = field(default_factory=KalmanFilter)
    age: int = 0
    time_since_update: int = 0
    hits: int = 1
    features: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Kalman Filterを初期化"""
        self.kalman.init(self.center)

    def predict(self) -> np.ndarray:
        """Kalman Filterで次フレームの位置を予測"""
        state = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return state[:2]  # (x, y) position

    def update(self, bbox: tuple[float, float, float, float], center: np.ndarray) -> None:
        """観測値で状態を更新"""
        self.bbox = bbox
        self.center = center
        self.kalman.update(center)
        self.time_since_update = 0
        self.hits += 1


class OpticalFlowTracker:
    """Sparse Optical Flowを使用した軽量追跡補間

    Lucas-Kanade法を使用して、検出フレーム間の動きを追跡します。
    """

    def __init__(
        self,
        max_corners: int = 100,
        quality_level: float = 0.3,
        min_distance: float = 7.0,
        block_size: int = 7,
    ):
        """OpticalFlowTrackerを初期化

        Args:
            max_corners: 追跡する特徴点の最大数
            quality_level: 特徴点品質の閾値
            min_distance: 特徴点間の最小距離
            block_size: ブロックサイズ
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # Lucas-Kanade Optical Flowパラメータ
        self.lk_params: dict[str, tuple[int, int] | int | tuple[int, int, float]] = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }

        # 前フレームの情報
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.prev_track_ids: list[int] = []

        logger.debug(
            "OpticalFlowTracker initialized: max_corners=%d, quality=%.2f",
            max_corners,
            quality_level,
        )

    def initialize(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        """検出結果から追跡ポイントを初期化

        Args:
            frame: 現在のフレーム (BGR)
            detections: 検出結果のリスト
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 各検出のbbox中心を追跡ポイントとして設定
        points = []
        track_ids = []

        for det in detections:
            if det.track_id is not None:
                x, y, w, h = det.bbox
                cx = x + w / 2
                cy = y + h / 2
                points.append([cx, cy])
                track_ids.append(det.track_id)

        if points:
            self.prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            self.prev_track_ids = track_ids
        else:
            self.prev_points = None
            self.prev_track_ids = []

        self.prev_gray = gray

        logger.debug("OpticalFlowTracker initialized with %d points", len(points))

    def track(
        self,
        frame: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Optical Flowで追跡を実行

        Args:
            frame: 現在のフレーム (BGR)

        Returns:
            track_id -> 予測位置 (x, y) のマッピング
        """
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            return {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flowを計算
        win_size: tuple[int, int] = self.lk_params["winSize"]  # type: ignore[assignment]
        max_level: int = self.lk_params["maxLevel"]  # type: ignore[assignment]
        criteria: tuple[int, int, float] = self.lk_params["criteria"]  # type: ignore[assignment]
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=win_size,
            maxLevel=max_level,
            criteria=criteria,
        )

        # 追跡成功したポイントを抽出
        tracked_positions: dict[int, np.ndarray] = {}
        good_points = []
        good_ids = []

        if next_points is not None:
            for i, (point, st) in enumerate(zip(next_points, status, strict=False)):
                if st[0] == 1:  # 追跡成功
                    track_id = self.prev_track_ids[i]
                    position = point[0]
                    tracked_positions[track_id] = position
                    good_points.append(point)
                    good_ids.append(track_id)

        # 次フレーム用に状態を更新
        if good_points:
            self.prev_points = np.array(good_points, dtype=np.float32)
            self.prev_track_ids = good_ids
        else:
            self.prev_points = None
            self.prev_track_ids = []

        self.prev_gray = gray

        logger.debug(
            "OpticalFlow tracked %d/%d points",
            len(tracked_positions),
            len(self.prev_track_ids) if self.prev_track_ids else 0,
        )

        return tracked_positions

    def reset(self) -> None:
        """追跡状態をリセット"""
        self.prev_gray = None
        self.prev_points = None
        self.prev_track_ids = []


class LightweightTracker:
    """ByteTrackスタイルの軽量トラッカー

    - 検出フレーム: IoUベースのマッチングでID割当
    - 非検出フレーム: Optical FlowまたはKalman予測で補間
    """

    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        use_optical_flow: bool = True,
    ):
        """LightweightTrackerを初期化

        Args:
            max_age: トラックが消失してから削除するまでの最大フレーム数
            iou_threshold: IoUマッチングの閾値
            use_optical_flow: Optical Flowを使用するか
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.use_optical_flow = use_optical_flow

        self.tracks: list[LightweightTrack] = []
        self.next_id = 1

        # Optical Flowトラッカー（オプション）
        self.of_tracker = OpticalFlowTracker() if use_optical_flow else None

        logger.info(
            "LightweightTracker initialized: max_age=%d, iou_threshold=%.2f, optical_flow=%s",
            max_age,
            iou_threshold,
            use_optical_flow,
        )

    def update_with_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """検出結果でトラックを更新（重い処理）

        Args:
            frame: 現在のフレーム
            detections: 検出結果のリスト

        Returns:
            track_idが割り当てられた検出結果
        """
        # 全トラックを予測
        for track in self.tracks:
            track.predict()

        # IoUマッチング
        matched, unmatched_dets, _unmatched_tracks = self._match_detections(detections)

        # マッチしたトラックを更新
        for track_idx, det_idx in matched:
            det = detections[det_idx]
            x, y, w, h = det.bbox
            center = np.array([x + w / 2, y + h / 2])

            self.tracks[track_idx].update(det.bbox, center)
            det.track_id = self.tracks[track_idx].track_id

        # 新規トラック作成
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            x, y, w, h = det.bbox
            center = np.array([x + w / 2, y + h / 2])

            new_track = LightweightTrack(
                track_id=self.next_id,
                bbox=det.bbox,
                center=center,
                features=det.features,
            )
            self.tracks.append(new_track)
            det.track_id = self.next_id
            self.next_id += 1

        # 古いトラックを削除
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Optical Flow用に初期化
        if self.of_tracker is not None:
            self.of_tracker.initialize(frame, detections)

        logger.debug(
            "Updated with detections: %d matched, %d new, %d total tracks",
            len(matched),
            len(unmatched_dets),
            len(self.tracks),
        )

        return detections

    def interpolate(self, frame: np.ndarray) -> list[tuple[int, tuple[float, float, float, float]]]:
        """非検出フレームでの位置補間（軽い処理）

        Args:
            frame: 現在のフレーム

        Returns:
            (track_id, bbox) のリスト。bboxはOptical Flow/Kalman予測による推定値
        """
        results: list[tuple[int, tuple[float, float, float, float]]] = []

        if self.of_tracker is not None:
            # Optical Flowで追跡
            tracked_positions = self.of_tracker.track(frame)

            for track in self.tracks:
                if track.track_id in tracked_positions:
                    # Optical Flow成功
                    new_center = tracked_positions[track.track_id]
                    _x, _y, w, h = track.bbox
                    new_bbox = (
                        float(new_center[0] - w / 2),
                        float(new_center[1] - h / 2),
                        w,
                        h,
                    )
                    results.append((track.track_id, new_bbox))
                    # Kalmanも更新
                    track.update(new_bbox, new_center)
                else:
                    # Optical Flow失敗 → Kalman予測を使用
                    predicted_center = track.predict()
                    _x, _y, w, h = track.bbox
                    new_bbox = (
                        float(predicted_center[0] - w / 2),
                        float(predicted_center[1] - h / 2),
                        w,
                        h,
                    )
                    results.append((track.track_id, new_bbox))
        else:
            # Kalman予測のみ
            for track in self.tracks:
                predicted_center = track.predict()
                _x, _y, w, h = track.bbox
                new_bbox = (
                    float(predicted_center[0] - w / 2),
                    float(predicted_center[1] - h / 2),
                    w,
                    h,
                )
                results.append((track.track_id, new_bbox))

        logger.debug("Interpolated %d tracks", len(results))
        return results

    def _match_detections(
        self,
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """IoUベースのマッチング

        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_track_indices)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # IoU行列を計算
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(track.bbox, det.bbox)

        # グリーディマッチング（簡易版）
        matched = []
        matched_tracks = set()
        matched_dets = set()

        # IoU降順でマッチング
        while True:
            if iou_matrix.size == 0:
                break

            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            flat_idx = int(iou_matrix.argmax())
            t_idx, d_idx = divmod(flat_idx, iou_matrix.shape[1])

            matched.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

            # マッチ済みを除外
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _compute_iou(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """IoUを計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 交差領域
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # 各領域
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self) -> None:
        """トラッカーをリセット"""
        self.tracks = []
        self.next_id = 1
        if self.of_tracker:
            self.of_tracker.reset()
        logger.info("LightweightTracker reset")

    def get_active_tracks(self) -> list[LightweightTrack]:
        """アクティブなトラックを取得"""
        return [t for t in self.tracks if t.time_since_update == 0]
