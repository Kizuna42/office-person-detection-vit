"""MOT (Multiple Object Tracking) metrics evaluation module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.tracking.track import Track

logger = logging.getLogger(__name__)


class MOTMetrics:
    """MOT (Multiple Object Tracking) メトリクス評価クラス

    追跡精度を評価するためのMOTメトリクスを計算します。
    """

    def __init__(self):
        """MOTMetricsを初期化"""
        logger.info("MOTMetrics initialized")

    def calculate_mota(
        self,
        ground_truth_tracks: list[dict],
        predicted_tracks: list[Track],
        _frame_count: int,
    ) -> float:
        """MOTA (Multiple Object Tracking Accuracy) を計算

        Args:
            ground_truth_tracks: Ground Truthトラックのリスト
            predicted_tracks: 予測トラックのリスト
            frame_count: 総フレーム数

        Returns:
            MOTAスコア (0.0-1.0、高いほど良い)
        """
        # 簡易実装: 実際の実装ではより詳細なマッチングが必要
        # ここでは基本的な計算のみ実装

        total_objects = sum(len(gt_track.get("trajectory", [])) for gt_track in ground_truth_tracks)
        if total_objects == 0:
            return 0.0

        # マッチング数を計算（簡易版）
        matches = self._count_matches(ground_truth_tracks, predicted_tracks)
        false_positives = len(predicted_tracks) - matches
        false_negatives = len(ground_truth_tracks) - matches
        id_switches = self._count_id_switches(predicted_tracks)

        # MOTA = 1 - (FN + FP + IDSW) / GT
        mota = 1.0 - (false_negatives + false_positives + id_switches) / total_objects
        mota = max(0.0, min(1.0, mota))  # 0-1の範囲にクリップ

        logger.info(
            f"MOTA calculation: matches={matches}, FP={false_positives}, "
            f"FN={false_negatives}, IDSW={id_switches}, MOTA={mota:.3f}"
        )

        return float(mota)

    def calculate_idf1(
        self,
        ground_truth_tracks: list[dict],
        predicted_tracks: list[Track],
    ) -> float:
        """IDF1 (ID F1 Score) を計算

        Args:
            ground_truth_tracks: Ground Truthトラックのリスト
            predicted_tracks: 予測トラックのリスト

        Returns:
            IDF1スコア (0.0-1.0、高いほど良い)
        """
        # 簡易実装
        total_gt_ids = len(ground_truth_tracks)
        total_pred_ids = len(predicted_tracks)

        if total_gt_ids == 0 and total_pred_ids == 0:
            return 1.0

        if total_gt_ids == 0 or total_pred_ids == 0:
            return 0.0

        # IDマッチング数を計算（簡易版）
        id_matches = self._count_id_matches(ground_truth_tracks, predicted_tracks)

        # IDF1 = 2 * IDTP / (IDTP + IDFP + IDFN)
        idtp = id_matches
        idfp = total_pred_ids - id_matches
        idfn = total_gt_ids - id_matches

        if idtp + idfp + idfn == 0:
            return 1.0

        idf1 = 2.0 * idtp / (idtp + idfp + idfn)
        idf1 = max(0.0, min(1.0, idf1))

        logger.info(f"IDF1 calculation: IDTP={idtp}, IDFP={idfp}, IDFN={idfn}, IDF1={idf1:.3f}")

        return float(idf1)

    def calculate_tracking_metrics(
        self,
        ground_truth_tracks: list[dict],
        predicted_tracks: list[Track],
        frame_count: int,
    ) -> dict[str, float]:
        """追跡メトリクスを一括計算

        Args:
            ground_truth_tracks: Ground Truthトラックのリスト
            predicted_tracks: 予測トラックのリスト
            frame_count: 総フレーム数

        Returns:
            メトリクスの辞書
        """
        mota = self.calculate_mota(ground_truth_tracks, predicted_tracks, frame_count)
        idf1 = self.calculate_idf1(ground_truth_tracks, predicted_tracks)
        id_switches = self._count_id_switches(predicted_tracks)

        return {
            "MOTA": mota,
            "IDF1": idf1,
            "ID_Switches": float(id_switches),
        }

    def _count_matches(self, ground_truth_tracks: list[dict], predicted_tracks: list[Track]) -> int:
        """マッチング数を計算（簡易版）

        Args:
            ground_truth_tracks: Ground Truthトラックのリスト
            predicted_tracks: 予測トラックのリスト

        Returns:
            マッチング数
        """
        # 簡易実装: 実際の実装ではIoUベースのマッチングが必要
        matches = 0
        for gt_track in ground_truth_tracks:
            gt_trajectory = gt_track.get("trajectory", [])
            if not gt_trajectory:
                continue

            # 最初の点で最も近い予測トラックを見つける
            gt_first_pt = gt_trajectory[0]
            min_distance = float("inf")
            best_match = None

            for pred_track in predicted_tracks:
                if not pred_track.trajectory:
                    continue

                pred_first_pt = pred_track.trajectory[0]
                distance = np.sqrt(
                    (gt_first_pt.get("x", 0) - pred_first_pt[0]) ** 2
                    + (gt_first_pt.get("y", 0) - pred_first_pt[1]) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    best_match = pred_track

            if best_match and min_distance < 50.0:  # 閾値: 50ピクセル
                matches += 1

        return matches

    def _count_id_matches(self, ground_truth_tracks: list[dict], predicted_tracks: list[Track]) -> int:
        """IDマッチング数を計算（簡易版）

        Args:
            ground_truth_tracks: Ground Truthトラックのリスト
            predicted_tracks: 予測トラックのリスト

        Returns:
            IDマッチング数
        """
        # 簡易実装: 実際の実装ではより詳細なマッチングが必要
        return min(len(ground_truth_tracks), len(predicted_tracks))

    def _count_id_switches(self, _predicted_tracks: list[Track]) -> int:
        """IDスイッチ数を計算

        Args:
            predicted_tracks: 予測トラックのリスト

        Returns:
            IDスイッチ数（簡易実装では0を返す）
        """
        # 簡易実装: 実際の実装では軌跡の連続性をチェック
        # ここでは基本的な実装のみ
        return 0
