"""Tracking phase of the pipeline."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.detection import ViTDetector
from src.models import Detection
from src.pipeline.base_phase import BasePhase
from src.tracking import Tracker
from src.tracking.track import Track
from src.utils.export_utils import TrajectoryExporter


class TrackingPhase(BasePhase):
    """オブジェクト追跡フェーズ"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.tracker: Optional[Tracker] = None
        self.detector: Optional[ViTDetector] = None  # 特徴量抽出用
        self.tracks: List[Track] = []

    def initialize(self) -> None:
        """トラッカーを初期化"""
        self.log_phase_start("フェーズ2.5: オブジェクト追跡")

        # 追跡が有効かチェック
        tracking_enabled = self.config.get("tracking.enabled", False)
        if not tracking_enabled:
            self.logger.info("追跡機能が無効です（tracking.enabled=false）")
            return

        # Trackerの初期化
        max_age = self.config.get("tracking.max_age", 30)
        min_hits = self.config.get("tracking.min_hits", 3)
        iou_threshold = self.config.get("tracking.iou_threshold", 0.3)
        appearance_weight = self.config.get("tracking.appearance_weight", 0.7)
        motion_weight = self.config.get("tracking.motion_weight", 0.3)

        self.tracker = Tracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
        )

        self.logger.info(f"Tracker initialized: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
        self.logger.info(f"  appearance_weight={appearance_weight}, motion_weight={motion_weight}")

        # 特徴量抽出用の検出器を初期化（既存の検出器があれば再利用）
        # ここでは新規に作成（検出フェーズとは別インスタンス）
        model_name = self.config.get("detection.model_name")
        confidence_threshold = self.config.get("detection.confidence_threshold")
        device = self.config.get("detection.device")

        self.detector = ViTDetector(model_name, confidence_threshold, device)
        self.detector.load_model()
        self.logger.info("特徴量抽出用の検出器を初期化しました")

    def execute(
        self,
        detection_results: list[tuple[int, str, list[Detection]]],
        sample_frames: list[tuple[int, str, np.ndarray]],
    ) -> list[tuple[int, str, list[Detection]]]:
        """追跡処理を実行

        Args:
            detection_results: 検出結果のリスト [(frame_num, timestamp, detections), ...]
            sample_frames: サンプルフレームのリスト [(frame_num, timestamp, frame), ...]

        Returns:
            track_idが割り当てられた検出結果のリスト
        """
        if not self.config.get("tracking.enabled", False):
            self.logger.info("追跡機能が無効のため、検出結果をそのまま返します")
            return detection_results

        if self.tracker is None or self.detector is None:
            raise RuntimeError("トラッカーまたは検出器が初期化されていません。initialize()を先に呼び出してください。")

        tracked_results = []

        # フレームをframe_numでインデックス化
        frame_dict = {frame_num: frame for frame_num, _, frame in sample_frames}

        for frame_num, timestamp, detections in tqdm(detection_results, desc="追跡処理中"):
            # frame_numでフレームを取得
            frame = frame_dict.get(frame_num)
            if frame is None:
                self.logger.warning(f"フレーム #{frame_num}: 対応する画像が見つかりません")
                tracked_results.append((frame_num, timestamp, detections))
                continue

            # 特徴量抽出（検出結果がある場合のみ）
            if detections:
                try:
                    features = self.detector.extract_features(frame, detections)
                    # 各検出結果に特徴量を割り当て
                    for i, detection in enumerate(detections):
                        if i < len(features):
                            detection.features = features[i]
                except Exception as e:
                    self.logger.warning(f"フレーム #{frame_num}: 特徴量抽出に失敗しました: {e}")
                    # 特徴量なしで続行

            # トラッカーで更新
            tracked_detections = self.tracker.update(detections)

            # 結果を保存
            tracked_results.append((frame_num, timestamp, tracked_detections))

            self.logger.debug(
                f"フレーム #{frame_num} ({timestamp}): " f"{len(detections)}検出 → {len(tracked_detections)}追跡"
            )

        # 最終的なトラックを取得
        self.tracks = self.tracker.get_confirmed_tracks()

        self.logger.info(f"追跡処理が完了: {len(tracked_results)}フレーム, {len(self.tracks)}トラック")

        return tracked_results

    def export_results(self, output_path: Path) -> None:
        """追跡結果をエクスポート

        Args:
            output_path: 出力ディレクトリ
        """
        if not self.config.get("tracking.enabled", False):
            return

        if not self.tracks:
            self.logger.warning("エクスポートするトラックがありません")
            return

        # TrajectoryExporterを使用してエクスポート
        exporter = TrajectoryExporter(output_path)

        # JSON形式でエクスポート
        try:
            json_path = exporter.export_json(self.tracks, filename="tracks.json")
            self.logger.info(f"追跡結果をJSONに出力しました: {json_path}")
        except Exception as e:
            self.logger.error(f"追跡結果のJSON出力に失敗しました: {e}")

        # CSV形式でエクスポート
        try:
            csv_path = exporter.export_csv(self.tracks, filename="tracks.csv")
            self.logger.info(f"追跡結果をCSVに出力しました: {csv_path}")
        except Exception as e:
            self.logger.error(f"追跡結果のCSV出力に失敗しました: {e}")

        # 統計情報をログ出力
        total_points = sum(len(track.trajectory) for track in self.tracks)
        avg_trajectory_length = total_points / len(self.tracks) if self.tracks else 0

        self.logger.info("=" * 80)
        self.logger.info("追跡統計:")
        self.logger.info(f"  総トラック数: {len(self.tracks)}")
        self.logger.info(f"  総軌跡点数: {total_points}")
        self.logger.info(f"  平均軌跡長: {avg_trajectory_length:.2f}点/トラック")
        self.logger.info("=" * 80)

        # 統計情報をJSONファイルに出力
        stats_path = output_path / "tracking_statistics.json"
        try:
            stats_dict = {
                "total_tracks": len(self.tracks),
                "total_trajectory_points": total_points,
                "avg_trajectory_length": avg_trajectory_length,
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "age": track.age,
                        "hits": track.hits,
                        "trajectory_length": len(track.trajectory),
                    }
                    for track in self.tracks
                ],
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_dict, f, indent=2, ensure_ascii=False)
            self.logger.info(f"追跡統計情報をJSONに出力しました: {stats_path}")
        except Exception as e:
            self.logger.error(f"追跡統計情報のJSON出力に失敗しました: {e}")

    def get_tracks(self) -> List[Track]:
        """追跡結果のトラックを取得

        Returns:
            トラックのリスト
        """
        return self.tracks.copy()

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.detector:
            # 検出器のクリーンアップ（必要に応じて）
            pass
