"""Tracking phase of the pipeline."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import ConfigManager
from src.core.policy import OutputPolicy
from src.detection import ViTDetector, YOLOv8Detector
from src.models import Detection
from src.pipeline.phases.base import BasePhase
from src.tracking import LightweightTracker, ReIDFeatureExtractor, Tracker
from src.tracking.track import Track
from src.utils.export_utils import TrajectoryExporter
from src.utils.image_utils import save_tracked_detection_image


class TrackingPhase(BasePhase):
    """オブジェクト追跡フェーズ"""

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.tracker: Tracker | None = None
        self.lightweight_tracker: LightweightTracker | None = None  # ハイブリッドモード用
        self.detector: ViTDetector | YOLOv8Detector | None = None  # 特徴量抽出用
        self.reid_extractor: ReIDFeatureExtractor | None = None  # Phase2: Re-ID特徴抽出
        self._detector_shared: bool = False  # 検出器が共有されているかどうか
        self._hybrid_mode: bool = False  # ハイブリッドモード使用フラグ
        self._reid_enabled: bool = False  # Re-ID特徴抽出使用フラグ
        self._dense_tracking_enabled: bool = False  # 高密度トラッキング（Phase 2）
        self._output_interval_minutes: int = 5  # 出力集約間隔
        self.tracks: list[Track] = []
        self.tracked_results: list[tuple[int, str, list[Detection]]] = []  # 画像出力用
        self.sample_frames: list[tuple[int, str, np.ndarray]] = []  # 画像出力用

    def set_detector(self, detector: ViTDetector | YOLOv8Detector) -> None:
        """検出器を外部から設定（共有用）

        Args:
            detector: 共有する検出器インスタンス
        """
        self.detector = detector
        self._detector_shared = True
        self.logger.info("検出器が外部から共有されました（メモリ効率化）")

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
        max_position_distance = self.config.get("tracking.max_position_distance", 150.0)

        self.tracker = Tracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
            max_position_distance=max_position_distance,
        )

        self.logger.info(f"Tracker initialized: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
        self.logger.info(f"  appearance_weight={appearance_weight}, motion_weight={motion_weight}")
        self.logger.info(f"  max_position_distance={max_position_distance}")

        # ハイブリッドモード: LightweightTrackerを追加初期化
        self._hybrid_mode = self.config.get("tracking.hybrid_mode.enabled", False)
        if self._hybrid_mode:
            use_optical_flow = self.config.get("tracking.hybrid_mode.use_optical_flow", True)
            self.lightweight_tracker = LightweightTracker(
                max_age=max_age,
                iou_threshold=iou_threshold,
                use_optical_flow=use_optical_flow,
            )
            self.logger.info(
                "ハイブリッドモード有効: LightweightTracker初期化 (optical_flow=%s)",
                use_optical_flow,
            )

        # 特徴量抽出用の検出器を初期化（既存の検出器があれば再利用）
        if self.detector is not None:
            self.logger.info("検出器は既にセットされています（共有モード）")
        else:
            # 検出器が未設定の場合のみ新規作成
            detector_type = self.config.get("detection.detector_type", "yolov8")
            confidence_threshold = self.config.get("detection.confidence_threshold", 0.25)
            device = self.config.get("detection.device")

            if detector_type == "yolov8":
                model_path = self.config.get("detection.yolov8_model_path", "runs/detect/person_ft/weights/best.pt")
                iou_threshold = self.config.get("detection.iou_threshold", 0.45)
                self.detector = YOLOv8Detector(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    iou_threshold=iou_threshold,
                )
                self.logger.info(f"YOLOv8Detector初期化: model={model_path}, conf={confidence_threshold}")
            else:
                model_name = self.config.get("detection.model_name", "facebook/detr-resnet-50")
                self.detector = ViTDetector(model_name, confidence_threshold, device)
                self.logger.info(f"ViTDetector初期化: model={model_name}, conf={confidence_threshold}")

            self.detector.load_model()
            self.logger.info("特徴量抽出用の検出器を新規初期化しました")

        # Phase2: Re-ID特徴抽出の初期化
        self._reid_enabled = self.config.get("tracking.reid.enabled", False)
        if self._reid_enabled:
            model_type = self.config.get("tracking.reid.model_type", "clip")
            model_name = self.config.get("tracking.reid.model_name", "openai/clip-vit-base-patch32")
            model_path = self.config.get("tracking.reid.model_path", None)
            device = self.config.get("detection.device", "mps")
            self.reid_extractor = ReIDFeatureExtractor(
                model_type=model_type,
                model_name=model_name,
                model_path=model_path,
                device=device,
            )
            self.reid_extractor.load_model()
            self.logger.info(f"Re-ID特徴抽出を有効化: model_type={model_type}")

        # Phase 2: 高密度トラッキング設定
        self._dense_tracking_enabled = self.config.get("video.dense_tracking.enabled", False)
        if self._dense_tracking_enabled:
            self._output_interval_minutes = self.config.get("video.dense_tracking.output_interval_minutes", 5)
            tracking_interval = self.config.get("video.dense_tracking.tracking_interval_seconds", 10)
            self.logger.info(
                f"高密度トラッキング有効: 処理間隔={tracking_interval}秒, 出力間隔={self._output_interval_minutes}分"
            )

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
        # 画像出力用にsample_framesを保存
        self.sample_frames = sample_frames

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
                # Phase2: Re-ID特徴抽出が有効な場合はCLIP特徴量を使用
                if self._reid_enabled and self.reid_extractor is not None:
                    bboxes = [det.bbox for det in detections]
                    reid_features = self.reid_extractor.extract_features(frame, bboxes)

                    # 各検出結果にRe-ID特徴量を割り当て
                    for i, det in enumerate(detections):
                        if i < len(reid_features):
                            det.features = reid_features[i]

                    self.logger.debug(f"フレーム #{frame_num}: Re-ID特徴量抽出完了 ({len(detections)} detections)")
                else:
                    # 従来のDETR特徴量を使用
                    missing_indices = [i for i, det in enumerate(detections) if det.features is None]
                    if missing_indices:
                        self.logger.warning(
                            "フレーム #%d: %d/%d 検出で特徴量が欠落しています（再推論は実施しません） indices=%s",
                            frame_num,
                            len(missing_indices),
                            len(detections),
                            missing_indices,
                        )
                    else:
                        self.logger.debug(f"フレーム #{frame_num}: 検出フェーズの特徴量を再利用します")

            # トラッカーで更新
            tracked_detections = self.tracker.update(detections)

            # 結果を保存
            tracked_results.append((frame_num, timestamp, tracked_detections))

            self.logger.debug(
                f"フレーム #{frame_num} ({timestamp}): {len(detections)}検出 → {len(tracked_detections)}追跡"
            )

        # 最終的なトラックを取得
        self.tracks = self.tracker.get_confirmed_tracks()
        # 画像出力用にtracked_resultsを保存
        self.tracked_results = tracked_results

        self.logger.info(f"追跡処理が完了: {len(tracked_results)}フレーム, {len(self.tracks)}トラック")

        # Phase 2: 高密度トラッキングの場合は出力間隔に集約
        if self._dense_tracking_enabled and len(tracked_results) > 0:
            aggregated_results = self._aggregate_to_output_intervals(tracked_results)
            self.logger.info(
                f"出力集約: {len(tracked_results)}フレーム → {len(aggregated_results)}フレーム "
                f"({self._output_interval_minutes}分間隔)"
            )
            return aggregated_results

        return tracked_results

    def export_results(self, output_path: Path, output_policy: OutputPolicy | None = None) -> None:
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

        # MOTChallenge形式での出力（評価用）
        try:
            mot_path = self._export_mot_challenge_csv(output_path)
            self.logger.info(f"MOTChallenge形式で追跡結果を出力しました: {mot_path}")
        except Exception as e:
            self.logger.error(f"MOTChallenge形式の出力に失敗しました: {e}")

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

        # ID付き検出画像を出力（オプション）
        save_tracking_images = (
            output_policy.save_tracking_images
            if output_policy is not None
            else self.config.get("output.save_tracking_images", True)
        )
        if save_tracking_images and self.tracked_results and self.sample_frames:
            try:
                images_dir = output_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                # フレームをframe_numでインデックス化
                frame_dict = {frame_num: frame for frame_num, _, frame in self.sample_frames}

                # 各フレームの画像を保存
                saved_count = 0
                for frame_num, timestamp, detections in self.tracked_results:
                    frame = frame_dict.get(frame_num)
                    if frame is None:
                        self.logger.warning(
                            f"フレーム #{frame_num}: 対応する画像が見つかりません（画像出力をスキップ）"
                        )
                        continue

                    if detections:
                        save_tracked_detection_image(
                            frame,
                            detections,
                            timestamp,
                            images_dir,
                            self.logger,
                        )
                        saved_count += 1
                    else:
                        self.logger.debug(f"フレーム #{frame_num}: 検出結果が空のため画像を保存しません")

                self.logger.info(f"ID付き検出画像を {saved_count} 枚保存しました: {images_dir}")
            except Exception as e:
                self.logger.error(f"ID付き検出画像の保存に失敗しました: {e}", exc_info=True)

    def _export_mot_challenge_csv(self, output_path: Path) -> Path:
        """MOTChallenge形式の追跡結果CSVを出力する."""
        mot_path = output_path / "tracks_mot.csv"
        output_path.mkdir(parents=True, exist_ok=True)

        with mot_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

            for frame_num, _timestamp, detections in self.tracked_results:
                for det in detections:
                    if det.track_id is None:
                        continue

                    x, y, w, h = det.bbox
                    writer.writerow(
                        [
                            frame_num,
                            det.track_id,
                            f"{x:.2f}",
                            f"{y:.2f}",
                            f"{w:.2f}",
                            f"{h:.2f}",
                            f"{det.confidence:.4f}",
                            -1,
                            -1,
                            -1,
                        ]
                    )

        return mot_path

    def get_tracks(self) -> list[Track]:
        """追跡結果のトラックを取得

        Returns:
            トラックのリスト
        """
        return self.tracks.copy()

    def _aggregate_to_output_intervals(
        self,
        tracked_results: list[tuple[int, str, list[Detection]]],
    ) -> list[tuple[int, str, list[Detection]]]:
        """高密度トラッキング結果を出力間隔に集約

        10秒間隔の追跡結果を5分間隔に集約。
        各ウィンドウ内のtrack_idを保持し、代表フレームを選択。

        Args:
            tracked_results: 10秒間隔の追跡結果

        Returns:
            5分間隔に集約された追跡結果
        """
        from datetime import datetime

        if not tracked_results:
            return []

        # タイムスタンプをパース
        def parse_ts(ts_str: str) -> datetime | None:
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%H:%M:%S"]:
                try:
                    return datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
            return None

        # 出力間隔（秒）
        interval_seconds = self._output_interval_minutes * 60

        # 結果をタイムスタンプでグループ化
        aggregated = []
        current_window_start: datetime | None = None
        current_window_detections: list[Detection] = []
        current_window_frame_num: int = 0
        current_window_timestamp: str = ""
        seen_track_ids: set[int] = set()

        for frame_num, timestamp, detections in tracked_results:
            ts = parse_ts(timestamp)
            if ts is None:
                # パース失敗時は最初のフォーマットでそのまま使用
                if not aggregated:
                    current_window_frame_num = frame_num
                    current_window_timestamp = timestamp
                current_window_detections.extend(detections)
                continue

            if current_window_start is None:
                # 最初のウィンドウ開始
                current_window_start = ts
                current_window_frame_num = frame_num
                current_window_timestamp = timestamp

            # ウィンドウ境界をチェック
            time_diff = (ts - current_window_start).total_seconds()
            if time_diff >= interval_seconds:
                # 現在のウィンドウを確定
                if current_window_detections:
                    # 重複track_idを除去（最新の検出を優先）
                    unique_detections = self._deduplicate_detections(current_window_detections)
                    aggregated.append(
                        (
                            current_window_frame_num,
                            current_window_timestamp,
                            unique_detections,
                        )
                    )

                # 新しいウィンドウ開始
                current_window_start = ts
                current_window_detections = list(detections)
                current_window_frame_num = frame_num
                current_window_timestamp = timestamp
                seen_track_ids = {d.track_id for d in detections if d.track_id is not None}
            else:
                # 現在のウィンドウに追加
                for det in detections:
                    if det.track_id is not None and det.track_id not in seen_track_ids:
                        current_window_detections.append(det)
                        seen_track_ids.add(det.track_id)

        # 最後のウィンドウを確定
        if current_window_detections:
            unique_detections = self._deduplicate_detections(current_window_detections)
            aggregated.append(
                (
                    current_window_frame_num,
                    current_window_timestamp,
                    unique_detections,
                )
            )

        return aggregated

    def _deduplicate_detections(self, detections: list[Detection]) -> list[Detection]:
        """重複track_idの検出結果を除去（最新を優先）"""
        seen: dict[int, Detection] = {}
        for det in detections:
            if det.track_id is not None:
                seen[det.track_id] = det  # 後の検出で上書き（最新優先）
        return list(seen.values())

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        # 共有された検出器は削除しない（所有者が管理する）
        if self.detector and not self._detector_shared:
            del self.detector
            self.detector = None
        elif self._detector_shared:
            # 共有された検出器は参照を解除するだけ
            self.detector = None
            self._detector_shared = False

        # Re-ID extractorのクリーンアップ
        if self.reid_extractor is not None:
            self.reid_extractor.cleanup()
            self.reid_extractor = None
