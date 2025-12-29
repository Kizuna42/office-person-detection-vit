"""Detection phase of the pipeline."""

import gc
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import ConfigManager
from src.core.policy import OutputPolicy
from src.detection import YOLOv8Detector
from src.models import Detection
from src.pipeline.phases.base import BasePhase
from src.utils import PerformanceMonitor, calculate_detection_statistics, save_detection_image


class DetectionPhase(BasePhase):
    """人物検出フェーズ（YOLOv8）"""

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.detector: YOLOv8Detector | None = None
        self.output_path: Path | None = None  # 検出画像の保存先
        self.performance_monitor = PerformanceMonitor()

    def initialize(self) -> None:
        """検出器を初期化"""
        self.log_phase_start("フェーズ2: YOLOv8人物検出")

        confidence_threshold = self.config.get("detection.confidence_threshold", 0.25)
        device = self.config.get("detection.device")
        model_path = self.config.get("detection.yolov8_model_path", "runs/detect/person_ft/weights/best.pt")
        iou_threshold = self.config.get("detection.iou_threshold", 0.45)

        self.logger.info(f"モデル: {model_path}")
        self.logger.info(f"信頼度閾値: {confidence_threshold}")
        self.logger.info(f"IoU閾値: {iou_threshold}")
        self.logger.info(f"デバイス: {device}")

        self.detector = YOLOv8Detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            iou_threshold=iou_threshold,
        )
        self.detector.load_model()

    def execute(
        self,
        sample_frames: list[tuple[int, str, np.ndarray]],
        output_policy: OutputPolicy | None = None,
    ) -> list[tuple[int, str, list[Detection]]]:
        """人物検出処理を実行

        Args:
            sample_frames: サンプルフレームのリスト [(frame_num, timestamp, frame), ...]

        Returns:
            検出結果のリスト [(frame_num, timestamp, detections), ...]
        """
        if self.detector is None:
            raise RuntimeError("検出器が初期化されていません。initialize()を先に呼び出してください。")

        results = []
        save_detection_images = (
            output_policy.save_detection_images
            if output_policy is not None
            else self.config.get("output.save_detection_images", True)
        )
        # output_pathが設定されている場合はそれを使用、なければ設定から取得
        if self.output_path:
            detection_images_dir = self.output_path / "images"
        else:
            detection_images_dir = Path(self.config.get("output.directory", "output")) / "detections"

        # ディレクトリを確実に作成
        detection_images_dir.mkdir(parents=True, exist_ok=True)

        if save_detection_images:
            self.logger.info(f"検出画像の保存先: {detection_images_dir}")

        # フレームごとに特徴量付き検出を実行
        for frame_num, timestamp, frame in tqdm(sample_frames, desc="人物検出中"):
            try:
                with self.performance_monitor.measure("detection_batch"):
                    detections, features = self.detector.detect_with_features(frame)

                if features.shape[0] != len(detections):
                    self.logger.warning(
                        f"フレーム #{frame_num}: 特徴量数と検出数が不一致 "
                        f"(features={features.shape[0]}, detections={len(detections)})"
                    )

                results.append((frame_num, timestamp, detections))
                self.logger.info(f"フレーム #{frame_num} ({timestamp}): {len(detections)}人検出")

                # 検出画像を保存（オプション）
                if save_detection_images:
                    if detections:
                        self.logger.debug(
                            f"検出画像を保存します: {detection_images_dir}, "
                            f"タイムスタンプ={timestamp}, 検出数={len(detections)}"
                        )
                        save_detection_image(
                            frame,
                            detections,
                            timestamp,
                            detection_images_dir,
                            self.logger,
                        )
                    else:
                        self.logger.debug(f"フレーム #{frame_num}: 検出結果が空のため画像を保存しません")
                else:
                    self.logger.debug(f"フレーム #{frame_num}: save_detection_imagesがFalseのため画像を保存しません")

            except Exception as e:
                self.logger.error(f"フレーム #{frame_num} の検出処理に失敗しました: {e}", exc_info=True)
                results.append((frame_num, timestamp, []))
                self.logger.warning(f"フレーム #{frame_num} をスキップしました")
            finally:
                # 定期的にガベージコレクションを実行
                if (frame_num + 1) % 10 == 0:
                    gc.collect()

        return results

    def log_statistics(
        self,
        detection_results: list[tuple[int, str, list[Detection]]],
        output_path: Path,
    ) -> None:
        """統計情報を計算・出力

        Args:
            detection_results: 検出結果のリスト
            output_path: 出力ディレクトリ
        """
        stats = calculate_detection_statistics(detection_results)

        # パフォーマンス統計を表示
        perf_stats = self.performance_monitor.get_metrics("detection_batch")
        if perf_stats and perf_stats.get("count", 0) > 0:
            avg_batch_time = perf_stats["total_time"] / perf_stats["count"]
            self.logger.info(f"バッチ検出平均処理時間: {avg_batch_time:.3f}秒/バッチ")

        # 統計情報を表示
        self.logger.info("=" * 80)
        self.logger.info("検出統計:")
        self.logger.info(f"  総検出数: {stats.total_detections}人")
        self.logger.info(f"  平均検出数: {stats.avg_detections_per_frame:.2f}人/フレーム")

        if stats.total_detections > 0:
            self.logger.info("  信頼度スコア統計:")
            self.logger.info(f"    平均: {stats.confidence_mean:.4f}")
            self.logger.info(f"    最小: {stats.confidence_min:.4f}")
            self.logger.info(f"    最大: {stats.confidence_max:.4f}")
            self.logger.info(f"    標準偏差: {stats.confidence_std:.4f}")
            self.logger.info(f"    中央値: {stats.confidence_median:.4f}")
        self.logger.info("=" * 80)

        # 統計情報をJSONファイルに出力
        detection_stats_path = output_path / "detection_statistics.json"
        try:
            stats_dict = {
                "total_detections": stats.total_detections,
                "avg_detections_per_frame": stats.avg_detections_per_frame,
                "confidence": {
                    "mean": stats.confidence_mean,
                    "min": stats.confidence_min,
                    "max": stats.confidence_max,
                    "std": stats.confidence_std,
                    "median": stats.confidence_median,
                },
                "frame_count": stats.frame_count,
            }
            with open(detection_stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_dict, f, indent=2, ensure_ascii=False)
            self.logger.info(f"検出統計情報をJSONに出力しました: {detection_stats_path}")
        except Exception as e:
            self.logger.error(f"検出統計情報のJSON出力に失敗しました: {e}")

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.detector is not None:
            del self.detector
            self.detector = None
        gc.collect()
