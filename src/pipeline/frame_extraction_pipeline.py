"""Frame extraction pipeline for 5-minute interval timestamp-based sampling."""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2
from src.video.frame_sampler import CoarseSampler, FineSampler

logger = logging.getLogger(__name__)


class FrameExtractionPipeline:
    """5分刻みフレーム抽出のメインパイプライン

    タイムラプス動画から5分刻みのタイムスタンプを持つフレームを
    高精度で抽出します。
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        interval_minutes: int = 5,
        tolerance_seconds: float = 10.0,
        confidence_threshold: float = 0.7,
        coarse_interval_seconds: float = 10.0,
        fine_search_window_seconds: float = 30.0,
        fps: float = 30.0,
        roi_config: Dict[str, float] = None,
        enabled_ocr_engines: List[str] = None,
    ):
        """FrameExtractionPipelineを初期化

        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ
            start_datetime: 開始日時（Noneの場合は動画の最初から）
            end_datetime: 終了日時（Noneの場合は動画の最後まで）
            interval_minutes: 抽出間隔（分）
            tolerance_seconds: 許容誤差（秒）
            confidence_threshold: 信頼度閾値
            coarse_interval_seconds: 粗サンプリング間隔（秒）
            fine_search_window_seconds: 精密サンプリングの探索ウィンドウ（秒）
            fps: 動画のフレームレート
            roi_config: ROI設定
            enabled_ocr_engines: 有効にするOCRエンジンのリスト
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interval_minutes = interval_minutes
        self.tolerance_seconds = tolerance_seconds

        # サンプラーと抽出器を初期化
        self.coarse_sampler = CoarseSampler(
            video_path, interval_seconds=coarse_interval_seconds
        )
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.fine_sampler = FineSampler(
            self.video_cap, search_window=fine_search_window_seconds
        )

        self.extractor = TimestampExtractorV2(
            confidence_threshold=confidence_threshold,
            roi_config=roi_config,
            fps=fps,
            enabled_ocr_engines=enabled_ocr_engines,
        )

        # 目標タイムスタンプ生成
        if start_datetime is None:
            # デフォルト: 計画書に記載の日時
            start_datetime = datetime(2025, 8, 26, 16, 5, 0)
        if end_datetime is None:
            # デフォルト: 計画書に記載の日時
            end_datetime = datetime(2025, 8, 29, 13, 45, 0)

        self.target_timestamps = self._generate_target_timestamps(
            start=start_datetime, end=end_datetime, interval_minutes=interval_minutes
        )

    def _generate_target_timestamps(
        self, start: datetime, end: datetime, interval_minutes: int
    ) -> List[datetime]:
        """5分刻みの目標タイムスタンプリストを生成

        Args:
            start: 開始日時
            end: 終了日時
            interval_minutes: 間隔（分）

        Returns:
            目標タイムスタンプのリスト
        """
        targets = []
        current = start
        while current <= end:
            targets.append(current)
            current += timedelta(minutes=interval_minutes)
        return targets

    def run(self) -> List[Dict[str, any]]:
        """パイプライン実行

        Returns:
            抽出結果のリスト
        """
        results = []

        logger.info(
            f"Starting frame extraction for {len(self.target_timestamps)} target timestamps"
        )

        try:
            for target_ts in tqdm(self.target_timestamps, desc="Extracting frames"):
                result = self._extract_frame_for_target(target_ts)
                if result:
                    results.append(result)
                    self._save_frame(result)
                else:
                    logger.warning(f"Failed to extract frame for {target_ts}")

            # 結果をCSV保存
            self._save_results_csv(results)

            logger.info(
                f"Extraction completed: {len(results)}/{len(self.target_timestamps)} frames extracted"
            )
            return results

        finally:
            self.cleanup()

    def _extract_frame_for_target(
        self, target_ts: datetime
    ) -> Optional[Dict[str, any]]:
        """目標タイムスタンプに最も近いフレームを抽出

        Args:
            target_ts: 目標タイムスタンプ

        Returns:
            抽出結果の辞書。失敗した場合はNone
        """
        # Phase 1: 粗サンプリングで近傍を探す
        approx_frame_idx = self._find_approximate_frame(target_ts)

        if approx_frame_idx is None:
            return None

        # Phase 2: 精密サンプリングでベストフレームを探す
        best_frame = self._find_best_frame_around(target_ts, approx_frame_idx)

        return best_frame

    def _find_approximate_frame(self, target_ts: datetime) -> Optional[int]:
        """粗サンプリングで目標時刻の近傍フレームを特定

        Args:
            target_ts: 目標タイムスタンプ

        Returns:
            近似フレーム番号。見つからない場合はNone
        """
        min_diff = timedelta(days=999)
        approx_frame_idx = None

        # 粗サンプリングは進捗が分かりにくいため、サイレントモードで実行
        # （メインループでプログレスバーが表示されるため）
        for frame_idx, frame in self.coarse_sampler.sample():
            result = self.extractor.extract(frame, frame_idx)

            if result and result.get("timestamp"):
                timestamp = result["timestamp"]
                diff = abs(timestamp - target_ts)

                if diff < min_diff:
                    min_diff = diff
                    approx_frame_idx = frame_idx

                # 目標時刻を過ぎたら終了
                if timestamp > target_ts + timedelta(minutes=1):
                    break

        return approx_frame_idx

    def _find_best_frame_around(
        self, target_ts: datetime, approx_frame_idx: int
    ) -> Optional[Dict[str, any]]:
        """精密サンプリングで±10秒以内のベストフレームを探す

        Args:
            target_ts: 目標タイムスタンプ
            approx_frame_idx: 近似フレーム番号

        Returns:
            最良のフレーム結果。見つからない場合はNone
        """
        candidates = []

        for frame_idx, frame in self.fine_sampler.sample_around_target(
            approx_frame_idx
        ):
            result = self.extractor.extract(frame, frame_idx)

            if result and result.get("timestamp"):
                timestamp = result["timestamp"]
                diff = abs((timestamp - target_ts).total_seconds())

                # ±10秒以内なら候補に追加
                if diff <= self.tolerance_seconds:
                    candidates.append(
                        {
                            **result,
                            "frame": frame,
                            "time_diff": diff,
                            "target_timestamp": target_ts,
                        }
                    )

        if not candidates:
            logger.warning(
                f"No frames within ±{self.tolerance_seconds}s of {target_ts}"
            )
            return None

        # 時間差が最小のフレームを選択
        best = min(candidates, key=lambda x: x["time_diff"])
        logger.info(
            f"Best frame for {target_ts}: {best['timestamp']} "
            f"(diff={best['time_diff']:.1f}s, confidence={best['confidence']:.2f})"
        )

        return best

    def _save_frame(self, result: Dict[str, any]) -> None:
        """抽出したフレームを保存

        Args:
            result: 抽出結果の辞書
        """
        timestamp = result["timestamp"]
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"frame_{timestamp_str}.jpg"

        frame = result.get("frame")
        if frame is not None:
            cv2.imwrite(str(output_path), frame)
            logger.debug(f"Saved frame: {output_path}")

    def _save_results_csv(self, results: List[Dict[str, any]]) -> None:
        """結果をCSVで保存

        Args:
            results: 抽出結果のリスト
        """
        csv_path = self.output_dir / "extraction_results.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "target_timestamp",
                "extracted_timestamp",
                "frame_index",
                "confidence",
                "time_diff_seconds",
                "ocr_text",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                timestamp = r["timestamp"]
                target_ts = r.get("target_timestamp", timestamp)
                writer.writerow(
                    {
                        "target_timestamp": target_ts.strftime("%Y/%m/%d %H:%M:%S"),
                        "extracted_timestamp": timestamp.strftime("%Y/%m/%d %H:%M:%S"),
                        "frame_index": r["frame_idx"],
                        "confidence": f"{r['confidence']:.4f}",
                        "time_diff_seconds": f"{r.get('time_diff', 0):.2f}",
                        "ocr_text": r.get("ocr_text", ""),
                    }
                )

        logger.info(f"Results saved: {csv_path}")

    def cleanup(self) -> None:
        """リソースを解放"""
        self.coarse_sampler.close()
        if self.video_cap is not None:
            self.video_cap.release()
        self.extractor.reset_validator()
