"""Frame extraction pipeline for 5-minute interval timestamp-based sampling."""

import contextlib
import csv
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2, TimestampValidator
from src.video import VideoProcessor
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
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        interval_minutes: int = 5,
        tolerance_seconds: float = 10.0,
        confidence_threshold: float = 0.7,
        coarse_interval_seconds: float = 2.0,
        fine_search_window_seconds: float = 60.0,
        fine_interval_seconds: float = 0.1,
        fps: float = 30.0,
        roi_config: dict[str, float] | None = None,
        enabled_ocr_engines: list[str] | None = None,
        use_improved_validator: bool = False,
        base_tolerance_seconds: float = 10.0,
        history_size: int = 10,
        z_score_threshold: float = 2.0,
        use_weighted_consensus: bool = False,
        use_voting_consensus: bool = False,
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
            fine_interval_seconds: 精密サンプリング間隔（秒）
            fps: 動画のフレームレート
            roi_config: ROI設定
            enabled_ocr_engines: 有効にするOCRエンジンのリスト
            use_improved_validator: TemporalValidatorV2を使用するか（デフォルト: False）
            base_tolerance_seconds: ベース許容範囲（秒、TemporalValidatorV2用）
            history_size: 履歴サイズ（TemporalValidatorV2用）
            z_score_threshold: Z-score閾値（TemporalValidatorV2用）
            use_weighted_consensus: 重み付けスキームを使用するか（デフォルト: False）
            use_voting_consensus: 投票ロジックを使用するか（デフォルト: False）
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interval_minutes = interval_minutes
        self.tolerance_seconds = tolerance_seconds

        # サンプラーと抽出器を初期化
        self.coarse_sampler = CoarseSampler(video_path, interval_seconds=coarse_interval_seconds)
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        # 精密サンプリング間隔を設定
        self.fine_sampler = FineSampler(
            self.video_cap,
            search_window=fine_search_window_seconds,
            interval_seconds=fine_interval_seconds,
        )

        self.extractor = TimestampExtractorV2(
            confidence_threshold=confidence_threshold,
            roi_config=roi_config,
            fps=fps,
            enabled_ocr_engines=enabled_ocr_engines,
            use_improved_validator=use_improved_validator,
            base_tolerance_seconds=base_tolerance_seconds,
            history_size=history_size,
            z_score_threshold=z_score_threshold,
            use_weighted_consensus=use_weighted_consensus,
            use_voting_consensus=use_voting_consensus,
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

    def _generate_target_timestamps(self, start: datetime, end: datetime, interval_minutes: int) -> list[datetime]:
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

    def run(self) -> list[dict[str, Any]]:
        """パイプライン実行

        Returns:
            抽出結果のリスト
        """
        results: list[dict[str, Any]] = []

        logger.info(f"Starting frame extraction for {len(self.target_timestamps)} target timestamps")

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

            logger.info(f"Extraction completed: {len(results)}/{len(self.target_timestamps)} frames extracted")
            return results

        finally:
            self.cleanup()

    def _extract_frame_for_target(self, target_ts: datetime) -> dict[str, Any] | None:
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

    def _find_approximate_frame(self, target_ts: datetime) -> int | None:
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

    def _find_best_frame_around(self, target_ts: datetime, approx_frame_idx: int) -> dict[str, Any] | None:
        """精密サンプリングで±10秒以内のベストフレームを探す

        Args:
            target_ts: 目標タイムスタンプ
            approx_frame_idx: 近似フレーム番号

        Returns:
            最良のフレーム結果。見つからない場合はNone
        """
        candidates: list[dict[str, Any]] = []

        for frame_idx, frame in self.fine_sampler.sample_around_target(approx_frame_idx):
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
            logger.warning(f"No frames within ±{self.tolerance_seconds}s of {target_ts}")
            return None

        # 時間差が最小のフレームを選択
        best = min(candidates, key=lambda x: x["time_diff"])
        logger.info(
            f"Best frame for {target_ts}: {best['timestamp']} "
            f"(diff={best['time_diff']:.1f}s, confidence={best['confidence']:.2f})"
        )

        return best

    def _save_frame(self, result: dict[str, Any]) -> None:
        """抽出したフレームを保存

        Args:
            result: 抽出結果の辞書
        """
        # フレーム保存用のディレクトリ（frames/サブディレクトリを使用）
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result["timestamp"]
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        output_path = frames_dir / f"frame_{timestamp_str}.jpg"

        frame = result.get("frame")
        if frame is not None:
            cv2.imwrite(str(output_path), frame)
            logger.debug(f"Saved frame: {output_path}")

    def _save_results_csv(self, results: list[dict[str, Any]]) -> None:
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

    def run_with_auto_targets(
        self,
        max_frames: int | None = None,
        disable_validation: bool = False,
        parallel_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """5分刻みフレーム抽出（自動目標タイムスタンプ生成）

        指定範囲のフレームからタイムスタンプを全て抽出し、
        5分刻みの目標タイムスタンプを自動生成して、
        各目標に最も近いフレームを選択・保存します。

        Args:
            max_frames: 最大処理フレーム数（Noneの場合は全フレーム）
            disable_validation: タイムスタンプ検証を無効化するか（デフォルト: False）
            parallel_workers: 並列ワーカー数（デフォルト: 4）

        Returns:
            抽出結果のリスト
        """
        # 検証を無効化する場合のダミーバリデーター
        original_validator: TimestampValidator | None = None
        if disable_validation:

            class NoOpValidator:
                def validate(self, _timestamp: datetime, _frame_idx: int) -> tuple[bool, float, str]:
                    return True, 1.0, "Validation disabled"

                def reset(self) -> None:
                    pass

            original_validator = self.extractor.validator
            self.extractor.validator = NoOpValidator()

        try:
            # ステップ1: 指定範囲のフレームからタイムスタンプを全て抽出
            logger.info("ステップ1: フレームからタイムスタンプを抽出中...")
            video_processor = VideoProcessor(self.video_path)
            video_processor.open()

            try:
                total_video_frames = video_processor.total_frames
                if total_video_frames is None:
                    logger.error("動画の総フレーム数を取得できませんでした")
                    return []

                frames_limit = total_video_frames
                if isinstance(max_frames, int):
                    frames_limit = max_frames

                frames_to_process = min(frames_limit, total_video_frames)

                all_extracted_frames: list[dict[str, Any]] = []

                # 並列処理用のバッチサイズ
                batch_size = parallel_workers * 4  # 各ワーカーに4フレームずつ

                # 一時フレーム保存用ディレクトリ（ストリーミング処理用）
                temp_frames_dir = self.output_dir / "_temp_frames"
                temp_frames_dir.mkdir(parents=True, exist_ok=True)

                for batch_start in tqdm(range(0, frames_to_process, batch_size), desc="タイムスタンプ抽出中（並列）"):
                    batch_end = min(batch_start + batch_size, frames_to_process)

                    # バッチ内のフレームを取得
                    batch_frames: list[tuple[int, Any]] = []
                    for frame_idx in range(batch_start, batch_end):
                        frame = video_processor.get_frame(frame_idx)
                        if frame is not None:
                            batch_frames.append((frame_idx, frame))

                    if not batch_frames:
                        continue

                    # 並列抽出
                    batch_results = self.extractor.extract_batch_parallel(batch_frames, max_workers=parallel_workers)

                    # 結果を収集（フレームはディスクに一時保存してメモリを解放）
                    for (frame_idx, frame), result in zip(batch_frames, batch_results, strict=True):
                        if result and result.get("timestamp"):
                            # フレームを一時保存
                            temp_path = temp_frames_dir / f"temp_{frame_idx}.jpg"
                            cv2.imwrite(str(temp_path), frame)

                            all_extracted_frames.append(
                                {
                                    "frame_index": frame_idx,
                                    "timestamp": result["timestamp"],
                                    "confidence": result.get("confidence", 0.0),
                                    "ocr_text": result.get("ocr_text", ""),
                                    "frame_path": str(temp_path),  # フレームパスを保存
                                }
                            )

            finally:
                video_processor.release()

            if not all_extracted_frames:
                logger.error("タイムスタンプを抽出できたフレームがありません")
                return []

            logger.info(f"タイムスタンプ抽出成功: {len(all_extracted_frames)}フレーム")

            # ステップ2: 5分刻みの目標タイムスタンプを生成
            first_timestamp = all_extracted_frames[0]["timestamp"]
            last_timestamp = all_extracted_frames[-1]["timestamp"]

            # 先頭フレームの時刻を5分刻みに切り上げ（例: 16:04:16 -> 16:05:00）
            start_minute = (first_timestamp.minute // self.interval_minutes + 1) * self.interval_minutes
            if start_minute >= 60:
                start_minute = 0
                start_hour = first_timestamp.hour + 1
            else:
                start_hour = first_timestamp.hour

            start_target = first_timestamp.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)

            # 終了時刻を5分刻みに切り下げ
            end_minute = (last_timestamp.minute // self.interval_minutes) * self.interval_minutes
            end_target = last_timestamp.replace(minute=end_minute, second=0, microsecond=0)

            # 5分刻みの目標タイムスタンプを生成
            target_timestamps = []
            current = start_target
            while current <= end_target:
                target_timestamps.append(current)
                current += timedelta(minutes=self.interval_minutes)

            logger.info(f"ステップ2: 5分刻みの目標タイムスタンプを生成 ({len(target_timestamps)}個)")
            logger.info(
                f"  抽出範囲: {first_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ～ "
                f"{last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # ステップ3: 各目標タイムスタンプに最も近いフレームを選択
            logger.info("ステップ3: 各目標タイムスタンプに最も近いフレームを選択中...")
            selected_frames: list[dict[str, Any]] = []
            tolerance_seconds = max(self.tolerance_seconds, 60.0)  # 最低60秒の許容範囲

            for target_ts in target_timestamps:
                best_frame: dict[str, Any] | None = None
                min_diff = float("inf")

                for extracted in all_extracted_frames:
                    timestamp = extracted["timestamp"]
                    diff = abs((timestamp - target_ts).total_seconds())

                    if diff < min_diff and diff <= tolerance_seconds:
                        min_diff = diff
                        best_frame = extracted

                if best_frame:
                    selected_frames.append(
                        {
                            "target_timestamp": target_ts,
                            "frame_index": best_frame["frame_index"],
                            "timestamp": best_frame["timestamp"],
                            "time_diff": min_diff,
                            "confidence": best_frame["confidence"],
                            "ocr_text": best_frame["ocr_text"],
                            "frame_path": best_frame.get("frame_path"),  # パスを保持
                        }
                    )
                    logger.info(
                        f"  目標: {target_ts.strftime('%Y-%m-%d %H:%M:%S')} -> "
                        f"フレーム{best_frame['frame_index']}: "
                        f"{best_frame['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                        f"(差={min_diff:.1f}秒)"
                    )
                else:
                    logger.warning(
                        f"  目標: {target_ts.strftime('%Y-%m-%d %H:%M:%S')} -> "
                        f"±{tolerance_seconds}秒以内のフレームが見つかりませんでした"
                    )

            # ステップ4: 選択されたフレームのみを最終保存先に移動
            logger.info(f"ステップ4: {len(selected_frames)}枚のフレームを保存中...")
            results: list[dict[str, Any]] = []

            # フレーム保存用のディレクトリ（frames/サブディレクトリを使用）
            frames_dir = self.output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # 選択されたフレームのパスを記録（後でクリーンアップ時に保持）
            selected_paths: set[str] = set()

            for selected in selected_frames:
                target_str = selected["target_timestamp"].strftime("%Y%m%d_%H%M%S")
                output_path_frame = frames_dir / f"frame_{target_str}_idx{selected['frame_index']}.jpg"

                # 一時ファイルから読み込んで最終保存先にコピー
                frame_path = selected.get("frame_path")
                if frame_path and Path(frame_path).exists():
                    frame = cv2.imread(frame_path)
                    selected_paths.add(frame_path)
                else:
                    logger.warning(f"フレームパスが見つかりません: {frame_path}")
                    frame = None

                if frame is not None:
                    cv2.imwrite(str(output_path_frame), frame)

                    results.append(
                        {
                            "target_timestamp": selected["target_timestamp"],
                            "timestamp": selected["timestamp"],
                            "frame_idx": selected["frame_index"],
                            "confidence": selected["confidence"],
                            "ocr_text": selected["ocr_text"],
                            "time_diff": selected["time_diff"],
                            "frame": frame,  # 後続フェーズで使用
                            "frame_path": str(output_path_frame),  # 最終保存パス
                        }
                    )

                    logger.debug(f"保存: {output_path_frame.name}")

            # 一時ファイルのクリーンアップ（選択されなかったフレームを削除）
            if temp_frames_dir.exists():
                for temp_file in temp_frames_dir.glob("temp_*.jpg"):
                    if str(temp_file) not in selected_paths:
                        temp_file.unlink()
                with contextlib.suppress(OSError):
                    temp_frames_dir.rmdir()

            # 結果をCSV保存
            self._save_results_csv(results)

            logger.info(f"抽出完了: {len(results)}/{len(target_timestamps)}フレームを抽出・保存しました")

            return results

        finally:
            if disable_validation and original_validator is not None:
                self.extractor.validator = original_validator
            self.extractor.reset_validator()

    def cleanup(self) -> None:
        """リソースを解放"""
        self.coarse_sampler.close()
        if self.video_cap is not None:
            self.video_cap.release()
        self.extractor.reset_validator()
