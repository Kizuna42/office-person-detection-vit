"""Frame sampling module for the office person detection system."""

import gc
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.timestamp import TimestampExtractor
from src.video.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class FrameSampler:
    """フレームサンプリングクラス（タイムスタンプベース）

    動画全体をスキャンし、5分刻みの目標タイムスタンプに最も近いフレームを抽出する。

    要件3に準拠: タイムスタンプは「HH:MM:SS」形式で秒を00に固定
    （例: 12:10:00, 12:15:00, 12:20:00）

    Attributes:
        interval_minutes: サンプリング間隔（分）
        tolerance_seconds: 許容誤差（秒）
    """

    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"

    def __init__(self, interval_minutes: int = 5, tolerance_seconds: int = 10):
        """FrameSamplerを初期化する

        Args:
            interval_minutes: サンプリング間隔（分）、デフォルトは5分
            tolerance_seconds: 許容誤差（秒）、デフォルトは10秒
        """
        self.interval_minutes = interval_minutes
        self.tolerance_seconds = tolerance_seconds
        self._scan_diagnostics: Dict[int, Dict[str, any]] = {}
        logger.info(
            f"FrameSampler初期化: 間隔={interval_minutes}分, 許容誤差=±{tolerance_seconds}秒"
        )

    def find_target_timestamps(
        self, start_dt: Union[datetime, str], end_dt: Union[datetime, str]
    ) -> List[Union[datetime, str]]:
        """開始・終了時刻から目標タイムスタンプリストを生成する

        要件3に準拠: 生成されるタイムスタンプは「HH:MM:SS」形式で秒を00に固定
        （例: 12:10:00, 12:15:00, 12:20:00）

        Args:
            start_dt: 目標タイムスタンプ生成開始の日時
            end_dt: 目標タイムスタンプ生成終了の日時

        Returns:
            目標タイムスタンプのリスト (datetime、秒は00に固定)
        """
        try:
            start_dt_dt, start_fmt = self._coerce_to_datetime(start_dt, reference=None)
            end_dt_dt, end_fmt = self._coerce_to_datetime(end_dt, reference=start_dt_dt)

            return_as_str = isinstance(start_dt, str) and isinstance(end_dt, str)
            output_fmt = start_fmt or end_fmt

            if end_dt_dt < start_dt_dt:
                logger.warning("終了時刻が開始時刻より前です。終了時刻を同日または翌日として補正します")
                end_dt_dt = end_dt_dt + timedelta(days=1)

            interval_seconds = self.interval_minutes * 60
            start_seconds = (
                start_dt_dt.hour * 3600 + start_dt_dt.minute * 60 + start_dt_dt.second
            )
            remainder = start_seconds % interval_seconds
            if remainder != 0:
                start_dt_dt += timedelta(seconds=interval_seconds - remainder)
            else:
                start_dt_dt += timedelta(minutes=self.interval_minutes)

            # 要件3に準拠: 秒を00に設定（HH:MM:SS形式）
            start_dt_dt = start_dt_dt.replace(second=0, microsecond=0)

            target_datetimes: List[datetime] = []
            current_dt = start_dt_dt
            while current_dt <= end_dt_dt:
                target_datetimes.append(current_dt)
                # 要件3に準拠: 5分刻みで秒を00に保証（HH:MM:SS形式）
                current_dt = (
                    current_dt + timedelta(minutes=self.interval_minutes)
                ).replace(second=0, microsecond=0)

            logger.info(
                "目標タイムスタンプ生成: %d個 (%s -> %s)",
                len(target_datetimes),
                start_dt_dt.strftime(self.TIMESTAMP_FORMAT),
                end_dt_dt.strftime(self.TIMESTAMP_FORMAT),
            )
            logger.debug(
                "目標タイムスタンプ: %s",
                [dt.strftime(self.TIMESTAMP_FORMAT) for dt in target_datetimes],
            )

            if return_as_str and output_fmt is not None:
                return [dt.strftime(output_fmt) for dt in target_datetimes]

            return target_datetimes

        except Exception as e:
            logger.error(f"目標タイムスタンプの生成中にエラーが発生しました: {e}")
            return []

    def find_closest_frame(
        self,
        target_dt: Union[datetime, str],
        frame_timestamps: Dict[int, Union[datetime, str]],
    ) -> Optional[int]:
        """目標タイムスタンプに最も近いフレーム番号を返す

        ±tolerance_seconds以内で最も近いフレームを検索する。
        見つからない場合は±15→±30→±60秒まで漸増探索する。

        Args:
            target_dt: 目標タイムスタンプ
            frame_timestamps: {フレーム番号: タイムスタンプ(datetime)} の辞書

        Returns:
            最も近いフレーム番号、見つからない場合None
        """
        try:
            normalized = self._normalize_frame_timestamps(frame_timestamps)
            if not normalized:
                logger.warning("提供されたフレームタイムスタンプが空です")
                return None

            first_dt = next(iter(normalized.values()))
            target_dt_dt, _ = self._coerce_to_datetime(target_dt, reference=first_dt)

            # 漸増探索: まず±tolerance_seconds、次に±15/30/60秒
            search_tolerances = [
                self.tolerance_seconds,
                15,  # ±15秒
                30,  # ±30秒
                60,  # ±60秒
            ]

            closest_frame = None
            min_abs_diff = float("inf")
            best_signed_diff = 0.0
            used_tolerance = None

            for tolerance in search_tolerances:
                for frame_num, frame_dt in normalized.items():
                    try:
                        signed_diff = (frame_dt - target_dt_dt).total_seconds()
                        abs_diff = abs(signed_diff)

                        # 許容誤差内かつより近い、または同距離で未来側を優先
                        if abs_diff <= tolerance:
                            if abs_diff < min_abs_diff or (
                                abs_diff == min_abs_diff
                                and (
                                    closest_frame is None
                                    or signed_diff > best_signed_diff
                                )
                            ):
                                min_abs_diff = abs_diff
                                best_signed_diff = signed_diff
                                closest_frame = frame_num
                                used_tolerance = tolerance

                    except ValueError:
                        continue

                # 見つかった場合は探索を終了
                if closest_frame is not None:
                    break

            if closest_frame is not None:
                logger.debug(
                    "目標 %s に最も近いフレーム: #%s (%s, 差=%.1f秒, 許容=±%d秒)",
                    target_dt_dt.strftime(self.TIMESTAMP_FORMAT),
                    closest_frame,
                    normalized[closest_frame].strftime(self.TIMESTAMP_FORMAT),
                    min_abs_diff,
                    used_tolerance or self.tolerance_seconds,
                )
            else:
                logger.warning(
                    "目標 %s に対応するフレームが見つかりませんでした（最大許容誤差±%d秒）",
                    target_dt_dt.strftime(self.TIMESTAMP_FORMAT),
                    search_tolerances[-1],
                )

            return closest_frame

        except Exception as e:
            logger.error(f"最近接フレーム検索中にエラーが発生しました: {e}")
            return None

    def _parse_user_time(self, value: str, reference: datetime) -> datetime:
        """ユーザー指定時刻文字列をdatetimeに変換する"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("空の時刻文字列です")

        candidate_formats = [
            self.TIMESTAMP_FORMAT,
            "%Y/%m/%d %H:%M",
            "%H:%M:%S",
            "%H:%M",
        ]

        last_error: Optional[Exception] = None

        for fmt in candidate_formats:
            try:
                parsed = datetime.strptime(normalized, fmt)

                if fmt.startswith("%H"):
                    second = parsed.second if fmt == "%H:%M:%S" else 0
                    parsed = reference.replace(
                        hour=parsed.hour,
                        minute=parsed.minute,
                        second=second,
                        microsecond=0,
                    )
                elif fmt == "%Y/%m/%d %H:%M":
                    parsed = parsed.replace(second=0, microsecond=0)
                else:
                    parsed = parsed.replace(microsecond=0)

                if fmt.startswith("%H"):
                    # 基準から大きく離れている場合は日付を補正
                    if parsed < reference - timedelta(hours=12):
                        parsed += timedelta(days=1)
                    elif parsed > reference + timedelta(hours=12):
                        parsed -= timedelta(days=1)

                return parsed
            except ValueError as exc:
                last_error = exc
                continue

        raise ValueError(f"不正な時刻形式です: '{value}'") from last_error

    def _default_reference(self) -> datetime:
        today = datetime.now()
        return today.replace(hour=0, minute=0, second=0, microsecond=0)

    def _coerce_to_datetime(
        self, value: Union[datetime, str], reference: Optional[datetime]
    ) -> Tuple[datetime, Optional[str]]:
        if isinstance(value, datetime):
            return value.replace(microsecond=0), None

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("空の時刻文字列です")

            candidate_formats = [
                self.TIMESTAMP_FORMAT,
                "%Y/%m/%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%H:%M:%S",
                "%H:%M",
            ]

            last_error: Optional[Exception] = None
            for fmt in candidate_formats:
                try:
                    parsed = datetime.strptime(normalized, fmt)

                    if fmt.startswith("%H"):
                        base = reference or self._default_reference()
                        second = parsed.second if fmt == "%H:%M:%S" else 0
                        parsed = base.replace(
                            hour=parsed.hour,
                            minute=parsed.minute,
                            second=second,
                            microsecond=0,
                        )

                        if reference is not None:
                            if parsed < reference - timedelta(hours=12):
                                parsed += timedelta(days=1)
                            elif parsed > reference + timedelta(hours=12):
                                parsed -= timedelta(days=1)
                    else:
                        parsed = parsed.replace(microsecond=0)

                    if fmt == "%Y/%m/%d %H:%M" or fmt == "%Y-%m-%d %H:%M":
                        parsed = parsed.replace(second=0)

                    return parsed, fmt
                except ValueError as exc:
                    last_error = exc
                    continue

            raise ValueError(f"不正な時刻形式です: '{value}'") from last_error

        raise TypeError(f"サポートされていない型です: {type(value)!r}")

    def _normalize_frame_timestamps(
        self, frame_timestamps: Dict[int, Union[datetime, str]]
    ) -> Dict[int, datetime]:
        normalized: Dict[int, datetime] = {}
        previous: Optional[datetime] = None

        for frame_num, timestamp in sorted(frame_timestamps.items()):
            dt, _ = self._coerce_to_datetime(timestamp, reference=previous)
            previous = dt
            normalized[frame_num] = dt

        return normalized

    def extract_sample_frames(
        self,
        video_processor: VideoProcessor,
        timestamp_extractor: TimestampExtractor,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        scan_interval: int = 30,
    ) -> List[Tuple[int, str, np.ndarray]]:
        """動画全体をスキャンし、5分刻みのフレームを抽出する

        Args:
            video_processor: VideoProcessorインスタンス
            timestamp_extractor: TimestampExtractorインスタンス
            start_time: 開始時刻 (HH:MM形式)、Noneの場合は自動検出
            end_time: 終了時刻 (HH:MM形式)、Noneの場合は自動検出
            scan_interval: スキャン間隔（フレーム数）、デフォルトは30フレーム

        Returns:
            [(フレーム番号, タイムスタンプ, フレーム画像), ...] のリスト
        """
        logger.info("動画のスキャンを開始します...")

        # 動画を先頭に戻す
        video_processor.reset()

        # 全フレームのタイムスタンプを抽出
        frame_timestamps = self._scan_all_timestamps(
            video_processor, timestamp_extractor, scan_interval
        )

        if not frame_timestamps:
            logger.error("タイムスタンプの抽出に失敗しました")
            return []

        # 開始・終了時刻を決定
        sorted_frames = sorted(frame_timestamps.items(), key=lambda item: item[1])
        timeline = [dt for _, dt in sorted_frames]
        detected_start_dt = timeline[0]
        detected_end_dt = timeline[-1]

        if start_time is None:
            start_dt = detected_start_dt
            logger.info(
                "開始時刻を自動検出: %s",
                start_dt.strftime(self.TIMESTAMP_FORMAT),
            )
        else:
            try:
                start_dt = self._parse_user_time(start_time, detected_start_dt)
            except ValueError as exc:
                logger.error(f"開始時刻の解析に失敗しました: {exc}")
                return []

        if end_time is None:
            end_dt = detected_end_dt
            logger.info(
                "終了時刻を自動検出: %s",
                end_dt.strftime(self.TIMESTAMP_FORMAT),
            )
        else:
            try:
                end_dt = self._parse_user_time(end_time, detected_end_dt)
            except ValueError as exc:
                logger.error(f"終了時刻の解析に失敗しました: {exc}")
                return []

        if start_dt < detected_start_dt:
            logger.info(
                "開始時刻が検出範囲より前のため補正します (%s -> %s)",
                start_dt.strftime(self.TIMESTAMP_FORMAT),
                detected_start_dt.strftime(self.TIMESTAMP_FORMAT),
            )
            start_dt = detected_start_dt

        if end_dt > detected_end_dt:
            logger.info(
                "終了時刻が検出範囲より後のため補正します (%s -> %s)",
                end_dt.strftime(self.TIMESTAMP_FORMAT),
                detected_end_dt.strftime(self.TIMESTAMP_FORMAT),
            )
            end_dt = detected_end_dt

        if end_dt < start_dt:
            logger.error("終了時刻が開始時刻より前です")
            return []

        # 目標タイムスタンプリストを生成
        target_datetimes = self.find_target_timestamps(start_dt, end_dt)

        if not target_datetimes:
            logger.error("目標タイムスタンプの生成に失敗しました")
            return []

        # 各目標タイムスタンプに最も近いフレームを検索
        sample_frames = []
        try:
            for target_dt in target_datetimes:
                frame_num = self.find_closest_frame(target_dt, frame_timestamps)

                if frame_num is not None:
                    # フレームを取得
                    frame = video_processor.get_frame(frame_num)

                    if frame is not None:
                        actual_dt = frame_timestamps[frame_num]
                        actual_ts_str = actual_dt.strftime(self.TIMESTAMP_FORMAT)
                        # フレーム画像のコピーを保存（参照ではなく）
                        frame_copy = frame.copy()
                        sample_frames.append((frame_num, actual_ts_str, frame_copy))
                        # 元のフレーム参照を削除（メモリ節約）
                        del frame
                        logger.info(
                            "サンプルフレーム抽出: #%d (%s)",
                            frame_num,
                            actual_ts_str,
                        )
                    else:
                        logger.warning(f"フレーム #{frame_num} の取得に失敗しました")
        finally:
            # 不要なデータをクリーンアップ
            if "timeline" in locals():
                del timeline
            if "sorted_frames" in locals():
                del sorted_frames
            # タイムスタンプ辞書は後で使用する可能性があるため保持

        logger.info(f"サンプルフレーム抽出完了: {len(sample_frames)}個")
        return sample_frames

    def _scan_all_timestamps(
        self,
        video_processor: VideoProcessor,
        timestamp_extractor: TimestampExtractor,
        scan_interval: int = 30,
    ) -> Dict[int, datetime]:
        """動画全体をスキャンし、各フレームのタイムスタンプを抽出する

        信頼度指標、近傍再OCR、補間、外れ値検知を実装。

        Args:
            video_processor: VideoProcessorインスタンス
            timestamp_extractor: TimestampExtractorインスタンス
            scan_interval: スキャン間隔（フレーム数）

        Returns:
            {フレーム番号: タイムスタンプ} の辞書
        """
        frame_timestamps: Dict[int, datetime] = {}
        frame_diagnostics: Dict[int, Dict[str, any]] = {}  # 診断情報
        frame_count = 0
        failed_count = 0
        last_valid_timestamp: Optional[datetime] = None
        last_valid_frame: Optional[int] = None

        # フレームバッファ（近傍再OCR用）
        frame_buffer: Dict[int, np.ndarray] = {}
        max_buffer_size = 10

        logger.info(f"タイムスタンプスキャン開始（間隔: {scan_interval}フレーム）")

        try:
            while True:
                ret, frame = video_processor.read_next_frame()

                if not ret or frame is None:
                    break

                try:
                    # scan_interval間隔でタイムスタンプを抽出
                    if frame_count % scan_interval == 0:
                        # フレームをバッファに保存（近傍再OCR用）
                        if len(frame_buffer) >= max_buffer_size:
                            # 最も古いフレームを削除
                            oldest_frame = min(frame_buffer.keys())
                            del frame_buffer[oldest_frame]
                        frame_buffer[frame_count] = frame.copy()

                        # 信頼度付きで抽出
                        (
                            timestamp,
                            confidence,
                        ) = timestamp_extractor.extract_with_confidence(frame)

                        timestamp_dt = None
                        needs_retry = False
                        source = "direct"

                        if timestamp:
                            try:
                                timestamp_dt = datetime.strptime(
                                    timestamp, self.TIMESTAMP_FORMAT
                                )
                            except ValueError:
                                logger.warning("タイムスタンプ文字列の解析に失敗: %s", timestamp)
                                timestamp_dt = None

                        # 低信頼度または外れ値検知の場合、近傍再OCRを試行
                        if timestamp_dt is not None:
                            # 時系列の整合性チェック（前回との差が不自然な場合）
                            if last_valid_timestamp is not None:
                                time_diff = abs(
                                    (
                                        timestamp_dt - last_valid_timestamp
                                    ).total_seconds()
                                )
                                frame_diff = (
                                    frame_count - last_valid_frame
                                    if last_valid_frame is not None
                                    else scan_interval
                                )
                                expected_diff = frame_diff * (
                                    scan_interval / 30.0
                                )  # 仮のFPS推定

                                # 外れ値検知: 秒差が不自然（負の値、または120秒以上）
                                if time_diff < 0 or time_diff > 120:
                                    logger.debug(
                                        f"フレーム {frame_count}: 時系列外れ値検知 "
                                        f"(差={time_diff:.1f}秒, 期待={expected_diff:.1f}秒)"
                                    )
                                    needs_retry = True
                        elif confidence < 0.5:  # 低信頼度
                            needs_retry = True
                            logger.debug(f"フレーム {frame_count}: 低信頼度 ({confidence:.2f})")

                        success = False

                        if timestamp_dt is not None and not needs_retry:
                            # そのまま採用
                            frame_timestamps[frame_count] = timestamp_dt
                            last_valid_timestamp = timestamp_dt
                            last_valid_frame = frame_count
                            frame_diagnostics[frame_count] = {
                                "timestamp": timestamp,
                                "confidence": confidence,
                                "source": source,
                                "corrections": timestamp_extractor.get_last_corrections(),
                            }
                            success = True

                        # 近傍再OCRを試行
                        if not success and needs_retry and timestamp_dt is None:
                            retry_success = False
                            for offset in [-3, -2, -1, 1, 2, 3]:
                                retry_frame_num = frame_count + offset
                                if retry_frame_num in frame_buffer:
                                    retry_frame = frame_buffer[retry_frame_num]
                                    (
                                        retry_ts,
                                        retry_conf,
                                    ) = timestamp_extractor.extract_with_confidence(
                                        retry_frame
                                    )
                                    if retry_ts:
                                        try:
                                            retry_dt = datetime.strptime(
                                                retry_ts, self.TIMESTAMP_FORMAT
                                            )
                                            # 時系列の整合性を再チェック
                                            if (
                                                last_valid_timestamp is None
                                                or abs(
                                                    (
                                                        retry_dt - last_valid_timestamp
                                                    ).total_seconds()
                                                )
                                                < 120
                                            ):
                                                timestamp_dt = retry_dt
                                                timestamp = retry_ts
                                                confidence = retry_conf
                                                source = f"retry_offset_{offset}"
                                                retry_success = True
                                                logger.debug(
                                                    f"フレーム {frame_count}: 近傍再OCR成功 "
                                                    f"(offset={offset}, conf={retry_conf:.2f})"
                                                )
                                                break
                                        except ValueError:
                                            continue

                            if not retry_success:
                                source = "failed"

                            # タイムスタンプが取得できた場合
                            if timestamp_dt is not None:
                                frame_timestamps[frame_count] = timestamp_dt
                                last_valid_timestamp = timestamp_dt
                                success = True
                            last_valid_frame = frame_count

                            # 診断情報を記録
                            frame_diagnostics[frame_count] = {
                                "timestamp": timestamp,
                                "confidence": confidence,
                                "source": source,
                                "corrections": timestamp_extractor.get_last_corrections(),
                            }

                        if success:
                            last_valid_frame = frame_count
                        else:
                            failed_count += 1

                            # 補間を試行（直前直後の有効時刻から）
                            if (
                                last_valid_timestamp is not None
                                and len(frame_timestamps) > 0
                            ):
                                # 次の有効フレームを探す（最大100フレーム先まで）
                                next_valid_frame = None
                                next_valid_timestamp = None
                                for i in range(1, min(100, scan_interval * 5)):
                                    check_frame = frame_count + i * scan_interval
                                    if check_frame in frame_buffer:
                                        (
                                            check_ts,
                                            check_conf,
                                        ) = timestamp_extractor.extract_with_confidence(
                                            frame_buffer[check_frame]
                                        )
                                        if check_ts:
                                            try:
                                                check_dt = datetime.strptime(
                                                    check_ts, self.TIMESTAMP_FORMAT
                                                )
                                                next_valid_frame = check_frame
                                                next_valid_timestamp = check_dt
                                                break
                                            except ValueError:
                                                continue

                                if next_valid_timestamp is not None:
                                    # 線形補間
                                    frame_diff = next_valid_frame - last_valid_frame
                                    time_diff = (
                                        next_valid_timestamp - last_valid_timestamp
                                    ).total_seconds()
                                    interpolated_dt = last_valid_timestamp + timedelta(
                                        seconds=time_diff
                                        * (frame_count - last_valid_frame)
                                        / frame_diff
                                    )
                                    frame_timestamps[frame_count] = interpolated_dt
                                    source = "interpolated"
                                    logger.debug(
                                        f"フレーム {frame_count}: 補間成功 "
                                        f"({last_valid_timestamp} -> {interpolated_dt} -> {next_valid_timestamp})"
                                    )

                                    frame_diagnostics[frame_count] = {
                                        "timestamp": interpolated_dt.strftime(
                                            self.TIMESTAMP_FORMAT
                                        ),
                                        "confidence": 0.0,
                                        "source": source,
                                        "corrections": [],
                                    }
                                else:
                                    frame_diagnostics[frame_count] = {
                                        "timestamp": None,
                                        "confidence": confidence
                                        if "confidence" in locals()
                                        else 0.0,
                                        "source": "failed",
                                        "corrections": [],
                                    }
                            else:
                                frame_diagnostics[frame_count] = {
                                    "timestamp": None,
                                    "confidence": confidence
                                    if "confidence" in locals()
                                    else 0.0,
                                    "source": "failed",
                                    "corrections": [],
                                }

                    frame_count += 1

                    # 進捗表示とメモリ管理（1000フレームごと）
                    if frame_count % 1000 == 0:
                        logger.info(f"スキャン進捗: {frame_count}フレーム処理済み")
                        # ガベージコレクションを実行（5000フレームごと）
                        if frame_count % 5000 == 0:
                            gc.collect()
                finally:
                    # フレーム画像の参照を削除（メモリ節約）
                    # ただし、バッファに保存したものは保持
                    if frame_count not in frame_buffer:
                        del frame

        except Exception as e:
            logger.error(f"タイムスタンプスキャン中にエラーが発生しました: {e}")

        logger.info(
            f"タイムスタンプスキャン完了: "
            f"総フレーム数={frame_count}, "
            f"抽出成功={len(frame_timestamps)}, "
            f"抽出失敗={failed_count}"
        )

        # 診断情報をインスタンス変数に保存（後でサマリー生成に使用）
        self._scan_diagnostics = frame_diagnostics

        return frame_timestamps

    def get_sampling_info(self) -> Dict[str, any]:
        """サンプリング設定情報を取得する

        Returns:
            サンプリング設定の辞書
        """
        return {
            "interval_minutes": self.interval_minutes,
            "tolerance_seconds": self.tolerance_seconds,
            "description": f"{self.interval_minutes}分間隔、±{self.tolerance_seconds}秒許容",
        }

    def write_timestamp_verification_summary(
        self,
        frame_timestamps: Dict[int, datetime],
        video_processor: VideoProcessor,
        output_dir: Union[str, Path],
        scan_interval: int = 10,
    ) -> None:
        """タイムスタンプ検証サマリーを生成・出力する

        Args:
            frame_timestamps: {フレーム番号: タイムスタンプ} の辞書
            video_processor: VideoProcessorインスタンス
            output_dir: 出力ディレクトリ
            scan_interval: スキャン間隔
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 基本統計
        total_frames = video_processor.total_frames or 0
        recognized_frames = len(frame_timestamps)
        failed_frames = len(self._scan_diagnostics) - recognized_frames
        recognition_rate = (
            recognized_frames / len(self._scan_diagnostics)
            if self._scan_diagnostics
            else 0.0
        )

        # 時系列統計
        sorted_timestamps = sorted(frame_timestamps.items(), key=lambda x: x[1])
        if sorted_timestamps:
            earliest = sorted_timestamps[0][1]
            latest = sorted_timestamps[-1][1]
            timeline_days = (latest - earliest).days + 1

            # 時間差の統計
            time_diffs = []
            for i in range(len(sorted_timestamps) - 1):
                frame1, ts1 = sorted_timestamps[i]
                frame2, ts2 = sorted_timestamps[i + 1]
                diff_seconds = (ts2 - ts1).total_seconds()
                time_diffs.append(diff_seconds)
        else:
            earliest = None
            latest = None
            timeline_days = 0
            time_diffs = []

        # フレームサンプリング検証
        start_dt = earliest if earliest else datetime.now()
        end_dt = latest if latest else datetime.now()
        target_timestamps = self.find_target_timestamps(start_dt, end_dt)

        matched_samples = []
        matched_count = 0
        within_tolerance_count = 0
        exact_match_count = 0

        for target_dt in target_timestamps[:50]:  # 最初の50件をサンプル
            closest_frame = self.find_closest_frame(target_dt, frame_timestamps)
            if closest_frame is not None:
                matched_ts = frame_timestamps[closest_frame]
                target_dt_dt = (
                    target_dt
                    if isinstance(target_dt, datetime)
                    else datetime.strptime(target_dt, self.TIMESTAMP_FORMAT)
                )
                abs_diff = abs((matched_ts - target_dt_dt).total_seconds())
                within_tolerance = abs_diff <= self.tolerance_seconds
                exact_match = abs_diff == 0.0

                matched_samples.append(
                    {
                        "target": target_dt_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        "matched_frame": closest_frame,
                        "matched_timestamp": matched_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        "abs_diff_seconds": abs_diff,
                        "within_tolerance": within_tolerance,
                    }
                )

                matched_count += 1
                if within_tolerance:
                    within_tolerance_count += 1
                if exact_match:
                    exact_match_count += 1

        # 補正ログ（ocr_corrections.jsonl）
        corrections_path = output_path / "ocr_corrections.jsonl"
        with open(corrections_path, "w", encoding="utf-8") as f:
            for frame_num, diag in sorted(self._scan_diagnostics.items()):
                correction_entry = {
                    "frame": frame_num,
                    "timestamp": diag.get("timestamp"),
                    "confidence": diag.get("confidence", 0.0),
                    "source": diag.get("source", "unknown"),
                    "corrections": diag.get("corrections", []),
                }
                f.write(json.dumps(correction_entry, ensure_ascii=False) + "\n")

        logger.info(f"補正ログを出力しました: {corrections_path}")

        # サマリーJSON
        summary = {
            "scan_interval_frames": scan_interval,
            "video_total_frames": total_frames,
            "frames_sampled": len(self._scan_diagnostics),
            "recognized_frames": recognized_frames,
            "failed_frames": failed_frames,
            "recognition_rate": recognition_rate,
            "timeline_earliest": earliest.strftime("%Y-%m-%dT%H:%M:%S")
            if earliest
            else None,
            "timeline_latest": latest.strftime("%Y-%m-%dT%H:%M:%S") if latest else None,
            "timeline_days_covered": timeline_days,
            "timestamp_delta_seconds": {
                "min": min(time_diffs) if time_diffs else 0.0,
                "max": max(time_diffs) if time_diffs else 0.0,
                "median": sorted(time_diffs)[len(time_diffs) // 2]
                if time_diffs
                else 0.0,
                "mean": sum(time_diffs) / len(time_diffs) if time_diffs else 0.0,
            },
            "frame_sampling_validation": {
                "target_count": len(target_timestamps),
                "matched_count": matched_count,
                "within_tolerance_count": within_tolerance_count,
                "exact_match_count": exact_match_count,
                "missing_count": len(target_timestamps) - matched_count,
                "max_abs_diff_seconds": max(
                    [s["abs_diff_seconds"] for s in matched_samples]
                )
                if matched_samples
                else 0.0,
                "median_abs_diff_seconds": sorted(
                    [s["abs_diff_seconds"] for s in matched_samples]
                )[len(matched_samples) // 2]
                if matched_samples
                else 0.0,
            },
            "frame_sampling_sample": matched_samples[:20],  # 最初の20件
            "frame_sampler_execution": {
                "requested_start": start_dt.strftime(self.TIMESTAMP_FORMAT)
                if start_dt
                else None,
                "requested_end": end_dt.strftime(self.TIMESTAMP_FORMAT)
                if end_dt
                else None,
                "sample_count": len(frame_timestamps),
            },
            "parameters": {
                "interval_minutes": self.interval_minutes,
                "tolerance_seconds": self.tolerance_seconds,
            },
        }

        summary_path = output_path / "timestamp_verification_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"検証サマリーを出力しました: {summary_path}")
