"""Frame sampling module for the office person detection system."""

import logging
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from src.video_processor import VideoProcessor
from src.timestamp_extractor import TimestampExtractor


logger = logging.getLogger(__name__)


class FrameSampler:
    """フレームサンプリングクラス（タイムスタンプベース）
    
    動画全体をスキャンし、5分刻みの目標タイムスタンプに最も近いフレームを抽出する。
    
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
        logger.info(f"FrameSampler初期化: 間隔={interval_minutes}分, 許容誤差=±{tolerance_seconds}秒")
    
    def find_target_timestamps(
        self,
        start_dt: Union[datetime, str],
        end_dt: Union[datetime, str]
    ) -> List[Union[datetime, str]]:
        """開始・終了時刻から目標タイムスタンプリストを生成する
        
        Args:
            start_dt: 目標タイムスタンプ生成開始の日時
            end_dt: 目標タイムスタンプ生成終了の日時
            
        Returns:
            目標タイムスタンプのリスト (datetime)
        """
        try:
            start_dt_dt, start_fmt = self._coerce_to_datetime(start_dt, reference=None)
            end_dt_dt, end_fmt = self._coerce_to_datetime(end_dt, reference=start_dt_dt)

            return_as_str = isinstance(start_dt, str) and isinstance(end_dt, str)
            output_fmt = start_fmt or end_fmt

            if end_dt_dt < start_dt_dt:
                logger.warning(
                    "終了時刻が開始時刻より前です。終了時刻を同日または翌日として補正します"
                )
                end_dt_dt = end_dt_dt + timedelta(days=1)

            interval_seconds = self.interval_minutes * 60
            start_seconds = (
                start_dt_dt.hour * 3600 + start_dt_dt.minute * 60 + start_dt_dt.second
            )
            remainder = start_seconds % interval_seconds
            if remainder != 0:
                start_dt_dt += timedelta(seconds=interval_seconds - remainder)
                start_dt_dt = start_dt_dt.replace(second=0)
            else:
                start_dt_dt += timedelta(minutes=self.interval_minutes)

            target_datetimes: List[datetime] = []
            current_dt = start_dt_dt
            while current_dt <= end_dt_dt:
                target_datetimes.append(current_dt)
                current_dt += timedelta(minutes=self.interval_minutes)

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
        frame_timestamps: Dict[int, Union[datetime, str]]
    ) -> Optional[int]:
        """目標タイムスタンプに最も近いフレーム番号を返す
        
        ±tolerance_seconds以内で最も近いフレームを検索する。
        
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

            closest_frame = None
            min_abs_diff = float("inf")
            best_signed_diff = 0.0

            for frame_num, frame_dt in normalized.items():
                try:
                    signed_diff = (frame_dt - target_dt_dt).total_seconds()
                    abs_diff = abs(signed_diff)

                    # 許容誤差内かつより近い、または同距離で未来側を優先
                    if abs_diff <= self.tolerance_seconds:
                        if (
                            abs_diff < min_abs_diff
                            or (
                                abs_diff == min_abs_diff
                                and (closest_frame is None or signed_diff > best_signed_diff)
                            )
                        ):
                            min_abs_diff = abs_diff
                            best_signed_diff = signed_diff
                            closest_frame = frame_num

                except ValueError:
                    continue

            if closest_frame is not None:
                logger.debug(
                    "目標 %s に最も近いフレーム: #%s (%s, 差=%.1f秒)",
                    target_dt_dt.strftime(self.TIMESTAMP_FORMAT),
                    closest_frame,
                    normalized[closest_frame].strftime(self.TIMESTAMP_FORMAT),
                    min_abs_diff,
                )
            else:
                logger.warning(
                    "目標 %s に対応するフレームが見つかりませんでした（許容誤差±%d秒）",
                    target_dt_dt.strftime(self.TIMESTAMP_FORMAT),
                    self.tolerance_seconds,
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
        self,
        value: Union[datetime, str],
        reference: Optional[datetime]
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
        self,
        frame_timestamps: Dict[int, Union[datetime, str]]
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
        scan_interval: int = 30
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
            video_processor,
            timestamp_extractor,
            scan_interval
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
            if 'timeline' in locals():
                del timeline
            if 'sorted_frames' in locals():
                del sorted_frames
            # タイムスタンプ辞書は後で使用する可能性があるため保持
        
        logger.info(f"サンプルフレーム抽出完了: {len(sample_frames)}個")
        return sample_frames
    
    def _scan_all_timestamps(
        self,
        video_processor: VideoProcessor,
        timestamp_extractor: TimestampExtractor,
        scan_interval: int = 30
    ) -> Dict[int, datetime]:
        """動画全体をスキャンし、各フレームのタイムスタンプを抽出する
        
        Args:
            video_processor: VideoProcessorインスタンス
            timestamp_extractor: TimestampExtractorインスタンス
            scan_interval: スキャン間隔（フレーム数）
            
        Returns:
            {フレーム番号: タイムスタンプ} の辞書
        """
        frame_timestamps: Dict[int, datetime] = {}
        frame_count = 0
        failed_count = 0
        last_valid_timestamp = None
        
        logger.info(f"タイムスタンプスキャン開始（間隔: {scan_interval}フレーム）")
        
        try:
            while True:
                ret, frame = video_processor.read_next_frame()
                
                if not ret or frame is None:
                    break
                
                try:
                    # scan_interval間隔でタイムスタンプを抽出
                    if frame_count % scan_interval == 0:
                        timestamp = timestamp_extractor.extract(frame, frame_index=frame_count)
                        
                        if timestamp:
                            try:
                                timestamp_dt = datetime.strptime(timestamp, self.TIMESTAMP_FORMAT)
                            except ValueError:
                                logger.warning(
                                    "タイムスタンプ文字列の解析に失敗: %s", timestamp
                                )
                                timestamp_dt = None
                            
                            if timestamp_dt is not None:
                                frame_timestamps[frame_count] = timestamp_dt
                                last_valid_timestamp = timestamp_dt
                        else:
                            failed_count += 1
                            # 前回の有効なタイムスタンプがあれば補間
                            if last_valid_timestamp:
                                logger.debug(f"フレーム {frame_count}: タイムスタンプ抽出失敗、前回値を使用")
                    
                    frame_count += 1
                    
                    # 進捗表示とメモリ管理（1000フレームごと）
                    if frame_count % 1000 == 0:
                        logger.info(f"スキャン進捗: {frame_count}フレーム処理済み")
                        # ガベージコレクションを実行（5000フレームごと）
                        if frame_count % 5000 == 0:
                            gc.collect()
                finally:
                    # フレーム画像の参照を削除（メモリ節約）
                    del frame
        
        except Exception as e:
            logger.error(f"タイムスタンプスキャン中にエラーが発生しました: {e}")
        
        logger.info(
            f"タイムスタンプスキャン完了: "
            f"総フレーム数={frame_count}, "
            f"抽出成功={len(frame_timestamps)}, "
            f"抽出失敗={failed_count}"
        )
        
        return frame_timestamps
    
    def get_sampling_info(self) -> Dict[str, any]:
        """サンプリング設定情報を取得する
        
        Returns:
            サンプリング設定の辞書
        """
        return {
            'interval_minutes': self.interval_minutes,
            'tolerance_seconds': self.tolerance_seconds,
            'description': f'{self.interval_minutes}分間隔、±{self.tolerance_seconds}秒許容'
        }
