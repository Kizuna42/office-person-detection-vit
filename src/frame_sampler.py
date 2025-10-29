"""Frame sampling module for the office person detection system."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
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
    
    def __init__(self, interval_minutes: int = 5, tolerance_seconds: int = 10):
        """FrameSamplerを初期化する
        
        Args:
            interval_minutes: サンプリング間隔（分）、デフォルトは5分
            tolerance_seconds: 許容誤差（秒）、デフォルトは10秒
        """
        self.interval_minutes = interval_minutes
        self.tolerance_seconds = tolerance_seconds
        logger.info(f"FrameSampler初期化: 間隔={interval_minutes}分, 許容誤差=±{tolerance_seconds}秒")
    
    def find_target_timestamps(self, start_time: str, end_time: str) -> List[str]:
        """開始・終了時刻から目標タイムスタンプリストを生成する
        
        5分刻みの目標タイムスタンプリストを生成する。
        例: start_time="12:08", end_time="12:32" の場合
            -> ["12:10", "12:15", "12:20", "12:25", "12:30"]
        
        Args:
            start_time: 開始時刻 (HH:MM形式)
            end_time: 終了時刻 (HH:MM形式)
            
        Returns:
            目標タイムスタンプのリスト (HH:MM形式)
        """
        try:
            # 時刻をdatetimeオブジェクトに変換
            start_dt = datetime.strptime(start_time, "%H:%M")
            end_dt = datetime.strptime(end_time, "%H:%M")
            
            # 終了時刻が開始時刻より前の場合（日をまたぐ場合）
            if end_dt < start_dt:
                end_dt += timedelta(days=1)
            
            # 開始時刻を次の目標時刻に切り上げ
            start_minute = start_dt.minute
            next_target_minute = ((start_minute // self.interval_minutes) + 1) * self.interval_minutes
            
            if next_target_minute >= 60:
                current_dt = start_dt.replace(minute=0) + timedelta(hours=1)
            else:
                current_dt = start_dt.replace(minute=next_target_minute, second=0, microsecond=0)
            
            # 目標タイムスタンプリストを生成
            target_timestamps = []
            while current_dt <= end_dt:
                timestamp = current_dt.strftime("%H:%M")
                target_timestamps.append(timestamp)
                current_dt += timedelta(minutes=self.interval_minutes)
            
            logger.info(f"目標タイムスタンプ生成: {len(target_timestamps)}個 ({start_time} -> {end_time})")
            logger.debug(f"目標タイムスタンプ: {target_timestamps}")
            
            return target_timestamps
            
        except Exception as e:
            logger.error(f"目標タイムスタンプの生成中にエラーが発生しました: {e}")
            return []
    
    def find_closest_frame(
        self,
        target_timestamp: str,
        frame_timestamps: Dict[int, str]
    ) -> Optional[int]:
        """目標タイムスタンプに最も近いフレーム番号を返す
        
        ±tolerance_seconds以内で最も近いフレームを検索する。
        
        Args:
            target_timestamp: 目標タイムスタンプ (HH:MM形式)
            frame_timestamps: {フレーム番号: タイムスタンプ} の辞書
            
        Returns:
            最も近いフレーム番号、見つからない場合None
        """
        try:
            target_dt = datetime.strptime(target_timestamp, "%H:%M")
            
            closest_frame = None
            min_diff_seconds = float('inf')
            
            for frame_num, timestamp in frame_timestamps.items():
                try:
                    frame_dt = datetime.strptime(timestamp, "%H:%M")
                    
                    # 時刻差を計算（秒単位）
                    diff = abs((frame_dt - target_dt).total_seconds())
                    
                    # 日をまたぐ場合の処理
                    if diff > 12 * 3600:  # 12時間以上の差がある場合
                        diff = 24 * 3600 - diff
                    
                    # 許容誤差内かつ最小差の場合
                    if diff <= self.tolerance_seconds and diff < min_diff_seconds:
                        min_diff_seconds = diff
                        closest_frame = frame_num
                        
                except ValueError:
                    continue
            
            if closest_frame is not None:
                logger.debug(
                    f"目標 {target_timestamp} に最も近いフレーム: "
                    f"#{closest_frame} ({frame_timestamps[closest_frame]}, 差={min_diff_seconds:.1f}秒)"
                )
            else:
                logger.warning(f"目標 {target_timestamp} に対応するフレームが見つかりませんでした（許容誤差±{self.tolerance_seconds}秒）")
            
            return closest_frame
            
        except Exception as e:
            logger.error(f"最近接フレーム検索中にエラーが発生しました: {e}")
            return None
    
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
        if start_time is None or end_time is None:
            timestamps = sorted(frame_timestamps.values())
            detected_start = timestamps[0]
            detected_end = timestamps[-1]
            
            if start_time is None:
                start_time = detected_start
                logger.info(f"開始時刻を自動検出: {start_time}")
            
            if end_time is None:
                end_time = detected_end
                logger.info(f"終了時刻を自動検出: {end_time}")
        
        # 目標タイムスタンプリストを生成
        target_timestamps = self.find_target_timestamps(start_time, end_time)
        
        if not target_timestamps:
            logger.error("目標タイムスタンプの生成に失敗しました")
            return []
        
        # 各目標タイムスタンプに最も近いフレームを検索
        sample_frames = []
        for target_ts in target_timestamps:
            frame_num = self.find_closest_frame(target_ts, frame_timestamps)
            
            if frame_num is not None:
                # フレームを取得
                frame = video_processor.get_frame(frame_num)
                
                if frame is not None:
                    actual_ts = frame_timestamps[frame_num]
                    sample_frames.append((frame_num, actual_ts, frame))
                    logger.info(f"サンプルフレーム抽出: #{frame_num} ({actual_ts})")
                else:
                    logger.warning(f"フレーム #{frame_num} の取得に失敗しました")
        
        logger.info(f"サンプルフレーム抽出完了: {len(sample_frames)}個")
        return sample_frames
    
    def _scan_all_timestamps(
        self,
        video_processor: VideoProcessor,
        timestamp_extractor: TimestampExtractor,
        scan_interval: int = 30
    ) -> Dict[int, str]:
        """動画全体をスキャンし、各フレームのタイムスタンプを抽出する
        
        Args:
            video_processor: VideoProcessorインスタンス
            timestamp_extractor: TimestampExtractorインスタンス
            scan_interval: スキャン間隔（フレーム数）
            
        Returns:
            {フレーム番号: タイムスタンプ} の辞書
        """
        frame_timestamps = {}
        frame_count = 0
        failed_count = 0
        last_valid_timestamp = None
        
        logger.info(f"タイムスタンプスキャン開始（間隔: {scan_interval}フレーム）")
        
        try:
            while True:
                ret, frame = video_processor.read_next_frame()
                
                if not ret or frame is None:
                    break
                
                # scan_interval間隔でタイムスタンプを抽出
                if frame_count % scan_interval == 0:
                    timestamp = timestamp_extractor.extract(frame)
                    
                    if timestamp:
                        frame_timestamps[frame_count] = timestamp
                        last_valid_timestamp = timestamp
                    else:
                        failed_count += 1
                        # 前回の有効なタイムスタンプがあれば補間
                        if last_valid_timestamp:
                            logger.debug(f"フレーム {frame_count}: タイムスタンプ抽出失敗、前回値を使用")
                
                frame_count += 1
                
                # 進捗表示（1000フレームごと）
                if frame_count % 1000 == 0:
                    logger.info(f"スキャン進捗: {frame_count}フレーム処理済み")
        
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
