"""Ground Truthスケジュール生成モジュール

frame0のタイムスタンプを基準に、+10秒（稀に+9秒）のスケジュールを生成。
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"


def load_reference_timestamp(csv_path: str) -> Optional[datetime]:
    """CSVからframe0のタイムスタンプを取得

    Args:
        csv_path: frames_ocr.csvのパス

    Returns:
        frame0のタイムスタンプ（datetime）、失敗時None
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_number"]) == 0:
                    timestamp_str = row.get("timestamp", "").strip()
                    if timestamp_str:
                        return datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
        logger.warning("frame0のタイムスタンプが見つかりませんでした")
        return None
    except Exception as e:
        logger.error(f"参照タイムスタンプの読み込みに失敗: {e}")
        return None


def estimate_interval_map(
    csv_path: str, reference_timestamp: datetime, default_interval: int = 10
) -> Dict[int, int]:
    """成功フレームの時刻差から9秒箇所を推定

    Args:
        csv_path: frames_ocr.csvのパス
        reference_timestamp: 基準タイムスタンプ
        default_interval: デフォルト間隔（秒）

    Returns:
        {フレーム番号: 間隔（秒）} の辞書
    """
    interval_map = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prev_frame = 0
            prev_timestamp = reference_timestamp

            for row in reader:
                frame_num = int(row["frame_number"])
                recognized = row.get("recognized", "False").lower() == "true"
                timestamp_str = row.get("timestamp", "").strip()

                if recognized and timestamp_str:
                    try:
                        current_timestamp = datetime.strptime(
                            timestamp_str, TIMESTAMP_FORMAT
                        )

                        # 前回との時刻差を計算
                        time_diff = (current_timestamp - prev_timestamp).total_seconds()
                        frame_diff = frame_num - prev_frame

                        if frame_diff > 0:
                            # 1フレームあたりの平均間隔を計算
                            avg_interval = time_diff / frame_diff

                            # 9秒または10秒に近いかチェック
                            if 8.5 <= avg_interval <= 9.5:
                                # 9秒間隔と判定
                                for frame_idx in range(prev_frame + 1, frame_num + 1):
                                    interval_map[frame_idx] = 9
                            elif 9.5 < avg_interval <= 10.5:
                                # 10秒間隔と判定
                                for frame_idx in range(prev_frame + 1, frame_num + 1):
                                    if frame_idx not in interval_map:
                                        interval_map[frame_idx] = 10

                            prev_frame = frame_num
                            prev_timestamp = current_timestamp
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"間隔マップの推定に失敗: {e}")

    return interval_map


def generate_ground_truth_schedule(
    csv_path: str,
    num_frames: int = 100,
    default_interval: int = 10,
    estimate_9s: bool = True,
    use_csv_timestamps: bool = False,
) -> Tuple[datetime, Dict[int, datetime]]:
    """Ground Truthスケジュールを生成
    
    Args:
        csv_path: frames_ocr.csvのパス
        num_frames: フレーム数
        default_interval: デフォルト間隔（秒）
        estimate_9s: 9秒箇所を推定するか
        use_csv_timestamps: CSVのタイムスタンプをそのまま使用するか（Trueの場合、OCR結果をGTとして使用）
        
    Returns:
        (基準タイムスタンプ, {フレーム番号: タイムスタンプ}) のタプル
    """
    # CSVからタイムスタンプを直接使用する場合
    if use_csv_timestamps:
        schedule = {}
        reference_timestamp = None
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row["frame_number"])
                timestamp_str = row.get("timestamp", "").strip()
                
                if timestamp_str:
                    try:
                        dt = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
                        schedule[frame_num] = dt
                        if reference_timestamp is None:
                            reference_timestamp = dt
                    except ValueError:
                        continue
        
        if reference_timestamp is None:
            raise ValueError("有効なタイムスタンプが見つかりませんでした")
        
        return reference_timestamp, schedule
    
    # 基準タイムスタンプを取得
    reference_timestamp = load_reference_timestamp(csv_path)

    if reference_timestamp is None:
        raise ValueError("基準タイムスタンプの取得に失敗しました")

    # 間隔マップを推定
    interval_map = {}
    if estimate_9s:
        interval_map = estimate_interval_map(
            csv_path, reference_timestamp, default_interval
        )

    # スケジュールを生成
    schedule = {0: reference_timestamp}
    current_timestamp = reference_timestamp

    for frame_num in range(1, num_frames):
        interval = interval_map.get(frame_num, default_interval)
        current_timestamp += timedelta(seconds=interval)
        schedule[frame_num] = current_timestamp

    return reference_timestamp, schedule


def save_interval_map(interval_map: Dict[int, int], output_path: str) -> None:
    """間隔マップをJSONファイルに保存

    Args:
        interval_map: {フレーム番号: 間隔（秒）} の辞書
        output_path: 出力パス
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(interval_map, f, indent=2, ensure_ascii=False)

    logger.info(f"間隔マップを保存しました: {output_path}")


def load_interval_map(input_path: str) -> Dict[int, int]:
    """間隔マップをJSONファイルから読み込み

    Args:
        input_path: 入力パス

    Returns:
        {フレーム番号: 間隔（秒）} の辞書
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # JSONのキーは文字列なので整数に変換
            return {int(k): int(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"間隔マップの読み込みに失敗: {e}")
        return {}
