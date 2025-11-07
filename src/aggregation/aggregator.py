"""集計モジュール - ゾーン別人数カウントと統計情報の計算"""

from collections import defaultdict
import csv
import logging
from typing import Dict, List

from src.models.data_models import AggregationResult, Detection

logger = logging.getLogger(__name__)


class Aggregator:
    """集計クラス

    フレームごとのゾーン別人数カウント、集計結果の蓄積、
    CSV出力、統計情報計算を担当する。

    Attributes:
        results: 集計結果のリスト
        _zone_data: ゾーン別の時系列データ {zone_id: [counts]}
    """

    def __init__(self):
        """Aggregatorを初期化"""
        self.results: list[AggregationResult] = []
        self._zone_data: dict[str, list[int]] = defaultdict(list)
        logger.info("Aggregator initialized")

    def aggregate_frame(self, timestamp: str, detections: list[Detection]) -> dict[str, int]:
        """1フレームの集計結果を追加

        Args:
            timestamp: タイムスタンプ (HH:MM形式)
            detections: 検出結果のリスト

        Returns:
            ゾーン別人数カウント {zone_id: count}
        """
        zone_counts = self.get_zone_counts(detections)

        # 集計結果を保存
        for zone_id, count in zone_counts.items():
            result = AggregationResult(timestamp=timestamp, zone_id=zone_id, count=count)
            self.results.append(result)
            self._zone_data[zone_id].append(count)

        logger.debug(f"Frame aggregated: timestamp={timestamp}, zone_counts={zone_counts}")
        return zone_counts

    def get_zone_counts(self, detections: list[Detection]) -> dict[str, int]:
        """ゾーン別人数をカウント

        各検出結果のzone_idsを集計し、ゾーンごとの人数を計算する。
        1人が複数のゾーンに属する場合、すべてのゾーンでカウントされる。

        Args:
            detections: 検出結果のリスト

        Returns:
            ゾーン別人数カウント {zone_id: count}
        """
        zone_counts: dict[str, int] = defaultdict(int)

        for detection in detections:
            if detection.zone_ids:
                # 各ゾーンでカウント
                for zone_id in detection.zone_ids:
                    zone_counts[zone_id] += 1
            else:
                # どのゾーンにも属さない場合は"未分類"としてカウント
                zone_counts["unclassified"] += 1

        return dict(zone_counts)

    def export_csv(self, output_path: str, zone_ids: list[str] = None) -> None:
        """CSV形式でエクスポート

        集計結果をCSVファイルに出力する。
        フォーマット: timestamp, zone_1, zone_2, ..., unclassified

        Args:
            output_path: 出力ファイルパス
            zone_ids: ゾーンIDのリスト（順序指定用、Noneの場合は自動検出）

        Raises:
            IOError: ファイル書き込みに失敗した場合
        """
        try:
            # タイムスタンプごとに集計結果をグループ化
            timestamp_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for result in self.results:
                timestamp_data[result.timestamp][result.zone_id] = result.count

            # ゾーンIDの順序を決定
            if zone_ids is None:
                # 自動検出: 設定ファイルの順序 + unclassified
                all_zone_ids = set()
                for result in self.results:
                    all_zone_ids.add(result.zone_id)
                # zone_1, zone_2, ... の順序でソート
                sorted_zones = sorted([z for z in all_zone_ids if z.startswith("zone_")])
                if "unclassified" in all_zone_ids:
                    sorted_zones.append("unclassified")
                zone_ids = sorted_zones
            else:
                # 指定された順序を使用（unclassifiedが含まれていない場合は追加）
                if "unclassified" not in zone_ids:
                    zone_ids = list(zone_ids) + ["unclassified"]

            # CSV出力
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # ヘッダー行: timestamp, zone_1, zone_2, ..., unclassified
                header = ["timestamp"] + zone_ids
                writer.writerow(header)

                # データ行: タイムスタンプごとに1行
                for timestamp in sorted(timestamp_data.keys()):
                    row = [timestamp]
                    for zone_id in zone_ids:
                        count = timestamp_data[timestamp].get(zone_id, 0)
                        row.append(count)
                    writer.writerow(row)

            logger.info(f"Aggregation results exported to CSV: {output_path} ({len(timestamp_data)} rows)")

        except OSError as e:
            logger.error(f"Failed to export CSV: {e}")
            raise

    def get_statistics(self) -> dict[str, dict]:
        """各ゾーンの統計情報（平均、最大、最小）を計算

        Returns:
            ゾーン別統計情報
            {
                zone_id: {
                    'average': float,
                    'max': int,
                    'min': int,
                    'total_frames': int
                }
            }
        """
        statistics: dict[str, dict] = {}

        for zone_id, counts in self._zone_data.items():
            if counts:
                statistics[zone_id] = {
                    "average": sum(counts) / len(counts),
                    "max": max(counts),
                    "min": min(counts),
                    "total_frames": len(counts),
                }
            else:
                statistics[zone_id] = {
                    "average": 0.0,
                    "max": 0,
                    "min": 0,
                    "total_frames": 0,
                }

        logger.info(f"Statistics calculated for {len(statistics)} zones")
        return statistics

    def get_total_detections(self) -> int:
        """全フレームの総検出数を取得

        Returns:
            総検出数
        """
        return sum(result.count for result in self.results)

    def get_zone_ids(self) -> list[str]:
        """集計されたゾーンIDのリストを取得

        Returns:
            ゾーンIDのリスト
        """
        return list(self._zone_data.keys())

    def clear(self) -> None:
        """集計結果をクリア"""
        self.results.clear()
        self._zone_data.clear()
        logger.info("Aggregation results cleared")
