"""集計モジュール - ゾーン別人数カウントと統計情報の計算"""

import csv
import logging
from collections import defaultdict
from typing import Dict, List

from src.data_models import AggregationResult, Detection

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
        self.results: List[AggregationResult] = []
        self._zone_data: Dict[str, List[int]] = defaultdict(list)
        logger.info("Aggregator initialized")

    def aggregate_frame(
        self, timestamp: str, detections: List[Detection]
    ) -> Dict[str, int]:
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
            result = AggregationResult(
                timestamp=timestamp, zone_id=zone_id, count=count
            )
            self.results.append(result)
            self._zone_data[zone_id].append(count)

        logger.debug(
            f"Frame aggregated: timestamp={timestamp}, zone_counts={zone_counts}"
        )
        return zone_counts

    def get_zone_counts(self, detections: List[Detection]) -> Dict[str, int]:
        """ゾーン別人数をカウント

        各検出結果のzone_idsを集計し、ゾーンごとの人数を計算する。
        1人が複数のゾーンに属する場合、すべてのゾーンでカウントされる。

        Args:
            detections: 検出結果のリスト

        Returns:
            ゾーン別人数カウント {zone_id: count}
        """
        zone_counts: Dict[str, int] = defaultdict(int)

        for detection in detections:
            if detection.zone_ids:
                # 各ゾーンでカウント
                for zone_id in detection.zone_ids:
                    zone_counts[zone_id] += 1
            else:
                # どのゾーンにも属さない場合は"未分類"としてカウント
                zone_counts["unclassified"] += 1

        return dict(zone_counts)

    def export_csv(self, output_path: str) -> None:
        """CSV形式でエクスポート

        集計結果をCSVファイルに出力する。
        フォーマット: timestamp, zone_id, count

        Args:
            output_path: 出力ファイルパス

        Raises:
            IOError: ファイル書き込みに失敗した場合
        """
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # ヘッダー行
                writer.writerow(["timestamp", "zone_id", "count"])

                # データ行
                for result in self.results:
                    writer.writerow([result.timestamp, result.zone_id, result.count])

            logger.info(
                f"Aggregation results exported to CSV: {output_path} ({len(self.results)} rows)"
            )

        except IOError as e:
            logger.error(f"Failed to export CSV: {e}")
            raise

    def get_statistics(self) -> Dict[str, dict]:
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
        statistics: Dict[str, dict] = {}

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

    def get_zone_ids(self) -> List[str]:
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
