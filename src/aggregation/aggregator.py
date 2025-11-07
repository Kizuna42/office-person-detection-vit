"""集計モジュール - ゾーン別人数カウントと統計情報の計算"""

from collections import defaultdict
import csv
import logging

import numpy as np

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

    def export_csv(self, output_path: str, zone_ids: list[str] | None = None) -> None:
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
                    zone_ids = [*list(zone_ids), "unclassified"]

            # CSV出力
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # ヘッダー行: timestamp, zone_1, zone_2, ..., unclassified
                header = ["timestamp", *zone_ids]
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
        """各ゾーンの統計情報（平均、最大、最小、標準偏差、中央値、四分位数）を計算

        Returns:
            ゾーン別統計情報
            {
                zone_id: {
                    'average': float,
                    'max': int,
                    'min': int,
                    'std': float,
                    'median': float,
                    'q1': float,
                    'q3': float,
                    'total_frames': int
                }
            }
        """
        statistics: dict[str, dict] = {}

        for zone_id, counts in self._zone_data.items():
            if counts:
                counts_array = np.array(counts)
                statistics[zone_id] = {
                    "average": float(np.mean(counts_array)),
                    "max": int(np.max(counts_array)),
                    "min": int(np.min(counts_array)),
                    "std": float(np.std(counts_array)),
                    "median": float(np.median(counts_array)),
                    "q1": float(np.percentile(counts_array, 25)),
                    "q3": float(np.percentile(counts_array, 75)),
                    "total_frames": len(counts),
                }
            else:
                statistics[zone_id] = {
                    "average": 0.0,
                    "max": 0,
                    "min": 0,
                    "std": 0.0,
                    "median": 0.0,
                    "q1": 0.0,
                    "q3": 0.0,
                    "total_frames": 0,
                }

        logger.info(f"Statistics calculated for {len(statistics)} zones")
        return statistics

    def get_trend_analysis(self) -> dict[str, dict]:
        """時系列トレンド分析を実行

        Returns:
            ゾーン別トレンド情報
            {
                zone_id: {
                    'trend': str,  # 'increasing', 'decreasing', 'stable'
                    'slope': float,  # 線形回帰の傾き
                    'r_squared': float,  # 決定係数
                }
            }
        """
        trends: dict[str, dict] = {}

        for zone_id, counts in self._zone_data.items():
            if len(counts) < 2:
                trends[zone_id] = {
                    "trend": "stable",
                    "slope": 0.0,
                    "r_squared": 0.0,
                }
                continue

            # 線形回帰でトレンドを計算
            x = np.arange(len(counts))
            y = np.array(counts)

            # 最小二乗法で線形回帰
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]

            # 決定係数を計算
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # トレンドを判定（傾きの閾値は0.01）
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"

            trends[zone_id] = {
                "trend": trend,
                "slope": float(slope),
                "r_squared": float(r_squared),
            }

        return trends

    def get_peak_times(self, top_n: int = 3) -> dict[str, list[tuple[str, int]]]:
        """ピーク時間帯を特定

        Args:
            top_n: 上位N個のピーク時間帯を返す

        Returns:
            ゾーン別ピーク時間帯のリスト
            {
                zone_id: [(timestamp, count), ...]
            }
        """
        peaks: dict[str, list[tuple[str, int]]] = {}

        # タイムスタンプごとの集計
        timestamp_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for result in self.results:
            timestamp_data[result.timestamp][result.zone_id] = result.count

        # 各ゾーンのピーク時間帯を計算
        for zone_id in self._zone_data:
            zone_peaks = []
            for timestamp, zone_counts in timestamp_data.items():
                count = zone_counts.get(zone_id, 0)
                zone_peaks.append((timestamp, count))

            # カウントでソートして上位N個を取得
            zone_peaks.sort(key=lambda x: x[1], reverse=True)
            peaks[zone_id] = zone_peaks[:top_n]

        return peaks

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
