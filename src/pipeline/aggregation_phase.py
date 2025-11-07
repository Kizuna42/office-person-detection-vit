"""Aggregation phase of the pipeline."""

from pathlib import Path

from tqdm import tqdm

from src.aggregation import Aggregator
from src.models import FrameResult
from src.pipeline.base_phase import BasePhase


class AggregationPhase(BasePhase):
    """集計フェーズ"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)

    def execute(self, frame_results: list[FrameResult], output_path: Path) -> Aggregator:
        """集計処理を実行

        Args:
            frame_results: FrameResultのリスト
            output_path: 出力ディレクトリ

        Returns:
            Aggregatorインスタンス
        """
        self.log_phase_start("フェーズ4: 集計とレポート生成")

        aggregator = Aggregator()

        # フレームごとに集計
        for frame_result in tqdm(frame_results, desc="集計中"):
            zone_counts = aggregator.aggregate_frame(frame_result.timestamp, frame_result.detections)
            frame_result.zone_counts = zone_counts

        # 統計情報を表示
        statistics = aggregator.get_statistics()
        zones = self.config.get("zones", [])
        self.logger.info("=" * 80)
        self.logger.info("集計統計:")
        for zone_id, stats in statistics.items():
            zone_name = next((z["name"] for z in zones if z["id"] == zone_id), zone_id)
            self.logger.info(f"  {zone_name} ({zone_id}):")
            self.logger.info(f"    平均: {stats['average']:.2f}人")
            self.logger.info(f"    最大: {stats['max']}人")
            self.logger.info(f"    最小: {stats['min']}人")
            self.logger.info(f"    標準偏差: {stats['std']:.2f}人")
            self.logger.info(f"    中央値: {stats['median']:.2f}人")
            self.logger.info(f"    第1四分位数: {stats['q1']:.2f}人")
            self.logger.info(f"    第3四分位数: {stats['q3']:.2f}人")
        self.logger.info("=" * 80)

        # トレンド分析を表示
        trends = aggregator.get_trend_analysis()
        self.logger.info("時系列トレンド分析:")
        for zone_id, trend_info in trends.items():
            zone_name = next((z["name"] for z in zones if z["id"] == zone_id), zone_id)
            trend_str = {"increasing": "増加傾向", "decreasing": "減少傾向", "stable": "安定"}.get(
                trend_info["trend"], trend_info["trend"]
            )
            self.logger.info(
                f"  {zone_name}: {trend_str} " f"(傾き={trend_info['slope']:.3f}, R²={trend_info['r_squared']:.3f})"
            )

        # ピーク時間帯を表示
        peaks = aggregator.get_peak_times(top_n=3)
        self.logger.info("ピーク時間帯（上位3位）:")
        for zone_id, peak_list in peaks.items():
            zone_name = next((z["name"] for z in zones if z["id"] == zone_id), zone_id)
            self.logger.info(f"  {zone_name}:")
            for i, (timestamp, count) in enumerate(peak_list, 1):
                self.logger.info(f"    {i}. {timestamp}: {count}人")

        # CSV出力
        csv_path = output_path / "zone_counts.csv"
        # 設定からゾーンIDの順序を取得
        zones = self.config.get("zones", [])
        zone_ids = [zone["id"] for zone in zones] if zones else None
        aggregator.export_csv(str(csv_path), zone_ids=zone_ids)
        self.logger.info(f"集計結果をCSVに出力しました: {csv_path}")

        return aggregator

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
