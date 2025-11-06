"""Aggregation phase of the pipeline."""

from pathlib import Path
from typing import List

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

    def execute(
        self, frame_results: List[FrameResult], output_path: Path
    ) -> Aggregator:
        """集計処理を実行

        Args:
            frame_results: FrameResultのリスト
            output_path: 出力ディレクトリ

        Returns:
            Aggregatorインスタンス
        """
        self.logger.info("=" * 80)
        self.logger.info("フェーズ4: 集計とレポート生成")
        self.logger.info("=" * 80)

        aggregator = Aggregator()

        # フレームごとに集計
        for frame_result in tqdm(frame_results, desc="集計中"):
            zone_counts = aggregator.aggregate_frame(
                frame_result.timestamp, frame_result.detections
            )
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
        self.logger.info("=" * 80)

        # CSV出力
        csv_path = output_path / "zone_counts.csv"
        aggregator.export_csv(str(csv_path))
        self.logger.info(f"集計結果をCSVに出力しました: {csv_path}")

        return aggregator
