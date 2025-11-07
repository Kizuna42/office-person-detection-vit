"""Visualization phase of the pipeline."""

from pathlib import Path
from typing import List

from tqdm import tqdm

from src.aggregation import Aggregator
from src.models import FrameResult
from src.pipeline.base_phase import BasePhase
from src.visualization import FloormapVisualizer, Visualizer


class VisualizationPhase(BasePhase):
    """可視化フェーズ"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)

    def execute(
        self,
        aggregator: Aggregator,
        frame_results: list[FrameResult],
        output_path: Path,
    ) -> None:
        """可視化を実行

        Args:
            aggregator: Aggregatorインスタンス
            frame_results: FrameResultのリスト
            output_path: 出力ディレクトリ
        """
        self.log_phase_start("フェーズ5: 可視化")

        # Visualizerの初期化
        visualizer = Visualizer(debug_mode=self.config.get("output.debug_mode", False))

        # 時系列グラフの生成
        time_series_path = output_path / "graphs" / "time_series.png"
        if visualizer.plot_time_series(aggregator, str(time_series_path)):
            self.logger.info(f"時系列グラフを生成しました: {time_series_path}")

        # 統計グラフの生成
        statistics_path = output_path / "graphs" / "statistics.png"
        if visualizer.plot_zone_statistics(aggregator, str(statistics_path)):
            self.logger.info(f"統計グラフを生成しました: {statistics_path}")

        # FloormapVisualizerの初期化と可視化
        save_floormap_images = self.config.get("output.save_floormap_images", True)

        if save_floormap_images:
            floormap_path = self.config.get("floormap.image_path")
            floormap_config = self.config.get("floormap")
            zones = self.config.get("zones", [])
            camera_config = self.config.get("camera", {})

            try:
                floormap_visualizer = FloormapVisualizer(floormap_path, floormap_config, zones, camera_config)

                # 各フレームのフロアマップ画像を生成
                for frame_result in tqdm(frame_results, desc="フロアマップ可視化中"):
                    # フロアマップ上に描画
                    floormap_image = floormap_visualizer.visualize_frame(
                        frame_result, draw_zones=True, draw_labels=True
                    )

                    # 保存
                    floormap_output = (
                        output_path / "floormaps" / f"floormap_{frame_result.timestamp.replace(':', '')}.png"
                    )
                    floormap_visualizer.save_visualization(floormap_image, str(floormap_output))

            except FileNotFoundError as e:
                self.logger.warning(f"フロアマップ画像が見つかりません: {e}")
            except Exception as e:
                self.logger.error(f"フロアマップ可視化エラー: {e}", exc_info=True)
