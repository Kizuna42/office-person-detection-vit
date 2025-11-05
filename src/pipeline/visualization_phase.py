"""Visualization phase of the pipeline."""

import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.config import ConfigManager
from src.models import FrameResult
from src.aggregation import Aggregator
from src.visualization import FloormapVisualizer, Visualizer


class VisualizationPhase:
    """可視化フェーズ"""
    
    def __init__(
        self,
        config: ConfigManager,
        logger: logging.Logger
    ):
        """初期化
        
        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        self.config = config
        self.logger = logger
    
    def execute(
        self,
        aggregator: Aggregator,
        frame_results: List[FrameResult],
        output_path: Path
    ) -> None:
        """可視化を実行
        
        Args:
            aggregator: Aggregatorインスタンス
            frame_results: FrameResultのリスト
            output_path: 出力ディレクトリ
        """
        self.logger.info("=" * 80)
        self.logger.info("フェーズ5: 可視化")
        self.logger.info("=" * 80)
        
        # Visualizerの初期化
        visualizer = Visualizer(debug_mode=self.config.get('output.debug_mode', False))
        
        # 時系列グラフの生成
        time_series_path = output_path / 'graphs' / 'time_series.png'
        if visualizer.plot_time_series(aggregator, str(time_series_path)):
            self.logger.info(f"時系列グラフを生成しました: {time_series_path}")
        
        # 統計グラフの生成
        statistics_path = output_path / 'graphs' / 'statistics.png'
        if visualizer.plot_zone_statistics(aggregator, str(statistics_path)):
            self.logger.info(f"統計グラフを生成しました: {statistics_path}")
        
        # ヒートマップの生成
        heatmap_path = output_path / 'graphs' / 'heatmap.png'
        if visualizer.plot_heatmap(aggregator, str(heatmap_path)):
            self.logger.info(f"ヒートマップを生成しました: {heatmap_path}")
        
        # FloormapVisualizerの初期化と可視化
        save_floormap_images = self.config.get('output.save_floormap_images', True)
        
        if save_floormap_images:
            floormap_path = self.config.get('floormap.image_path')
            floormap_config = self.config.get('floormap')
            zones = self.config.get('zones', [])
            camera_config = self.config.get('camera', {})
            
            try:
                floormap_visualizer = FloormapVisualizer(
                    floormap_path,
                    floormap_config,
                    zones,
                    camera_config
                )
                
                # 各フレームのフロアマップ画像を生成
                for frame_result in tqdm(frame_results, desc="フロアマップ可視化中"):
                    # フロアマップ上に描画
                    floormap_image = floormap_visualizer.visualize_frame(
                        frame_result,
                        draw_zones=True,
                        draw_labels=True
                    )
                    
                    # 保存
                    floormap_output = output_path / 'floormaps' / f"floormap_{frame_result.timestamp.replace(':', '')}.png"
                    floormap_visualizer.save_visualization(floormap_image, str(floormap_output))
                
                # 凡例を生成
                legend_image = floormap_visualizer.create_legend()
                legend_path = output_path / 'floormaps' / 'legend.png'
                floormap_visualizer.save_visualization(legend_image, str(legend_path))
                self.logger.info(f"フロアマップ凡例を生成しました: {legend_path}")
                
            except FileNotFoundError as e:
                self.logger.warning(f"フロアマップ画像が見つかりません: {e}")
            except Exception as e:
                self.logger.error(f"フロアマップ可視化エラー: {e}", exc_info=True)

