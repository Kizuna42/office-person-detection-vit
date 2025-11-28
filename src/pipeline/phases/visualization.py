"""Visualization phase of the pipeline."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

from tqdm import tqdm

from src.aggregation import Aggregator
from src.config import ConfigManager
from src.models import FrameResult
from src.pipeline.phases.base import BasePhase
from src.utils.export_utils import SideBySideVideoExporter
from src.visualization import FloormapVisualizer, Visualizer


class VisualizationPhase(BasePhase):
    """可視化フェーズ"""

    def __init__(self, config: ConfigManager, logger: logging.Logger):
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
            parallel_workers = self.config.get("output.visualization_workers", 4)

            try:
                # フロアマップディレクトリを事前に作成
                floormaps_dir = output_path / "floormaps"
                floormaps_dir.mkdir(parents=True, exist_ok=True)

                def generate_floormap(frame_result: FrameResult) -> str | None:
                    """単一フレームのフロアマップを生成"""
                    try:
                        # 各スレッドで独自のVisualizerインスタンスを使用
                        visualizer = FloormapVisualizer(floormap_path, floormap_config, zones, camera_config)
                        floormap_image = visualizer.visualize_frame(frame_result, draw_zones=True, draw_labels=True)
                        floormap_output = floormaps_dir / f"floormap_{frame_result.timestamp.replace(':', '')}.png"
                        visualizer.save_visualization(floormap_image, str(floormap_output))
                        return str(floormap_output)
                    except Exception as e:
                        self.logger.warning(f"フロアマップ生成エラー（{frame_result.timestamp}）: {e}")
                        return None

                # 並列生成
                completed_count = 0
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    futures = {executor.submit(generate_floormap, fr): fr for fr in frame_results}

                    for future in tqdm(as_completed(futures), total=len(futures), desc="フロアマップ可視化中（並列）"):
                        result = future.result()
                        if result:
                            completed_count += 1

                self.logger.info(f"フロアマップ画像を {completed_count}/{len(frame_results)} 枚生成しました")

            except FileNotFoundError as e:
                self.logger.warning(f"フロアマップ画像が見つかりません: {e}")
            except Exception as e:
                self.logger.error(f"フロアマップ可視化エラー: {e}", exc_info=True)

        # Side-by-side動画の生成（オプション）
        save_side_by_side_video = self.config.get("output.save_side_by_side_video", True)
        tracking_enabled = self.config.get("tracking.enabled", False)

        if save_side_by_side_video and tracking_enabled and frame_results:
            try:
                # 検出画像ディレクトリとフロアマップ画像ディレクトリのパスを取得
                # output_pathはphase5_visualizationディレクトリ
                # 検出画像はphase2_detection/images/にある
                detection_images_dir = output_path.parent / "phase2_detection" / "images"
                floormap_images_dir = output_path / "floormaps"

                if not detection_images_dir.exists():
                    self.logger.warning(f"検出画像ディレクトリが見つかりません: {detection_images_dir}")
                elif not floormap_images_dir.exists():
                    self.logger.warning(f"フロアマップ画像ディレクトリが見つかりません: {floormap_images_dir}")
                else:
                    # SideBySideVideoExporterを使用して動画を生成
                    video_fps = self.config.get("output.side_by_side_video_fps", 1.0)
                    exporter = SideBySideVideoExporter(output_path)

                    video_path = exporter.export_side_by_side_video(
                        frame_results=frame_results,
                        detection_images_dir=detection_images_dir,
                        floormap_images_dir=floormap_images_dir,
                        filename="side_by_side_tracking.mp4",
                        fps=video_fps,
                    )

                    self.logger.info(f"Side-by-side動画を生成しました: {video_path}")

            except Exception as e:
                self.logger.error(f"Side-by-side動画生成エラー: {e}", exc_info=True)

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
