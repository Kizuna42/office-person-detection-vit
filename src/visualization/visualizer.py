"""Visualization module for detection results and attention maps."""

import logging
import os

import cv2
import matplotlib
import numpy as np

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from src.aggregation import Aggregator
from src.models.data_models import Detection

logger = logging.getLogger(__name__)


class Visualizer:
    """検出結果の可視化クラス

    バウンディングボックス付き画像の生成、Attention Mapのオーバーレイ表示、
    デバッグモード時の中間処理結果表示を担当する。
    """

    def __init__(self, debug_mode: bool = False):
        """Visualizerを初期化

        Args:
            debug_mode: デバッグモードの有効化
        """
        self.debug_mode = debug_mode
        logger.info(f"Visualizer initialized (debug_mode={debug_mode})")

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        show_confidence: bool = True,
        show_coords: bool = False,
    ) -> np.ndarray:
        """フレームに検出結果（バウンディングボックス）を描画

        Args:
            frame: 入力画像 (numpy array, BGR format)
            detections: 検出結果のリスト
            show_confidence: 信頼度を表示するか
            show_coords: 座標情報を表示するか（デバッグモード用）

        Returns:
            検出結果を描画した画像
        """
        image = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # バウンディングボックスを描画
            color = (0, 255, 0)  # 緑色
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # ラベルテキストを作成
            label_parts = [detection.class_name]
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            label = " ".join(label_parts)

            # ラベル背景を描画
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)

            # ラベルテキストを描画
            cv2.putText(
                image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 足元座標を描画（カメラ座標）
            foot_x, foot_y = detection.camera_coords
            foot_x, foot_y = int(foot_x), int(foot_y)
            cv2.circle(image, (foot_x, foot_y), 5, (0, 0, 255), -1)

            # デバッグモード: 座標情報を表示
            if self.debug_mode or show_coords:
                coord_text = f"Cam: ({foot_x},{foot_y})"
                cv2.putText(
                    image,
                    coord_text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # フロアマップ座標も表示
                if detection.floor_coords is not None:
                    floor_x, floor_y = detection.floor_coords
                    floor_text = f"Floor: ({floor_x:.1f},{floor_y:.1f})"
                    cv2.putText(
                        image,
                        floor_text,
                        (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                # ゾーン情報も表示
                if detection.zone_ids:
                    zone_text = f"Zones: {','.join(detection.zone_ids)}"
                    cv2.putText(
                        image,
                        zone_text,
                        (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

        # デバッグモード: フレーム情報を表示
        if self.debug_mode:
            info_text = f"Detections: {len(detections)}"
            cv2.putText(
                image,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return image

    def draw_attention_map(self, frame: np.ndarray, attention_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Attention Mapをオーバーレイ表示

        Args:
            frame: 入力画像 (numpy array, BGR format)
            attention_map: Attention Map (1D or 2D numpy array)
            alpha: オーバーレイの透明度 (0.0-1.0)

        Returns:
            Attention Mapをオーバーレイした画像
        """
        image = frame.copy()
        height, width = frame.shape[:2]

        try:
            # Attention mapが1Dの場合、2Dに変形
            if attention_map.ndim == 1:
                num_patches = len(attention_map)
                patch_size = int(np.sqrt(num_patches))

                if patch_size * patch_size != num_patches:
                    logger.warning(
                        f"Attention map size {num_patches} is not a perfect square. "
                        f"Using closest square: {patch_size}x{patch_size}"
                    )
                    # パディングまたはトリミング
                    target_size = patch_size * patch_size
                    if num_patches < target_size:
                        attention_map = np.pad(
                            attention_map,
                            (0, target_size - num_patches),
                            mode="constant",
                        )
                    else:
                        attention_map = attention_map[:target_size]

                attention_grid = attention_map.reshape(patch_size, patch_size)
            else:
                attention_grid = attention_map

            # 元画像サイズにリサイズ
            attention_resized = cv2.resize(attention_grid, (width, height), interpolation=cv2.INTER_LINEAR)

            # 正規化（0-255）
            attention_normalized = (attention_resized * 255).astype(np.uint8)

            # ヒートマップに変換（JETカラーマップ）
            attention_colored = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)

            # 元画像とブレンド
            blended = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)

            # カラーバーを追加（デバッグモード時）
            if self.debug_mode:
                blended = self._add_colorbar(blended, attention_normalized)

            return blended

        except Exception as e:
            logger.error(f"Failed to draw attention map: {e}")
            return image

    def _add_colorbar(self, image: np.ndarray, _attention_normalized: np.ndarray) -> np.ndarray:
        """カラーバーを画像に追加

        Args:
            image: 画像
            attention_normalized: 正規化されたAttention Map

        Returns:
            カラーバー付き画像
        """
        _height, width = image.shape[:2]
        colorbar_width = 30
        colorbar_height = 200

        # カラーバーを作成
        colorbar = np.linspace(255, 0, colorbar_height, dtype=np.uint8)
        colorbar = np.tile(colorbar.reshape(-1, 1), (1, colorbar_width))
        colorbar_colored = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET)

        # 画像の右側に配置
        x_pos = width - colorbar_width - 20
        y_pos = 50

        # 白い背景を描画
        cv2.rectangle(
            image,
            (x_pos - 5, y_pos - 5),
            (x_pos + colorbar_width + 5, y_pos + colorbar_height + 5),
            (255, 255, 255),
            -1,
        )

        # カラーバーを配置
        image[y_pos : y_pos + colorbar_height, x_pos : x_pos + colorbar_width] = colorbar_colored

        # ラベルを追加
        cv2.putText(
            image,
            "High",
            (x_pos - 40, y_pos + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            "Low",
            (x_pos - 35, y_pos + colorbar_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return image

    def visualize_with_attention(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        attention_map: np.ndarray | None = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """検出結果とAttention Mapを同時に可視化

        Args:
            frame: 入力画像 (numpy array, BGR format)
            detections: 検出結果のリスト
            attention_map: Attention Map (optional)
            alpha: Attention Mapの透明度

        Returns:
            可視化画像
        """
        # まずAttention Mapをオーバーレイ
        image = self.draw_attention_map(frame, attention_map, alpha) if attention_map is not None else frame.copy()

        # 検出結果を描画
        image = self.draw_detections(image, detections, show_coords=self.debug_mode)

        return image

    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """画像を保存

        Args:
            image: 保存する画像
            output_path: 保存先パス

        Returns:
            保存成功時True、失敗時False
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            success = bool(cv2.imwrite(output_path, image))
            if success:
                logger.info(f"Image saved: {output_path}")
            else:
                logger.error(f"Failed to save image: {output_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def create_comparison_view(
        self,
        original: np.ndarray,
        with_detections: np.ndarray,
        with_attention: np.ndarray | None = None,
    ) -> np.ndarray:
        """比較ビューを作成（デバッグモード用）

        Args:
            original: 元画像
            with_detections: 検出結果付き画像
            with_attention: Attention Map付き画像 (optional)

        Returns:
            比較ビュー画像
        """
        images = [original, with_detections]
        labels = ["Original", "Detections"]

        if with_attention is not None:
            images.append(with_attention)
            labels.append("Attention Map")

        # 各画像にラベルを追加
        labeled_images = []
        for img, label in zip(images, labels, strict=False):
            img_copy = img.copy()
            cv2.putText(
                img_copy,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
            )
            cv2.putText(img_copy, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            labeled_images.append(img_copy)

        # 水平方向に連結
        comparison = np.hstack(labeled_images)

        return comparison

    def plot_time_series(self, aggregator: Aggregator, output_path: str, figsize: tuple = (12, 6)) -> bool:
        """ゾーン別人数の時系列グラフを生成

        Args:
            aggregator: 集計データを持つAggregatorインスタンス
            output_path: 保存先パス（PNG形式）
            figsize: グラフのサイズ (width, height)

        Returns:
            保存成功時True、失敗時False
        """
        try:
            # 集計結果からデータを抽出
            if not aggregator.results:
                logger.warning("No aggregation results to plot")
                return False

            # タイムスタンプとゾーン別データを整理
            timestamps = []
            zone_data: dict[str, list[int]] = {}

            # ユニークなタイムスタンプを取得
            unique_timestamps = sorted({r.timestamp for r in aggregator.results})

            # 各ゾーンのデータを整理
            for timestamp in unique_timestamps:
                timestamps.append(timestamp)

                # このタイムスタンプの各ゾーンのカウントを取得
                timestamp_results = [r for r in aggregator.results if r.timestamp == timestamp]

                for result in timestamp_results:
                    if result.zone_id not in zone_data:
                        zone_data[result.zone_id] = []
                    zone_data[result.zone_id].append(result.count)

            # グラフを作成
            plt.figure(figsize=figsize)

            # 各ゾーンのデータをプロット
            for zone_id, counts in zone_data.items():
                plt.plot(
                    range(len(counts)),
                    counts,
                    marker="o",
                    label=zone_id,
                    linewidth=2,
                    markersize=6,
                )

            # グラフの設定
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Number of People", fontsize=12)
            plt.title("Zone Occupancy Over Time", fontsize=14, fontweight="bold")
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.3)

            # X軸のラベルを設定（タイムスタンプ）
            if len(timestamps) <= 20:
                # タイムスタンプが少ない場合は全て表示
                plt.xticks(range(len(timestamps)), timestamps, rotation=45, ha="right")
            else:
                # タイムスタンプが多い場合は間引いて表示
                step = len(timestamps) // 10
                indices = range(0, len(timestamps), step)
                plt.xticks(indices, [timestamps[i] for i in indices], rotation=45, ha="right")

            # レイアウトを調整
            plt.tight_layout()

            # 保存
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Time series graph saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create time series graph: {e}")
            plt.close()
            return False

    def plot_zone_statistics(self, aggregator: Aggregator, output_path: str, figsize: tuple = (10, 6)) -> bool:
        """ゾーン別統計情報のグラフを生成

        Args:
            aggregator: 集計データを持つAggregatorインスタンス
            output_path: 保存先パス（PNG形式）
            figsize: グラフのサイズ (width, height)

        Returns:
            保存成功時True、失敗時False
        """
        try:
            # 統計情報を取得
            statistics = aggregator.get_statistics()

            if not statistics:
                logger.warning("No statistics to plot")
                return False

            # データを抽出
            zone_ids = list(statistics.keys())
            averages = [statistics[z]["average"] for z in zone_ids]
            maxs = [statistics[z]["max"] for z in zone_ids]
            mins = [statistics[z]["min"] for z in zone_ids]

            # グラフを作成
            _fig, ax = plt.subplots(figsize=figsize)

            x = np.arange(len(zone_ids))
            width = 0.25

            # 棒グラフを描画
            ax.bar(x - width, averages, width, label="Average", alpha=0.8)
            ax.bar(x, maxs, width, label="Maximum", alpha=0.8)
            ax.bar(x + width, mins, width, label="Minimum", alpha=0.8)

            # グラフの設定
            ax.set_xlabel("Zone", fontsize=12)
            ax.set_ylabel("Number of People", fontsize=12)
            ax.set_title("Zone Statistics", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(zone_ids, rotation=45, ha="right")
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            # レイアウトを調整
            plt.tight_layout()

            # 保存
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Zone statistics graph saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create zone statistics graph: {e}")
            plt.close()
            return False

    def plot_heatmap(self, aggregator: Aggregator, output_path: str, figsize: tuple = (12, 8)) -> bool:
        """ゾーン×時間のヒートマップを生成

        Args:
            aggregator: 集計データを持つAggregatorインスタンス
            output_path: 保存先パス（PNG形式）
            figsize: グラフのサイズ (width, height)

        Returns:
            保存成功時True、失敗時False
        """
        try:
            if not aggregator.results:
                logger.warning("No aggregation results to plot")
                return False

            # タイムスタンプとゾーンIDを取得
            unique_timestamps = sorted({r.timestamp for r in aggregator.results})
            unique_zones = sorted({r.zone_id for r in aggregator.results})

            # ヒートマップ用のマトリックスを作成
            heatmap_data = np.zeros((len(unique_zones), len(unique_timestamps)))

            for i, zone_id in enumerate(unique_zones):
                for j, timestamp in enumerate(unique_timestamps):
                    # このゾーンとタイムスタンプの組み合わせのカウントを取得
                    matching_results = [
                        r for r in aggregator.results if r.zone_id == zone_id and r.timestamp == timestamp
                    ]
                    if matching_results:
                        heatmap_data[i, j] = matching_results[0].count

            # ヒートマップを作成
            _fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")

            # カラーバーを追加
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Number of People", rotation=270, labelpad=20, fontsize=12)

            # 軸ラベルを設定
            ax.set_xticks(np.arange(len(unique_timestamps)))
            ax.set_yticks(np.arange(len(unique_zones)))

            # タイムスタンプが多い場合は間引いて表示
            if len(unique_timestamps) <= 20:
                ax.set_xticklabels(unique_timestamps, rotation=45, ha="right")
            else:
                step = len(unique_timestamps) // 10
                labels = [""] * len(unique_timestamps)
                for i in range(0, len(unique_timestamps), step):
                    labels[i] = unique_timestamps[i]
                ax.set_xticklabels(labels, rotation=45, ha="right")

            ax.set_yticklabels(unique_zones)

            # グラフの設定
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Zone", fontsize=12)
            ax.set_title("Zone Occupancy Heatmap", fontsize=14, fontweight="bold")

            # レイアウトを調整
            plt.tight_layout()

            # 保存
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Heatmap saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            plt.close()
            return False
