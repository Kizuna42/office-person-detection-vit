"""Floor map visualization module for the office person detection system."""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.data_models import Detection, FrameResult

logger = logging.getLogger(__name__)


class FloormapVisualizer:
    """フロアマップ可視化クラス
    
    フロアマップ画像上に検出結果やゾーンを描画する。
    
    Attributes:
        floormap_image: フロアマップ画像
        floormap_config: フロアマップ設定
        zone_colors: ゾーンごとの色マップ
    """
    
    def __init__(self, floormap_path: str, floormap_config: Dict, zones: List[Dict]):
        """FloormapVisualizerを初期化する
        
        Args:
            floormap_path: フロアマップ画像のパス
            floormap_config: フロアマップ設定
            zones: ゾーン定義のリスト
            
        Raises:
            FileNotFoundError: フロアマップ画像が見つからない場合
        """
        self.floormap_config = floormap_config
        self.zones = zones
        
        # フロアマップ画像を読み込み
        if not Path(floormap_path).exists():
            raise FileNotFoundError(f"フロアマップ画像が見つかりません: {floormap_path}")
        
        self.floormap_image = cv2.imread(floormap_path)
        if self.floormap_image is None:
            raise ValueError(f"フロアマップ画像の読み込みに失敗しました: {floormap_path}")
        
        logger.info(f"フロアマップ画像を読み込みました: {floormap_path} "
                   f"({self.floormap_image.shape[1]}x{self.floormap_image.shape[0]})")
        
        # ゾーンごとの色を生成
        self.zone_colors = self._generate_zone_colors()
    
    def _generate_zone_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """ゾーンごとの色を生成する
        
        Returns:
            ゾーンIDと色のマッピング
        """
        colors = {}
        color_palette = [
            (255, 100, 100),  # 赤
            (100, 255, 100),  # 緑
            (100, 100, 255),  # 青
            (255, 255, 100),  # 黄
            (255, 100, 255),  # マゼンタ
            (100, 255, 255),  # シアン
            (255, 150, 100),  # オレンジ
            (150, 100, 255),  # 紫
        ]
        
        for i, zone in enumerate(self.zones):
            zone_id = zone['id']
            colors[zone_id] = color_palette[i % len(color_palette)]
        
        # 未分類用の色
        colors['unclassified'] = (128, 128, 128)  # グレー
        
        return colors
    
    def draw_zones(self, image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """ゾーンを描画する
        
        Args:
            image: 描画対象の画像
            alpha: ゾーンの透明度（0.0-1.0）
            
        Returns:
            ゾーンを描画した画像
        """
        overlay = image.copy()
        
        for zone in self.zones:
            zone_id = zone['id']
            polygon = zone['polygon']
            color = self.zone_colors.get(zone_id, (128, 128, 128))
            
            # 多角形を描画
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
            
            # 境界線を描画
            cv2.polylines(image, [points], True, color, 2)
            
            # ゾーン名を描画
            centroid = points.mean(axis=0).astype(int)
            zone_name = zone.get('name', zone_id)
            cv2.putText(image, zone_name, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 透明度を適用
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection], 
                       draw_labels: bool = True) -> np.ndarray:
        """検出結果を描画する
        
        Args:
            image: 描画対象の画像
            detections: 検出結果のリスト
            draw_labels: ラベルを描画するか
            
        Returns:
            検出結果を描画した画像
        """
        for detection in detections:
            if detection.floor_coords is None:
                continue
            
            x, y = detection.floor_coords
            x, y = int(x), int(y)
            
            # 画像範囲外の場合はスキップ
            if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                continue
            
            # ゾーンに応じた色を取得
            if detection.zone_ids:
                color = self.zone_colors.get(detection.zone_ids[0], (0, 255, 0))
            else:
                color = self.zone_colors['unclassified']
            
            # 円を描画（足元位置）
            cv2.circle(image, (x, y), 8, color, -1)
            cv2.circle(image, (x, y), 10, (255, 255, 255), 2)
            
            # ラベルを描画
            if draw_labels:
                label = f"{detection.confidence:.2f}"
                if detection.zone_ids:
                    label += f" ({','.join(detection.zone_ids)})"
                
                cv2.putText(image, label, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, label, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def visualize_frame(self, frame_result: FrameResult, 
                       draw_zones: bool = True,
                       draw_labels: bool = True) -> np.ndarray:
        """フレーム結果を可視化する
        
        Args:
            frame_result: フレーム処理結果
            draw_zones: ゾーンを描画するか
            draw_labels: ラベルを描画するか
            
        Returns:
            可視化画像
        """
        # フロアマップ画像をコピー
        image = self.floormap_image.copy()
        
        # ゾーンを描画
        if draw_zones:
            image = self.draw_zones(image)
        
        # 検出結果を描画
        image = self.draw_detections(image, frame_result.detections, draw_labels)
        
        # フレーム情報を描画
        info_text = f"Frame: {frame_result.frame_number} | Time: {frame_result.timestamp}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # ゾーン別カウントを描画
        y_offset = 70
        for zone_id, count in frame_result.zone_counts.items():
            zone_name = next((z['name'] for z in self.zones if z['id'] == zone_id), zone_id)
            count_text = f"{zone_name}: {count}"
            color = self.zone_colors.get(zone_id, (128, 128, 128))
            
            cv2.putText(image, count_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, count_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 30
        
        return image
    
    def save_visualization(self, image: np.ndarray, output_path: str):
        """可視化画像を保存する
        
        Args:
            image: 保存する画像
            output_path: 保存先パス
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(output_path, image)
        if success:
            logger.info(f"可視化画像を保存しました: {output_path}")
        else:
            logger.error(f"可視化画像の保存に失敗しました: {output_path}")
    
    def create_legend(self, width: int = 300, height: int = None) -> np.ndarray:
        """凡例画像を作成する
        
        Args:
            width: 凡例の幅
            height: 凡例の高さ（Noneの場合は自動計算）
            
        Returns:
            凡例画像
        """
        if height is None:
            height = 50 + len(self.zones) * 40
        
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # タイトル
        cv2.putText(legend, "Zone Legend", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 各ゾーンの凡例
        y_offset = 60
        for zone in self.zones:
            zone_id = zone['id']
            zone_name = zone.get('name', zone_id)
            color = self.zone_colors.get(zone_id, (128, 128, 128))
            
            # 色のボックス
            cv2.rectangle(legend, (10, y_offset - 15), (40, y_offset + 5), color, -1)
            cv2.rectangle(legend, (10, y_offset - 15), (40, y_offset + 5), (0, 0, 0), 1)
            
            # ゾーン名
            cv2.putText(legend, zone_name, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            y_offset += 40
        
        return legend
