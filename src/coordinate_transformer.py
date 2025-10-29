"""Coordinate transformation module for the office person detection system."""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """座標変換クラス
    
    ホモグラフィ変換を使用して、カメラ座標系からフロアマップ座標系への
    射影変換を実行する。フロアマップの原点オフセットとスケール変換にも対応。
    
    Attributes:
        H: 3x3ホモグラフィ変換行列
        floormap_config: フロアマップ設定（原点、スケール等）
    """
    
    def __init__(self, homography_matrix: List[List[float]], floormap_config: Optional[Dict] = None):
        """CoordinateTransformerを初期化する
        
        Args:
            homography_matrix: 3x3ホモグラフィ変換行列
            floormap_config: フロアマップ設定（オプション）
                - image_width: 画像幅（ピクセル）
                - image_height: 画像高さ（ピクセル）
                - image_origin_x: 原点X座標オフセット（ピクセル）
                - image_origin_y: 原点Y座標オフセット（ピクセル）
                - image_x_mm_per_pixel: X軸スケール（mm/pixel）
                - image_y_mm_per_pixel: Y軸スケール（mm/pixel）
            
        Raises:
            ValueError: 変換行列が不正な形式の場合
        """
        self.H = self._validate_and_convert_matrix(homography_matrix)
        self.floormap_config = floormap_config or {}
        
        # フロアマップパラメータを取得
        self.origin_x = self.floormap_config.get('image_origin_x', 0)
        self.origin_y = self.floormap_config.get('image_origin_y', 0)
        self.x_mm_per_pixel = self.floormap_config.get('image_x_mm_per_pixel', 1.0)
        self.y_mm_per_pixel = self.floormap_config.get('image_y_mm_per_pixel', 1.0)
        self.image_width = self.floormap_config.get('image_width', 0)
        self.image_height = self.floormap_config.get('image_height', 0)
        
        logger.info(f"CoordinateTransformerを初期化しました。原点オフセット: ({self.origin_x}, {self.origin_y})")
    
    def _validate_and_convert_matrix(self, matrix: List[List[float]]) -> np.ndarray:
        """ホモグラフィ変換行列を検証し、numpy配列に変換する
        
        Args:
            matrix: 3x3変換行列（リスト形式）
            
        Returns:
            numpy配列形式の3x3変換行列
            
        Raises:
            ValueError: 行列が不正な形式の場合
        """
        if not isinstance(matrix, (list, np.ndarray)):
            raise ValueError("ホモグラフィ行列はリストまたはnumpy配列である必要があります。")
        
        H = np.array(matrix, dtype=np.float64)
        
        if H.shape != (3, 3):
            raise ValueError(f"ホモグラフィ行列は3x3である必要があります。現在の形状: {H.shape}")
        
        # 行列式が0に近い場合は警告
        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            logger.warning(f"ホモグラフィ行列の行列式が0に近い値です: {det}")
        
        logger.debug(f"ホモグラフィ行列:\n{H}")
        return H
    
    def transform(self, camera_point: Tuple[float, float], apply_origin_offset: bool = True) -> Tuple[float, float]:
        """カメラ座標をフロアマップ座標に変換する
        
        同次座標系を使用した射影変換を実行し、原点オフセットを適用する。
        
        Args:
            camera_point: カメラ座標 (x, y)
            apply_origin_offset: 原点オフセットを適用するか（デフォルト: True）
            
        Returns:
            フロアマップ座標 (x, y) ピクセル単位
            
        Raises:
            ValueError: 変換に失敗した場合
        """
        try:
            # 同次座標に変換 [x, y, 1]
            point_homogeneous = np.array([camera_point[0], camera_point[1], 1.0])
            
            # ホモグラフィ変換を適用
            transformed = self.H @ point_homogeneous
            
            # w成分で正規化
            if abs(transformed[2]) < 1e-10:
                raise ValueError(f"変換後のw成分が0に近い値です: {transformed[2]}")
            
            floor_x = transformed[0] / transformed[2]
            floor_y = transformed[1] / transformed[2]
            
            # 原点オフセットを適用
            if apply_origin_offset:
                floor_x += self.origin_x
                floor_y += self.origin_y
            
            return (float(floor_x), float(floor_y))
            
        except Exception as e:
            logger.error(f"座標変換に失敗しました: camera_point={camera_point}, error={e}")
            raise ValueError(f"座標変換エラー: {e}")
    
    def transform_batch(self, camera_points: List[Tuple[float, float]], apply_origin_offset: bool = True) -> List[Tuple[float, float]]:
        """複数のカメラ座標をバッチ変換する
        
        効率的な行列演算により、複数の座標を一度に変換する。
        
        Args:
            camera_points: カメラ座標のリスト [(x1, y1), (x2, y2), ...]
            apply_origin_offset: 原点オフセットを適用するか（デフォルト: True）
            
        Returns:
            フロアマップ座標のリスト [(x1, y1), (x2, y2), ...] ピクセル単位
        """
        if not camera_points:
            return []
        
        try:
            # 同次座標行列を作成 (N x 3)
            points_array = np.array([[p[0], p[1], 1.0] for p in camera_points])
            
            # バッチ変換 (3 x 3) @ (N x 3).T = (3 x N)
            transformed = self.H @ points_array.T
            
            # w成分で正規化
            w = transformed[2, :]
            
            # w成分が0に近い点をチェック
            if np.any(np.abs(w) < 1e-10):
                logger.warning("一部の点でw成分が0に近い値です。")
            
            floor_x = transformed[0, :] / w
            floor_y = transformed[1, :] / w
            
            # 原点オフセットを適用
            if apply_origin_offset:
                floor_x += self.origin_x
                floor_y += self.origin_y
            
            # リスト形式に変換
            floor_points = [(float(x), float(y)) for x, y in zip(floor_x, floor_y)]
            
            logger.debug(f"{len(camera_points)}個の座標をバッチ変換しました。")
            return floor_points
            
        except Exception as e:
            logger.error(f"バッチ座標変換に失敗しました: error={e}")
            # フォールバック: 個別に変換
            logger.info("個別変換にフォールバックします。")
            return [self.transform(p, apply_origin_offset) for p in camera_points]
    
    def get_foot_position(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """バウンディングボックスから足元座標を計算する
        
        バウンディングボックスの中心下端を足元座標として使用する。
        
        Args:
            bbox: バウンディングボックス (x, y, width, height)
            
        Returns:
            足元座標 (x, y)
        """
        x, y, width, height = bbox
        
        # 中心下端の座標を計算
        foot_x = x + width / 2.0
        foot_y = y + height
        
        return (foot_x, foot_y)
    
    def transform_detection(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """検出結果のバウンディングボックスをフロアマップ座標に変換する
        
        バウンディングボックスから足元座標を計算し、フロアマップ座標に変換する。
        
        Args:
            bbox: バウンディングボックス (x, y, width, height)
            
        Returns:
            フロアマップ上の足元座標 (x, y)
        """
        camera_foot = self.get_foot_position(bbox)
        floor_foot = self.transform(camera_foot)
        return floor_foot
    
    def transform_detections_batch(self, bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float]]:
        """複数の検出結果をバッチ変換する
        
        Args:
            bboxes: バウンディングボックスのリスト
            
        Returns:
            フロアマップ上の足元座標のリスト（ピクセル単位）
        """
        camera_feet = [self.get_foot_position(bbox) for bbox in bboxes]
        floor_feet = self.transform_batch(camera_feet)
        return floor_feet
    
    def pixel_to_mm(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """フロアマップのピクセル座標をmm座標に変換する
        
        Args:
            pixel_point: ピクセル座標 (x, y)
            
        Returns:
            mm座標 (x, y)
        """
        x_mm = pixel_point[0] * self.x_mm_per_pixel
        y_mm = pixel_point[1] * self.y_mm_per_pixel
        return (float(x_mm), float(y_mm))
    
    def mm_to_pixel(self, mm_point: Tuple[float, float]) -> Tuple[float, float]:
        """mm座標をフロアマップのピクセル座標に変換する
        
        Args:
            mm_point: mm座標 (x, y)
            
        Returns:
            ピクセル座標 (x, y)
        """
        x_pixel = mm_point[0] / self.x_mm_per_pixel
        y_pixel = mm_point[1] / self.y_mm_per_pixel
        return (float(x_pixel), float(y_pixel))
    
    def is_within_bounds(self, floor_point: Tuple[float, float]) -> bool:
        """座標がフロアマップの範囲内にあるか判定する
        
        Args:
            floor_point: フロアマップ座標 (x, y) ピクセル単位
            
        Returns:
            範囲内の場合True、範囲外の場合False
        """
        if self.image_width == 0 or self.image_height == 0:
            # 画像サイズが設定されていない場合は常にTrue
            return True
        
        x, y = floor_point
        return (0 <= x < self.image_width) and (0 <= y < self.image_height)
    
    def get_floormap_info(self) -> Dict:
        """フロアマップの情報を取得する
        
        Returns:
            フロアマップ情報の辞書
        """
        return {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'x_mm_per_pixel': self.x_mm_per_pixel,
            'y_mm_per_pixel': self.y_mm_per_pixel
        }
