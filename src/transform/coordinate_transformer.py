"""Coordinate transformation module for the office person detection system."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """座標変換クラス

    ホモグラフィ変換を使用して、カメラ座標系からフロアマップ座標系への
    射影変換を実行する。フロアマップの原点オフセットとスケール変換にも対応。
    歪み補正にも対応。

    Attributes:
        H: 3x3ホモグラフィ変換行列
        floormap_config: フロアマップ設定（原点、スケール等）
        camera_matrix: カメラ内部パラメータ行列（オプション）
        dist_coeffs: 歪み係数（オプション）
        use_distortion_correction: 歪み補正を使用するか
    """

    def __init__(
        self,
        homography_matrix: list[list[float]],
        floormap_config: Optional[dict] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        use_distortion_correction: bool = False,
    ):
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
            camera_matrix: カメラ内部パラメータ行列（3x3、オプション）
            dist_coeffs: 歪み係数（オプション）
            use_distortion_correction: 歪み補正を使用するか

        Raises:
            ValueError: 変換行列が不正な形式の場合
        """
        self.H = self._validate_and_convert_matrix(homography_matrix)
        self.floormap_config = floormap_config or {}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.use_distortion_correction = use_distortion_correction

        # フロアマップパラメータを取得
        self.origin_x = self.floormap_config.get("image_origin_x", 0)
        self.origin_y = self.floormap_config.get("image_origin_y", 0)
        self.x_mm_per_pixel = self.floormap_config.get("image_x_mm_per_pixel", 1.0)
        self.y_mm_per_pixel = self.floormap_config.get("image_y_mm_per_pixel", 1.0)
        self.image_width = self.floormap_config.get("image_width", 0)
        self.image_height = self.floormap_config.get("image_height", 0)

        # 歪み補正の設定確認
        if self.use_distortion_correction:
            if self.camera_matrix is None or self.dist_coeffs is None:
                logger.warning(
                    "歪み補正が有効ですが、カメラ行列または歪み係数が設定されていません。"
                    "歪み補正を無効化します。"
                )
                self.use_distortion_correction = False
            else:
                logger.info("歪み補正が有効です")

        logger.info(f"CoordinateTransformerを初期化しました。原点オフセット: ({self.origin_x}, {self.origin_y})")

    def _validate_and_convert_matrix(self, matrix: list[list[float]]) -> np.ndarray:
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
            raise ValueError(f"ホモグラフィ行列が特異行列です（行列式={det}）。変換が正しく動作しません。")

        # 条件数をチェック（数値安定性の指標）
        cond = np.linalg.cond(H)
        if cond > 1e12:
            logger.warning(f"ホモグラフィ行列の条件数が大きすぎます: {cond}。数値誤差が大きくなる可能性があります。")

        # 最後の行が[0, 0, 1]に近いかチェック（射影変換の標準形式）
        last_row = H[2, :]
        expected_last_row = np.array([0, 0, 1])
        if np.linalg.norm(last_row - expected_last_row) > 1e-6:
            logger.debug(f"ホモグラフィ行列の最後の行が標準形式ではありません: {last_row}")

        logger.debug(f"ホモグラフィ行列:\n{H}")
        logger.debug(f"行列式: {det}, 条件数: {cond}")
        return H

    def _undistort_point(self, x: float, y: float) -> tuple[float, float]:
        """単一の点の歪みを補正

        Args:
            x: カメラ座標X
            y: カメラ座標Y

        Returns:
            歪み補正後の座標 (x, y)
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return (x, y)

        # 点を配列形式に変換
        point = np.array([[x, y]], dtype=np.float32)

        # 歪み補正
        undistorted = cv2.undistortPoints(
            point, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix
        )

        return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """画像の歪みを補正

        Args:
            image: 入力画像

        Returns:
            歪み補正された画像

        Raises:
            RuntimeError: カメラ行列または歪み係数が設定されていない場合
        """
        if not self.use_distortion_correction:
            return image

        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("カメラ行列または歪み係数が設定されていません。")

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return dst

    def transform(self, camera_point: tuple[float, float], apply_origin_offset: bool = True) -> tuple[float, float]:
        """カメラ座標をフロアマップ座標に変換する

        同次座標系を使用した射影変換を実行し、原点オフセットを適用する。
        歪み補正が有効な場合は、変換前に座標の歪みを補正する。

        Args:
            camera_point: カメラ座標 (x, y)
            apply_origin_offset: 原点オフセットを適用するか（デフォルト: True）

        Returns:
            フロアマップ座標 (x, y) ピクセル単位

        Raises:
            ValueError: 変換に失敗した場合
        """
        try:
            # 入力値の検証
            if not isinstance(camera_point, (tuple, list)) or len(camera_point) != 2:
                raise ValueError(f"カメラ座標は2要素のタプルまたはリストである必要があります: {camera_point}")

            camera_x, camera_y = float(camera_point[0]), float(camera_point[1])

            # 歪み補正が有効な場合は座標を補正
            if self.use_distortion_correction:
                camera_x, camera_y = self._undistort_point(camera_x, camera_y)

            # 同次座標に変換 [x, y, 1]
            point_homogeneous = np.array([camera_x, camera_y, 1.0])

            # ホモグラフィ変換を適用
            transformed = self.H @ point_homogeneous

            # w成分で正規化
            w = transformed[2]
            if abs(w) < 1e-10:
                error_msg = (
                    f"変換後のw成分が0に近い値です: {w}. "
                    f"カメラ座標: ({camera_x}, {camera_y}), "
                    f"変換後: {transformed}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            floor_x = transformed[0] / w
            floor_y = transformed[1] / w

            # 原点オフセットを適用
            if apply_origin_offset:
                floor_x += self.origin_x
                floor_y += self.origin_y

            # 変換後の座標が範囲外の場合の警告
            floor_point = (float(floor_x), float(floor_y))
            if not self.is_within_bounds(floor_point):
                logger.warning(
                    f"変換後の座標がフロアマップ範囲外です: "
                    f"カメラ座標=({camera_x:.2f}, {camera_y:.2f}), "
                    f"フロアマップ座標=({floor_x:.2f}, {floor_y:.2f}), "
                    f"範囲=[0, {self.image_width}) x [0, {self.image_height})"
                )

            return floor_point

        except ValueError:
            # ValueErrorはそのまま再スロー
            raise
        except Exception as e:
            error_msg = f"座標変換に失敗しました: camera_point={camera_point}, " f"error={type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(f"座標変換エラー: {e}") from e

    def transform_batch(
        self, camera_points: list[tuple[float, float]], apply_origin_offset: bool = True
    ) -> list[tuple[float, float]]:
        """複数のカメラ座標をバッチ変換する

        効率的な行列演算により、複数の座標を一度に変換する。
        歪み補正が有効な場合は、変換前に座標の歪みを補正する。

        Args:
            camera_points: カメラ座標のリスト [(x1, y1), (x2, y2), ...]
            apply_origin_offset: 原点オフセットを適用するか（デフォルト: True）

        Returns:
            フロアマップ座標のリスト [(x1, y1), (x2, y2), ...] ピクセル単位
        """
        if not camera_points:
            return []

        try:
            # 歪み補正が有効な場合は座標を補正
            if self.use_distortion_correction:
                points_array = np.array([[p[0], p[1]] for p in camera_points], dtype=np.float32)
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    undistorted = cv2.undistortPoints(
                        points_array, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix
                    )
                    # 同次座標行列を作成 (N x 3)
                    points_array = np.hstack([undistorted.reshape(-1, 2), np.ones((len(camera_points), 1))])
                else:
                    points_array = np.array([[p[0], p[1], 1.0] for p in camera_points])
            else:
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
            floor_points = [(float(x), float(y)) for x, y in zip(floor_x, floor_y, strict=False)]

            logger.debug(f"{len(camera_points)}個の座標をバッチ変換しました。")
            return floor_points

        except Exception as e:
            logger.error(f"バッチ座標変換に失敗しました: error={e}")
            # フォールバック: 個別に変換
            logger.info("個別変換にフォールバックします。")
            return [self.transform(p, apply_origin_offset) for p in camera_points]

    def get_foot_position(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
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

    def transform_detection(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
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

    def transform_detections_batch(self, bboxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float]]:
        """複数の検出結果をバッチ変換する

        Args:
            bboxes: バウンディングボックスのリスト

        Returns:
            フロアマップ上の足元座標のリスト（ピクセル単位）
        """
        camera_feet = [self.get_foot_position(bbox) for bbox in bboxes]
        floor_feet = self.transform_batch(camera_feet)
        return floor_feet

    def pixel_to_mm(self, pixel_point: tuple[float, float]) -> tuple[float, float]:
        """フロアマップのピクセル座標をmm座標に変換する

        Args:
            pixel_point: ピクセル座標 (x, y)

        Returns:
            mm座標 (x, y)
        """
        x_mm = pixel_point[0] * self.x_mm_per_pixel
        y_mm = pixel_point[1] * self.y_mm_per_pixel
        return (float(x_mm), float(y_mm))

    def mm_to_pixel(self, mm_point: tuple[float, float]) -> tuple[float, float]:
        """mm座標をフロアマップのピクセル座標に変換する

        Args:
            mm_point: mm座標 (x, y)

        Returns:
            ピクセル座標 (x, y)
        """
        x_pixel = mm_point[0] / self.x_mm_per_pixel
        y_pixel = mm_point[1] / self.y_mm_per_pixel
        return (float(x_pixel), float(y_pixel))

    def is_within_bounds(self, floor_point: tuple[float, float]) -> bool:
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

    def get_floormap_info(self) -> dict:
        """フロアマップの情報を取得する

        Returns:
            フロアマップ情報の辞書
        """
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "x_mm_per_pixel": self.x_mm_per_pixel,
            "y_mm_per_pixel": self.y_mm_per_pixel,
        }
