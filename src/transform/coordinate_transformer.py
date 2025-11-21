"""Coordinate transformation module for the office person detection system."""

import logging

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
        floormap_config: dict | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
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
                    "歪み補正が有効ですが、カメラ行列または歪み係数が設定されていません。歪み補正を無効化します。"
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
        if not isinstance(matrix, list | np.ndarray):
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
        undistorted = cv2.undistortPoints(point, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

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
        new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

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
            if not isinstance(camera_point, tuple | list) or len(camera_point) != 2:
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
                    f"変換後のw成分が0に近い値です: {w}. カメラ座標: ({camera_x}, {camera_y}), 変換後: {transformed}"
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
            error_msg = f"座標変換に失敗しました: camera_point={camera_point}, error={type(e).__name__}: {e}"
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
        result: bool = (0 <= x < self.image_width) and (0 <= y < self.image_height)
        return result

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

    @staticmethod
    def compute_homography_from_params(
        camera_params: dict,
        floormap_config: dict | None = None,
    ) -> np.ndarray:
        """カメラパラメータからホモグラフィ行列を計算する

        Args:
            camera_params: カメラパラメータ辞書
                - height_m: カメラ高さ (m)
                - pitch_deg: 俯角 (度、負=下向き、正=上向き)
                - yaw_deg: 方位角 (度、0=右下方向、正=時計回り/右方向、負=反時計回り/左方向)
                - roll_deg: 回転角 (度、0=水平)
                - focal_length_x: 焦点距離X (px)
                - focal_length_y: 焦点距離Y (px)
                - center_x: 画像中心X (px)
                - center_y: 画像中心Y (px)
                - position_x: カメラ位置X (px、フロアマップ座標)
                - position_y: カメラ位置Y (px、フロアマップ座標)
            floormap_config: フロアマップ設定（オプション）
                - image_x_mm_per_pixel: X軸スケール (mm/px)
                - image_y_mm_per_pixel: Y軸スケール (mm/px)
                - image_origin_x: 原点X (px)
                - image_origin_y: 原点Y (px)

        Returns:
            3x3 ホモグラフィ行列 (Camera -> FloorMap)

        Note:
            仕様: yaw=0, roll=0 の時にフロアマップ座標系で右下方向を向く。
            yawを調整することで、右下方向から左右にパンできる。
            - 正の値: 時計回り（右方向に回転）
            - 負の値: 反時計回り（左方向に回転）
        """
        # パラメータの取得
        h = float(camera_params.get("height_m", 2.2))
        pitch = np.radians(float(camera_params.get("pitch_deg", 45.0)))
        yaw = np.radians(float(camera_params.get("yaw_deg", 0.0)))
        roll = np.radians(float(camera_params.get("roll_deg", 0.0)))

        fx = float(camera_params.get("focal_length_x", 1000.0))
        fy = float(camera_params.get("focal_length_y", 1000.0))
        cx = float(camera_params.get("center_x", 640.0))
        cy = float(camera_params.get("center_y", 360.0))

        # カメラ内部パラメータ行列 K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # 回転行列 R (World -> Camera)
        # World座標系: X右, Y奥, Z上 (右手系)
        # Camera座標系: X右, Y下, Z奥 (右手系)

        # 基本回転 (World -> Camera aligned with World)
        # World X -> Camera X
        # World Y -> Camera Z (Forward) -> Camera Z is depth
        # World Z -> Camera -Y (Up is -Down)
        # しかし、一般的な定義では:
        # Camera looking down -Z axis of World?
        # ここではシンプルに、カメラ座標系で回転を考える

        # 回転行列の構築 (RX * RY * RZ)
        # Pitch: X軸周り回転（正=下向き、負=上向き）
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

        # Yaw: Y軸周り回転（正=時計回り/右方向、負=反時計回り/左方向）
        # 符号を反転して直感的な操作を実現（正の値で右方向に回転）
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])

        # Roll: Z軸周り回転（正=時計回り、負=反時計回り）
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

        # 座標変換行列 (World -> Camera) を構築する
        # World: X=Right, Y=Forward, Z=Up
        # Camera: X=Right, Y=Down, Z=Forward
        # Floormap: X=Right, Y=Down, Z=Up
        #
        # 仕様: yaw=0, roll=0 の時にフロアマップ座標系で右下方向を向く
        # フロアマップ座標系での右下方向 = +X方向（右）かつ +Y方向（下）
        # World座標系では、X+方向（右）と-Z方向（下）の組み合わせ

        # World -> Camera Base Transformation (Look Down-Right)
        # yaw=0, roll=0 の時にフロアマップ座標系で右下方向（+X, +Y）を向くように設定
        # フロアマップの右下方向は、World座標系では (X+方向, -Z方向) = 45度回転
        # ただし、実際の座標系変換を考慮して調整が必要

        # 基底変換: World座標系からCamera座標系への変換
        # さらに、yaw=0の時に右下方向を向くように45度回転を加える
        R_base_world_to_cam = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],  # Y_camera = -Z_world (下方向)
                [0, 1, 0],  # Z_camera = Y_world (前方)
            ]
        )

        # 右下方向を向くための追加回転
        # フロアマップ座標系での右下方向 = +X方向（右）かつ +Y方向（下）
        # World座標系では、X軸（右）と-Z軸（下）の組み合わせ
        #
        # R_base_world_to_cam は「前方（World Y軸）を見る」状態
        # 右下方向を見るには、Y軸周りに+120度回転を適用
        # 座標系変換を考慮して調整
        yaw_offset_rad = np.radians(110.0)  # 右下方向へのオフセット
        R_yaw_offset = np.array(
            [
                [np.cos(yaw_offset_rad), 0, np.sin(yaw_offset_rad)],
                [0, 1, 0],
                [-np.sin(yaw_offset_rad), 0, np.cos(yaw_offset_rad)],
            ]
        )

        # ユーザー指定の回転 (Pitch, Yaw, Roll)
        # Pitch: 俯角（正=下向き、負=上向き）
        # Yaw: 方位角（正=時計回り/右方向、負=反時計回り/左方向）
        #      yaw=0 で右下方向を向く
        # Roll: 回転角（正=時計回り、負=反時計回り）

        R_user = Rz @ Ry @ Rx  # 適用順序: Roll → Yaw → Pitch

        # 最終的な回転行列 R (World -> Camera)
        # yaw=0, roll=0 の時に右下方向を向くように、オフセット回転を適用
        R = R_user @ R_yaw_offset @ R_base_world_to_cam

        # 並進ベクトル t (World origin in Camera frame)
        # Camera position C in World: [0, 0, h]
        C = np.array([0, 0, h])
        t = -R @ C

        # 平面ホモグラフィ (Floor Z=0 -> Camera)
        # P = K [R|t]
        # p_c ~ P [X, Y, 0, 1]^T = K [r1 r2 t] [X, Y, 1]^T
        # H_floor2cam = K [r1 r2 t]

        r1 = R[:, 0]
        r2 = R[:, 1]

        H_floor2cam = K @ np.column_stack((r1, r2, t))

        # ホモグラフィの逆変換 (Camera -> Floor Metric)
        try:
            H_cam2floor_metric = np.linalg.inv(H_floor2cam)
        except np.linalg.LinAlgError:
            logger.error("ホモグラフィ行列の逆行列計算に失敗しました")
            return np.eye(3)

        # Floor Metric -> Floor Pixel 変換
        # Metric (m) -> Pixel
        # X_pix = X_metric * (1000 / mm_per_pixel_x) + position_x
        # Y_pix = Y_metric * (1000 / mm_per_pixel_y) + position_y

        floormap_config = floormap_config or {}
        mm_per_pixel_x = floormap_config.get("image_x_mm_per_pixel", 1.0)
        mm_per_pixel_y = floormap_config.get("image_y_mm_per_pixel", 1.0)

        # カメラ位置（ピクセル）を原点として使用
        # camera_paramsにposition_x/yが含まれていればそれを使用し、
        # なければfloormap_configのoriginを使用（後方互換性）
        pos_x = camera_params.get("position_x")
        pos_y = camera_params.get("position_y")

        if pos_x is None:
            pos_x = floormap_config.get("image_origin_x", 0)
        if pos_y is None:
            pos_y = floormap_config.get("image_origin_y", 0)

        scale_x = 1000.0 / mm_per_pixel_x
        scale_y = 1000.0 / mm_per_pixel_y

        # S = [[sx, 0, ox], [0, sy, oy], [0, 0, 1]]
        S = np.array([[scale_x, 0, pos_x], [0, scale_y, pos_y], [0, 0, 1]])

        # 最終的なホモグラフィ H = S * H_cam2floor_metric
        H_final = S @ H_cam2floor_metric

        return H_final
