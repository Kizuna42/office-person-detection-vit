"""レンズ歪み補正モジュール

カメラレンズによる歪みを補正するための機能を提供します。
チェスボードキャリブレーションまたは手動設定された歪み係数を使用できます。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DistortionParams:
    """歪み係数パラメータ

    OpenCVの歪みモデルに対応:
    - k1, k2, k3: 放射歪み係数
    - p1, p2: 接線歪み係数

    歪み式:
    x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
    y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def to_array(self) -> np.ndarray:
        """OpenCV形式の配列に変換 [k1, k2, p1, p2, k3]"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray | list) -> DistortionParams:
        """配列から作成"""
        arr = np.array(arr).flatten()
        if len(arr) >= 5:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3], k3=arr[4])
        if len(arr) >= 4:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3])
        if len(arr) >= 2:
            return cls(k1=arr[0], k2=arr[1])
        return cls()

    def is_zero(self) -> bool:
        """すべての係数がゼロか確認"""
        return (
            abs(self.k1) < 1e-10
            and abs(self.k2) < 1e-10
            and abs(self.k3) < 1e-10
            and abs(self.p1) < 1e-10
            and abs(self.p2) < 1e-10
        )

    def to_dict(self) -> dict[str, float]:
        """辞書に変換"""
        return {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "p1": self.p1,
            "p2": self.p2,
        }


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ

    Attributes:
        fx, fy: 焦点距離 [pixels]
        cx, cy: 主点 [pixels]
        width, height: 画像サイズ [pixels]
        distortion: 歪み係数
    """

    fx: float = 1250.0
    fy: float = 1250.0
    cx: float = 640.0
    cy: float = 360.0
    width: int = 1280
    height: int = 720
    distortion: DistortionParams = field(default_factory=DistortionParams)

    def get_camera_matrix(self) -> np.ndarray:
        """カメラ行列を取得"""
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_config(cls, config: dict) -> CameraIntrinsics:
        """設定辞書から作成"""
        dist_config = config.get("distortion", {})
        if isinstance(dist_config, list):
            distortion = DistortionParams.from_array(dist_config)
        elif isinstance(dist_config, dict):
            distortion = DistortionParams(
                k1=dist_config.get("k1", 0.0),
                k2=dist_config.get("k2", 0.0),
                k3=dist_config.get("k3", 0.0),
                p1=dist_config.get("p1", 0.0),
                p2=dist_config.get("p2", 0.0),
            )
        else:
            distortion = DistortionParams()

        return cls(
            fx=config.get("focal_length_x", 1250.0),
            fy=config.get("focal_length_y", 1250.0),
            cx=config.get("center_x", 640.0),
            cy=config.get("center_y", 360.0),
            width=config.get("image_width", 1280),
            height=config.get("image_height", 720),
            distortion=distortion,
        )


class LensDistortionCorrector:
    """レンズ歪み補正器

    画像座標の歪み補正を行います。
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """初期化

        Args:
            intrinsics: カメラ内部パラメータ
        """
        self.intrinsics = intrinsics
        self.camera_matrix = intrinsics.get_camera_matrix()
        self.dist_coeffs = intrinsics.distortion.to_array()

        # 補正が必要かどうか
        self.enabled = not intrinsics.distortion.is_zero()

        if self.enabled:
            logger.info(f"LensDistortionCorrector enabled: {intrinsics.distortion.to_dict()}")
        else:
            logger.info("LensDistortionCorrector disabled (zero distortion)")

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """点群の歪みを補正

        Args:
            points: 入力点 (N, 2) または (N, 1, 2)

        Returns:
            補正された点 (N, 2)
        """
        if not self.enabled:
            if points.ndim == 3:
                return points.reshape(-1, 2)
            return points

        # OpenCVの形式に変換
        pts = points.reshape(-1, 1, 2).astype(np.float64) if points.ndim == 2 else points.astype(np.float64)

        # 歪み補正
        undistorted = cv2.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix,  # 正規化座標に戻さず、ピクセル座標を出力
        )

        return undistorted.reshape(-1, 2)

    def undistort_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """1点の歪みを補正

        Args:
            point: (x, y)

        Returns:
            補正された点 (x, y)
        """
        if not self.enabled:
            return point

        pts = np.array([[point]], dtype=np.float64)
        undistorted = self.undistort_points(pts)
        return (float(undistorted[0, 0]), float(undistorted[0, 1]))

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """画像全体の歪みを補正

        Args:
            image: 入力画像

        Returns:
            補正された画像
        """
        if not self.enabled:
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            alpha=1.0,  # すべてのピクセルを保持
        )

        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix,
        )

        return undistorted

    def visualize_distortion_grid(
        self,
        grid_size: int = 20,
        output_path: Path | str | None = None,
    ) -> np.ndarray:
        """歪みをグリッドで可視化

        Args:
            grid_size: グリッドの間隔 [pixels]
            output_path: 出力パス

        Returns:
            可視化画像
        """
        w, h = self.intrinsics.width, self.intrinsics.height
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # グリッド点を生成
        xs = np.arange(0, w, grid_size)
        ys = np.arange(0, h, grid_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

        # 歪み補正後の点
        undistorted = self.undistort_points(points) if self.enabled else points

        # 元のグリッド（青）
        for i in range(len(xs)):
            for j in range(len(ys) - 1):
                pt1 = (int(xs[i]), int(ys[j]))
                pt2 = (int(xs[i]), int(ys[j + 1]))
                cv2.line(img, pt1, pt2, (255, 200, 200), 1)
        for j in range(len(ys)):
            for i in range(len(xs) - 1):
                pt1 = (int(xs[i]), int(ys[j]))
                pt2 = (int(xs[i + 1]), int(ys[j]))
                cv2.line(img, pt1, pt2, (255, 200, 200), 1)

        # 補正後のグリッド（赤）
        undist_grid = undistorted.reshape(len(ys), len(xs), 2)
        for i in range(len(xs)):
            for j in range(len(ys) - 1):
                pt1 = (int(undist_grid[j, i, 0]), int(undist_grid[j, i, 1]))
                pt2 = (int(undist_grid[j + 1, i, 0]), int(undist_grid[j + 1, i, 1]))
                cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        for j in range(len(ys)):
            for i in range(len(xs) - 1):
                pt1 = (int(undist_grid[j, i, 0]), int(undist_grid[j, i, 1]))
                pt2 = (int(undist_grid[j, i + 1, 0]), int(undist_grid[j, i + 1, 1]))
                cv2.line(img, pt1, pt2, (0, 0, 255), 1)

        # ラベル
        cv2.putText(img, "Blue: Original | Red: Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(
            img,
            f"k1={self.intrinsics.distortion.k1:.4f}, k2={self.intrinsics.distortion.k2:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        if output_path:
            cv2.imwrite(str(output_path), img)

        return img

    @classmethod
    def from_config(cls, config: dict) -> LensDistortionCorrector:
        """設定から作成

        Args:
            config: camera_params または calibration セクション

        Returns:
            LensDistortionCorrector
        """
        intrinsics = CameraIntrinsics.from_config(config)
        return cls(intrinsics)

    @classmethod
    def from_calibration_file(cls, file_path: Path | str) -> LensDistortionCorrector:
        """キャリブレーション結果ファイルから作成

        Args:
            file_path: キャリブレーション結果JSONファイル

        Returns:
            LensDistortionCorrector
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # カメラ行列と歪み係数を抽出
        camera_matrix = np.array(data.get("camera_matrix", [[1250, 0, 640], [0, 1250, 360], [0, 0, 1]]))
        dist_coeffs = np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]))

        intrinsics = CameraIntrinsics(
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2],
            distortion=DistortionParams.from_array(dist_coeffs),
        )

        return cls(intrinsics)


def estimate_distortion_from_lines(
    image: np.ndarray,
    min_line_length: int = 100,
) -> DistortionParams:
    """直線検出から歪みを推定（実験的）

    画像中の直線が曲がっていることから歪みを推定します。
    チェスボードがない場合の代替手法です。

    Args:
        image: 入力画像
        min_line_length: 最小直線長さ

    Returns:
        推定された歪み係数
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # 直線検出（Hough変換）
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10,
    )

    if lines is None or len(lines) < 10:
        logger.warning("Not enough lines detected for distortion estimation")
        return DistortionParams()

    # 画像中心からの距離と曲率の関係から k1 を推定
    # 注: h, w, cx, cy は将来の完全な実装で使用予定

    # 簡易的な推定（実際にはより複雑な最適化が必要）
    logger.info(f"Detected {len(lines)} lines for distortion estimation")

    # ここでは仮の値を返す（本格的な実装は別途必要）
    return DistortionParams(k1=-0.1, k2=0.05)


def calibrate_from_chessboard_images(
    image_paths: list[Path | str],
    chessboard_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> tuple[CameraIntrinsics, dict]:
    """チェスボード画像からキャリブレーション

    Args:
        image_paths: チェスボード画像のパスリスト
        chessboard_size: チェスボードの内部コーナー数 (width, height)
        square_size: マス目のサイズ [任意単位]

    Returns:
        (CameraIntrinsics, 詳細情報の辞書)
    """
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    img_size = None

    # チェスボードの3D座標
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2) * square_size

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # サブピクセル精度に改善
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            logger.info(f"Found corners in: {img_path}")
        else:
            logger.warning(f"No corners found in: {img_path}")

    if len(objpoints) < 3:
        raise ValueError(f"Insufficient images with detected corners: {len(objpoints)} < 3")

    if img_size is None:
        raise ValueError("No valid images found for calibration")

    # キャリブレーション
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None,
    )

    # 再投影誤差を計算
    total_error = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(objpoints)

    # 結果を作成
    intrinsics = CameraIntrinsics(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        width=img_size[0],
        height=img_size[1],
        distortion=DistortionParams.from_array(dist_coeffs),
    )

    info = {
        "num_images": len(objpoints),
        "mean_reprojection_error": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }

    logger.info(f"Calibration completed: mean error = {mean_error:.4f}px")
    logger.info(f"Distortion: {intrinsics.distortion.to_dict()}")

    return intrinsics, info
