"""Piecewise Affine 変換モジュール。

Delaunay三角形分割を使用した高精度座標変換。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class PiecewiseAffineModel:
    """Piecewise Affine 変換モデル（Pickle可能）。

    Attributes:
        src_pts: 入力座標 (N, 2)
        dst_pts: 出力座標 (N, 2)
        simplices: Delaunay三角形のインデックス
        transforms: 各三角形のアフィン変換行列
    """

    src_pts: np.ndarray
    dst_pts: np.ndarray
    simplices: np.ndarray
    transforms: list[np.ndarray]

    @classmethod
    def from_correspondences(cls, src_pts: np.ndarray, dst_pts: np.ndarray) -> PiecewiseAffineModel:
        """対応点から学習。

        Args:
            src_pts: 入力座標 (N, 2)
            dst_pts: 出力座標 (N, 2)

        Returns:
            PiecewiseAffineModel
        """
        # Delaunay三角形分割
        tri = Delaunay(src_pts)

        # 各三角形のアフィン変換を計算
        transforms = []
        for simplex in tri.simplices:
            src_tri = src_pts[simplex]
            dst_tri = dst_pts[simplex]

            # アフィン変換行列を計算
            src_h = np.column_stack([src_tri, np.ones(3)])
            A = np.linalg.lstsq(src_h, dst_tri, rcond=None)[0]
            transforms.append(A.T)

        return cls(
            src_pts=src_pts,
            dst_pts=dst_pts,
            simplices=tri.simplices,
            transforms=transforms,
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        """座標変換。

        Args:
            points: 入力座標 (N, 2)

        Returns:
            変換後の座標 (N, 2)
        """
        # Delaunay三角形を再構築
        tri = Delaunay(self.src_pts)

        # 各点がどの三角形に属するか
        simplex_indices = tri.find_simplex(points)

        result = np.zeros_like(points)
        for i, (pt, idx) in enumerate(zip(points, simplex_indices, strict=False)):
            if idx == -1:
                # 三角形外の点は最近傍の三角形を使用
                distances = np.linalg.norm(self.src_pts - pt, axis=1)
                nearest = np.argmin(distances)
                # 最近傍点を含む三角形を見つける
                for j, simplex in enumerate(self.simplices):
                    if nearest in simplex:
                        idx = j
                        break

            if idx == -1:
                idx = 0  # フォールバック

            A = self.transforms[idx]
            pt_h = np.array([pt[0], pt[1], 1])
            result[i] = A @ pt_h

        return result


def train_piecewise_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> PiecewiseAffineModel:
    """Piecewise Affine変換を学習。

    Args:
        src_pts: 入力座標 (N, 2)
        dst_pts: 出力座標 (N, 2)

    Returns:
        PiecewiseAffineModel
    """
    return PiecewiseAffineModel.from_correspondences(src_pts, dst_pts)


def save_model(model: PiecewiseAffineModel, path: str) -> None:
    """モデルを保存。"""
    import pickle

    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> PiecewiseAffineModel:
    """モデルを読み込み。"""
    import pickle

    with open(path, "rb") as f:
        return cast("PiecewiseAffineModel", pickle.load(f))
