"""対応点データ管理モジュール。

線分-点対応データの読み込みと管理を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LinePointCorrespondence:
    """線分-点対応データ。

    画像上の垂直線分（人物の立ち位置を示す）と、
    フロアマップ上の対応する床面点を表す。

    Attributes:
        src_line: 画像上の線分 [(x1, y1), (x2, y2)]
        dst_point: フロアマップ上の点 (px, py)
    """

    src_line: tuple[tuple[float, float], tuple[float, float]]
    dst_point: tuple[float, float]

    @property
    def line_top(self) -> tuple[float, float]:
        """線分の上端点を返す。"""
        p1, p2 = self.src_line
        return p1 if p1[1] < p2[1] else p2

    @property
    def line_bottom(self) -> tuple[float, float]:
        """線分の下端点（足元）を返す。"""
        p1, p2 = self.src_line
        return p2 if p1[1] < p2[1] else p1

    @property
    def line_center(self) -> tuple[float, float]:
        """線分の中点を返す。"""
        p1, p2 = self.src_line
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    @property
    def line_length(self) -> float:
        """線分の長さを返す。"""
        p1, p2 = self.src_line
        return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    def to_dict(self) -> dict:
        """辞書形式に変換。"""
        return {
            "src_line": [list(self.src_line[0]), list(self.src_line[1])],
            "dst_point": list(self.dst_point),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LinePointCorrespondence:
        """辞書から作成。"""
        src_line = data["src_line"]
        dst_point = data["dst_point"]
        return cls(
            src_line=(tuple(src_line[0]), tuple(src_line[1])),
            dst_point=tuple(dst_point),
        )


@dataclass
class PointCorrespondence:
    """点-点対応データ。

    画像上の点とフロアマップ上の点の対応を表す。

    Attributes:
        src_point: 画像上の点 (u, v)
        dst_point: フロアマップ上の点 (px, py)
    """

    src_point: tuple[float, float]
    dst_point: tuple[float, float]

    def to_dict(self) -> dict:
        """辞書形式に変換。"""
        return {
            "src_point": list(self.src_point),
            "dst_point": list(self.dst_point),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PointCorrespondence:
        """辞書から作成。"""
        return cls(
            src_point=tuple(data["src_point"]),
            dst_point=tuple(data["dst_point"]),
        )


@dataclass
class CorrespondenceData:
    """対応点データセット。

    Attributes:
        camera_id: カメラID
        description: 説明
        image_size: 画像サイズ (width, height)
        floormap_size: フロアマップサイズ (width, height)
        line_point_pairs: 線分-点対応のリスト
        point_pairs: 点-点対応のリスト
        metadata: その他のメタデータ
    """

    camera_id: str = ""
    description: str = ""
    image_size: tuple[int, int] = (1280, 720)
    floormap_size: tuple[int, int] = (1878, 1369)
    line_point_pairs: list[LinePointCorrespondence] = field(default_factory=list)
    point_pairs: list[PointCorrespondence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_correspondences(self) -> int:
        """対応点の総数を返す。"""
        return len(self.line_point_pairs) + len(self.point_pairs)

    def get_foot_points(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """線分-点対応と点-点対応から足元点の対応リストを返す。

        Returns:
            [(image_foot_point, floormap_point), ...] のリスト
        """
        result = []
        # 線分-点対応から足元点を取得
        for lpc in self.line_point_pairs:
            result.append((lpc.line_bottom, lpc.dst_point))
        # 点-点対応も追加
        for pc in self.point_pairs:
            result.append((pc.src_point, pc.dst_point))
        return result

    def to_dict(self) -> dict:
        """辞書形式に変換。"""
        return {
            "camera_id": self.camera_id,
            "description": self.description,
            "metadata": {
                "image_size": {
                    "width": self.image_size[0],
                    "height": self.image_size[1],
                },
                "floormap_size": {
                    "width": self.floormap_size[0],
                    "height": self.floormap_size[1],
                },
                "num_line_segment_correspondences": len(self.line_point_pairs),
                "num_point_correspondences": len(self.point_pairs),
                **self.metadata,
            },
            "line_segment_correspondences": [lpc.to_dict() for lpc in self.line_point_pairs],
            "point_correspondences": [pc.to_dict() for pc in self.point_pairs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CorrespondenceData:
        """辞書から作成。"""
        metadata = data.get("metadata", {})
        image_size = metadata.get("image_size", {})
        floormap_size = metadata.get("floormap_size", {})

        line_point_pairs = [
            LinePointCorrespondence.from_dict(lpc) for lpc in data.get("line_segment_correspondences", [])
        ]
        point_pairs = [PointCorrespondence.from_dict(pc) for pc in data.get("point_correspondences", [])]

        return cls(
            camera_id=data.get("camera_id", ""),
            description=data.get("description", ""),
            image_size=(
                image_size.get("width", 1280),
                image_size.get("height", 720),
            ),
            floormap_size=(
                floormap_size.get("width", 1878),
                floormap_size.get("height", 1369),
            ),
            line_point_pairs=line_point_pairs,
            point_pairs=point_pairs,
            metadata={k: v for k, v in metadata.items() if k not in ("image_size", "floormap_size")},
        )


def load_correspondence_file(file_path: str | Path) -> CorrespondenceData:
    """対応点ファイルを読み込む。

    Args:
        file_path: JSONファイルのパス

    Returns:
        CorrespondenceData インスタンス

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        json.JSONDecodeError: JSONの解析に失敗した場合
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Correspondence file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = CorrespondenceData.from_dict(data)
    logger.info(
        f"Loaded correspondence data from {path}: "
        f"{len(result.line_point_pairs)} line-point pairs, "
        f"{len(result.point_pairs)} point pairs"
    )
    return result


def save_correspondence_file(
    data: CorrespondenceData,
    file_path: str | Path,
) -> None:
    """対応点データをファイルに保存。

    Args:
        data: CorrespondenceData インスタンス
        file_path: 出力JSONファイルのパス
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved correspondence data to {path}")
