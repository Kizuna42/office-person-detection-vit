#!/usr/bin/env python3
"""ゾーン可視化ツール。

フロアマップ上にゾーン定義と変換された検出結果をオーバーレイして表示します。

機能:
- ゾーン境界を描画
- 検出結果の変換位置を表示
- ゾーン内/外の分類結果を色分け
- 統計情報を表示

使用方法:
    python tools/zone_visualizer.py [--input DETECTIONS_JSON] [--output OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import ClassVar

import cv2
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ZoneVisualizer:
    """ゾーン可視化クラス。"""

    # ゾーンごとの色（BGR）
    ZONE_COLORS: ClassVar[list[tuple[int, int, int]]] = [
        (255, 100, 100),  # 青
        (100, 255, 100),  # 緑
        (100, 100, 255),  # 赤
        (255, 255, 100),  # シアン
        (255, 100, 255),  # マゼンタ
        (100, 255, 255),  # 黄
    ]

    def __init__(self, config: ConfigManager):
        """初期化。

        Args:
            config: ConfigManager インスタンス
        """
        self.config = config

        # フロアマップ画像を読み込み
        floormap_path = config.get("floormap.image_path", "data/floormap.png")
        if Path(floormap_path).exists():
            self.floormap_image = cv2.imread(str(floormap_path))
            logger.info(f"フロアマップ画像を読み込みました: {floormap_path}")
        else:
            raise FileNotFoundError(f"フロアマップ画像が見つかりません: {floormap_path}")

        # ゾーン定義を読み込み
        self.zones = config.get("zones", [])
        logger.info(f"ゾーン数: {len(self.zones)}")

        # カメラ位置を読み込み
        camera_params = config.get("camera_params", {})
        self.camera_position_px = (
            float(camera_params.get("position_x_px", 859.0)),
            float(camera_params.get("position_y_px", 1040.0)),
        )

    def load_detections(self, json_path: str | Path) -> list[dict]:
        """検出結果を読み込み。

        Args:
            json_path: coordinate_transformations.json のパス

        Returns:
            検出結果のリスト
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        detections = []
        for frame in data:
            for det in frame.get("detections", []):
                floor_coords = det.get("floor_coords_px")
                if floor_coords:
                    detections.append(
                        {
                            "timestamp": frame.get("timestamp", ""),
                            "floor_coords": (floor_coords["x"], floor_coords["y"]),
                            "zone_ids": det.get("zone_ids", []),
                            "track_id": det.get("track_id"),
                            "confidence": det.get("confidence", 0),
                        }
                    )

        logger.info(f"検出結果を読み込みました: {len(detections)} 件")
        return detections

    def _point_in_polygon(self, point: tuple[float, float], polygon: list[list[float]]) -> bool:
        """点がポリゴン内にあるか判定（Ray Casting法）。"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def visualize(
        self,
        detections: list[dict],
        show_all_points: bool = True,
        show_statistics: bool = True,
    ) -> np.ndarray:
        """可視化画像を生成。

        Args:
            detections: 検出結果のリスト
            show_all_points: 全ての検出点を表示
            show_statistics: 統計情報を表示

        Returns:
            可視化画像
        """
        img = self.floormap_image.copy()

        # 統計情報を集計
        zone_counts = {zone["id"]: 0 for zone in self.zones}
        zone_counts["unclassified"] = 0
        total_detections = len(detections)

        # ゾーンを描画
        for i, zone in enumerate(self.zones):
            polygon = np.array(zone.get("polygon", []), dtype=np.int32)
            if len(polygon) > 2:
                color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]

                # ゾーン領域を半透明で塗りつぶし
                overlay = img.copy()
                cv2.fillPoly(overlay, [polygon], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

                # ゾーン境界を描画
                cv2.polylines(img, [polygon], True, color, 2)

                # ゾーン名を表示
                center = polygon.mean(axis=0).astype(int)
                cv2.putText(
                    img,
                    zone.get("name", zone.get("id", "")),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    zone.get("name", zone.get("id", "")),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                )

        # 検出結果を描画
        for det in detections:
            pt = tuple(map(int, det["floor_coords"]))
            zone_ids = det["zone_ids"]

            # 範囲外チェック
            if pt[0] < 0 or pt[0] >= img.shape[1] or pt[1] < 0 or pt[1] >= img.shape[0]:
                zone_counts["unclassified"] += 1
                continue

            if zone_ids:
                # ゾーン内
                zone_id = zone_ids[0]
                zone_idx = next((i for i, z in enumerate(self.zones) if z["id"] == zone_id), 0)
                color = self.ZONE_COLORS[zone_idx % len(self.ZONE_COLORS)]
                zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
            else:
                # ゾーン外（未分類）
                color = (128, 128, 128)  # グレー
                zone_counts["unclassified"] += 1

            if show_all_points:
                cv2.circle(img, pt, 5, color, -1)
                cv2.circle(img, pt, 6, (255, 255, 255), 1)

        # カメラ位置を描画
        cam_pos = tuple(map(int, self.camera_position_px))
        cv2.drawMarker(
            img,
            cam_pos,
            (0, 0, 255),
            cv2.MARKER_TRIANGLE_UP,
            20,
            3,
        )
        cv2.putText(
            img,
            "Camera",
            (cam_pos[0] + 15, cam_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        # 統計情報パネルを追加
        if show_statistics:
            panel_width = 250
            panel = np.zeros((img.shape[0], panel_width, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)

            y = 40
            cv2.putText(
                panel,
                "Zone Statistics",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            y += 40
            cv2.putText(
                panel,
                f"Total: {total_detections}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            y += 30
            for i, zone in enumerate(self.zones):
                zone_id = zone["id"]
                count = zone_counts.get(zone_id, 0)
                pct = count / total_detections * 100 if total_detections > 0 else 0
                color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]

                cv2.rectangle(panel, (10, y - 12), (25, y + 3), color, -1)
                cv2.putText(
                    panel,
                    f"{zone.get('name', zone_id)}: {count} ({pct:.1f}%)",
                    (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )
                y += 25

            # 未分類
            unclassified = zone_counts["unclassified"]
            pct = unclassified / total_detections * 100 if total_detections > 0 else 0
            y += 10
            cv2.rectangle(panel, (10, y - 12), (25, y + 3), (128, 128, 128), -1)
            cv2.putText(
                panel,
                f"Unclassified: {unclassified} ({pct:.1f}%)",
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            # 評価
            y += 50
            classified_pct = (1 - unclassified / total_detections) * 100 if total_detections > 0 else 0
            if classified_pct >= 80:
                status_color = (0, 255, 0)
                status = "GOOD"
            elif classified_pct >= 50:
                status_color = (0, 165, 255)
                status = "NEEDS WORK"
            else:
                status_color = (0, 0, 255)
                status = "POOR"

            cv2.putText(
                panel,
                f"Classification: {classified_pct:.1f}%",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
            )

            y += 25
            cv2.putText(
                panel,
                f"Status: {status}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            y += 50
            cv2.putText(
                panel,
                "Legend:",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            y += 25
            cv2.circle(panel, (20, y), 5, (128, 128, 128), -1)
            cv2.putText(
                panel,
                "Unclassified",
                (35, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )

            y += 20
            cv2.circle(panel, (20, y), 5, (100, 255, 100), -1)
            cv2.putText(
                panel,
                "In Zone",
                (35, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 255, 100),
                1,
            )

            img = np.hstack([img, panel])

        return img

    def save_visualization(
        self,
        detections: list[dict],
        output_path: str | Path,
    ) -> None:
        """可視化画像を保存。

        Args:
            detections: 検出結果のリスト
            output_path: 出力パス
        """
        vis_image = self.visualize(detections)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"可視化画像を保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ゾーン可視化ツール")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="検出結果JSONファイル (coordinate_transformations.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="出力ディレクトリ",
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)

    # 入力ファイルを取得
    input_path = args.input
    if not input_path:
        # output/latest から自動検出
        latest_path = Path("output/latest/phase3_transform/coordinate_transformations.json")
        if latest_path.exists():
            input_path = str(latest_path)
            logger.info(f"検出結果を自動検出: {input_path}")

    if not input_path or not Path(input_path).exists():
        logger.error("検出結果ファイルが見つかりません。--input オプションで指定してください。")
        sys.exit(1)

    print("=" * 60)
    print("ゾーン可視化ツール")
    print("=" * 60)
    print(f"入力ファイル: {input_path}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print("=" * 60)

    visualizer = ZoneVisualizer(config)
    detections = visualizer.load_detections(input_path)

    # 可視化
    output_path = Path(args.output_dir) / "zone_visualization.png"
    visualizer.save_visualization(detections, output_path)

    # 統計を表示
    zone_counts = {zone["id"]: 0 for zone in visualizer.zones}
    zone_counts["unclassified"] = 0

    for det in detections:
        zone_ids = det["zone_ids"]
        if zone_ids:
            for zid in zone_ids:
                zone_counts[zid] = zone_counts.get(zid, 0) + 1
        else:
            zone_counts["unclassified"] += 1

    print("\n" + "=" * 60)
    print("ゾーン別集計")
    print("=" * 60)
    total = len(detections)
    for zone in visualizer.zones:
        zid = zone["id"]
        count = zone_counts.get(zid, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {zone.get('name', zid)}: {count} ({pct:.1f}%)")

    unclassified = zone_counts["unclassified"]
    pct = unclassified / total * 100 if total > 0 else 0
    print(f"  未分類: {unclassified} ({pct:.1f}%)")
    print("=" * 60)

    classified_pct = (1 - unclassified / total) * 100 if total > 0 else 0
    if classified_pct >= 80:
        print(f"\n✓ 分類率 {classified_pct:.1f}% - 目標達成")
    else:
        print(f"\n× 分類率 {classified_pct:.1f}% - 目標80%に未達")

    print("\n出力ファイル:")
    print(f"  - {output_path}")


if __name__ == "__main__":
    main()
