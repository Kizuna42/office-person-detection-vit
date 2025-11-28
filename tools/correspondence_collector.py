#!/usr/bin/env python3
"""対応点収集・編集ツール。

カメラフレームとフロアマップを並べて表示し、
クリックで対応点を追加・編集できます。

機能:
- カメラフレームとフロアマップを並列表示
- クリックで点-点対応を追加
- 既存の対応点を表示・編集
- リアルタイムで変換誤差を表示
- 対応点をJSONで保存

使用方法:
    python tools/correspondence_collector.py --frame FRAME_IMAGE --floormap FLOORMAP_IMAGE
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.calibration.correspondence import (
    CorrespondenceData,
    PointCorrespondence,
    load_correspondence_file,
    save_correspondence_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CollectorState:
    """コレクタの状態。"""

    frame_image: np.ndarray
    floormap_image: np.ndarray
    frame_points: list[tuple[float, float]] = field(default_factory=list)
    floormap_points: list[tuple[float, float]] = field(default_factory=list)
    current_frame_point: tuple[float, float] | None = None
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    selected_index: int | None = None
    mode: str = "add"  # "add" or "select"


class CorrespondenceCollector:
    """対応点収集ツール。"""

    WINDOW_NAME = "Correspondence Collector"
    HELP_TEXT = """
対応点収集ツール - キー操作:
  左クリック (カメラ): 画像上の点を選択
  左クリック (フロアマップ): 対応するフロアマップ点を選択
  右クリック: 選択をキャンセル
  d: 選択した対応点を削除
  s: 対応点を保存
  u: 最後の対応点を削除 (Undo)
  r: 全ての対応点をクリア
  q/ESC: 終了
"""

    def __init__(
        self,
        frame_path: str | Path,
        floormap_path: str | Path,
        existing_file: str | Path | None = None,
        output_path: str | Path | None = None,
    ):
        """初期化。

        Args:
            frame_path: カメラフレーム画像パス
            floormap_path: フロアマップ画像パス
            existing_file: 既存の対応点ファイル（オプション）
            output_path: 出力ファイルパス
        """
        # 画像を読み込み
        self.frame_image = cv2.imread(str(frame_path))
        if self.frame_image is None:
            raise FileNotFoundError(f"フレーム画像が見つかりません: {frame_path}")

        self.floormap_image = cv2.imread(str(floormap_path))
        if self.floormap_image is None:
            raise FileNotFoundError(f"フロアマップ画像が見つかりません: {floormap_path}")

        self.frame_path = Path(frame_path)
        self.floormap_path = Path(floormap_path)
        self.output_path = (
            Path(output_path) if output_path else Path("output/calibration/correspondence_points_new.json")
        )

        # 状態を初期化
        self.state = CollectorState(
            frame_image=self.frame_image.copy(),
            floormap_image=self.floormap_image.copy(),
        )

        # 既存の対応点を読み込み
        if existing_file and Path(existing_file).exists():
            self._load_existing(existing_file)

        # 表示用のスケールを計算
        target_height = 700
        self.frame_scale = target_height / self.frame_image.shape[0]
        self.floormap_scale = target_height / self.floormap_image.shape[0]

        # 境界線の位置
        self.frame_width = int(self.frame_image.shape[1] * self.frame_scale)
        self.floormap_width = int(self.floormap_image.shape[1] * self.floormap_scale)

        logger.info(f"フレーム画像: {self.frame_image.shape}")
        logger.info(f"フロアマップ画像: {self.floormap_image.shape}")
        logger.info(f"既存の対応点: {len(self.state.correspondences)}")

    def _load_existing(self, file_path: str | Path) -> None:
        """既存の対応点を読み込み。"""
        try:
            data = load_correspondence_file(file_path)
            # 線分-点対応から足元点を抽出
            foot_points = data.get_foot_points()
            self.state.correspondences.extend(foot_points)
            # 点-点対応を追加
            for pc in data.point_pairs:
                self.state.correspondences.append((pc.src_point, pc.dst_point))
            logger.info(f"既存の対応点を読み込みました: {len(self.state.correspondences)} 点")
        except Exception as e:
            logger.warning(f"既存の対応点の読み込みに失敗: {e}")

    def _draw_frame(self) -> np.ndarray:
        """フレーム画像を描画。"""
        img = self.frame_image.copy()

        # 対応点を描画
        for i, (frame_pt, _) in enumerate(self.state.correspondences):
            pt = tuple(map(int, frame_pt))
            color = (0, 255, 0) if i != self.state.selected_index else (0, 0, 255)
            cv2.circle(img, pt, 8, color, 2)
            cv2.putText(
                img,
                str(i),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 選択中の点を描画
        if self.state.current_frame_point:
            pt = tuple(map(int, self.state.current_frame_point))
            cv2.circle(img, pt, 10, (0, 255, 255), 3)
            cv2.putText(
                img,
                "Selected",
                (pt[0] + 15, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        return img

    def _draw_floormap(self) -> np.ndarray:
        """フロアマップ画像を描画。"""
        img = self.floormap_image.copy()

        # 対応点を描画
        for i, (_, floormap_pt) in enumerate(self.state.correspondences):
            pt = tuple(map(int, floormap_pt))
            color = (0, 255, 0) if i != self.state.selected_index else (0, 0, 255)
            cv2.circle(img, pt, 8, color, -1)
            cv2.putText(
                img,
                str(i),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return img

    def _draw_info_panel(self) -> np.ndarray:
        """情報パネルを描画。"""
        panel_width = 250
        panel_height = 700
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        y = 30
        cv2.putText(
            panel,
            "Correspondence",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            panel,
            "Collector",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        y += 40
        cv2.putText(
            panel,
            f"Points: {len(self.state.correspondences)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        y += 30
        if self.state.current_frame_point:
            cv2.putText(
                panel,
                "Frame point set",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            y += 20
            cv2.putText(
                panel,
                "Click floormap",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        else:
            cv2.putText(
                panel,
                "Click frame first",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        y += 50
        cv2.putText(
            panel,
            "Controls:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        controls = [
            "LClick: Add point",
            "RClick: Cancel",
            "d: Delete selected",
            "u: Undo last",
            "s: Save",
            "q/ESC: Exit",
        ]
        for ctrl in controls:
            y += 20
            cv2.putText(
                panel,
                ctrl,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1,
            )

        y += 40
        if self.state.selected_index is not None:
            cv2.putText(
                panel,
                f"Selected: #{self.state.selected_index}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return panel

    def _create_display(self) -> np.ndarray:
        """表示画像を作成。"""
        # フレームとフロアマップを描画
        frame_vis = self._draw_frame()
        floormap_vis = self._draw_floormap()
        info_panel = self._draw_info_panel()

        # リサイズ
        frame_resized = cv2.resize(frame_vis, None, fx=self.frame_scale, fy=self.frame_scale)
        floormap_resized = cv2.resize(floormap_vis, None, fx=self.floormap_scale, fy=self.floormap_scale)

        # 高さを揃える
        target_height = 700
        if frame_resized.shape[0] < target_height:
            pad = target_height - frame_resized.shape[0]
            frame_resized = cv2.copyMakeBorder(frame_resized, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if floormap_resized.shape[0] < target_height:
            pad = target_height - floormap_resized.shape[0]
            floormap_resized = cv2.copyMakeBorder(floormap_resized, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 結合
        display = np.hstack([frame_resized, floormap_resized, info_panel])

        # ラベルを追加
        cv2.putText(
            display,
            "Camera Frame",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Floor Map",
            (self.frame_width + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return display

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """マウスイベントハンドラ。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.frame_width:
                # フレーム領域でクリック
                real_x = x / self.frame_scale
                real_y = y / self.frame_scale
                self.state.current_frame_point = (real_x, real_y)
                logger.info(f"フレーム点を選択: ({real_x:.1f}, {real_y:.1f})")
            elif x < self.frame_width + self.floormap_width:
                # フロアマップ領域でクリック
                if self.state.current_frame_point:
                    real_x = (x - self.frame_width) / self.floormap_scale
                    real_y = y / self.floormap_scale
                    floormap_pt = (real_x, real_y)

                    # 対応点を追加
                    self.state.correspondences.append((self.state.current_frame_point, floormap_pt))
                    logger.info(
                        f"対応点を追加: #{len(self.state.correspondences) - 1} "
                        f"({self.state.current_frame_point[0]:.1f}, {self.state.current_frame_point[1]:.1f}) -> "
                        f"({real_x:.1f}, {real_y:.1f})"
                    )
                    self.state.current_frame_point = None
                else:
                    # 既存の点を選択
                    min_dist = float("inf")
                    selected = None
                    for i, (_, fm_pt) in enumerate(self.state.correspondences):
                        real_x = (x - self.frame_width) / self.floormap_scale
                        real_y = y / self.floormap_scale
                        dist = np.linalg.norm(np.array([real_x, real_y]) - np.array(fm_pt))
                        if dist < 20 and dist < min_dist:
                            min_dist = dist
                            selected = i
                    self.state.selected_index = selected

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 選択をキャンセル
            self.state.current_frame_point = None
            self.state.selected_index = None

    def save(self) -> None:
        """対応点を保存。"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # CorrespondenceData を作成
        point_pairs = [
            PointCorrespondence(src_point=frame_pt, dst_point=fm_pt) for frame_pt, fm_pt in self.state.correspondences
        ]

        data = CorrespondenceData(
            camera_id="cam01",
            description=f"対応点データ（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}に作成）",
            image_size=(self.frame_image.shape[1], self.frame_image.shape[0]),
            floormap_size=(self.floormap_image.shape[1], self.floormap_image.shape[0]),
            line_point_pairs=[],
            point_pairs=point_pairs,
            metadata={
                "reference_image": str(self.frame_path),
                "floormap_image": str(self.floormap_path),
                "created_at": datetime.now().isoformat(),
            },
        )

        save_correspondence_file(data, self.output_path)
        logger.info(f"対応点を保存しました: {self.output_path} ({len(self.state.correspondences)} 点)")

    def run(self) -> None:
        """メインループを実行。"""
        print(self.HELP_TEXT)

        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        while True:
            display = self._create_display()
            cv2.imshow(self.WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                break
            if key == ord("s"):
                self.save()
            elif key == ord("d"):
                if self.state.selected_index is not None:
                    del self.state.correspondences[self.state.selected_index]
                    logger.info(f"対応点 #{self.state.selected_index} を削除")
                    self.state.selected_index = None
            elif key == ord("u"):
                if self.state.correspondences:
                    self.state.correspondences.pop()
                    logger.info("最後の対応点を削除")
            elif key == ord("r"):
                self.state.correspondences.clear()
                logger.info("全ての対応点をクリア")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="対応点収集ツール")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="カメラフレーム画像パス",
    )
    parser.add_argument(
        "--floormap",
        default=None,
        help="フロアマップ画像パス",
    )
    parser.add_argument(
        "--existing",
        default=None,
        help="既存の対応点ファイル",
    )
    parser.add_argument(
        "--output",
        default="output/calibration/correspondence_points_new.json",
        help="出力ファイルパス",
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)

    # フレーム画像を取得
    frame_path = args.frame
    if not frame_path:
        # output/latest/phase1_extraction/frames/ から最初のフレームを使用
        frames_dir = Path("output/latest/phase1_extraction/frames")
        if frames_dir.exists():
            frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
            if frames:
                frame_path = str(sorted(frames)[0])
                logger.info(f"フレーム画像を自動選択: {frame_path}")

    if not frame_path:
        logger.error("フレーム画像が指定されていません。--frame オプションで指定してください。")
        sys.exit(1)

    # フロアマップ画像を取得
    floormap_path = args.floormap
    if not floormap_path:
        floormap_path = config.get("floormap.image_path", "data/floormap.png")

    # 既存の対応点ファイル
    existing_file = args.existing
    if not existing_file:
        existing_file = config.get("calibration.correspondence_file", "")

    print("=" * 60)
    print("対応点収集ツール")
    print("=" * 60)
    print(f"フレーム画像: {frame_path}")
    print(f"フロアマップ: {floormap_path}")
    print(f"既存データ: {existing_file or 'なし'}")
    print(f"出力先: {args.output}")
    print("=" * 60)

    collector = CorrespondenceCollector(
        frame_path=frame_path,
        floormap_path=floormap_path,
        existing_file=existing_file,
        output_path=args.output,
    )
    collector.run()


if __name__ == "__main__":
    main()
