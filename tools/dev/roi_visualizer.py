#!/usr/bin/env python
"""ROI座標の可視化ツール

複数フレームでROI位置を可視化し、タイムスタンプがすべて含まれているか確認します。
ユーザーが目視で確認して、config.yamlのROI設定を調整できるようにします。
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


def visualize_roi_on_frame(
    frame: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    frame_idx: int,
    timestamp_text: str = None,
) -> np.ndarray:
    """フレームにROI矩形を描画

    Args:
        frame: フレーム画像
        roi_coords: ROI座標 (x, y, width, height)
        frame_idx: フレーム番号
        timestamp_text: タイムスタンプテキスト（オプション）

    Returns:
        描画済みフレーム画像
    """
    overlay = frame.copy()
    x, y, w, h = roi_coords

    # ROI矩形を描画（緑色、太めの線）
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 座標情報を表示
    info_text = f"Frame: {frame_idx}"
    cv2.putText(
        overlay,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
    )

    # ROI座標を表示
    coord_text = f"ROI: x={x}, y={y}, w={w}, h={h}"
    cv2.putText(
        overlay,
        coord_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        coord_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    # タイムスタンプテキストを表示（あれば）
    if timestamp_text:
        ts_text = f"Timestamp: {timestamp_text}"
        cv2.putText(
            overlay,
            ts_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ROI領域を拡大表示（右上に配置）
    roi_region = frame[y : y + h, x : x + w]
    if roi_region.size > 0:
        # ROIを拡大（3倍、最大サイズ制限あり）
        max_zoom_w = min(w * 3, overlay.shape[1] - 40)
        max_zoom_h = min(h * 3, overlay.shape[0] - 40)
        zoom_scale = min(max_zoom_w / w, max_zoom_h / h, 3.0)
        zoom_w = int(w * zoom_scale)
        zoom_h = int(h * zoom_scale)

        roi_zoomed = cv2.resize(roi_region, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

        # 右上に配置
        zoom_x = overlay.shape[1] - zoom_w - 20
        zoom_y = 20

        # 配置範囲のチェック
        if zoom_x >= 0 and zoom_y >= 0 and zoom_x + zoom_w <= overlay.shape[1] and zoom_y + zoom_h <= overlay.shape[0]:
            # 背景を白で塗りつぶし
            overlay[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w] = (
                255,
                255,
                255,
            )

            # ROI拡大画像を配置
            if len(roi_zoomed.shape) == 2:
                roi_zoomed_bgr = cv2.cvtColor(roi_zoomed, cv2.COLOR_GRAY2BGR)
            else:
                roi_zoomed_bgr = roi_zoomed

            overlay[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w] = roi_zoomed_bgr

            # 拡大ROIの枠を描画
            cv2.rectangle(
                overlay,
                (zoom_x, zoom_y),
                (zoom_x + zoom_w, zoom_y + zoom_h),
                (255, 0, 0),
                2,
            )

    return overlay


def extract_sample_frames(
    video_path: str,
    num_frames: int = 10,
    frame_indices: list[int] = None,
) -> list[tuple[int, np.ndarray]]:
    """動画からサンプルフレームを抽出

    Args:
        video_path: 動画ファイルのパス
        num_frames: 抽出するフレーム数（frame_indicesが指定されていない場合）
        frame_indices: 抽出するフレーム番号のリスト（指定されている場合）

    Returns:
        (フレーム番号, フレーム画像) のタプルのリスト
    """
    video_processor = VideoProcessor(video_path)
    video_processor.open()

    try:
        total_frames = video_processor.total_frames

        if frame_indices:
            indices = frame_indices
        else:
            # 均等にサンプリング
            step = max(1, total_frames // num_frames)
            indices = [i * step for i in range(num_frames)]

        frames = []
        for idx in tqdm(indices, desc="フレーム抽出中"):
            if idx >= total_frames:
                continue
            frame = video_processor.get_frame(idx)
            if frame is not None:
                frames.append((idx, frame))

        return frames
    finally:
        video_processor.release()


def visualize_roi_on_multiple_frames(
    video_path: str,
    roi_config: dict,
    output_dir: Path,
    num_frames: int = 10,
    frame_indices: list[int] = None,
):
    """複数フレームでROI位置を可視化

    Args:
        video_path: 動画ファイルのパス
        roi_config: ROI設定
        output_dir: 出力ディレクトリ
        num_frames: 抽出するフレーム数
        frame_indices: 抽出するフレーム番号のリスト
    """
    # サンプルフレームを抽出
    logger.info(f"動画からサンプルフレームを抽出中: {video_path}")
    sample_frames = extract_sample_frames(video_path, num_frames, frame_indices)

    if not sample_frames:
        logger.error("サンプルフレームを抽出できませんでした")
        return

    logger.info(f"抽出したフレーム数: {len(sample_frames)}")

    # ROI抽出器を初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各フレームでROIを可視化
    for frame_idx, frame in tqdm(sample_frames, desc="ROI可視化中"):
        # ROI抽出
        roi, roi_coords = roi_extractor.extract_roi(frame)

        # 前処理済みROIも取得（比較用）
        preprocessed_roi = roi_extractor.preprocess_roi(roi)

        # 元のフレームにROI矩形を描画
        overlay = visualize_roi_on_frame(frame, roi_coords, frame_idx)

        # 保存
        overlay_path = output_dir / f"roi_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        # ROI画像を保存
        roi_path = output_dir / f"roi_extracted_{frame_idx:06d}.jpg"
        cv2.imwrite(str(roi_path), roi)

        # 前処理済みROIを保存
        preprocessed_path = output_dir / f"roi_preprocessed_{frame_idx:06d}.jpg"
        cv2.imwrite(str(preprocessed_path), preprocessed_roi)

        logger.info(f"Frame {frame_idx}: ROI座標={roi_coords}, サイズ={roi.shape}")

    # サマリー画像を作成（複数フレームを並べて表示）
    create_summary_image(sample_frames, roi_extractor, output_dir)

    logger.info(f"ROI可視化結果を保存しました: {output_dir}")
    logger.info("=" * 80)
    logger.info("次のステップ:")
    logger.info("1. output/roi_visualization/ ディレクトリ内の画像を確認")
    logger.info("2. タイムスタンプがすべてROI矩形内に含まれているか確認")
    logger.info("3. 必要に応じて config.yaml の ROI 設定を調整")
    logger.info("   - timestamp.extraction.roi.x_ratio")
    logger.info("   - timestamp.extraction.roi.y_ratio")
    logger.info("   - timestamp.extraction.roi.width_ratio")
    logger.info("   - timestamp.extraction.roi.height_ratio")
    logger.info("=" * 80)


def create_summary_image(
    sample_frames: list[tuple[int, np.ndarray]],
    roi_extractor: TimestampROIExtractor,
    output_dir: Path,
):
    """複数フレームのROIを1つの画像にまとめて表示

    Args:
        sample_frames: サンプルフレームのリスト
        roi_extractor: ROI抽出器
        output_dir: 出力ディレクトリ
    """
    if len(sample_frames) == 0:
        return

    # グリッドレイアウト（最大5列）
    cols = min(5, len(sample_frames))
    rows = (len(sample_frames) + cols - 1) // cols

    # 各フレームのサイズ
    frame_h, frame_w = sample_frames[0][1].shape[:2]
    cell_w = frame_w // 2  # 縮小表示
    cell_h = frame_h // 2

    # サマリー画像のサイズ
    summary_w = cell_w * cols + 20 * (cols + 1)
    summary_h = cell_h * rows + 20 * (rows + 1) + 50  # タイトル用の余白

    summary = np.ones((summary_h, summary_w, 3), dtype=np.uint8) * 255

    # タイトル
    title = "ROI Visualization Summary"
    cv2.putText(
        summary,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )

    # 各フレームを配置
    for idx, (frame_idx, frame) in enumerate(sample_frames):
        row = idx // cols
        col = idx % cols

        x = 20 + col * (cell_w + 20)
        y = 60 + row * (cell_h + 20)

        # フレームを縮小
        frame_small = cv2.resize(frame, (cell_w, cell_h))

        # ROI矩形を描画
        roi, roi_coords = roi_extractor.extract_roi(frame)
        rx, ry, rw, rh = roi_coords
        # 縮小率に合わせて座標を調整
        scale_x = cell_w / frame_w
        scale_y = cell_h / frame_h
        rx_small = int(rx * scale_x)
        ry_small = int(ry * scale_y)
        rw_small = int(rw * scale_x)
        rh_small = int(rh * scale_y)

        cv2.rectangle(
            frame_small,
            (rx_small, ry_small),
            (rx_small + rw_small, ry_small + rh_small),
            (0, 255, 0),
            2,
        )

        # フレーム番号を表示
        frame_text = f"F{frame_idx}"
        cv2.putText(
            frame_small,
            frame_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_small,
            frame_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # サマリー画像に配置
        summary[y : y + cell_h, x : x + cell_w] = frame_small

    # 保存
    summary_path = output_dir / "roi_summary.jpg"
    cv2.imwrite(str(summary_path), summary)
    logger.info(f"サマリー画像を保存しました: {summary_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="ROI座標の可視化ツール")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--output-dir", type=str, help="出力ディレクトリ（デフォルト: output/roi_visualization）")
    parser.add_argument("--num-frames", type=int, default=10, help="抽出するフレーム数（デフォルト: 10）")
    parser.add_argument("--frame-indices", type=str, help="抽出するフレーム番号（カンマ区切り、例: 0,100,200）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 設定読み込み
    config = ConfigManager(args.config)

    # 動画パスの取得
    if args.video:
        video_path = args.video
    else:
        video_path = config.get("video.input_path")

    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1

    # ROI設定の取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    roi_config = extraction_config.get("roi", {})

    if not roi_config:
        logger.warning("ROI設定が見つかりません。デフォルト設定を使用します。")
        roi_config = {
            "x_ratio": 0.65,
            "y_ratio": 0.0,
            "width_ratio": 0.35,
            "height_ratio": 0.08,
        }

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("output.directory", "output")) / "roi_visualization"

    # フレーム番号のパース
    frame_indices = None
    if args.frame_indices:
        frame_indices = [int(x.strip()) for x in args.frame_indices.split(",")]

    # ROI可視化実行
    logger.info("=" * 80)
    logger.info("ROI座標の可視化")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"ROI設定: {roi_config}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info("=" * 80)

    visualize_roi_on_multiple_frames(
        video_path=video_path,
        roi_config=roi_config,
        output_dir=output_dir,
        num_frames=args.num_frames,
        frame_indices=frame_indices,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
