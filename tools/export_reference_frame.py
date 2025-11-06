"""Reference frame exporter for camera calibration.

This utility extracts a still frame from the configured input video at a
target timestamp (or specific frame number) and stores it under
`output/calibration/`. Optionally, it creates a floormap guide image with a
grid overlay to assist when selecting correspondence points.

Usage examples:

```bash
python tools/export_reference_frame.py --timestamp 12:20
python tools/export_reference_frame.py --frame-number 12345 --no-grid
python tools/export_reference_frame.py --timestamp 12:20 --grid-interval 80
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.utils import setup_logging
from src.video import FrameSampler, VideoProcessor
from src.timestamp import TimestampExtractor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="カメラキャリブレーション用の参照フレームを出力します。"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルのパス (default: config.yaml)",
    )

    parser.add_argument(
        "--timestamp",
        help="抽出したいタイムスタンプ (HH:MM)。" " --frame-number より優先されます。",
    )

    parser.add_argument(
        "--frame-number",
        type=int,
        help="直接フレーム番号を指定して抽出します。",
    )

    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="出力ディレクトリ (default: output/calibration)",
    )

    parser.add_argument(
        "--grid-interval",
        type=int,
        default=100,
        help="フロアマップガイドのグリッド間隔（ピクセル）。(default: 100)",
    )

    parser.add_argument(
        "--grid-color",
        default="0,255,0",
        help="グリッド線のBGRカラー。例: 0,255,0 (default)",
    )

    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="フロアマップガイドの生成をスキップします。",
    )

    parser.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        help="タイムスタンプ探索時のフレーム間隔。(default: 60)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを有効にします。",
    )

    return parser.parse_args()


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("出力ディレクトリを確認しました: %s", path)


def parse_color(color_str: str) -> Tuple[int, int, int]:
    try:
        b_str, g_str, r_str = color_str.split(",")
        return int(b_str), int(g_str), int(r_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--grid-color は B,G,R 形式で指定してください") from exc


def export_floormap_grid(
    floormap_path: Path,
    output_path: Path,
    interval: int,
    color: Tuple[int, int, int],
) -> None:
    logger.info("フロアマップガイドを生成しています: %s", output_path)

    image = cv2.imread(str(floormap_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"フロアマップ画像を読み込めません: {floormap_path}")

    height, width = image.shape[:2]
    overlay = image.copy()

    for x in range(0, width, interval):
        cv2.line(overlay, (x, 0), (x, height), color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(x),
            (x + 2, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    for y in range(0, height, interval):
        cv2.line(overlay, (0, y), (width, y), color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(y),
            (5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    alpha = 0.35
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    if not cv2.imwrite(str(output_path), blended):
        raise RuntimeError(f"フロアマップガイドの保存に失敗しました: {output_path}")

    logger.info("フロアマップガイドを出力しました: %s", output_path)


def export_reference_frame(
    frame: np.ndarray,
    output_path: Path,
    timestamp: Optional[str],
    frame_number: Optional[int],
) -> None:
    meta = []
    if timestamp:
        meta.append(f"ts={timestamp}")
    if frame_number is not None:
        meta.append(f"frame={frame_number}")
    logger.info("参照フレームを保存します: %s (%s)", output_path, ", ".join(meta))

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"参照フレームの保存に失敗しました: {output_path}")


def find_frame_by_timestamp(
    target_timestamp: str,
    video_processor: VideoProcessor,
    sampler: FrameSampler,
    extractor: TimestampExtractor,
    scan_interval: int,
) -> Tuple[int, np.ndarray, str]:
    logger.info("タイムスタンプ %s に最も近いフレームを探索します", target_timestamp)

    frame_timestamps = sampler._scan_all_timestamps(  # type: ignore[attr-defined]  # noqa: SLF001
        video_processor,
        extractor,
        scan_interval=scan_interval,
    )

    if not frame_timestamps:
        raise RuntimeError("タイムスタンプのスキャンに失敗しました")

    closest = sampler.find_closest_frame(target_timestamp, frame_timestamps)
    if closest is None:
        raise RuntimeError(f"指定タイムスタンプ {target_timestamp} 周辺のフレームが見つかりません")

    frame = video_processor.get_frame(closest)
    if frame is None:
        raise RuntimeError(f"フレーム#{closest} の読み込みに失敗しました")

    actual_timestamp = frame_timestamps.get(closest, target_timestamp)
    logger.info("フレーム #%d (実タイムスタンプ %s) を取得しました", closest, actual_timestamp)
    return closest, frame, actual_timestamp


def extract_reference_frame(
    args: argparse.Namespace,
    config: ConfigManager,
    output_dir: Path,
) -> Path:
    video_path = config.get("video.input_path")
    if not video_path:
        raise ValueError("config.video.input_path が設定されていません")

    video_processor = VideoProcessor(video_path)
    video_processor.open()

    try:
        if args.frame_number is not None:
            frame = video_processor.get_frame(args.frame_number)
            if frame is None:
                raise RuntimeError(f"フレーム#{args.frame_number} の取得に失敗しました")

            timestamp = None
            if args.timestamp:
                timestamp = args.timestamp
            else:
                extractor = TimestampExtractor()
                timestamp = extractor.extract(frame, frame_index=args.frame_number)

            sanitized_ts = (timestamp or "unknown").replace(":", "")
            filename = f"reference_{sanitized_ts}_f{args.frame_number:06d}.png"
            output_path = output_dir / filename
            export_reference_frame(frame, output_path, timestamp, args.frame_number)
            return output_path

        if not args.timestamp:
            raise ValueError("--timestamp または --frame-number のいずれかを指定してください")

        extractor = TimestampExtractor()
        if args.verbose:
            debug_dir = output_dir / "timestamp_debug"
            extractor.enable_debug(str(debug_dir))

        sampler = FrameSampler(
            interval_minutes=config.get("video.frame_interval_minutes", 5),
            tolerance_seconds=config.get("video.tolerance_seconds", 10),
        )

        frame_number, frame, actual_ts = find_frame_by_timestamp(
            args.timestamp,
            video_processor,
            sampler,
            extractor,
            args.scan_interval,
        )

        sanitized_ts = actual_ts.replace(":", "")
        filename = f"reference_{sanitized_ts}_f{frame_number:06d}.png"
        output_path = output_dir / filename
        export_reference_frame(frame, output_path, actual_ts, frame_number)
        return output_path

    finally:
        video_processor.release()


def main() -> None:
    args = parse_args()
    setup_logging(debug_mode=args.verbose)

    config = ConfigManager(args.config)
    output_dir = Path(args.output_dir)
    ensure_output_directory(output_dir)

    if not args.no_grid:
        floormap_path = config.get("floormap.image_path")
        if not floormap_path:
            logger.warning("floormap.image_path が設定されていないため、ガイドをスキップします。")
        else:
            color = parse_color(args.grid_color)
            grid_output = output_dir / "floormap_grid.png"
            export_floormap_grid(Path(floormap_path), grid_output, args.grid_interval, color)

    frame_path = extract_reference_frame(args, config, output_dir)
    logger.info("参照フレームの書き出しが完了しました: %s", frame_path)


if __name__ == "__main__":
    main()

