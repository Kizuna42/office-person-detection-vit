#!/usr/bin/env python3
"""COCO形式からGround Truthトラックを自動生成するスクリプト

COCO形式のアノテーションデータ（result_fixed.json）から、
Ground Truthトラック形式を自動生成します。
"""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.coordinate_transformer import CoordinateTransformer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_coco_data(coco_path: Path) -> dict:
    """COCO形式のJSONファイルを読み込む

    Args:
        coco_path: COCO形式JSONファイルのパス

    Returns:
        COCO形式のデータ辞書

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        json.JSONDecodeError: JSON形式が不正な場合
    """
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO形式ファイルが見つかりません: {coco_path}")

    with open(coco_path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"COCO形式データを読み込みました: {coco_path}")
    logger.info(f"  画像数: {len(data.get('images', []))}")
    logger.info(f"  アノテーション数: {len(data.get('annotations', []))}")

    return data


def get_foot_position(bbox: list[float]) -> tuple[float, float]:
    """バウンディングボックスから足元座標を計算

    COCO形式のbboxは [x, y, width, height] 形式。
    足元座標はバウンディングボックスの中心下端。

    Args:
        bbox: バウンディングボックス [x, y, width, height]

    Returns:
        足元座標 (x, y)
    """
    x, y, width, height = bbox
    foot_x = x + width / 2.0
    foot_y = y + height
    return (float(foot_x), float(foot_y))


def group_annotations_by_frame(coco_data: dict) -> dict[int, list[dict]]:
    """アノテーションをフレーム（image_id）ごとにグループ化

    Args:
        coco_data: COCO形式のデータ辞書

    Returns:
        {image_id: [annotations]} の辞書
    """
    annotations_by_frame = defaultdict(list)

    for ann in coco_data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is not None:
            annotations_by_frame[image_id].append(ann)

    # image_idでソート
    sorted_frames = dict(sorted(annotations_by_frame.items()))

    logger.info(f"フレーム数: {len(sorted_frames)}")
    return sorted_frames


def create_tracks_from_annotations(
    annotations_by_frame: dict[int, list[dict]],
    coordinate_transformer: CoordinateTransformer,
    min_distance: float = 50.0,
) -> list[dict]:
    """アノテーションからトラックを作成

    簡易的なトラッキングアルゴリズム：
    - 各フレームの人物を前フレームの人物と距離でマッチング
    - 距離がmin_distance以下の場合、同じトラックIDを割り当て
    - マッチしない場合は新しいトラックIDを割り当て

    Args:
        annotations_by_frame: フレームごとのアノテーション辞書
        coordinate_transformer: 座標変換器
        min_distance: 同一人物判定の距離閾値（ピクセル）

    Returns:
        トラックのリスト
    """
    tracks: dict[int, list[dict]] = {}  # {track_id: [trajectory_points]}
    next_track_id = 1

    # フレームを時系列順に処理
    sorted_frames = sorted(annotations_by_frame.keys())
    frame_to_index = {frame_id: idx for idx, frame_id in enumerate(sorted_frames)}

    for frame_id in sorted_frames:
        frame_idx = frame_to_index[frame_id]
        annotations = annotations_by_frame[frame_id]

        # 現在フレームの人物位置（フロアマップ座標）を計算
        current_positions = []
        for ann in annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            # 足元座標を計算
            camera_foot = get_foot_position(bbox)

            # フロアマップ座標に変換
            try:
                floor_foot = coordinate_transformer.transform(camera_foot, apply_origin_offset=True)
                current_positions.append((floor_foot, ann))
            except Exception as e:
                logger.warning(f"座標変換に失敗しました (frame={frame_id}, bbox={bbox}): {e}")
                continue

        # 前フレームのトラックとマッチング
        if frame_idx == 0:
            # 最初のフレーム：全て新しいトラックとして作成
            for floor_foot, _ann in current_positions:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [
                    {
                        "x": floor_foot[0],
                        "y": floor_foot[1],
                        "frame": frame_idx,
                    }
                ]
        else:
            # 前フレームのトラック位置を取得
            prev_track_positions = {}
            for track_id, trajectory in tracks.items():
                if trajectory and trajectory[-1]["frame"] == frame_idx - 1:
                    prev_track_positions[track_id] = (trajectory[-1]["x"], trajectory[-1]["y"])

            # 現在フレームの人物を前フレームのトラックとマッチング
            matched_tracks = set()
            for floor_foot, _ann in current_positions:
                best_match_id = None
                best_distance = float("inf")

                # 最も近いトラックを探す
                for track_id, prev_pos in prev_track_positions.items():
                    if track_id in matched_tracks:
                        continue

                    distance = np.sqrt((floor_foot[0] - prev_pos[0]) ** 2 + (floor_foot[1] - prev_pos[1]) ** 2)

                    if distance < min_distance and distance < best_distance:
                        best_match_id = track_id
                        best_distance = distance

                # マッチしたトラックに追加、または新しいトラックを作成
                if best_match_id is not None:
                    tracks[best_match_id].append(
                        {
                            "x": floor_foot[0],
                            "y": floor_foot[1],
                            "frame": frame_idx,
                        }
                    )
                    matched_tracks.add(best_match_id)
                else:
                    # 新しいトラックを作成
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = [
                        {
                            "x": floor_foot[0],
                            "y": floor_foot[1],
                            "frame": frame_idx,
                        }
                    ]

    # トラック形式に変換
    track_list = []
    for track_id, trajectory in sorted(tracks.items()):
        track_list.append(
            {
                "track_id": track_id,
                "trajectory": trajectory,
            }
        )

    logger.info(f"トラック数: {len(track_list)}")
    return track_list


def save_gt_tracks(output_path: Path, tracks: list[dict], source_file: str, num_frames: int) -> None:
    """Ground TruthトラックをJSONファイルに保存

    Args:
        output_path: 出力ファイルパス
        tracks: トラックのリスト
        source_file: 元のCOCO形式ファイルのパス
        num_frames: 総フレーム数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "tracks": tracks,
        "metadata": {
            "source_file": source_file,
            "num_frames": num_frames,
            "num_tracks": len(tracks),
            "coordinate_system": "floormap_pixels",
            "origin_offset_applied": True,
            "note": "自動生成されたデータ。手動編集が必要です。",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Ground Truthトラックを保存しました: {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="COCO形式からGround Truthトラックを自動生成")
    parser.add_argument("--input", type=str, required=True, help="COCO形式JSONファイルのパス")
    parser.add_argument(
        "--output",
        type=str,
        default="data/gt_tracks_auto.json",
        help="出力ファイルパス (default: data/gt_tracks_auto.json)",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルパス (default: config.yaml)")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=50.0,
        help="同一人物判定の距離閾値（ピクセル） (default: 50.0)",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    logger.info("=" * 80)
    logger.info("Ground Truthトラック自動生成を開始")
    logger.info("=" * 80)

    # ファイルパスの確認
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {input_path}")
        return 1

    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    try:
        # 設定の読み込み
        config = ConfigManager(str(config_path))
        homography_matrix = config.get("homography.matrix")
        floormap_config = config.get("floormap")

        if homography_matrix is None:
            logger.error("ホモグラフィ行列が設定されていません")
            return 1

        # 座標変換器の初期化
        coordinate_transformer = CoordinateTransformer(homography_matrix, floormap_config)
        logger.info("座標変換器を初期化しました")

        # COCO形式データの読み込み
        coco_data = load_coco_data(input_path)

        # アノテーションをフレームごとにグループ化
        annotations_by_frame = group_annotations_by_frame(coco_data)

        if not annotations_by_frame:
            logger.warning("アノテーションが見つかりませんでした")
            return 1

        # トラックを作成
        logger.info("トラックを作成中...")
        tracks = create_tracks_from_annotations(annotations_by_frame, coordinate_transformer, args.min_distance)

        if not tracks:
            logger.warning("トラックが作成されませんでした")
            return 1

        # Ground Truthトラックを保存
        num_frames = len(annotations_by_frame)
        save_gt_tracks(output_path, tracks, str(input_path), num_frames)

        logger.info("=" * 80)
        logger.info("Ground Truthトラック自動生成が完了しました")
        logger.info("=" * 80)
        logger.info(f"  出力ファイル: {output_path}")
        logger.info(f"  トラック数: {len(tracks)}")
        logger.info(f"  総フレーム数: {num_frames}")
        logger.info("")
        logger.info("注意: 自動生成されたデータは手動編集が必要です。")
        logger.info("      tools/edit_gt_tracks.py を使用してID割り振りや位置を修正してください。")

        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
