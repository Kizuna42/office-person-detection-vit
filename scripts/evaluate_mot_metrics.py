"""MOTメトリクス評価スクリプト

実際の動画データでMOTメトリクスを評価します。
"""

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.mot_metrics import MOTMetrics
from src.tracking.track import Track
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_ground_truth_tracks(gt_path: Path) -> list[dict]:
    """Ground Truthトラックを読み込む

    Args:
        gt_path: Ground Truthファイルのパス（JSON形式）

    Returns:
        Ground Truthトラックのリスト
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
        return data.get("tracks", [])


def load_predicted_tracks(tracks_path: Path) -> list[Track]:
    """予測トラックを読み込む

    Args:
        tracks_path: トラックファイルのパス（JSON形式）

    Returns:
        予測トラックのリスト
    """
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
        tracks_data = data.get("tracks", [])

    # Trackオブジェクトに変換
    from src.models.data_models import Detection
    from src.tracking.kalman_filter import KalmanFilter

    predicted_tracks = []
    for track_data in tracks_data:
        # Detectionオブジェクトを作成（簡易版）
        detection = Detection(
            bbox=(0, 0, 0, 0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(0, 0),
        )

        # Kalman Filterを作成
        kf = KalmanFilter()

        # Trackオブジェクトを作成
        track = Track(
            track_id=track_data["track_id"],
            detection=detection,
            kalman_filter=kf,
        )

        # 軌跡を設定
        trajectory = track_data.get("trajectory", [])
        track.trajectory = [(pt["x"], pt["y"]) for pt in trajectory]
        track.age = track_data.get("age", 1)
        track.hits = track_data.get("hits", 1)

        predicted_tracks.append(track)

    return predicted_tracks


def evaluate_mot_metrics(gt_tracks: list[dict], predicted_tracks: list[Track], frame_count: int) -> dict[str, float]:
    """MOTメトリクスを評価

    Args:
        gt_tracks: Ground Truthトラックのリスト
        predicted_tracks: 予測トラックのリスト
        frame_count: 総フレーム数

    Returns:
        MOTメトリクスの辞書
    """
    mot_metrics = MOTMetrics()
    return mot_metrics.calculate_tracking_metrics(gt_tracks, predicted_tracks, frame_count)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="MOTメトリクス評価スクリプト")
    parser.add_argument("--gt", type=str, required=True, help="Ground Truthファイルのパス（JSON形式）")
    parser.add_argument("--tracks", type=str, required=True, help="予測トラックファイルのパス（JSON形式）")
    parser.add_argument("--frames", type=int, required=True, help="総フレーム数")
    parser.add_argument("--output", type=str, default="mot_metrics.json", help="出力ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    logger.info("=" * 80)
    logger.info("MOTメトリクス評価を開始")
    logger.info("=" * 80)

    # ファイルパスの確認
    gt_path = Path(args.gt)
    tracks_path = Path(args.tracks)

    if not gt_path.exists():
        logger.error(f"Ground Truthファイルが見つかりません: {gt_path}")
        return 1

    if not tracks_path.exists():
        logger.error(f"予測トラックファイルが見つかりません: {tracks_path}")
        return 1

    # データの読み込み
    logger.info(f"Ground Truthトラックを読み込み中: {gt_path}")
    gt_tracks = load_ground_truth_tracks(gt_path)
    logger.info(f"  Ground Truthトラック数: {len(gt_tracks)}")

    logger.info(f"予測トラックを読み込み中: {tracks_path}")
    predicted_tracks = load_predicted_tracks(tracks_path)
    logger.info(f"  予測トラック数: {len(predicted_tracks)}")

    # MOTメトリクスの評価
    logger.info("MOTメトリクスを計算中...")
    metrics = evaluate_mot_metrics(gt_tracks, predicted_tracks, args.frames)

    # 結果の表示
    logger.info("=" * 80)
    logger.info("MOTメトリクス評価結果")
    logger.info("=" * 80)
    logger.info(f"  MOTA: {metrics['MOTA']:.3f}")
    logger.info(f"  IDF1: {metrics['IDF1']:.3f}")
    logger.info(f"  ID Switches: {metrics['ID_Switches']:.0f}")
    logger.info("=" * 80)

    # 目標値との比較
    mota_target = 0.7
    idf1_target = 0.8

    logger.info("目標値との比較:")
    logger.info(f"  MOTA目標: {mota_target:.1f} {'✅ 達成' if metrics['MOTA'] >= mota_target else '❌ 未達成'}")
    logger.info(f"  IDF1目標: {idf1_target:.1f} {'✅ 達成' if metrics['IDF1'] >= idf1_target else '❌ 未達成'}")

    # 結果をJSONファイルに保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metrics": metrics,
        "targets": {
            "MOTA": mota_target,
            "IDF1": idf1_target,
        },
        "achieved": {
            "MOTA": metrics["MOTA"] >= mota_target,
            "IDF1": metrics["IDF1"] >= idf1_target,
        },
        "frame_count": args.frames,
        "num_gt_tracks": len(gt_tracks),
        "num_predicted_tracks": len(predicted_tracks),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"評価結果を保存しました: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
