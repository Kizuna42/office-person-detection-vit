"""再投影誤差評価スクリプト

実際のホモグラフィ行列で再投影誤差を評価します。
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.calibration.reprojection_error import ReprojectionErrorEvaluator
from src.config import ConfigManager
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_correspondence_points(points_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """対応点を読み込む

    Args:
        points_path: 対応点ファイルのパス（JSON形式）

    Returns:
        (src_points, dst_points) のタプル
    """
    with open(points_path, encoding="utf-8") as f:
        data = json.load(f)
        src_points = [tuple(pt) for pt in data.get("src_points", [])]
        dst_points = [tuple(pt) for pt in data.get("dst_points", [])]
        return src_points, dst_points


def load_homography_matrix(config: ConfigManager) -> np.ndarray:
    """ホモグラフィ行列を読み込む

    Args:
        config: ConfigManagerインスタンス

    Returns:
        ホモグラフィ行列（3x3）
    """
    matrix_data = config.get("homography.matrix")
    return np.array(matrix_data, dtype=np.float64)


def evaluate_reprojection_error(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
    homography_matrix: np.ndarray,
) -> dict[str, float]:
    """再投影誤差を評価

    Args:
        src_points: 変換元の点のリスト（カメラ座標）
        dst_points: 変換先の点のリスト（フロアマップ座標）
        homography_matrix: ホモグラフィ変換行列（3x3）

    Returns:
        評価結果の辞書
    """
    evaluator = ReprojectionErrorEvaluator()
    return evaluator.evaluate_homography(src_points, dst_points, homography_matrix)


def create_error_map(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
    homography_matrix: np.ndarray,
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """誤差マップを生成

    Args:
        src_points: 変換元の点のリスト
        dst_points: 変換先の点のリスト
        homography_matrix: ホモグラフィ変換行列
        image_shape: 画像形状 (height, width)
        output_path: 出力ファイルのパス
    """
    evaluator = ReprojectionErrorEvaluator()
    error_map = evaluator.create_error_map(src_points, dst_points, homography_matrix, image_shape)

    # 誤差マップを可視化して保存
    import cv2

    # 誤差を0-255の範囲に正規化
    if np.max(error_map) > 0:
        error_map_normalized = (error_map / np.max(error_map) * 255).astype(np.uint8)
    else:
        error_map_normalized = error_map.astype(np.uint8)

    # カラーマップを適用
    error_map_colored = cv2.applyColorMap(error_map_normalized, cv2.COLORMAP_JET)

    cv2.imwrite(str(output_path), error_map_colored)
    logger.info(f"誤差マップを保存しました: {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="再投影誤差評価スクリプト")
    parser.add_argument("--points", type=str, required=True, help="対応点ファイルのパス（JSON形式）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--output", type=str, default="reprojection_error.json", help="出力ファイルのパス")
    parser.add_argument("--error-map", type=str, help="誤差マップの出力パス（オプション）")
    parser.add_argument("--image-shape", type=int, nargs=2, default=[1369, 1878], help="画像形状 (height, width)")

    args = parser.parse_args()

    # ロギング設定
    setup_logging()

    logger.info("=" * 80)
    logger.info("再投影誤差評価を開始")
    logger.info("=" * 80)

    # ファイルパスの確認
    points_path = Path(args.points)
    config_path = Path(args.config)

    if not points_path.exists():
        logger.error(f"対応点ファイルが見つかりません: {points_path}")
        return 1

    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1

    # 設定の読み込み
    config = ConfigManager(str(config_path))

    # 対応点の読み込み
    logger.info(f"対応点を読み込み中: {points_path}")
    src_points, dst_points = load_correspondence_points(points_path)
    logger.info(f"  対応点数: {len(src_points)}")

    # ホモグラフィ行列の読み込み
    logger.info("ホモグラフィ行列を読み込み中")
    homography_matrix = load_homography_matrix(config)
    logger.info(f"  ホモグラフィ行列:\n{homography_matrix}")

    # 再投影誤差の評価
    logger.info("再投影誤差を計算中...")
    error_result = evaluate_reprojection_error(src_points, dst_points, homography_matrix)

    # 結果の表示
    logger.info("=" * 80)
    logger.info("再投影誤差評価結果")
    logger.info("=" * 80)
    logger.info(f"  平均誤差: {error_result['mean_error']:.2f} ピクセル")
    logger.info(f"  最大誤差: {error_result['max_error']:.2f} ピクセル")
    logger.info(f"  最小誤差: {error_result['min_error']:.2f} ピクセル")
    logger.info(f"  標準偏差: {error_result['std_error']:.2f} ピクセル")
    logger.info("=" * 80)

    # 目標値との比較
    error_threshold = 2.0  # ピクセル単位

    logger.info("目標値との比較:")
    logger.info(f"  目標誤差: {error_threshold:.1f} ピクセル")
    logger.info(f"  平均誤差: {'✅ 達成' if error_result['mean_error'] <= error_threshold else '❌ 未達成'}")
    logger.info(f"  最大誤差: {'✅ 達成' if error_result['max_error'] <= error_threshold else '❌ 未達成'}")

    # 誤差マップの生成（オプション）
    if args.error_map:
        logger.info("誤差マップを生成中...")
        error_map_path = Path(args.error_map)
        error_map_path.parent.mkdir(parents=True, exist_ok=True)
        create_error_map(src_points, dst_points, homography_matrix, tuple(args.image_shape), error_map_path)

    # 結果をJSONファイルに保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "error_metrics": error_result,
        "target": {
            "mean_error_threshold": error_threshold,
        },
        "achieved": {
            "mean_error": error_result["mean_error"] <= error_threshold,
            "max_error": error_result["max_error"] <= error_threshold,
        },
        "num_points": len(src_points),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"評価結果を保存しました: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
