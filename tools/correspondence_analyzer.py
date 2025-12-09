"""対応点の品質分析・分布可視化ツール

Phase 3座標変換精度改善のための分析ツール
"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """NumPy型をJSONシリアライズするカスタムエンコーダー"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_correspondence_points(json_path: Path) -> dict:
    """対応点JSONを読み込む"""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def analyze_distribution(points: list[dict], image_size: tuple[int, int]) -> dict:
    """対応点の分布を分析

    Args:
        points: 対応点リスト [{"src_point": [x, y], "dst_point": [x, y]}, ...]
        image_size: (width, height)

    Returns:
        分析結果辞書
    """
    width, height = image_size
    src_points = np.array([p["src_point"] for p in points])

    # グリッド分割 (5x4)
    grid_cols, grid_rows = 5, 4
    cell_width = width / grid_cols
    cell_height = height / grid_rows

    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    grid_points = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]

    for i, pt in enumerate(src_points):
        col = min(int(pt[0] / cell_width), grid_cols - 1)
        row = min(int(pt[1] / cell_height), grid_rows - 1)
        grid_counts[row, col] += 1
        grid_points[row][col].append(i)

    # 統計情報
    stats = {
        "total_points": len(points),
        "image_size": image_size,
        "grid_size": (grid_cols, grid_rows),
        "grid_counts": grid_counts.tolist(),
        "empty_cells": int(np.sum(grid_counts == 0)),
        "min_count": int(np.min(grid_counts)),
        "max_count": int(np.max(grid_counts)),
        "mean_count": float(np.mean(grid_counts)),
        "std_count": float(np.std(grid_counts)),
        "coverage_ratio": float(np.sum(grid_counts > 0) / (grid_cols * grid_rows)),
        "grid_points": grid_points,
    }

    # 各領域の詳細
    region_stats = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x_range = (col * cell_width, (col + 1) * cell_width)
            y_range = (row * cell_height, (row + 1) * cell_height)
            count = grid_counts[row, col]
            region_stats.append(
                {
                    "row": row,
                    "col": col,
                    "x_range": x_range,
                    "y_range": y_range,
                    "count": int(count),
                    "is_sparse": count < 3,
                    "is_empty": count == 0,
                }
            )

    stats["regions"] = region_stats

    # 疎な領域（点が少ない領域）を特定
    sparse_regions = [r for r in region_stats if r["is_sparse"]]
    stats["sparse_regions"] = sparse_regions
    stats["num_sparse_regions"] = len(sparse_regions)

    return stats


def compute_homography_error(points: list[dict]) -> dict:
    """ホモグラフィ変換の誤差を計算

    Args:
        points: 対応点リスト

    Returns:
        誤差分析結果
    """
    src_pts = np.array([p["src_point"] for p in points], dtype=np.float64)
    dst_pts = np.array([p["dst_point"] for p in points], dtype=np.float64)

    # RANSAC でホモグラフィを推定
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"error": "Failed to compute homography"}

    inliers = mask.ravel().astype(bool)
    num_inliers = int(np.sum(inliers))

    # 変換を適用
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    transformed = (H @ src_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    # 誤差を計算
    errors = np.sqrt(np.sum((transformed - dst_pts) ** 2, axis=1))

    # 各点の誤差詳細
    point_errors = []
    for i, (src, dst, trans, err, is_inlier) in enumerate(
        zip(src_pts, dst_pts, transformed, errors, inliers, strict=False)
    ):
        point_errors.append(
            {
                "index": i,
                "src_point": src.tolist(),
                "dst_point": dst.tolist(),
                "transformed_point": trans.tolist(),
                "error": float(err),
                "is_inlier": bool(is_inlier),  # numpy.bool_ -> Python bool
                "error_vector": (trans - dst).tolist(),
            }
        )

    # ソートして外れ値を特定
    sorted_errors = sorted(point_errors, key=lambda x: x["error"], reverse=True)

    return {
        "homography_matrix": H.tolist(),
        "num_inliers": num_inliers,
        "num_outliers": len(points) - num_inliers,
        "inlier_ratio": num_inliers / len(points),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "std_error": float(np.std(errors)),
        "percentile_90": float(np.percentile(errors, 90)),
        "percentile_95": float(np.percentile(errors, 95)),
        "point_errors": point_errors,
        "worst_points": sorted_errors[:10],  # 上位10件の外れ値
    }


def visualize_distribution(
    points: list[dict],
    image_size: tuple[int, int],
    reference_image_path: Path | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """対応点の分布を可視化

    Args:
        points: 対応点リスト
        image_size: (width, height)
        reference_image_path: 参照画像のパス（オプション）
        output_path: 出力パス（オプション）

    Returns:
        可視化画像
    """
    width, height = image_size

    # ベース画像を作成
    if reference_image_path and reference_image_path.exists():
        img = cv2.imread(str(reference_image_path))
        if img is None:
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # グリッドを描画
    grid_cols, grid_rows = 5, 4
    cell_width = width / grid_cols
    cell_height = height / grid_rows

    # グリッド線
    for col in range(1, grid_cols):
        x = int(col * cell_width)
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 1)
    for row in range(1, grid_rows):
        y = int(row * cell_height)
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 1)

    # 対応点をプロット
    src_points = np.array([p["src_point"] for p in points])

    # グリッドごとの色を計算
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    for pt in src_points:
        col = min(int(pt[0] / cell_width), grid_cols - 1)
        row = min(int(pt[1] / cell_height), grid_rows - 1)
        grid_counts[row, col] += 1

    # セルの背景色（疎な領域を赤く）
    overlay = img.copy()
    for row in range(grid_rows):
        for col in range(grid_cols):
            count = grid_counts[row, col]
            x1, y1 = int(col * cell_width), int(row * cell_height)
            x2, y2 = int((col + 1) * cell_width), int((row + 1) * cell_height)

            if count == 0:
                color = (0, 0, 255)  # 赤：空
            elif count < 3:
                color = (0, 165, 255)  # オレンジ：疎
            else:
                color = (0, 255, 0)  # 緑：十分

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # 対応点をプロット
    for _i, pt in enumerate(src_points):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(img, (x, y), 5, (0, 0, 0), 1)

    # 凡例
    cv2.putText(img, f"Total: {len(points)} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(
        img, "Red: Empty | Orange: Sparse (<3) | Green: OK (>=3)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )

    # グリッドごとの点数を表示
    for row in range(grid_rows):
        for col in range(grid_cols):
            count = grid_counts[row, col]
            x = int((col + 0.5) * cell_width)
            y = int((row + 0.5) * cell_height)
            color = (0, 0, 0) if count >= 3 else (0, 0, 255)
            cv2.putText(img, str(count), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if output_path:
        cv2.imwrite(str(output_path), img)

    return img


def visualize_error_map(
    points: list[dict],
    error_results: dict,
    floormap_size: tuple[int, int],
    floormap_path: Path | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """誤差をフロアマップ上に可視化

    Args:
        points: 対応点リスト
        error_results: compute_homography_error の結果
        floormap_size: (width, height)
        floormap_path: フロアマップ画像のパス
        output_path: 出力パス

    Returns:
        可視化画像
    """
    width, height = floormap_size

    # ベース画像
    if floormap_path and floormap_path.exists():
        img = cv2.imread(str(floormap_path))
        if img is None:
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 誤差の最大値でスケーリング
    max_error = error_results["max_error"]
    point_errors = error_results["point_errors"]

    for pe in point_errors:
        dst = pe["dst_point"]
        trans = pe["transformed_point"]
        error = pe["error"]
        is_inlier = pe["is_inlier"]

        # 誤差に応じた色 (緑→黄→赤)
        ratio = min(error / max(max_error, 1), 1.0)
        if ratio < 0.5:
            # 緑→黄
            r = int(255 * ratio * 2)
            g = 255
            b = 0
        else:
            # 黄→赤
            r = 255
            g = int(255 * (1 - (ratio - 0.5) * 2))
            b = 0
        color = (b, g, r)

        # 期待位置と変換位置を描画
        dst_pt = (int(dst[0]), int(dst[1]))
        trans_pt = (int(trans[0]), int(trans[1]))

        # 誤差ベクトル
        cv2.arrowedLine(img, dst_pt, trans_pt, color, 2, tipLength=0.3)

        # 期待位置
        marker = cv2.MARKER_CROSS if is_inlier else cv2.MARKER_TILTED_CROSS
        cv2.drawMarker(img, dst_pt, (0, 0, 0), marker, 10, 2)

    # 凡例
    cv2.putText(img, f"RMSE: {error_results['rmse']:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Max: {error_results['max_error']:.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(
        img,
        f"Inliers: {error_results['num_inliers']}/{len(points)}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    if output_path:
        cv2.imwrite(str(output_path), img)

    return img


def generate_report(
    data: dict,
    distribution_stats: dict,
    error_results: dict,
    output_path: Path,
) -> str:
    """分析レポートを生成

    Args:
        data: 対応点データ
        distribution_stats: 分布分析結果
        error_results: 誤差分析結果
        output_path: 出力パス

    Returns:
        レポート文字列
    """
    report = []
    report.append("# 対応点品質分析レポート\n")
    report.append(f"生成日時: {data['metadata'].get('created_at', 'N/A')}\n")

    report.append("\n## 1. 基本情報\n")
    report.append(f"- 対応点数: {distribution_stats['total_points']}")
    report.append(f"- 画像サイズ: {distribution_stats['image_size']}")
    report.append(f"- フロアマップサイズ: {data['metadata']['floormap_size']}")

    report.append("\n## 2. 分布分析\n")
    report.append(f"- グリッド分割: {distribution_stats['grid_size'][0]}x{distribution_stats['grid_size'][1]}")
    report.append(f"- カバレッジ率: {distribution_stats['coverage_ratio']:.1%}")
    report.append(f"- 空セル数: {distribution_stats['empty_cells']}")
    report.append(f"- 疎なセル数 (<3点): {distribution_stats['num_sparse_regions']}")

    report.append("\n### グリッドごとの点数\n")
    report.append("```")
    grid = distribution_stats["grid_counts"]
    for row in grid:
        report.append("  ".join(f"{c:2d}" for c in row))
    report.append("```")

    report.append("\n### 改善が必要な領域\n")
    for r in distribution_stats["sparse_regions"]:
        status = "空" if r["is_empty"] else f"{r['count']}点"
        report.append(f"- [{r['row']},{r['col']}] x={r['x_range']}, y={r['y_range']}: {status}")

    report.append("\n## 3. ホモグラフィ誤差分析\n")
    report.append(f"- RMSE: {error_results['rmse']:.2f} px")
    report.append(f"- MAE: {error_results['mae']:.2f} px")
    report.append(f"- 最大誤差: {error_results['max_error']:.2f} px")
    report.append(f"- 最小誤差: {error_results['min_error']:.2f} px")
    report.append(f"- 標準偏差: {error_results['std_error']:.2f} px")
    report.append(f"- 90パーセンタイル: {error_results['percentile_90']:.2f} px")
    report.append(f"- 95パーセンタイル: {error_results['percentile_95']:.2f} px")
    report.append(f"- インライア数: {error_results['num_inliers']}/{distribution_stats['total_points']}")
    report.append(f"- インライア率: {error_results['inlier_ratio']:.1%}")

    report.append("\n### 誤差が大きい上位10点\n")
    report.append("| # | src (x, y) | dst (x, y) | 誤差 (px) | インライア |")
    report.append("|---|------------|------------|-----------|------------|")
    for i, wp in enumerate(error_results["worst_points"]):
        src = wp["src_point"]
        dst = wp["dst_point"]
        report.append(
            f"| {i + 1} | ({src[0]:.0f}, {src[1]:.0f}) | "
            f"({dst[0]:.0f}, {dst[1]:.0f}) | {wp['error']:.1f} | "
            f"{'Yes' if wp['is_inlier'] else 'No'} |"
        )

    report.append("\n## 4. 推奨事項\n")

    recommendations = []
    if distribution_stats["empty_cells"] > 0:
        recommendations.append(f"- 空のグリッドセル（{distribution_stats['empty_cells']}個）に対応点を追加してください")
    if distribution_stats["num_sparse_regions"] > 5:
        recommendations.append(f"- 疎な領域（{distribution_stats['num_sparse_regions']}個）に追加の対応点が必要です")
    if error_results["rmse"] > 50:
        recommendations.append(
            f"- RMSE ({error_results['rmse']:.1f}px) が目標 (<20px) を大きく超えています。PWA変換への移行を推奨します"
        )
    if error_results["num_outliers"] > 5:
        recommendations.append(f"- 外れ値 ({error_results['num_outliers']}点) の確認・修正を推奨します")

    if recommendations:
        for rec in recommendations:
            report.append(rec)
    else:
        report.append("- 対応点の品質は良好です")

    report_text = "\n".join(report)

    # ファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    """メイン処理"""
    # パス設定
    correspondence_path = PROJECT_ROOT / "output/calibration/correspondence_points_new.json"
    reference_image_path = PROJECT_ROOT / "output/latest/phase1_extraction/frames/frame_20250826_160500_idx4.jpg"
    floormap_path = PROJECT_ROOT / "data/floormap.png"
    output_dir = PROJECT_ROOT / "output/calibration/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("対応点品質分析ツール")
    print("=" * 60)

    # データ読み込み
    print("\n[1/5] 対応点データを読み込み中...")
    data = load_correspondence_points(correspondence_path)
    points = data["point_correspondences"]
    image_size = (
        data["metadata"]["image_size"]["width"],
        data["metadata"]["image_size"]["height"],
    )
    floormap_size = (
        data["metadata"]["floormap_size"]["width"],
        data["metadata"]["floormap_size"]["height"],
    )
    print(f"  - 対応点数: {len(points)}")
    print(f"  - 画像サイズ: {image_size}")
    print(f"  - フロアマップサイズ: {floormap_size}")

    # 分布分析
    print("\n[2/5] 分布を分析中...")
    dist_stats = analyze_distribution(points, image_size)
    print(f"  - カバレッジ率: {dist_stats['coverage_ratio']:.1%}")
    print(f"  - 空セル数: {dist_stats['empty_cells']}")
    print(f"  - 疎なセル数: {dist_stats['num_sparse_regions']}")

    # 誤差分析
    print("\n[3/5] ホモグラフィ誤差を計算中...")
    error_results = compute_homography_error(points)
    print(f"  - RMSE: {error_results['rmse']:.2f} px")
    print(f"  - 最大誤差: {error_results['max_error']:.2f} px")
    print(f"  - インライア率: {error_results['inlier_ratio']:.1%}")

    # 可視化
    print("\n[4/5] 可視化を生成中...")
    visualize_distribution(
        points,
        image_size,
        reference_image_path if reference_image_path.exists() else None,
        output_dir / "distribution_map.png",
    )
    print(f"  - 分布マップ: {output_dir / 'distribution_map.png'}")

    visualize_error_map(
        points,
        error_results,
        floormap_size,
        floormap_path if floormap_path.exists() else None,
        output_dir / "error_map.png",
    )
    print(f"  - 誤差マップ: {output_dir / 'error_map.png'}")

    # レポート生成
    print("\n[5/5] レポートを生成中...")
    report = generate_report(data, dist_stats, error_results, output_dir / "quality_report.md")
    print(f"  - レポート: {output_dir / 'quality_report.md'}")

    # JSON出力
    analysis_result = {
        "distribution": dist_stats,
        "homography_error": {k: v for k, v in error_results.items() if k != "point_errors"},
        "point_errors": error_results["point_errors"],
    }
    with open(output_dir / "analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  - 分析結果JSON: {output_dir / 'analysis_result.json'}")

    print("\n" + "=" * 60)
    print("分析完了！")
    print("=" * 60)
    print("\n" + report)

    return dist_stats, error_results


if __name__ == "__main__":
    main()
