"""座標変換精度評価モジュール

Phase 3座標変換の精度を定量的に評価するためのツール
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.transform import TransformResult

logger = logging.getLogger(__name__)


class Transformer(Protocol):
    """座標変換器のプロトコル"""

    def transform_pixel(self, image_point: tuple[float, float]) -> TransformResult:
        """画像座標を変換"""
        ...


@dataclass
class EvaluationMetrics:
    """評価メトリクス

    Attributes:
        rmse: Root Mean Square Error (px)
        mae: Mean Absolute Error (px)
        max_error: 最大誤差 (px)
        min_error: 最小誤差 (px)
        std_error: 標準偏差 (px)
        percentile_90: 90パーセンタイル (px)
        percentile_95: 95パーセンタイル (px)
        num_points: 評価点数
        num_valid: 有効変換点数
        valid_ratio: 有効変換率
    """

    rmse: float = 0.0
    mae: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    std_error: float = 0.0
    percentile_90: float = 0.0
    percentile_95: float = 0.0
    num_points: int = 0
    num_valid: int = 0
    valid_ratio: float = 0.0
    per_point_errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "max_error": self.max_error,
            "min_error": self.min_error,
            "std_error": self.std_error,
            "percentile_90": self.percentile_90,
            "percentile_95": self.percentile_95,
            "num_points": self.num_points,
            "num_valid": self.num_valid,
            "valid_ratio": self.valid_ratio,
        }

    def meets_target(
        self,
        rmse_target: float = 20.0,
        max_error_target: float = 50.0,
    ) -> bool:
        """目標精度を満たしているか"""
        return self.rmse <= rmse_target and self.max_error <= max_error_target

    def summary(self) -> str:
        """サマリー文字列"""
        status = "✓ OK" if self.meets_target() else "✗ NG"
        return (
            f"RMSE: {self.rmse:.2f}px | "
            f"MAE: {self.mae:.2f}px | "
            f"Max: {self.max_error:.2f}px | "
            f"Valid: {self.num_valid}/{self.num_points} | "
            f"{status}"
        )


class TransformEvaluator:
    """座標変換精度評価器

    対応点データを使用して座標変換の精度を評価します。
    """

    def __init__(
        self,
        correspondence_points: list[dict] | None = None,
        correspondence_file: Path | str | None = None,
    ):
        """初期化

        Args:
            correspondence_points: 対応点リスト [{"src_point": [x,y], "dst_point": [x,y]}, ...]
            correspondence_file: 対応点JSONファイルのパス
        """
        self.points: list[dict] = []

        if correspondence_points:
            self.points = correspondence_points
        elif correspondence_file:
            self.load_correspondence_file(correspondence_file)

        logger.info(f"TransformEvaluator initialized with {len(self.points)} points")

    def load_correspondence_file(self, file_path: Path | str) -> None:
        """対応点ファイルを読み込み

        Args:
            file_path: JSONファイルのパス
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Correspondence file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.points = data.get("point_correspondences", [])
        logger.info(f"Loaded {len(self.points)} correspondence points from {path}")

    def evaluate(
        self,
        transformer: Transformer,
    ) -> EvaluationMetrics:
        """座標変換の精度を評価

        Args:
            transformer: 座標変換器

        Returns:
            評価メトリクス
        """
        if not self.points:
            logger.warning("No correspondence points available for evaluation")
            return EvaluationMetrics()

        errors = []
        per_point_errors = []
        num_valid = 0

        for i, point in enumerate(self.points):
            src = point["src_point"]
            dst_expected = point["dst_point"]

            # 変換を実行
            result = transformer.transform_pixel((src[0], src[1]))

            if result.is_valid and result.floor_coords_px:
                num_valid += 1
                dst_actual = result.floor_coords_px

                # 誤差を計算
                error = np.sqrt((dst_actual[0] - dst_expected[0]) ** 2 + (dst_actual[1] - dst_expected[1]) ** 2)
                errors.append(error)

                per_point_errors.append(
                    {
                        "index": i,
                        "src_point": src,
                        "dst_expected": dst_expected,
                        "dst_actual": list(dst_actual),
                        "error": float(error),
                        "error_vector": [
                            dst_actual[0] - dst_expected[0],
                            dst_actual[1] - dst_expected[1],
                        ],
                        "is_valid": True,
                    }
                )
            else:
                per_point_errors.append(
                    {
                        "index": i,
                        "src_point": src,
                        "dst_expected": dst_expected,
                        "dst_actual": None,
                        "error": None,
                        "error_vector": None,
                        "is_valid": False,
                    }
                )

        if not errors:
            logger.warning("No valid transformations for evaluation")
            return EvaluationMetrics(
                num_points=len(self.points),
                num_valid=0,
                valid_ratio=0.0,
                per_point_errors=per_point_errors,
            )

        errors_array = np.array(errors)

        return EvaluationMetrics(
            rmse=float(np.sqrt(np.mean(errors_array**2))),
            mae=float(np.mean(errors_array)),
            max_error=float(np.max(errors_array)),
            min_error=float(np.min(errors_array)),
            std_error=float(np.std(errors_array)),
            percentile_90=float(np.percentile(errors_array, 90)),
            percentile_95=float(np.percentile(errors_array, 95)),
            num_points=len(self.points),
            num_valid=num_valid,
            valid_ratio=num_valid / len(self.points),
            per_point_errors=per_point_errors,
        )

    def evaluate_homography(
        self,
        homography_matrix: np.ndarray,
    ) -> EvaluationMetrics:
        """ホモグラフィ行列の精度を評価

        Args:
            homography_matrix: 3x3ホモグラフィ行列

        Returns:
            評価メトリクス
        """
        if not self.points:
            return EvaluationMetrics()

        H = np.array(homography_matrix, dtype=np.float64)
        errors = []
        per_point_errors = []

        for i, point in enumerate(self.points):
            src = point["src_point"]
            dst_expected = point["dst_point"]

            # ホモグラフィ変換
            src_h = np.array([src[0], src[1], 1.0])
            dst_h = H @ src_h
            if abs(dst_h[2]) < 1e-10:
                per_point_errors.append(
                    {
                        "index": i,
                        "src_point": src,
                        "dst_expected": dst_expected,
                        "dst_actual": None,
                        "error": None,
                        "is_valid": False,
                    }
                )
                continue

            dst_actual = [dst_h[0] / dst_h[2], dst_h[1] / dst_h[2]]

            error = np.sqrt((dst_actual[0] - dst_expected[0]) ** 2 + (dst_actual[1] - dst_expected[1]) ** 2)
            errors.append(error)

            per_point_errors.append(
                {
                    "index": i,
                    "src_point": src,
                    "dst_expected": dst_expected,
                    "dst_actual": dst_actual,
                    "error": float(error),
                    "error_vector": [
                        dst_actual[0] - dst_expected[0],
                        dst_actual[1] - dst_expected[1],
                    ],
                    "is_valid": True,
                }
            )

        if not errors:
            return EvaluationMetrics(
                num_points=len(self.points),
                per_point_errors=per_point_errors,
            )

        errors_array = np.array(errors)

        return EvaluationMetrics(
            rmse=float(np.sqrt(np.mean(errors_array**2))),
            mae=float(np.mean(errors_array)),
            max_error=float(np.max(errors_array)),
            min_error=float(np.min(errors_array)),
            std_error=float(np.std(errors_array)),
            percentile_90=float(np.percentile(errors_array, 90)),
            percentile_95=float(np.percentile(errors_array, 95)),
            num_points=len(self.points),
            num_valid=len(errors),
            valid_ratio=len(errors) / len(self.points),
            per_point_errors=per_point_errors,
        )

    def visualize_errors(
        self,
        metrics: EvaluationMetrics,
        floormap_image: np.ndarray | Path | str,
        output_path: Path | str | None = None,
        max_error_scale: float | None = None,
    ) -> np.ndarray:
        """誤差を可視化

        Args:
            metrics: 評価メトリクス
            floormap_image: フロアマップ画像またはパス
            output_path: 出力パス（オプション）
            max_error_scale: 誤差スケールの最大値（色付け用）

        Returns:
            可視化画像
        """
        # 画像を読み込み
        if isinstance(floormap_image, str | Path):
            img = cv2.imread(str(floormap_image))
            if img is None:
                raise FileNotFoundError(f"Image not found: {floormap_image}")
        else:
            img = floormap_image.copy()

        if not metrics.per_point_errors:
            return img

        # 誤差のスケール
        max_err = max_error_scale or metrics.max_error or 100.0

        for pe in metrics.per_point_errors:
            if not pe["is_valid"]:
                continue

            dst_expected = pe["dst_expected"]
            dst_actual = pe["dst_actual"]
            error = pe["error"]

            # 誤差に応じた色 (緑→黄→赤)
            ratio = min(error / max_err, 1.0)
            if ratio < 0.5:
                r = int(255 * ratio * 2)
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (1 - (ratio - 0.5) * 2))
                b = 0
            color = (b, g, r)

            pt_expected = (int(dst_expected[0]), int(dst_expected[1]))
            pt_actual = (int(dst_actual[0]), int(dst_actual[1]))

            # 誤差ベクトル
            cv2.arrowedLine(img, pt_expected, pt_actual, color, 2, tipLength=0.3)
            cv2.drawMarker(img, pt_expected, (0, 0, 0), cv2.MARKER_CROSS, 10, 2)

        # メトリクス表示
        cv2.putText(img, f"RMSE: {metrics.rmse:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Max: {metrics.max_error:.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(
            img,
            f"Valid: {metrics.num_valid}/{metrics.num_points}",
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
        self,
        metrics: EvaluationMetrics,
        output_path: Path | str | None = None,
        title: str = "座標変換精度評価レポート",
    ) -> str:
        """評価レポートを生成

        Args:
            metrics: 評価メトリクス
            output_path: 出力パス（オプション）
            title: レポートタイトル

        Returns:
            レポート文字列
        """
        lines = [
            f"# {title}\n",
            "## 評価サマリー\n",
            f"- **ステータス**: {'✓ 目標達成' if metrics.meets_target() else '✗ 改善が必要'}",
            f"- **評価点数**: {metrics.num_points}",
            f"- **有効変換数**: {metrics.num_valid} ({metrics.valid_ratio:.1%})\n",
            "## メトリクス\n",
            "| 指標 | 値 | 目標 |",
            "|------|-----|-----|",
            f"| RMSE | {metrics.rmse:.2f} px | < 20 px |",
            f"| MAE | {metrics.mae:.2f} px | - |",
            f"| 最大誤差 | {metrics.max_error:.2f} px | < 50 px |",
            f"| 最小誤差 | {metrics.min_error:.2f} px | - |",
            f"| 標準偏差 | {metrics.std_error:.2f} px | - |",
            f"| 90%タイル | {metrics.percentile_90:.2f} px | - |",
            f"| 95%タイル | {metrics.percentile_95:.2f} px | - |",
        ]

        # 上位10件の誤差
        if metrics.per_point_errors:
            sorted_errors = sorted(
                [e for e in metrics.per_point_errors if e["is_valid"]],
                key=lambda x: x["error"],
                reverse=True,
            )[:10]

            lines.append("\n## 誤差上位10点\n")
            lines.append("| # | src (x,y) | dst期待 (x,y) | dst実際 (x,y) | 誤差 (px) |")
            lines.append("|---|-----------|---------------|---------------|-----------|")

            for i, e in enumerate(sorted_errors):
                src = e["src_point"]
                dst_exp = e["dst_expected"]
                dst_act = e["dst_actual"]
                lines.append(
                    f"| {i + 1} | ({src[0]:.0f}, {src[1]:.0f}) | "
                    f"({dst_exp[0]:.0f}, {dst_exp[1]:.0f}) | "
                    f"({dst_act[0]:.0f}, {dst_act[1]:.0f}) | "
                    f"{e['error']:.1f} |"
                )

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def cross_validate(
        self,
        transformer_factory,
        k_folds: int = 5,
    ) -> dict[str, Any]:
        """クロスバリデーションで精度を評価

        Args:
            transformer_factory: 対応点から変換器を作成する関数
            k_folds: 分割数

        Returns:
            クロスバリデーション結果
        """
        if len(self.points) < k_folds:
            logger.warning(f"Not enough points for {k_folds}-fold CV")
            return {}

        indices = np.arange(len(self.points))
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)

        fold_results = []

        for i, test_indices in enumerate(folds):
            train_indices = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            train_points = [self.points[idx] for idx in train_indices]
            test_points = [self.points[idx] for idx in test_indices]

            # 訓練データで変換器を作成
            transformer = transformer_factory(train_points)

            # テストデータで評価
            test_evaluator = TransformEvaluator(correspondence_points=test_points)
            metrics = test_evaluator.evaluate(transformer)

            fold_results.append(
                {
                    "fold": i,
                    "train_size": len(train_points),
                    "test_size": len(test_points),
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "max_error": metrics.max_error,
                }
            )

        # 集計
        rmses = [r["rmse"] for r in fold_results]
        maes = [r["mae"] for r in fold_results]
        maxes = [r["max_error"] for r in fold_results]

        return {
            "k_folds": k_folds,
            "fold_results": fold_results,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "mean_max_error": float(np.mean(maxes)),
            "std_max_error": float(np.std(maxes)),
        }
