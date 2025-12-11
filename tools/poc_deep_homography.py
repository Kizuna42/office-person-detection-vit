"""
Deep Homography (Kornia) と既存ホモグラフィの誤差・速度を比較する PoC スクリプト。

使い方:
  python tools/poc_deep_homography.py \
    --correspondence data/correspondence_points_cam01.json.template \
    --config config/calibration_template.yaml

出力:
 - 既存/DeepHomography それぞれの RMSE/Max エラーと推論時間（秒）を標準出力
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from src.config import ConfigManager
from src.evaluation.transform_evaluator import TransformEvaluator
from src.transform import HomographyTransformer
from src.transform.floormap_config import FloorMapConfig

try:
    import kornia as K
    import torch

    _HAS_KORNIA = True
except Exception:  # pragma: no cover - 任意依存
    _HAS_KORNIA = False


def _load_homography_from_config(config: ConfigManager) -> np.ndarray:
    """設定ファイルからホモグラフィ行列を取得（無ければ単位行列）。"""
    matrix = config.get("homography.matrix")
    if matrix is None:
        return np.eye(3, dtype=np.float64)
    return np.array(matrix, dtype=np.float64)


def _estimate_homography_kornia(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Kornia でホモグラフィを推定。"""
    if not _HAS_KORNIA:
        raise RuntimeError("kornia/torch がインストールされていません。pip install kornia torch")
    # Kornia は (B, N, 2) 形状を要求
    src_t = torch.from_numpy(src)[None, ...].float()
    dst_t = torch.from_numpy(dst)[None, ...].float()
    H, _ = K.geometry.homography.find_homography_dlt(src_t, dst_t)
    return H[0].cpu().numpy()


def _load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """対応点 JSON から src/dst をロード。"""
    evaluator = TransformEvaluator(correspondence_file=path)
    src_points = []
    dst_points = []
    for p in evaluator.points:
        src_points.append(p["src_point"])
        dst_points.append(p["dst_point"])
    return np.array(src_points, dtype=np.float64), np.array(dst_points, dtype=np.float64)


def _evaluate_matrix(H: np.ndarray, floormap_cfg: FloorMapConfig, corr_path: Path) -> tuple[float, float, float]:
    """与えられた H で RMSE/Max/計測時間を返す。"""
    transformer = HomographyTransformer(H, floormap_cfg)
    evaluator = TransformEvaluator(correspondence_file=corr_path)
    start = time.time()
    metrics = evaluator.evaluate(transformer)
    elapsed = time.time() - start
    return metrics.rmse, metrics.max_error, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Homography PoC")
    parser.add_argument("--correspondence", type=Path, required=True, help="対応点 JSON (src/dst)")
    parser.add_argument("--config", type=Path, default=Path("config/calibration_template.yaml"), help="設定ファイル")
    args = parser.parse_args()

    config = ConfigManager(str(args.config))
    floormap_cfg = FloorMapConfig.from_config(config.get("floormap", {}))

    src, dst = _load_correspondences(args.correspondence)
    if len(src) < 4:
        raise ValueError("対応点は 4 点以上が必要です。")

    # 既存ホモグラフィ
    base_H = _load_homography_from_config(config)
    base_rmse, base_max, base_t = _evaluate_matrix(base_H, floormap_cfg, args.correspondence)

    # Kornia 推定
    if _HAS_KORNIA:
        est_H = _estimate_homography_kornia(src, dst)
        k_rmse, k_max, k_t = _evaluate_matrix(est_H, floormap_cfg, args.correspondence)
    else:
        est_H = None
        k_rmse = k_max = k_t = None

    print("=== Homography PoC (CPU) ===")
    print(f"対応点数: {len(src)}")
    print(f"[既存] RMSE={base_rmse:.2f}px, Max={base_max:.2f}px, time={base_t:.3f}s")
    if est_H is not None:
        print(f"[Kornia] RMSE={k_rmse:.2f}px, Max={k_max:.2f}px, time={k_t:.3f}s")
    else:
        print("[Kornia] 未実行: kornia/torch が未インストール")


if __name__ == "__main__":
    main()
