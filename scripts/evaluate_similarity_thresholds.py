#!/usr/bin/env python3
"""Evaluate similarity thresholds for merging tracklets.

This script loads tracking結果 (`tracks.json`) とラベル付き tracklet ペア
(`similarity_pairs.json` など) を読み込み、外観/挙動に基づく類似度スコアを計算し
ROC / Precision-Recall 曲線と推奨閾値を出力する。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


def load_tracks(tracks_path: Path) -> dict[int, dict[str, Any]]:
    with tracks_path.open(encoding="utf-8") as f:
        data = json.load(f)
    tracks = data.get("tracks", data)
    return {int(track["track_id"]): track for track in tracks if "track_id" in track}


def load_pairs(pairs_path: Path) -> list[dict[str, Any]]:
    with pairs_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", data)


def load_embeddings(emb_path: Path | None) -> dict[int, np.ndarray]:
    if emb_path is None:
        return {}
    with emb_path.open(encoding="utf-8") as f:
        data = json.load(f)
    embeddings: dict[int, np.ndarray] = {}
    for key, value in data.items():
        try:
            embeddings[int(key)] = np.asarray(value, dtype=np.float32)
        except ValueError:
            continue
    return embeddings


def compute_motion_descriptor(track: dict[str, Any]) -> np.ndarray:
    trajectory = track.get("trajectory", [])
    if not trajectory:
        return np.zeros(6, dtype=np.float32)

    points = np.array([[pt.get("x", 0.0), pt.get("y", 0.0)] for pt in trajectory], dtype=np.float32)
    deltas = np.diff(points, axis=0) if len(points) > 1 else np.zeros((1, 2), dtype=np.float32)
    distances = np.linalg.norm(deltas, axis=1)

    mean_pos = points.mean(axis=0)
    std_pos = points.std(axis=0)
    total_disp = points[-1] - points[0]
    mean_speed = distances.mean() if len(distances) else 0.0

    return np.concatenate([mean_pos, std_pos, total_disp, [mean_speed]]).astype(np.float32)


def motion_similarity(desc_a: np.ndarray, desc_b: np.ndarray, epsilon: float = 1e-6) -> float:
    distance = np.linalg.norm(desc_a - desc_b)
    return 1.0 / (1.0 + distance + epsilon)


def appearance_similarity(emb_a: np.ndarray | None, emb_b: np.ndarray | None) -> float:
    if emb_a is None or emb_b is None:
        return 0.0
    denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
    if denom == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / denom)


def evaluate_pairs(
    tracks: dict[int, dict[str, Any]],
    pairs: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    appearance_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []

    motion_cache: dict[int, np.ndarray] = {}

    for pair in pairs:
        track_id_a = int(pair["track_a"])
        track_id_b = int(pair["track_b"])
        label = int(pair.get("label", pair.get("same_identity", 0)))

        track_a = tracks.get(track_id_a)
        track_b = tracks.get(track_id_b)
        if track_a is None or track_b is None:
            logger.warning("Track %s or %s not found; skipping pair.", track_id_a, track_id_b)
            continue

        desc_a = motion_cache.setdefault(track_id_a, compute_motion_descriptor(track_a))
        desc_b = motion_cache.setdefault(track_id_b, compute_motion_descriptor(track_b))

        motion_sim = motion_similarity(desc_a, desc_b)
        app_sim = appearance_similarity(embeddings.get(track_id_a), embeddings.get(track_id_b))

        combined = appearance_weight * app_sim + (1 - appearance_weight) * motion_sim

        scores.append(combined)
        labels.append(label)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


def summarize_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if len(scores) == 0:
        return {"error": "no_valid_pairs"}

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    f1_scores = []
    # precision is one element longer than recall, so we align them
    for p, r in zip(precision[:-1], recall, strict=False):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    best_f1_idx = int(np.argmax(f1_scores))

    metrics = {
        "num_pairs": len(scores),
        "positive_pairs": int(labels.sum()),
        "positive_ratio": float(labels.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": {
            "threshold": float(pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]),
            "f1": float(f1_scores[best_f1_idx]),
            "precision": float(precision[best_f1_idx]),
            "recall": float(recall[best_f1_idx]),
        },
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate similarity thresholds for tracklet merging.")
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("output/latest/phase2.5_tracking/tracks.json"),
        help="Path to tracks.json (default: output/latest/phase2.5_tracking/tracks.json)",
    )
    parser.add_argument("--pairs", type=Path, required=True, help="Path to labelled similarity pairs JSON")
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Optional path to aggregated appearance embeddings (JSON: {track_id: [vector]})",
    )
    parser.add_argument(
        "--appearance-weight",
        type=float,
        default=0.7,
        help="Weight for appearance similarity (0-1). Motion weight = 1 - appearance_weight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/similarity_metrics.json"),
        help="Output path for metrics JSON",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.tracks.exists():
        parser.error(f"tracks file not found: {args.tracks}")
    if not args.pairs.exists():
        parser.error(f"pairs file not found: {args.pairs}")

    tracks = load_tracks(args.tracks)
    pairs = load_pairs(args.pairs)
    embeddings = load_embeddings(args.embeddings)

    scores, labels = evaluate_pairs(tracks, pairs, embeddings, args.appearance_weight)
    metrics = summarize_metrics(scores, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Saved metrics to %s", args.output)
    logger.info(
        "ROC AUC: %.3f | PR AUC: %.3f | Best F1: %.3f (threshold=%.3f)",
        metrics.get("roc_auc", 0.0),
        metrics.get("pr_auc", 0.0),
        metrics.get("best_f1", {}).get("f1", 0.0),
        metrics.get("best_f1", {}).get("threshold", 0.0),
    )


if __name__ == "__main__":
    main()
