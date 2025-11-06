#!/usr/bin/env python3
"""Phase 1 frame sampling & timestamp OCR review utility.

This script executes the following steps:

1. Run frame sampling on the configured input video to obtain 5-minute
   interval frames (using the existing FrameSampler implementation).
2. Evaluate timestamp OCR accuracy under the baseline pipeline and several
   single-factor variations.
3. Generate detailed per-frame diagnostics (CSV), failure overlays, and
   experiment metadata for downstream analysis.

Outputs (relative to project root):

- ``output/diagnostics/experiments/phase1_results.csv``
- ``output/diagnostics/experiments/phase1_failures/`` (overlays & metadata)
- ``output/diagnostics/experiments/<experiment_id>_config.json``
- ``output/diagnostics/experiments/phase1_baseline_summary.json``

The script is intentionally opinionated so that it can be re-run deterministically.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.timestamp import TimestampExtractor
from src.timestamp.ocr_engines import (
    PADDLEOCR_AVAILABLE,
    EASYOCR_AVAILABLE,
    run_ocr,
    run_tesseract,
)
from src.timestamp.timestamp_postprocess import parse_flexible_timestamp
from src.utils.image_utils import create_timestamp_overlay
from src.video import FrameSampler, VideoProcessor
from src.detection.preprocessing import apply_pipeline


def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=None)


def _timestamp() -> str:
    return _now_utc().strftime("%Y%m%dT%H%M%SZ")


def round_to_nearest_five_minutes(dt: datetime) -> datetime:
    """Round a datetime to the nearest 5-minute mark."""

    base = dt.replace(second=0, microsecond=0)
    seconds_since_hour = dt.minute * 60 + dt.second
    lower = (seconds_since_hour // 300) * 300
    upper = lower + 300

    if abs(seconds_since_hour - lower) <= abs(upper - seconds_since_hour):
        target = lower
    else:
        target = upper

    # ``target`` is seconds elapsed since start of the hour
    rounded = base.replace(minute=0) + timedelta(seconds=target)
    return rounded


def floor_to_five_minutes(dt: datetime) -> datetime:
    """Floor a datetime to the nearest lower 5-minute boundary."""

    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def compute_expected_schedule(actuals: Sequence[datetime]) -> List[datetime]:
    if not actuals:
        return []

    start = floor_to_five_minutes(actuals[0])
    end = floor_to_five_minutes(actuals[-1])

    # Ensure end is >= start
    if end < start:
        end = start

    expected = []
    current = start
    while current <= end + timedelta(seconds=1):
        expected.append(current)
        current += timedelta(minutes=5)

    return expected


def deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively merge dictionaries (without modifying inputs)."""

    result = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def compute_digit_accuracy(
    expected: Sequence[str],
    predictions: Sequence[Optional[str]],
) -> Dict[str, float]:
    """Compute per-digit accuracy (positions 1-14 over YYYYMMDDHHMMSS)."""

    totals = Counter()
    correct = Counter()

    for exp, pred in zip(expected, predictions):
        exp_digits = "".join(ch for ch in exp if ch.isdigit())
        pred_digits = "".join(ch for ch in (pred or "") if ch.isdigit())

        for idx, exp_digit in enumerate(exp_digits):
            totals[idx] += 1
            if idx < len(pred_digits) and pred_digits[idx] == exp_digit:
                correct[idx] += 1

    accuracy = {}
    for idx in range(14):
        if totals[idx]:
            accuracy[f"digit_{idx+1:02d}"] = correct[idx] / totals[idx]
        else:
            accuracy[f"digit_{idx+1:02d}"] = 0.0

    return accuracy


def estimate_rotation_angle(binary_image: np.ndarray) -> float:
    """Estimate rotation angle from a binary image via min-area rectangle."""

    try:
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return 0.0

        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        if angle < -45:
            angle += 90
        return float(abs(angle))
    except Exception:
        return 0.0


def classify_failure(
    raw_text: str,
    normalized: Optional[str],
    expected: str,
    mean_intensity: float,
    std_intensity: float,
    rotation_deg: float,
) -> str:
    """Heuristic failure categorisation."""

    if normalized:
        if normalized != expected:
            exp_digits = "".join(ch for ch in expected if ch.isdigit())
            pred_digits = "".join(ch for ch in normalized if ch.isdigit())
            diff = sum(1 for a, b in zip(exp_digits, pred_digits) if a != b)
            diff += abs(len(exp_digits) - len(pred_digits))
            if diff <= 2:
                return "誤字"
            return "その他"
        return ""

    raw = (raw_text or "").strip()
    digits_only = "".join(ch for ch in raw if ch.isdigit())

    if not raw:
        if std_intensity < 18.0:
            return "低コントラスト"
        return "部分欠損"

    if "/" not in raw and ":" in raw:
        return "スラッシュ欠落"

    if len(digits_only) >= 12 and ("/" not in raw or " " not in raw):
        return "数字結合"

    if rotation_deg >= 2.5:
        return "回転"

    return "誤字"


@dataclass
class FrameRecord:
    frame_number: int
    actual_ts: datetime
    frame: np.ndarray


@dataclass
class FrameResult:
    frame_number: int
    actual_ts: datetime
    expected_ts: datetime
    raw_text: str
    normalized_text: Optional[str]
    confidence: float
    success: bool
    exact_match: bool
    fail_reason: str
    time_diff_seconds: float
    roi_mean: float
    roi_std: float
    rotation_deg: float


def run_single_extraction(
    record: FrameRecord,
    extractor: TimestampExtractor,
    preproc_params: Dict,
    engine: str,
    engine_params: Dict,
    reference_timestamp: Optional[datetime],
    tolerance_seconds: float,
) -> FrameResult:
    frame = record.frame
    x, y, w, h = extractor.roi
    roi = frame[y : y + h, x : x + w]

    if roi.size == 0:
        return FrameResult(
            frame_number=record.frame_number,
            actual_ts=record.actual_ts,
            expected_ts=round_to_nearest_five_minutes(record.actual_ts),
            raw_text="",
            normalized_text=None,
            confidence=0.0,
            success=False,
            exact_match=False,
            fail_reason="ROI空領域",
            time_diff_seconds=0.0,
            roi_mean=0.0,
            roi_std=0.0,
            rotation_deg=0.0,
        )

    pipeline_config = copy.deepcopy(preproc_params) if preproc_params else copy.deepcopy(extractor.preproc_params)
    preprocessed = apply_pipeline(roi, pipeline_config)

    raw_text: Optional[str]
    raw_conf: float

    if engine == "tesseract" and not engine_params:
        raw_text, raw_conf = run_tesseract(preprocessed)
    else:
        raw_text, raw_conf = run_ocr(preprocessed, engine=engine, **engine_params)

    normalized = parse_flexible_timestamp(
        raw_text or "",
        confidence=raw_conf,
        reference_timestamp=reference_timestamp,
    )

    if normalized is None:
        normalized, _ = extractor.extract_with_confidence(
            frame,
            preproc_params=preproc_params,
        )
        confidence = raw_conf if raw_conf > 0 else 0.0
    else:
        confidence = raw_conf

    expected_ts = round_to_nearest_five_minutes(record.actual_ts)
    time_diff = abs((record.actual_ts - expected_ts).total_seconds())
    success = bool(normalized) and time_diff <= tolerance_seconds
    exact_match = bool(normalized) and normalized == expected_ts.strftime("%Y/%m/%d %H:%M:%S")

    if preprocessed.ndim == 2:
        roi_gray = preprocessed
    else:
        roi_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    roi_mean = float(np.mean(roi_gray))
    roi_std = float(np.std(roi_gray))
    rotation_deg = estimate_rotation_angle(preprocessed)

    fail_reason = ""
    if not success:
        fail_reason = classify_failure(
            raw_text or "",
            normalized,
            expected_ts.strftime("%Y/%m/%d %H:%M:%S"),
            roi_mean,
            roi_std,
            rotation_deg,
        )

    return FrameResult(
        frame_number=record.frame_number,
        actual_ts=record.actual_ts,
        expected_ts=expected_ts,
        raw_text=raw_text or "",
        normalized_text=normalized,
        confidence=confidence,
        success=success,
        exact_match=exact_match,
        fail_reason=fail_reason,
        time_diff_seconds=time_diff,
        roi_mean=roi_mean,
        roi_std=roi_std,
        rotation_deg=rotation_deg,
    )


def _cache_dir(config: ConfigManager) -> Path:
    base = Path(config.get("output.directory", "output")) / "diagnostics" / "cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_sample_frames(
    config: ConfigManager,
    start_time: Optional[str],
    end_time: Optional[str],
    scan_interval: int,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Tuple[List[FrameRecord], Dict[int, Dict[str, object]]]:
    video_path = config.get("video.input_path")
    interval_minutes = config.get("video.frame_interval_minutes", 5)
    tolerance_seconds = config.get("video.tolerance_seconds", 10)

    cache_path = _cache_dir(config) / "phase1_frames_cache.json"
    cache_payload: Optional[Dict[str, object]] = None

    if use_cache and not refresh_cache and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if (
            data.get("video_path") == video_path
            and data.get("interval_minutes") == interval_minutes
            and data.get("tolerance_seconds") == tolerance_seconds
            and data.get("scan_interval") == scan_interval
            and data.get("start_time") == start_time
            and data.get("end_time") == end_time
        ):
            cache_payload = data

    if cache_payload:
        processor = VideoProcessor(video_path)
        processor.open()

        frames_meta = cache_payload.get("frames", [])
        records: List[FrameRecord] = []
        for entry in frames_meta:
            frame_number = int(entry["frame_number"])
            timestamp_str = entry["timestamp"]
            frame = processor.get_frame(frame_number)
            if frame is None:
                continue
            actual_dt = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
            records.append(
                FrameRecord(
                    frame_number=frame_number,
                    actual_ts=actual_dt,
                    frame=frame,
                )
            )

        diagnostics_data = cache_payload.get("diagnostics", {})
        diagnostics: Dict[int, Dict[str, object]] = {}
        for key, value in diagnostics_data.items():
            try:
                idx = int(key)
            except ValueError:
                continue
            diag_entry = dict(value)
            ts_val = diag_entry.get("timestamp")
            if ts_val and not isinstance(ts_val, str):
                diag_entry["timestamp"] = str(ts_val)
            diagnostics[idx] = diag_entry

        processor.release()

        if not records:
            raise RuntimeError("キャッシュからサンプルフレームを復元できませんでした")

        return records, diagnostics

    processor = VideoProcessor(video_path)
    processor.open()

    extractor = TimestampExtractor()
    sampler = FrameSampler(interval_minutes, tolerance_seconds)

    samples = sampler.extract_sample_frames(
        processor,
        extractor,
        start_time=start_time,
        end_time=end_time,
        scan_interval=scan_interval,
    )

    if not samples:
        processor.release()
        raise RuntimeError(
            "タイムスタンプOCRからサンプルフレームを取得できませんでした。ROIやOCR設定を確認してください。"
        )

    records: List[FrameRecord] = []
    for frame_num, ts_str, frame in samples:
        actual_dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
        records.append(
            FrameRecord(
                frame_number=frame_num,
                actual_ts=actual_dt,
                frame=frame,
            )
        )

    diagnostics = copy.deepcopy(getattr(sampler, "_scan_diagnostics", {}))

    if use_cache:
        frames_meta = [
            {"frame_number": rec.frame_number, "timestamp": rec.actual_ts.strftime("%Y/%m/%d %H:%M:%S")}
            for rec in records
        ]
        diagnostics_serialisable = {}
        for key, value in diagnostics.items():
            diag_entry = dict(value)
            ts_val = diag_entry.get("timestamp")
            if isinstance(ts_val, datetime):
                diag_entry["timestamp"] = ts_val.strftime("%Y/%m/%d %H:%M:%S")
            diagnostics_serialisable[str(key)] = diag_entry

        cache_payload = {
            "video_path": video_path,
            "interval_minutes": interval_minutes,
            "tolerance_seconds": tolerance_seconds,
            "scan_interval": scan_interval,
            "start_time": start_time,
            "end_time": end_time,
            "frames": frames_meta,
            "diagnostics": diagnostics_serialisable,
            "generated_at": _now_utc().isoformat() + "Z",
        }

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    processor.release()

    return records, diagnostics


def save_results_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "frame",
        "extracted_timestamp_raw",
        "extracted_timestamp_normalized",
        "expected_timestamp",
        "success_bool",
        "ocr_confidence",
        "fail_reason",
        "experiment_id",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_failure_artifacts(
    output_dir: Path,
    extractor: TimestampExtractor,
    results: Sequence[FrameResult],
    frames: Sequence[FrameRecord],
    experiment_id: str,
) -> Dict[str, List[Dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_map = {rec.frame_number: rec for rec in frames}
    clusters: Dict[str, List[FrameResult]] = defaultdict(list)
    for res in results:
        if not res.success:
            clusters[res.fail_reason or "未分類"].append(res)

    saved_metadata: Dict[str, List[Dict[str, object]]] = {}
    for category, items in clusters.items():
        saved_metadata[category] = []
        for res in items[:5]:
            frame_record = frame_map[res.frame_number]
            overlay = create_timestamp_overlay(
                frame_record.frame,
                extractor.roi,
                res.normalized_text,
                res.confidence,
                res.frame_number,
            )

            summary_text = (
                f"Expected: {res.expected_ts.strftime('%H:%M:%S')}"
                f" | Δt={res.time_diff_seconds:.2f}s"
            )
            cv2.putText(
                overlay,
                summary_text,
                (10, overlay.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                f"Reason: {category or 'N/A'}",
                (10, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            overlay_name = (
                f"frame_{res.frame_number:06d}_{category or 'NA'}_overlay.png"
            )
            overlay_path = output_dir / overlay_name
            cv2.imwrite(str(overlay_path), overlay)

            meta = {
                "frame": res.frame_number,
                "category": category,
                "experiment_id": experiment_id,
                "actual_timestamp": res.actual_ts.strftime("%Y/%m/%d %H:%M:%S"),
                "expected_timestamp": res.expected_ts.strftime("%Y/%m/%d %H:%M:%S"),
                "raw_text": res.raw_text,
                "normalized_text": res.normalized_text,
                "confidence": res.confidence,
                "time_diff_seconds": res.time_diff_seconds,
                "roi_mean": res.roi_mean,
                "roi_std": res.roi_std,
                "rotation_deg": res.rotation_deg,
            }

            meta_name = overlay_name.replace("_overlay.png", "_meta.json")
            with (output_dir / meta_name).open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            saved_metadata[category].append(meta)

    return saved_metadata


def run_experiments(
    frames: Sequence[FrameRecord],
    tolerance_seconds: float,
    base_extractor: TimestampExtractor,
) -> Tuple[List[FrameResult], Dict[str, Dict[str, object]]]:
    experiments_config: Dict[str, Dict[str, object]] = {}

    baseline_params = copy.deepcopy(base_extractor.preproc_params)

    experiments = [
        {
            "id": "baseline",
            "label": "Baseline (CLAHE+Resize+Otsu)",
            "preproc": baseline_params,
            "engine": "tesseract",
            "engine_params": {},
        },
        {
            "id": "clahe_off",
            "label": "CLAHE disabled",
            "preproc": deep_update(
                baseline_params,
                {"clahe": {"enabled": False}},
            ),
            "engine": "tesseract",
            "engine_params": {},
        },
        {
            "id": "adaptive_threshold",
            "label": "Adaptive threshold (block=17, C=2)",
            "preproc": deep_update(
                baseline_params,
                {
                    "threshold": {
                        "enabled": True,
                        "method": "adaptive",
                        "block_size": 17,
                        "C": 2,
                    }
                },
            ),
            "engine": "tesseract",
            "engine_params": {},
        },
        {
            "id": "deskew_on",
            "label": "Deskew enabled (max_angle=4)",
            "preproc": deep_update(
                baseline_params,
                {
                    "deskew": {
                        "enabled": True,
                        "max_angle": 4.0,
                    }
                },
            ),
            "engine": "tesseract",
            "engine_params": {},
        },
    ]

    if PADDLEOCR_AVAILABLE:
        experiments.append(
            {
                "id": "paddleocr",
                "label": "PaddleOCR engine",
                "preproc": baseline_params,
                "engine": "paddleocr",
                "engine_params": {},
            }
        )
    else:
        experiments.append(
            {
                "id": "paddleocr_unavailable",
                "label": "PaddleOCR (unavailable)",
                "preproc": baseline_params,
                "engine": "paddleocr",
                "engine_params": {},
            }
        )

    results_per_experiment: Dict[str, List[FrameResult]] = {}

    for exp in experiments:
        extractor = TimestampExtractor(
            roi=base_extractor.roi,
            preproc_params=copy.deepcopy(exp["preproc"]),
            ocr_params=copy.deepcopy(base_extractor.ocr_params),
        )

        exp_results: List[FrameResult] = []
        last_timestamp: Optional[datetime] = None
        for record in frames:
            frame_result = run_single_extraction(
                record,
                extractor,
                exp["preproc"],
                exp["engine"],
                exp["engine_params"],
                reference_timestamp=last_timestamp,
                tolerance_seconds=tolerance_seconds,
            )
            exp_results.append(frame_result)
            if frame_result.normalized_text:
                last_timestamp = datetime.strptime(
                    frame_result.normalized_text,
                    "%Y/%m/%d %H:%M:%S",
                )

        results_per_experiment[exp["id"]] = exp_results

        if exp_results:
            success_rate = sum(1 for r in exp_results if r.success) / len(exp_results)
            exact_rate = sum(1 for r in exp_results if r.exact_match) / len(exp_results)
            recognitions = [r.confidence for r in exp_results if r.normalized_text]
            avg_conf = sum(recognitions) / len(recognitions) if recognitions else 0.0
        else:
            success_rate = 0.0
            exact_rate = 0.0
            avg_conf = 0.0

        experiments_config[exp["id"]] = {
            "label": exp["label"],
            "engine": exp["engine"],
            "engine_params": exp["engine_params"],
            "success_rate": success_rate,
            "exact_match_rate": exact_rate,
            "average_confidence": avg_conf,
            "availability": {
                "paddleocr": PADDLEOCR_AVAILABLE,
                "easyocr": EASYOCR_AVAILABLE,
            },
        }

    baseline_results = results_per_experiment["baseline"]
    return baseline_results, experiments_config


def write_config(
    path: Path,
    experiment_id: str,
    config: ConfigManager,
    summary: Dict[str, object],
    experiments: Dict[str, Dict[str, object]],
    runtime_seconds: float,
) -> None:
    output = {
        "experiment_id": experiment_id,
        "generated_at_utc": _now_utc().isoformat() + "Z",
        "runtime_seconds": runtime_seconds,
        "config": {
            "video": {
                "input_path": config.get("video.input_path"),
                "frame_interval_minutes": config.get("video.frame_interval_minutes", 5),
                "tolerance_seconds_configured": config.get("video.tolerance_seconds", 10),
            },
            "output_directory": config.get("output.directory", "output"),
        },
        "summary": summary,
        "experiments": experiments,
        "random_seed": 42,
        "numpy_seed": 42,
        "opencl_device" : "",  # placeholder for completeness
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def summarise_baseline(results: Sequence[FrameResult]) -> Dict[str, object]:
    total = len(results)
    within_tolerance = sum(1 for r in results if r.success)
    exact_matches = sum(1 for r in results if r.exact_match)
    mean_conf = (
        sum(r.confidence for r in results if r.normalized_text)
        / max(1, sum(1 for r in results if r.normalized_text))
    )
    avg_time_diff = (
        sum(r.time_diff_seconds for r in results) / total if total else 0.0
    )

    expected_strings = [r.expected_ts.strftime("%Y/%m/%d %H:%M:%S") for r in results]
    predicted_strings = [r.normalized_text for r in results]
    digit_accuracy = compute_digit_accuracy(expected_strings, predicted_strings)

    categories = Counter(r.fail_reason or "" for r in results if not r.success)
    failure_breakdown = {
        cat or "成功": count / total for cat, count in categories.items()
    }

    return {
        "total_frames": total,
        "within_tolerance": within_tolerance,
        "exact_matches": exact_matches,
        "within_tolerance_rate": within_tolerance / total if total else 0.0,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "average_confidence": mean_conf,
        "average_time_diff_seconds": avg_time_diff,
        "digit_accuracy": digit_accuracy,
        "failure_breakdown": failure_breakdown,
    }


def build_csv_rows(
    results: Sequence[FrameResult],
    experiment_id: str,
) -> List[Dict[str, object]]:
    rows = []
    for res in results:
        rows.append(
            {
                "frame": res.frame_number,
                "extracted_timestamp_raw": res.raw_text,
                "extracted_timestamp_normalized": res.normalized_text or "",
                "expected_timestamp": res.expected_ts.strftime("%Y/%m/%d %H:%M:%S"),
                "success_bool": "true" if res.success else "false",
                "ocr_confidence": f"{res.confidence:.4f}",
                "fail_reason": res.fail_reason,
                "experiment_id": experiment_id,
            }
        )
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 review runner")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--experiment-id", default=None, help="Custom experiment ID")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Evaluation tolerance in seconds (default: 10)",
    )
    parser.add_argument("--start-time", default=None, help="Optional start time HH:MM")
    parser.add_argument("--end-time", default=None, help="Optional end time HH:MM")
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=30,
        help="Scan interval (frames) for initial timestamp sweep (default: 30)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached frame metadata and always rescan video",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force regeneration of cached frame metadata",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    random.seed(42)
    np.random.seed(42)

    config = ConfigManager(args.config)

    experiment_id = args.experiment_id or f"phase1_{_timestamp()}"

    base_output_dir = Path("output/diagnostics/experiments")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_output_dir / "phase1_results.csv"
    failure_dir = base_output_dir / "phase1_failures"
    summary_path = base_output_dir / "phase1_baseline_summary.json"
    config_path = base_output_dir / f"{experiment_id}_config.json"

    start_time = time.perf_counter()

    use_cache = not args.no_cache or args.refresh_cache

    frames, diagnostics = load_sample_frames(
        config,
        start_time=args.start_time,
        end_time=args.end_time,
        scan_interval=max(1, args.scan_interval),
        use_cache=use_cache,
        refresh_cache=args.refresh_cache,
    )

    if not frames:
        raise RuntimeError("サンプルフレームが0件のため、実験を継続できません。")

    extractor = TimestampExtractor()
    baseline_results, experiments = run_experiments(
        frames,
        tolerance_seconds=args.tolerance,
        base_extractor=extractor,
    )

    csv_rows = build_csv_rows(baseline_results, experiment_id)
    save_results_csv(csv_path, csv_rows)

    failure_metadata = save_failure_artifacts(
        failure_dir,
        extractor,
        baseline_results,
        frames,
        experiment_id,
    )

    baseline_summary = summarise_baseline(baseline_results)
    baseline_summary["failure_examples"] = failure_metadata
    baseline_summary["diagnostics"] = diagnostics

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_summary, f, ensure_ascii=False, indent=2)

    runtime = time.perf_counter() - start_time

    write_config(
        config_path,
        experiment_id,
        config,
        baseline_summary,
        experiments,
        runtime_seconds=runtime,
    )

    print(f"Experiment ID: {experiment_id}")
    print(f"Baseline frames: {len(baseline_results)}")
    print(f"Results CSV: {csv_path}")
    print(f"Failure overlays: {failure_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Config JSON: {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

