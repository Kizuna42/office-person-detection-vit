"""
PyTracking/DeepSORT ベースの簡易トラッキング PoC スクリプト。

目的:
 - 既存検出結果(JSON)とフレーム画像ディレクトリを入力し、track_id を付与した結果を出力する。
 - PyTracking がインストールされていればバックエンドに切り替え、未導入なら DeepSORT(既存 Tracker)を使用。

前提:
 - 検出 JSON フォーマット:
   [
     {"frame": 0, "timestamp": "2024-01-01 00:00:00", "detections": [
        {"bbox": [x, y, w, h], "score": 0.9}
     ]}
   ]
 - bbox 座標系は元フレームのピクセル。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

try:
    # PyTracking は任意依存。未インストールなら ImportError を握りつぶす。
    import importlib

    importlib.import_module("pytracking")
    _HAS_PYTRACKING = True
except Exception:  # pragma: no cover - 任意依存
    _HAS_PYTRACKING = False

from src.models import Detection
from src.tracking import Tracker as DeepSortTracker


def load_detections(json_path: Path) -> list[tuple[int, str, list[Detection]]]:
    """検出 JSON を読み込み、フレームごとに Detection を生成する。"""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    results: list[tuple[int, str, list[Detection]]] = []
    for frame_item in raw:
        frame_id = int(frame_item.get("frame", 0))
        ts = str(frame_item.get("timestamp", ""))
        detections: list[Detection] = []
        for det in frame_item.get("detections", []):
            bbox = det.get("bbox") or det.get("bbox_xywh")
            if not bbox or len(bbox) != 4:
                continue
            score = float(det.get("score", det.get("confidence", 0.0)))
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=score,
                    class_id=int(det.get("class_id", 0)),
                    class_name=str(det.get("class_name", "person")),
                    camera_coords=(0.0, 0.0),
                )
            )
        results.append((frame_id, ts, detections))
    return results


def run_deepsort(detections: list[tuple[int, str, list[Detection]]]) -> list[tuple[int, str, list[Detection]]]:
    """既存 DeepSORT トラッカーで track_id を付与。"""
    tracker = DeepSortTracker()
    tracked = []
    for _, ts, dets in detections:
        tracked_dets = tracker.update(dets)
        tracked.append((_, ts, tracked_dets))
    return tracked


def run_pytracking_stub(
    detections: list[tuple[int, str, list[Detection]]],
    tracker_name: str,
    tracker_params: str,
) -> list[tuple[int, str, list[Detection]]]:
    """
    PyTracking でのマルチオブジェクト追跡はケースによりセットアップが異なるため、
    ここでは API 呼び出しの最小骨組みのみ提供する。

    実際の利用時は、PyTracking が提供する MOT 拡張（TaMOs/RTS）に合わせて
    この関数を拡張することを想定。
    """
    if not _HAS_PYTRACKING:
        raise RuntimeError("PyTracking がインストールされていません。pip install pytracking==0.5 を実行してください。")

    # 参考: Tracker(tracker_name, tracker_params) は SOT 用。MOT 用は別途パラメータが必要。
    # ここでは DeepSORT 出力をそのまま返し、PyTracking 実装時の差し替えポイントだけ示す。
    print(
        "PyTracking バックエンドはサンプル実装のみです。"
        "TaMOs/RTS 用のパラメータを設定し、単一/複数対象の処理に合わせて拡張してください。"
    )
    return run_deepsort(detections)


def save_outputs(
    tracked: list[tuple[int, str, list[Detection]]],
    output_dir: Path,
    backend: str,
) -> None:
    """CSV/JSON 形式で保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"tracks_{backend}.json"
    csv_path = output_dir / f"tracks_{backend}.csv"
    mot_path = output_dir / f"tracks_{backend}_mot.csv"

    serializable = []
    for frame_id, ts, dets in tracked:
        for det in dets:
            serializable.append(
                {
                    "frame": frame_id,
                    "timestamp": ts,
                    "track_id": det.track_id,
                    "bbox": list(det.bbox),
                    "score": det.confidence,
                }
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # 簡易 CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,timestamp,track_id,x,y,w,h,score\n")
        for item in serializable:
            bbox = item["bbox"]
            f.write(
                f"{item['frame']},{item['timestamp']},{item['track_id']},"
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{item['score']}\n"
            )

    with open(mot_path, "w", encoding="utf-8") as f:
        f.write("frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n")
        for item in serializable:
            if item["track_id"] is None:
                continue
            bbox = item["bbox"]
            f.write(
                f"{item['frame']},{item['track_id']},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},"
                f"{item['score']},-1,-1,-1\n"
            )

    print(f"[OK] 出力: {json_path}, {csv_path}, {mot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTracking / DeepSORT PoC runner")
    parser.add_argument("--detections", type=Path, required=True, help="検出結果 JSON パス")
    parser.add_argument("--backend", choices=["deepsort", "pytracking"], default="deepsort", help="使用バックエンド")
    parser.add_argument("--tracker-name", default="rts", help="PyTracking tracker 名 (例: rts)")
    parser.add_argument("--tracker-params", default="default", help="PyTracking パラメータ名")
    parser.add_argument("--output-dir", type=Path, default=Path("output/poc_tracking"), help="出力ディレクトリ")
    args = parser.parse_args()

    dets = load_detections(args.detections)
    start = time.time()

    if args.backend == "pytracking":
        tracked = run_pytracking_stub(dets, args.tracker_name, args.tracker_params)
    else:
        tracked = run_deepsort(dets)

    elapsed = time.time() - start
    fps = len(dets) / elapsed if elapsed > 0 else 0.0
    print(f"[INFO] backend={args.backend}, frames={len(dets)}, elapsed={elapsed:.2f}s, fps={fps:.2f}")
    save_outputs(tracked, args.output_dir, args.backend)


if __name__ == "__main__":
    main()
