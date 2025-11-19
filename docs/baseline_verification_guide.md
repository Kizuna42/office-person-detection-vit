# ベースライン検証ガイド

最新セッションに対して `run_baseline.py` と `evaluate_baseline.py` を組み合わせて実行し、検出/追跡/座標変換/パフォーマンスを一括で定量化するための手順をまとめる。
作業端末のカレントディレクトリは常に `/Users/kizuna/Aeterlink/yolo3` を想定する。

---

## 1. 前提条件

- `input/merged_moviefiles.mov` が存在し、`config.yaml` の `video.input_path` が当該ファイルを指していること。
- `config.yaml` には最新の `floormap`, `homography`, `tracking`, `timestamp` 設定が反映されていること。
- Ground Truth トラックと対応点を用意する場合:
  - `data/gt_tracks_auto.json`（もしくは手動で整備した `gt_tracks.json`）
  - `output/calibration/correspondence_points_cam01.json`
- Python 依存関係（`make setup-deps`）と `tesseract` がインストール済みであること。

---

## 2. セッション生成 (`run_baseline.py`)

```bash
python scripts/run_baseline.py --config config.yaml --tag baseline-20251119
```

- `output/sessions/<SESSION_ID>/` が作成され、従来の `phase1_extraction`〜`phase5_visualization` を含む。
- 付随して `baseline_info.json` が生成され、使用した config とタグが記録される。

> 既存のセッション（例: `output/sessions/20251119_132130`）を再利用する場合、このステップはスキップ可能。

---

## 3. 統合評価 (`evaluate_baseline.py`)

代表的な実行例:

```bash
python scripts/evaluate_baseline.py \
  --session 20251119_132130 \
  --config config.yaml \
  --gt data/gt_tracks_auto.json \
  --points output/calibration/correspondence_points_cam01.json
```

- **MOT メトリクス** → `output/sessions/<SESSION_ID>/mot_metrics.json`
- **再投影誤差** → `output/sessions/<SESSION_ID>/reprojection_error.json` と `reprojection_error_map.png`
- **パフォーマンス** → `output/sessions/<SESSION_ID>/performance_metrics.json`
- **統合結果** → `output/sessions/<SESSION_ID>/baseline_metrics.json`

オプション:

- `--gt` または `--points` を省略した場合、それぞれの指標は `available=false` でスキップされる。
- `scripts/evaluate_baseline.py --help` で追加引数を参照。

---

## 4. `baseline_metrics.json` の読み方

最新セッション `20251119_132130` を用いた実行結果例:

```json
{
  "mot_metrics": { "MOTA": 0.0, "IDF1": 1.0, "available": true },
  "reprojection_error": {
    "mean_error": 63.75,
    "max_error": 545.65,
    "available": true
  },
  "performance": {
    "time_per_frame_seconds": 9.82,
    "memory_increase_mb": 749.69,
    "phase_times": {
      "phase1_extraction": 32.76,
      "phase2_detection": 3.41,
      "phase2.5_tracking": 2.59,
      "phase3_transform": 0.01,
      "phase4_aggregation": 0.00,
      "phase5_visualization": 0.50
    }
  },
  "targets": {
    "MOTA": 0.7,
    "IDF1": 0.8,
    "mean_error": 2.0,
    "max_error": 4.0,
    "time_per_frame": 2.0,
    "memory_mb": 12288
  },
  "achieved": {
    "MOTA": false,
    "IDF1": true,
    "mean_error": false,
    "max_error": false,
    "time_per_frame": false,
    "memory": true
  }
}
```

- `performance.time_per_frame_seconds` は `phase_times.total / num_frames` で計算される。
  例では 4 フレーム処理で **9.82 秒/フレーム** と目標 (2.0s) を大きく超過している。
- `reprojection_error` の mean/max は `correspondence_points_cam01.json` と `homography.matrix` の品質に依存する。
  現状 (mean 63.75px, max 545.65px) は仮の対応点による結果であり、実測データでの再キャリブレーションが必要。
- `mot_metrics` は `data/gt_tracks_auto.json` を基準に算出している。
  自動生成 GT ではオクルージョンが考慮されていないため、MOTA=0 となっている。実 GT を用意する場合は `--gt` を差し替えること。

---

## 5. 出力ファイルの所在一覧

| ファイル | 役割 | 備考 |
| --- | --- | --- |
| `output/sessions/<SESSION_ID>/mot_metrics.json` | MOTA / IDF1 / ID Switch の定量結果 | `frame_count`, `num_gt_tracks` も記録 |
| `output/sessions/<SESSION_ID>/reprojection_error.json` | 再投影誤差統計と各点の誤差リスト | 併せて `reprojection_error_map.png` で空間分布を可視化 |
| `output/sessions/<SESSION_ID>/performance_metrics.json` | フェーズ別処理時間・メモリ使用量 | `targets` と `achieved` を含む |
| `output/sessions/<SESSION_ID>/baseline_metrics.json` | 上記すべてを集約した統合レポート | `achieved.*` で閾値達成状況を判定 |

---

## 6. トラブルシューティング

- **パフォーマンス測定が `ValueError: too many values to unpack` で停止する**
  `scripts/measure_performance.py` を最新版に更新する。`sample_frames = orchestrator.prepare_frames_for_detection(...)` を挟み、`run_transform`/`run_aggregation` の戻り値を分解してから次のフェーズへ渡す実装に修正済み。

- **MOTA が常に 0 になる**
  自動生成 GT (`data/gt_tracks_auto.json`) は参照レベルのため、実写フレームに合わせた `tracks.json` を手動で整備して `--gt` を差し替える。
  `mot_metrics.json` の `num_gt_tracks` が想定と合っているか確認する。

- **再投影誤差が極端に大きい**
  対応点に外れ値が混入している可能性が高い。`output/calibration/correspondence_points_cam01.json` を見直し、`tools/homography_calibrator.py --update-config` で再取得する。
  誤差マップ (`reprojection_error_map.png`) で誤差が集中する領域を特定して、追加の参照点を配置する。

- **処理時間/フレームが高い**
  `config.yaml` の `video.frame_interval_minutes` や `detection.batch_size` を調整し、測定時は `--max-frames` オプションで評価対象を絞る。

---

これらの手順に従うことで、`baseline_metrics.json` を起点にモジュールごとの改善前後比較ができる。
将来的に `make baseline` ターゲットを追加する際も、本ガイドのコマンドをそのままラップすればよい。***
