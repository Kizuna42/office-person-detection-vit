# キャリブレーション・座標変換ワークフロー

`config/calibration_template.yaml` を起点に、ホモグラフィ行列の推定と再投影誤差の検証までを一気通貫で行うための手順をまとめる。

---

## ステップ 0: テンプレートのコピー

```bash
cp config/calibration_template.yaml config/calibration_cam01.yaml
# または JSON 版
cp config/calibration_template.json config/calibration_cam01.json
```

- `camera.*`, `floormap.*`, `homography.matrix`, `calibration.*`, `zones[]` を現場の値で更新する。
- 参照: `docs/guides/camera_setup_checklist.md`

---

## ステップ 1: 対応点の収集

1. `tools/export_reference_frame.py` などで静止フレームを出力
2. `tools/homography_calibrator.py` を起動し、カメラ画像とフロアマップの対応点をクリック

```bash
python tools/homography_calibrator.py \
  --config config/calibration_cam01.yaml \
  --reference-image output/calibration/reference_cam01.png \
  --update-config
```

- `output/calibration/correspondence_points_cam01.json` と `homography_*.yaml` が生成される。
- `--update-config` を指定すると `config.yaml` の `homography.matrix` がバックアップ付きで更新される。

---

## ステップ 2: 再投影誤差の検証

1. 歪み補正 OFF/ON それぞれで再投影誤差を計測し、差分を比較する。

```bash
# OFF（config.yamlをそのまま使用）
python scripts/evaluate_reprojection_error.py \
  --points output/calibration/correspondence_points_cam01.json \
  --config config.yaml \
  --output output/calibration/reprojection_cam01_nodist.json \
  --error-map output/calibration/reprojection_cam01_nodist.png

# ON（一時的に use_distortion_correction=true にした設定を渡す）
python scripts/evaluate_reprojection_error.py \
  --points output/calibration/correspondence_points_cam01.json \
  --config output/calibration/config_use_distortion.yaml \
  --output output/calibration/reprojection_cam01_dist.json \
  --error-map output/calibration/reprojection_cam01_dist.png
```

2. 目標値: 平均誤差 ≤ 2 px、最大誤差 ≤ 4 px。超過している場合は対応点を増やすか再計測。
3. 結果は `docs/architecture/floormap_integration.md` のベンチマーク表に記録する。

---

## ステップ 3: ConfigManager での検証

```python
from src.config import ConfigManager

config = ConfigManager("config.yaml")
config.validate()  # camera/calibration/floormap/homography の必須項目が漏れていないか確認
```

- `camera` または `calibration` セクションが欠落している場合、テンプレート値が補完されるが必ず現場値に置き換える。
- `src/config/config_manager.py` は `reprojection_error_threshold` や内部パラメータの範囲を検証するため、ここでエラーが出ない状態まで整備する。

---

## ステップ 4: ベースライン実行で最終確認

```bash
python scripts/run_baseline.py --config config.yaml --tag calibration-check
python scripts/evaluate_baseline.py \
  --session <SESSION_ID> \
  --config config.yaml \
  --gt data/gt_tracks_auto.json \
  --points output/calibration/correspondence_points_cam01.json
```

- `baseline_metrics.json` に `mot_metrics`, `reprojection_error`, `performance` がまとまる。
- 再投影誤差が目標内か、`performance.time_per_frame_seconds` が許容範囲かを確認し、必要に応じて `homography` や `tracking` パラメータを微調整する。

---

## ステップ 5: ドキュメント更新

- キャリブ結果・誤差マップを `docs/architecture/floormap_integration.md` に追記
- 撮影メモ、対応点 JSON、更新済み `config/calibration_cam01.*` をリポジトリ or Notion などで共有

このワークフローを踏むことで、カメラごとの座標変換パラメータを再現性のある形で管理できる。\*\*\*
