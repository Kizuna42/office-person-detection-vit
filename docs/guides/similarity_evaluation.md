# 同一オブジェクト類似度評価ガイド

DeepSORT で分割された tracklet を再結合するための閾値設計フローをまとめる。

---

## 1. ラベル付き tracklet ペアの作成

1. `output/sessions/<SESSION_ID>/phase2.5_tracking/tracks.json` をベースに、同一人物と異なる人物の tracklet ペアを手動ラベル付けする。
2. フォーマット例（`data/similarity_pairs_cam01.json`）:

```json
{
  "pairs": [
    { "track_a": 1, "track_b": 5, "label": 1 },
    { "track_a": 2, "track_b": 17, "label": 0 }
  ]
}
```

- `label=1` は同一人物、`0` は別人物。
- 100〜200 ペア以上を目標。

---

## 2. (任意) 外観特徴量の集約

- `phase2_detection` の ROI から切り出した特徴量を track 単位に平均化し、`{track_id: [embedding...]}` 形式で JSON 化する。
- ファイル例: `output/sessions/<SESSION_ID>/phase2.5_tracking/track_embeddings.json`

> 外観情報がない場合は、スクリプトが軌跡ベースのモーション類似度のみで評価を行う。

---

## 3. スコア計算と ROC / PR 評価

```bash
python scripts/evaluate_similarity_thresholds.py \
  --tracks output/sessions/20251119_132130/phase2.5_tracking/tracks.json \
  --pairs data/similarity_pairs_cam01.json \
  --embeddings output/sessions/20251119_132130/phase2.5_tracking/track_embeddings.json \
  --appearance-weight 0.7 \
  --output output/similarity_metrics_cam01.json
```

- `--appearance-weight`: 外観vsモーションの重み (0.0〜1.0)
- 出力例:
  - `roc_auc`, `pr_auc`
  - `best_f1.threshold`: F1 最大となる閾値
  - `roc_curve` / `pr_curve`: 可視化用にそのまま pandas / matplotlib に渡せるリスト

---

## 4. 閾値の決定と運用

1. `output/similarity_metrics_cam01.json` を開き、`best_f1.threshold` または PR 曲線で TPR / precision のバランスが良い点を採用。
2. `src/tracking/similarity.py` で使用している `SimilarityCalculator` に同閾値を適用し、tracklet マージ条件を実装（例: `score >= 0.82` でマージ）。
3. 更新後の tracking 結果を `scripts/evaluate_baseline.py` で再評価し、MOTA / IDF1 / false merge 件数の変化を確認する。

---

## 5. 今後の拡張アイデア

- `--auto-generate-negative` オプションを追加し、遠距離トラックの組み合わせから自動的にネガティブペアを抽出する。
- multi-camera 対応: `pairs` JSON に `camera_a`, `camera_b` を追加して、カメラ間 ReID の ROC を計測。
- Streamlit ダッシュボードから `scripts/evaluate_similarity_thresholds.py` を呼び出し、閾値候補を UI 上で確認できるようにする。
