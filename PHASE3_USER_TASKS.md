# Phase 3: ユーザーが行うタスク

Phase 3 の精度改善において、**ユーザーが手動で行う必要があるタスク**をまとめます。

## 🔧 タスク 2: 前処理パラメータの効果確認

### 実行手順

1. **前処理パラメータの A/B テストを実行**

   ```bash
   python tools/preprocessing_tuner.py --frame-indices "0,1000,2000,3000,4000"
   ```

2. **結果を確認**

   - `output/preprocessing_tuning/preprocessing_comparison.json` - 比較結果
   - `output/preprocessing_tuning/param_*_frame_*.jpg` - 各パラメータセットの前処理済み画像

3. **最良のパラメータを選択**

   - 平均信頼度が最も高いパラメータセットを確認
   - または、目視で最も読みやすい前処理結果を選択

4. **パラメータを適用**
   - `src/timestamp/roi_extractor.py` の `preprocess_roi` メソッドを更新
   - または、設定ファイル経由でパラメータを指定できるようにする（将来の拡張）

## 📊 タスク 3: Tesseract パラメータの調整（オプション）

### 確認事項

現在の設定（`config.yaml`）:

```yaml
ocr:
  tesseract:
    config: "--psm 7 --oem 3"
    whitelist: "0123456789/:  "
```

### 推奨テスト

異なる `--psm` モードを試行:

- `--psm 6`: 単一の均一なテキストブロック
- `--psm 7`: 単一のテキスト行（現在）
- `--psm 8`: 単一の単語
- `--psm 13`: 生の行（改行なし）

### 調整方法

1. `config.yaml` の `ocr.tesseract.config` を変更
2. タイムスタンプ抽出を再実行
3. 精度を比較

## 📝 まとめ

**ユーザーが行うタスク（優先順位順）:**

1. **🔴 最優先: ROI 座標の目視確認と調整**

   - ツール: `tools/roi_visualizer.py`
   - 出力: `output/roi_visualization/`
   - 確認: タイムスタンプがすべて ROI 矩形内に含まれているか

2. **🟡 中優先: 前処理パラメータの効果確認**

   - ツール: `tools/preprocessing_tuner.py`
   - 出力: `output/preprocessing_tuning/`
   - 確認: 最も読みやすい前処理結果を選択

3. **🟢 低優先: Tesseract パラメータの調整（必要に応じて）**
   - 手動で `config.yaml` を編集
   - 異なる `--psm` モードを試行

**自動化されたタスク:**

- ✅ ROI 可視化ツールの作成
- ✅ 前処理パラメータの A/B テストツールの作成
- ⏳ OCR エンジンの最適化（実装中）
- ⏳ コンセンサスアルゴリズムの改善（実装中）
- ⏳ 時系列検証の強化（実装中）
