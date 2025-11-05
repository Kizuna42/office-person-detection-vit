# タイムスタンプ OCR 実験システム - 使い方ガイド

## 概要

本システムは、タイムスタンプ OCR の精度向上を目的とした実験・評価フレームワークです。前処理パラメータと OCR 設定を柔軟に変更し、Ground Truth と比較して認識率を測定できます。

## 実装完了項目

### 1. 前処理パラメタ化 (`src/preprocessing.py`)

**機能**: OCR 精度向上のための画像前処理をパラメタ化

**利用可能な前処理**:

- `invert`: 画像反転（白文字 → 黒背景）
- `clahe`: コントラスト強調（CLAHE）
- `resize`: 画像リサイズ
- `threshold`: 二値化（Otsu / Adaptive）
- `blur`: ガウシアンブラー
- `unsharp`: アンシャープマスク（シャープ化）
- `morphology`: モルフォロジー変換（open/close）
- `deskew`: 傾き補正

**使用例**:

```python
from src.detection.preprocessing import apply_pipeline

preproc_params = {
    "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "resize": {"enabled": True, "fx": 2.0},
    "threshold": {"enabled": True, "method": "otsu"},
    "invert_after_threshold": {"enabled": True},
    "morphology": {"enabled": True, "operation": "close", "kernel_size": 2, "iterations": 1}
}

processed_image = apply_pipeline(roi_image, preproc_params)
```

### 2. OCR エンジン共通 IF (`src/ocr_engines.py`)

**機能**: Tesseract、PaddleOCR、EasyOCR を統一的に扱う

**対応エンジン**:

- `tesseract`: Tesseract OCR（デフォルト）
- `paddleocr`: PaddleOCR（オプショナル）
- `easyocr`: EasyOCR（オプショナル）

**使用例**:

```python
from src.timestamp.ocr_engines import run_ocr

# Tesseract
text, confidence = run_ocr(image, engine="tesseract", psm=7, whitelist="0123456789/:")

# PaddleOCR
text, confidence = run_ocr(image, engine="paddleocr")

# EasyOCR
text, confidence = run_ocr(image, engine="easyocr")
```

### 3. TimestampExtractor 更新 (`src/timestamp_extractor.py`)

**機能**: 前処理・OCR パラメータを受け取れるように拡張

**主な変更点**:

- `preproc_params`と`ocr_params`をコンストラクタで受け取れる
- `extract_with_confidence()`メソッドで信頼度付き抽出
- 複数 OCR 設定での多数決（`_multi_ocr_vote()`）

**使用例**:

```python
from src.timestamp import TimestampExtractor

extractor = TimestampExtractor(
    roi=(900, 30, 360, 45),
    preproc_params=preproc_params,
    ocr_params=ocr_params,
    use_flexible_postprocess=True
)

timestamp, confidence = extractor.extract_with_confidence(frame)
```

### 4. 柔軟な後処理 (`src/timestamp_postprocess.py`)

**機能**: 柔軟な正規表現、候補生成、Levenshtein 距離によるスコアリング

**主な機能**:

- `normalize_text()`: OCR テキストの正規化
- `generate_timestamp_candidates()`: 数字列からタイムスタンプ候補を生成
- `score_candidate()`: 候補のスコアリング
- `parse_flexible_timestamp()`: 柔軟なタイムスタンプ抽出

**使用例**:

```python
from src.timestamp.timestamp_postprocess import parse_flexible_timestamp
from datetime import datetime

reference = datetime(2025, 10, 8, 12, 0, 0)
timestamp = parse_flexible_timestamp(
    ocr_text="2025108/2616:05:26",
    confidence=0.8,
    reference_timestamp=reference
)
# 結果: "2025/10/08 16:05:26"
```

### 5. Ground Truth スケジュール生成 (`src/gt_schedule.py`)

**機能**: frame0 を基準に+10 秒（稀に+9 秒）のスケジュール生成

**主な関数**:

- `load_reference_timestamp()`: CSV から frame0 のタイムスタンプを取得
- `estimate_interval_map()`: 成功フレームの時刻差から 9 秒箇所を推定
- `generate_ground_truth_schedule()`: GT スケジュールを生成

**使用例**:

```python
from src.gt_schedule import generate_ground_truth_schedule

reference, schedule = generate_ground_truth_schedule(
    csv_path="output/diagnostics/simple_extract/frames_ocr.csv",
    num_frames=100,
    default_interval=10,
    estimate_9s=True,
    use_csv_timestamps=True  # CSVのタイムスタンプをGTとして使用
)
```

### 6. 実験ランナー (`tools/run_timestamp_experiment.py`)

**機能**: 実験実行、結果記録、失敗分析、McNemar 検定

**実行モード**:

- `run`: 実験を実行
- `analyze`: 失敗分析を実行
- `compare`: 2 つの実験を比較（McNemar 検定）

**使用例**:

#### 実験実行

```bash
python tools/run_timestamp_experiment.py \
    --mode run \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --config configs/experiments/exp-001.json \
    --experiment-id exp-001-baseline
```

#### 失敗分析

```bash
python tools/run_timestamp_experiment.py \
    --mode analyze \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --experiment-id exp-001-baseline
```

#### 実験比較

```bash
python tools/run_timestamp_experiment.py \
    --mode compare \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --compare exp-001-baseline exp-002-enhanced
```

### 7. 失敗分析機能

**機能**: クラスタリング、代表画像保存、割合算出

**失敗パターン**:

- `ocr_failed`: OCR が完全に失敗
- `slash_missing`: スラッシュ欠落
- `digit_concatenated`: 数字が連結
- `low_contrast`: コントラスト不足
- `skew`: 傾き
- `other`: その他

**出力**:

- `failure_analysis.json`: 失敗分析結果
- `failure_clusters/`: 各クラスタの代表画像（最大 5 枚）

### 8. レポート生成 (`tools/generate_report.py`)

**機能**: 実験一覧、ベスト設定、失敗 Top10、推奨アクション

**使用例**:

```bash
python tools/generate_report.py \
    --experiments output/diagnostics/experiments \
    --out output/diagnostics/experiments/report.md
```

**レポート内容**:

- 実験一覧（認識率順）
- ベスト設定（JSON 形式）
- 失敗 Top10（画像パス）
- 次の推奨アクション

## 実験設定ファイルの作成

実験設定は JSON 形式で記述します。`configs/experiments/`ディレクトリに配置してください。

### 設定例

#### exp-001.json（ベースライン）

```json
{
  "preprocessing": {
    "invert": { "enabled": true },
    "clahe": { "enabled": true, "clip_limit": 2.0, "tile_grid_size": [8, 8] },
    "resize": { "enabled": true, "fx": 2.0 },
    "threshold": { "enabled": true, "method": "otsu" },
    "invert_after_threshold": { "enabled": true },
    "morphology": {
      "enabled": true,
      "operation": "close",
      "kernel_size": 2,
      "iterations": 1
    }
  },
  "ocr": {
    "engine": "tesseract",
    "psm": 7,
    "whitelist": "0123456789/:",
    "lang": "jpn+eng",
    "oem": 3
  },
  "description": "invert + CLAHE + resize fx=2.0 + Otsu + Tesseract --psm 7 jpn+eng whitelist"
}
```

#### exp-002.json（PaddleOCR 使用）

```json
{
  "preprocessing": {
    "clahe": { "enabled": true, "clip_limit": 2.0, "tile_grid_size": [8, 8] },
    "resize": { "enabled": true, "fx": 3.0 },
    "threshold": {
      "enabled": true,
      "method": "adaptive",
      "block_size": 11,
      "C": 2
    },
    "morphology": {
      "enabled": true,
      "operation": "close",
      "kernel_size": 2,
      "iterations": 1
    }
  },
  "ocr": {
    "engine": "paddleocr"
  },
  "description": "CLAHE + adaptiveThreshold(block=11,C=2) + resize fx=3.0 + PaddleOCR default"
}
```

## ワークフロー

### 1. 準備

1. フレーム画像を準備（`output/diagnostics/simple_extract/frames/`）
2. Ground Truth CSV を準備（`output/diagnostics/simple_extract/frames_ocr.csv`）

### 2. 実験設定の作成

`configs/experiments/exp-XXX.json`を作成し、前処理・OCR パラメータを定義

### 3. 実験実行

```bash
python tools/run_timestamp_experiment.py \
    --mode run \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --config configs/experiments/exp-XXX.json \
    --experiment-id exp-XXX
```

### 4. 失敗分析

```bash
python tools/run_timestamp_experiment.py \
    --mode analyze \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --experiment-id exp-XXX
```

### 5. レポート生成

```bash
python tools/generate_report.py \
    --experiments output/diagnostics/experiments \
    --out output/diagnostics/experiments/report.md
```

## 出力ファイル構造

```
output/diagnostics/experiments/
├── exp-001-baseline/
│   ├── exp-001-baseline.json          # 実験設定
│   ├── exp-001-baseline_results.csv   # 結果CSV
│   ├── failure_analysis.json          # 失敗分析結果
│   ├── failures/                      # 失敗ROI画像
│   ├── overlays/                      # オーバーレイ画像
│   └── failure_clusters/              # クラスタ代表画像
├── exp-002-enhanced/
│   └── ...
└── report.md                          # 実験レポート
```

## 注意事項

### 現在の認識率が低い理由

現在の認識率が 1%と低いのは、**overlay 画像から ROI を抽出しているため**です。

**解決策**:

1. 実際のフレーム画像（`output/diagnostics/simple_extract/frames/`）を使用
2. 動画から直接フレームを抽出

### フレーム画像の準備

実験ランナーは以下の順序でフレーム画像を探します：

1. `frame_XXXXXX.png`（元のフレーム画像）
2. `frame_XXXXXX_overlay.png`（overlay 画像、フォールバック）
3. `frame_XXX.png`（短い形式）

**推奨**: 元のフレーム画像を使用してください。

## トラブルシューティング

### OCR エンジンがインストールされていない

```bash
# Tesseract（必須）
brew install tesseract  # macOS
# または apt-get install tesseract-ocr  # Linux

# PaddleOCR（オプション）
pip install paddleocr

# EasyOCR（オプション）
pip install easyocr

# Levenshtein距離計算（オプション）
pip install python-Levenshtein
```

### Ground Truth CSV が見つからない

`frames_ocr.csv`は以下の形式である必要があります：

```csv
frame_number,timestamp,recognized
0,2025/10/08 12:00:00,True
1,2025/10/08 12:00:10,True
...
```

### 実験結果が保存されない

- 出力ディレクトリの書き込み権限を確認
- フレーム画像が正しく読み込めているか確認（ログを確認）

## 次のステップ

1. **前処理パラメータの最適化**: グリッドサーチやベイズ最適化で最適パラメータを探索
2. **OCR エンジンの比較**: Tesseract、PaddleOCR、EasyOCR の性能比較
3. **アンサンブル**: 複数 OCR エンジンの結果を統合
4. **時系列補正**: 前後のフレーム情報を活用した補正

## 参考

- 設計書: `.kiro/specs/office-person-detection/design.md`
- 要件定義書: `.kiro/specs/office-person-detection/requirements.md`

