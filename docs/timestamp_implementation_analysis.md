# タイムスタンプ OCR 実験システム - 実装完了項目分析

## 実装概要

タイムスタンプ OCR の精度向上を目的とした実験・評価フレームワークが実装されました。前処理パラメータと OCR 設定を柔軟に変更し、Ground Truth と比較して認識率を測定できます。

## 実装完了項目の詳細分析

### 1. 前処理パラメタ化 (`src/preprocessing.py`)

#### 実装内容

- 8 種類の前処理をパラメタ化
- 各処理は`enabled`フラグで有効/無効を制御
- パイプライン形式で順次適用

#### 利用可能な前処理

| 処理名       | 機能               | 主要パラメータ                                        |
| ------------ | ------------------ | ----------------------------------------------------- |
| `invert`     | 画像反転           | `enabled`                                             |
| `clahe`      | コントラスト強調   | `clip_limit`, `tile_grid_size`                        |
| `resize`     | リサイズ           | `fx`, `fy`, `interpolation`                           |
| `threshold`  | 二値化             | `method` (otsu/adaptive), `block_size`, `C`           |
| `blur`       | ガウシアンブラー   | `kernel_size`, `sigma`                                |
| `unsharp`    | シャープ化         | `amount`, `radius`, `threshold`                       |
| `morphology` | モルフォロジー変換 | `operation` (open/close), `kernel_size`, `iterations` |
| `deskew`     | 傾き補正           | `max_angle`                                           |

#### 設計の特徴

- **モジュール性**: 各処理が独立した関数として実装
- **柔軟性**: パラメータを辞書形式で渡すことで、設定ファイルから制御可能
- **拡張性**: 新しい前処理を追加しやすい設計

#### 使用例

```python
preproc_params = {
    "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "resize": {"enabled": True, "fx": 2.0},
    "threshold": {"enabled": True, "method": "otsu"},
    "morphology": {"enabled": True, "operation": "close", "kernel_size": 2}
}
```

---

### 2. OCR エンジン共通 IF (`src/ocr_engines.py`)

#### 実装内容

- Tesseract、PaddleOCR、EasyOCR を統一的に扱うラッパー
- 各エンジンは`(text, confidence)`のタプルを返す統一インターフェース
- 遅延初期化でメモリ効率を向上

#### 対応エンジン

| エンジン  | 必須/オプション | 特徴                               |
| --------- | --------------- | ---------------------------------- |
| Tesseract | 必須            | デフォルト、PSM/whitelist 設定可能 |
| PaddleOCR | オプション      | 高精度、GPU 対応                   |
| EasyOCR   | オプション      | 多言語対応、GPU 対応               |

#### 設計の特徴

- **統一インターフェース**: `run_ocr(image, engine="tesseract", **kwargs)`で統一
- **エラーハンドリング**: エンジンが利用不可でもエラーにならず警告のみ
- **遅延初期化**: グローバルインスタンスをキャッシュしてメモリ効率化

#### 使用例

```python
# Tesseract（デフォルト）
text, conf = run_ocr(image, engine="tesseract", psm=7, whitelist="0123456789/:")

# PaddleOCR
text, conf = run_ocr(image, engine="paddleocr")

# EasyOCR
text, conf = run_ocr(image, engine="easyocr")
```

---

### 3. TimestampExtractor 更新 (`src/timestamp_extractor.py`)

#### 実装内容

- 前処理・OCR パラメータをコンストラクタで受け取れるように拡張
- `extract_with_confidence()`メソッドで信頼度付き抽出
- 複数 OCR 設定での多数決（`_multi_ocr_vote()`）

#### 主な変更点

| 機能     | 変更前       | 変更後                                |
| -------- | ------------ | ------------------------------------- |
| 前処理   | ハードコード | `preproc_params`で制御可能            |
| OCR 設定 | ハードコード | `ocr_params`で制御可能                |
| 信頼度   | 取得不可     | `extract_with_confidence()`で取得可能 |
| OCR 戦略 | 単一設定     | 複数 PSM 設定で多数決                 |

#### 多数決アルゴリズム

1. 複数の PSM 設定（7, 6, 8, 13, 11）で OCR 実行
2. 各結果を柔軟な後処理で正規化
3. 同じタイムスタンプの出現回数をカウント
4. 最も多く出現したタイムスタンプを選択（タイの場合は平均信頼度が高いものを選択）

#### 使用例

```python
extractor = TimestampExtractor(
    roi=(900, 30, 360, 45),
    preproc_params=preproc_params,
    ocr_params=ocr_params,
    use_flexible_postprocess=True
)

timestamp, confidence = extractor.extract_with_confidence(frame)
```

---

### 4. 柔軟な後処理 (`src/timestamp_postprocess.py`)

#### 実装内容

- 柔軟な正規表現パターンでタイムスタンプ抽出
- 数字列から候補生成
- Levenshtein 距離によるスコアリング

#### 主な機能

| 関数                              | 機能                                                 |
| --------------------------------- | ---------------------------------------------------- |
| `normalize_text()`                | OCR テキストの正規化（全角 → 半角、文字修正）        |
| `extract_digits()`                | 数字のみを抽出                                       |
| `generate_timestamp_candidates()` | 数字列からタイムスタンプ候補を生成                   |
| `score_candidate()`               | 候補のスコアリング（OCR 信頼度 + 妥当性 + 編集距離） |
| `parse_flexible_timestamp()`      | 柔軟なタイムスタンプ抽出                             |

#### スコアリング式

```
スコア = OCR_confidence × 0.6 + date_validity(0/1) × 0.4 - normed_edit_distance × 0.2
```

#### 設計の特徴

- **柔軟なパターンマッチ**: スラッシュ欠落、スペース欠落に対応
- **候補生成**: 数字列から複数の解釈を生成
- **参照タイムスタンプ**: 前のフレームのタイムスタンプを参照して妥当性チェック

#### 使用例

```python
from datetime import datetime

reference = datetime(2025, 10, 8, 12, 0, 0)
timestamp = parse_flexible_timestamp(
    ocr_text="2025108/2616:05:26",  # スラッシュ欠落
    confidence=0.8,
    reference_timestamp=reference
)
# 結果: "2025/10/08 16:05:26"
```

---

### 5. Ground Truth スケジュール生成 (`src/gt_schedule.py`)

#### 実装内容

- frame0 のタイムスタンプを基準に+10 秒（稀に+9 秒）のスケジュール生成
- CSV のタイムスタンプをそのまま GT として使用可能

#### 主な関数

| 関数                               | 機能                                    |
| ---------------------------------- | --------------------------------------- |
| `load_reference_timestamp()`       | CSV から frame0 のタイムスタンプを取得  |
| `estimate_interval_map()`          | 成功フレームの時刻差から 9 秒箇所を推定 |
| `generate_ground_truth_schedule()` | GT スケジュールを生成                   |

#### 間隔推定アルゴリズム

1. CSV から成功フレームのタイムスタンプを取得
2. 前回フレームとの時刻差を計算
3. 1 フレームあたりの平均間隔を計算
4. 8.5-9.5 秒なら 9 秒、9.5-10.5 秒なら 10 秒と判定

#### 使用例

```python
reference, schedule = generate_ground_truth_schedule(
    csv_path="frames_ocr.csv",
    num_frames=100,
    default_interval=10,
    estimate_9s=True,
    use_csv_timestamps=True  # CSVのタイムスタンプをGTとして使用
)
```

---

### 6. 実験ランナー (`tools/run_timestamp_experiment.py`)

#### 実装内容

- 実験実行、結果記録、失敗分析、McNemar 検定

#### 実行モード

| モード    | 機能                             |
| --------- | -------------------------------- |
| `run`     | 実験を実行し、結果を記録         |
| `analyze` | 失敗パターンを分析               |
| `compare` | 2 つの実験を比較（McNemar 検定） |

#### 実験実行フロー

1. 実験設定 JSON を読み込み
2. フレーム画像を順次処理
3. 前処理・OCR を実行
4. Ground Truth と比較
5. 結果を CSV に記録
6. 失敗時は ROI 画像とオーバーレイ画像を保存

#### 出力ファイル

- `{experiment_id}_results.csv`: 結果 CSV
- `{experiment_id}.json`: 実験設定
- `failures/`: 失敗 ROI 画像
- `overlays/`: オーバーレイ画像

#### 使用例

```bash
python tools/run_timestamp_experiment.py \
    --mode run \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --config configs/experiments/exp-001.json \
    --experiment-id exp-001-baseline
```

---

### 7. 失敗分析機能

#### 実装内容

- 失敗パターンをクラスタリング
- 代表画像を保存
- 割合を算出

#### 失敗パターン

| パターン             | 判定基準                           |
| -------------------- | ---------------------------------- |
| `ocr_failed`         | OCR が完全に失敗（空文字列）       |
| `slash_missing`      | スラッシュ欠落（数字列 12 桁以上） |
| `digit_concatenated` | 数字が連結（14 桁超）              |
| `low_contrast`       | コントラスト不足（将来実装）       |
| `skew`               | 傾き（将来実装）                   |
| `other`              | その他                             |

#### 出力

- `failure_analysis.json`: 失敗分析結果
- `failure_clusters/`: 各クラスタの代表画像（最大 5 枚）

#### 使用例

```bash
python tools/run_timestamp_experiment.py \
    --mode analyze \
    --input output/diagnostics/simple_extract/frames \
    --csv output/diagnostics/simple_extract/frames_ocr.csv \
    --output output/diagnostics/experiments \
    --experiment-id exp-001-baseline
```

---

### 8. レポート生成 (`tools/generate_report.py`)

#### 実装内容

- 実験一覧、ベスト設定、失敗 Top10、推奨アクション

#### レポート内容

| セクション     | 内容                                        |
| -------------- | ------------------------------------------- |
| 実験一覧       | 認識率順にソート、表形式で表示              |
| ベスト設定     | 最高認識率の実験設定（JSON 形式）           |
| 失敗 Top10     | 失敗画像のパス（上位 3 実験から最大 10 枚） |
| 推奨アクション | 認識率に応じた次のステップ                  |

#### 推奨アクションのロジック

- **90%以上**: 目標達成
- **70-90%**: 前処理パラメータの微調整、別 OCR エンジンの試行
- **70%未満**: 失敗分析の詳細確認、ROI 領域の再検討

#### 使用例

```bash
python tools/generate_report.py \
    --experiments output/diagnostics/experiments \
    --out output/diagnostics/experiments/report.md
```

---

## データフロー

```
フレーム画像
    ↓
[前処理パイプライン]
    ├─ invert
    ├─ CLAHE
    ├─ resize
    ├─ threshold
    ├─ morphology
    └─ deskew
    ↓
[OCRエンジン]
    ├─ Tesseract (複数PSM設定)
    ├─ PaddleOCR
    └─ EasyOCR
    ↓
[柔軟な後処理]
    ├─ 正規化
    ├─ 候補生成
    └─ スコアリング
    ↓
[多数決]
    ↓
タイムスタンプ + 信頼度
    ↓
[Ground Truth比較]
    ↓
結果記録（CSV）
```

## 現在の課題と解決策

### 課題 1: 認識率が 1%と低い

**原因**: overlay 画像から ROI を抽出しているため

**解決策**:

1. 実際のフレーム画像（`output/diagnostics/simple_extract/frames/`）を使用
2. 動画から直接フレームを抽出

### 課題 2: フレーム画像の準備

**現状**: 実験ランナーは以下の順序でフレーム画像を探します：

1. `frame_XXXXXX.png`（元のフレーム画像）← **推奨**
2. `frame_XXXXXX_overlay.png`（overlay 画像、フォールバック）
3. `frame_XXX.png`（短い形式）

**推奨**: 元のフレーム画像を使用してください。

## 今後の改善案

### 1. 前処理パラメータの最適化

- グリッドサーチで最適パラメータを探索
- ベイズ最適化で効率的に探索

### 2. OCR エンジンの比較

- Tesseract、PaddleOCR、EasyOCR の性能比較
- アンサンブル（複数エンジンの結果を統合）

### 3. 時系列補正

- 前後のフレーム情報を活用した補正
- カルマンフィルタによる平滑化

### 4. 深層学習モデル

- タイムスタンプ専用の CNN/Transformer モデル
- ファインチューニング

## まとめ

本システムは、タイムスタンプ OCR の精度向上を目的とした包括的な実験・評価フレームワークです。前処理パラメータと OCR 設定を柔軟に変更し、Ground Truth と比較して認識率を測定できます。

**主な特徴**:

- ✅ モジュール設計で拡張性が高い
- ✅ 設定ファイルから制御可能
- ✅ 失敗分析で問題点を特定
- ✅ レポート生成で結果を可視化

**次のステップ**:

1. 実際のフレーム画像を使用して実験を再実行
2. 前処理パラメータを最適化
3. 複数 OCR エンジンの性能比較
4. アンサンブル手法の検討

