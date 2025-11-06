# フェーズ 1 処理関連ファイル一覧

## 概要

フェーズ 1（フレームサンプリング）処理に関連するすべてのファイルとコード行数を列挙します。

## ファイル構成

### 1. メインエントリーポイント

- **`main.py`** (163 行)
  - フェーズ 1 呼び出し部分: 89-96 行（8 行）

### 2. パイプラインモジュール

- **`src/pipeline/frame_sampling_phase.py`** (72 行)
  - フレームサンプリングフェーズの統合クラス

### 3. 動画処理モジュール

- **`src/video/video_processor.py`** (211 行)
  - 動画ファイルの読み込み、フレーム取得、リソース管理
- **`src/video/frame_sampler.py`** (890 行)
  - 5 分刻みフレーム抽出、タイムスタンプスキャン、目標タイムスタンプ生成

### 4. タイムスタンプ抽出モジュール

- **`src/timestamp/timestamp_extractor.py`** (921 行)
  - OCR 実行、複数 PSM 多数決、タイムスタンプ正規化
- **`src/timestamp/ocr_engines.py`** (240 行)
  - Tesseract/PaddleOCR/EasyOCR 統合インターフェース
- **`src/timestamp/timestamp_postprocess.py`** (285 行)
  - 柔軟なタイムスタンプ正規化、候補生成、Levenshtein 距離補正

### 5. 前処理モジュール

- **`src/detection/preprocessing.py`** (357 行)
  - OCR 用画像前処理パイプライン（CLAHE、リサイズ、二値化、モルフォロジーなど）

### 6. モジュール初期化ファイル

- **`src/video/__init__.py`** (10 行)
- **`src/timestamp/__init__.py`** (24 行)
- **`src/pipeline/__init__.py`** (18 行)

## 行数集計

| カテゴリ             | ファイル                                 | 行数         |
| -------------------- | ---------------------------------------- | ------------ |
| メイン               | `main.py` (フェーズ 1 部分)              | 8 行         |
| パイプライン         | `src/pipeline/frame_sampling_phase.py`   | 72 行        |
| 動画処理             | `src/video/video_processor.py`           | 211 行       |
| フレームサンプリング | `src/video/frame_sampler.py`             | 890 行       |
| タイムスタンプ抽出   | `src/timestamp/timestamp_extractor.py`   | 921 行       |
| OCR エンジン         | `src/timestamp/ocr_engines.py`           | 240 行       |
| タイムスタンプ後処理 | `src/timestamp/timestamp_postprocess.py` | 285 行       |
| 前処理               | `src/detection/preprocessing.py`         | 357 行       |
| モジュール初期化     | `src/video/__init__.py`                  | 10 行        |
| モジュール初期化     | `src/timestamp/__init__.py`              | 24 行        |
| モジュール初期化     | `src/pipeline/__init__.py`               | 18 行        |
| **合計**             | **11 ファイル**                          | **3,046 行** |

## 処理フロー

```
main.py (89-96行)
  └─> FrameSamplingPhase.execute() (72行)
       ├─> VideoProcessor.open() (211行)
       ├─> TimestampExtractor (921行)
       │    ├─> apply_pipeline() (357行) - 前処理
       │    ├─> run_ocr() (240行) - OCR実行
       │    └─> parse_flexible_timestamp() (285行) - 後処理
       └─> FrameSampler.extract_sample_frames() (890行)
            ├─> _scan_all_timestamps() - 全フレームスキャン
            └─> find_target_timestamps() - 5分刻み目標生成
                 └─> find_closest_frame() - 最近接フレーム探索
```

## 依存関係

- **タイムスタンプ抽出が使用するモジュール**:

  - `src/detection/preprocessing` - 前処理パイプライン
  - `src/timestamp/ocr_engines` - OCR エンジン統合
  - `src/timestamp/timestamp_postprocess` - 柔軟な後処理

- **フレームサンプラーが使用するモジュール**:
  - `src/video/video_processor` - 動画ファイル操作
  - `src/timestamp/timestamp_extractor` - タイムスタンプ抽出

## 注意事項

- モジュール初期化ファイル（`__init__.py`）はエクスポート定義のみ
- 実際の処理コードは上記の主要モジュールに含まれる
- 合計行数は実装コード行数（コメント・空行含む）
