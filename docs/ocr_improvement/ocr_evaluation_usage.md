# OCR 精度評価ツール 使い方ガイド

## 📋 概要

OCR 精度評価ツールは、予測 OCR 出力と参照（ゴールド）テキストを比較して、タイムスタンプ整合性と内容一致を評価します。

## 📁 必要なファイル

### 1. 予測 OCR 出力ファイル（`pred.json` または `pred.tsv`）

**JSON 形式の例:**

```json
[
  {
    "segment_id": "seg_001",
    "start_sec": 0.0,
    "end_sec": 10.0,
    "text": "2025/08/26 16:04:16",
    "ocr_confidence": 0.85
  },
  {
    "segment_id": "seg_002",
    "start_sec": 10.0,
    "end_sec": 20.0,
    "text": "2025/08/26 16:04:26",
    "ocr_confidence": 0.92
  }
]
```

または、辞書形式:

```json
{
  "segments": [
    {
      "segment_id": "seg_001",
      "start_sec": 0.0,
      "end_sec": 10.0,
      "text": "2025/08/26 16:04:16",
      "ocr_confidence": 0.85
    }
  ]
}
```

**TSV 形式の例:**

```tsv
segment_id	start_sec	end_sec	text	ocr_confidence
seg_001	0.0	10.0	2025/08/26 16:04:16	0.85
seg_002	10.0	20.0	2025/08/26 16:04:26	0.92
```

### 2. 参照（ゴールド）テキストファイル（`gold.json` または `gold.tsv`）

予測 OCR 出力と同じ形式で、正解データを用意します。

**重要**:

- `segment_id` が一致するレコード同士が比較されます
- `text` フィールドにはタイムスタンプ文字列（`"2025/08/26 16:04:16"`形式）を記載
- `ocr_confidence` はオプション（予測ファイルのみ）

## 🚀 実行手順

### ステップ 1: OCR 精度分析

予測 OCR 出力と参照テキストを比較して分析します。

```bash
python3 tools/video_timestamp_analyzer.py \
  --pred pred.json \
  --ref gold.json \
  --out analyzer_output.json \
  --tolerance 0.5 1.0 2.0 5.0
```

**引数説明:**

- `--pred`: 予測 OCR 出力ファイル（JSON または TSV）
- `--ref`: 参照（ゴールド）テキストファイル（JSON または TSV）
- `--out`: 出力 JSON ファイルパス（オプション、省略時はコンソール出力）
- `--tolerance`: タイムスタンプ許容ズレ（秒）のリスト（デフォルト: 0.5 1.0 2.0 5.0）

**出力:**

- `analyzer_output.json`: 分析結果（JSON 形式）

### ステップ 2: 詳細スコア計算とレポート生成

分析結果から詳細スコアを計算し、Markdown レポートを生成します。

```bash
python3 tools/timestamp_score_evaluator.py \
  --input analyzer_output.json \
  --tolerance 0.5 \
  --out final_report.json
```

**引数説明:**

- `--input`: 分析結果 JSON ファイルパス（ステップ 1 の出力）
- `--tolerance`: タイムスタンプ許容ズレ（秒、デフォルト: 0.5）
- `--out`: 出力 JSON ファイルパス（オプション、省略時はコンソール出力）

**出力:**

- `final_report.json`: 詳細スコアと分析結果（JSON 形式）
- `final_report.md`: 人間向けサマリレポート（Markdown 形式）

## 📊 出力内容

### analyzer_output.json の構造

```json
{
  "overall": {
    "total_pred_segments": 100,
    "total_ref_segments": 100,
    "matched_segments": 95,
    "timestamp_consistency": {
      "by_tolerance": {
        "0.5": {
          "within_tolerance": 90,
          "total_matched": 95,
          "rate": 0.9474,
          "avg_time_diff": 0.25,
          "min_time_diff": 0.0,
          "max_time_diff": 2.5
        }
      }
    },
    "content_accuracy": {
      "cer": {
        "cer": 0.05,
        "substitutions": 10,
        "insertions": 5,
        "deletions": 3
      },
      "wer": {
        "wer": 0.08,
        "substitutions": 8,
        "insertions": 4,
        "deletions": 2
      },
      "token_metrics": {
        "precision": 0.95,
        "recall": 0.92,
        "f1": 0.935
      }
    },
    "critical_errors": {
      "total": 5,
      "by_type": {
        "time_mismatch": 3,
        "digit_mismatch": 2
      }
    }
  },
  "by_segment": [
    {
      "segment_id": "seg_001",
      "ref_text": "2025/08/26 16:04:16",
      "pred_text": "2025/08/26 16:04:16",
      "matched": true,
      "time_diff_seconds": 0.0,
      "cer": 0.0,
      "wer": 0.0,
      "token_f1": 1.0
    }
  ],
  "error_samples": {
    "time_mismatch": [],
    "digit_mismatch": [],
    "text_mismatch": [],
    "missing_segment": []
  }
}
```

### final_report.md の内容

- **エグゼクティブサマリー**: 総合スコア、タイムスタンプ整合性、内容一致スコア
- **定量結果**: タイムスタンプ整合性、内容一致評価（CER, WER, Token F1）
- **改善優先度付き課題一覧**: エラー種別ごとの件数と優先度
- **サンプル表示**: 正解例・失敗例（それぞれ上位 10 件）
- **次のアクション提案**: スコアに基づいた改善提案

## 📝 サンプルファイルの作成

### サンプル 1: 最小限の JSON ファイル

**pred.json:**

```json
[
  {
    "segment_id": "seg_001",
    "start_sec": 0.0,
    "end_sec": 10.0,
    "text": "2025/08/26 16:04:16"
  },
  {
    "segment_id": "seg_002",
    "start_sec": 10.0,
    "end_sec": 20.0,
    "text": "2025/08/26 16:04:26"
  }
]
```

**gold.json:**

```json
[
  {
    "segment_id": "seg_001",
    "start_sec": 0.0,
    "end_sec": 10.0,
    "text": "2025/08/26 16:04:16"
  },
  {
    "segment_id": "seg_002",
    "start_sec": 10.0,
    "end_sec": 20.0,
    "text": "2025/08/26 16:04:26"
  }
]
```

### サンプル 2: TSV 形式

**pred.tsv:**

```tsv
segment_id	start_sec	end_sec	text	ocr_confidence
seg_001	0.0	10.0	2025/08/26 16:04:16	0.85
seg_002	10.0	20.0	2025/08/26 16:04:26	0.92
```

**gold.tsv:**

```tsv
segment_id	start_sec	end_sec	text
seg_001	0.0	10.0	2025/08/26 16:04:16
seg_002	10.0	20.0	2025/08/26 16:04:26
```

## 🔍 評価指標の説明

### タイムスタンプ整合性

- **許容範囲内率**: 指定した許容ズレ（±0.5 秒など）内に収まる割合
- **平均時間差**: 予測と参照のタイムスタンプの平均時間差（秒）
- **最小/最大時間差**: 時間差の最小値と最大値

### 内容一致評価

- **CER（文字エラー率）**: 文字レベルの編集距離 / 参照文字数
  - 0.0 = 完全一致、1.0 = 完全不一致
- **WER（単語エラー率）**: 単語レベルの編集距離 / 参照単語数
  - 0.0 = 完全一致、1.0 = 完全不一致
- **Token F1**: トークンレベルでの Precision と Recall の調和平均
  - 1.0 = 完全一致、0.0 = 完全不一致

### スコア補正

タイムスタンプ誤差がある場合、内容一致スコアが補正されます：

- 許容ズレ内: 補正なし
- 許容ズレ超過: 誤差に応じて最大 50%減点

## ⚠️ 注意事項

1. **segment_id の一致**: 予測と参照で`segment_id`が一致する必要があります
2. **タイムスタンプ形式**: `text`フィールドは`"YYYY/MM/DD HH:MM:SS"`形式である必要があります
3. **ファイルエンコーディング**: UTF-8 で保存してください
4. **TSV 形式**: タブ区切り（`\t`）を使用してください

## 🐛 トラブルシューティング

### エラー: "ファイルが見つかりません"

- ファイルパスが正しいか確認
- 相対パスを使用する場合は、実行ディレクトリを確認

### エラー: "JSON 形式が不正です"

- JSON ファイルが正しい形式か確認
- JSON バリデーターで検証

### エラー: "segment_id が一致しません"

- 予測と参照で`segment_id`が一致しているか確認
- マッチしないセグメントは`unmatched_pred`/`unmatched_ref`として記録されます

## 📚 関連ファイル

- `tools/video_timestamp_analyzer.py`: OCR 精度分析ツール
- `tools/timestamp_score_evaluator.py`: 詳細スコア計算とレポート生成ツール
- `src/utils/text_metrics.py`: CER/WER 計算ユーティリティ
