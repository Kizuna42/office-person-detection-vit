# トラッキング評価レポート

## 評価設定

| 項目 | 値 |
|-----|-----|
| 評価モード | 密サンプリング（10秒間隔） |
| IoU閾値 | 0.5 |
| 評価フレーム数 | 14 |
| フレーム間遷移数 | 13 |

## MOT標準メトリクス

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **MOTA** | -111.76% | Multiple Object Tracking Accuracy |
| **IDF1** | 12.20% | ID F1 Score |
| IDP | 10.42% | ID Precision |
| IDR | 14.71% | ID Recall |
| **IDSW** | 0 | ID Switch回数 |
| FP | 172 | False Positives |
| FN | 116 | False Negatives |
| GT | 136 | Ground Truth総数 |

## 疎サンプリング指標

| メトリクス | 値 | 備考 |
|-----------|-----|------|
| **IDSW/遷移** | 0.0000 | 遷移あたりのID Switch率 |

## 診断サマリー

| 項目 | 件数 |
|-----|------|
| ID Switch | 0 |
| Lost Track | 0 |
| False Positive | 192 |
| Missed Detection | 136 |
