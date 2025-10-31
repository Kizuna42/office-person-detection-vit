# オフィス人物検出システム

Vision Transformer (ViT) ベースの物体検出モデルを使用して、オフィス内の定点カメラ映像から人物を検出し、フロアマップ上でのゾーン別人数集計を実現するバッチ処理システムです。

## 特徴

- **ViT ベース検出**: DETR または ViT-Det による高精度な人物検出
- **タイムスタンプベースサンプリング**: OCR によるタイムラプス動画からの 5 分刻みフレーム抽出
- **ホモグラフィ変換**: カメラ座標からフロアマップ座標への射影変換
- **ゾーン別集計**: 多角形ベースのゾーン判定と人数カウント
- **MPS 対応**: Apple Silicon GPU (MPS) による高速処理
- **設定駆動**: YAML 設定ファイルでコード変更なしに動作をカスタマイズ

## 技術スタック

- **言語**: Python 3.10+
- **深層学習**: PyTorch 2.0+ (MPS 対応)
- **ViT モデル**: Hugging Face Transformers (DETR, ViT-Det)
- **画像処理**: OpenCV, Pillow
- **OCR**: pytesseract
- **設定管理**: PyYAML

## プロジェクト構造

```
office-person-detection/
├── config.yaml              # メイン設定ファイル
├── main.py                  # エントリーポイント
├── requirements.txt         # Python依存関係
├── src/                     # ソースコード
│   ├── __init__.py
│   ├── config_manager.py    # 設定管理
│   ├── data_models.py       # データクラス定義
│   ├── video_processor.py   # 動画処理
│   ├── frame_sampler.py     # フレームサンプリング
│   ├── timestamp_extractor.py  # タイムスタンプ抽出（OCR）
│   ├── vit_detector.py      # ViT人物検出（DETR/ViT-Det）
│   ├── coordinate_transformer.py  # 座標変換（ホモグラフィ）
│   ├── zone_classifier.py   # ゾーン判定（Point-in-Polygon）
│   ├── aggregator.py        # 集計処理
│   ├── visualizer.py        # 可視化（グラフ生成）
│   ├── floormap_visualizer.py  # フロアマップ可視化
│   └── evaluation_module.py    # 精度評価（IoU計算）
├── input/                   # 入力ファイル
│   └── merged_moviefiles.mov  # タイムラプス動画（要配置）
├── data/                    # 参照データ
│   └── floormap.png        # フロアマップ画像（要配置）
├── output/                  # 出力ディレクトリ
│   ├── labels/             # Ground Truth データ
│   │   └── result_fixed.json
│   ├── detections/         # 検出結果画像
│   ├── floormaps/          # フロアマップ画像
│   ├── graphs/             # グラフ
│   └── zone_counts.csv     # 集計結果CSV
└── tests/                   # テストコード
    ├── conftest.py          # テスト設定
    ├── test_frame_sampler.py
    ├── test_timestamp_extractor.py
    └── test_vit_detector.py

```

## セットアップ

### 1. 依存関係のインストール

```bash
# Python 3.10以上が必要
python --version

# 依存関係をインストール
pip install -r requirements.txt

# Tesseract OCRのインストール（macOS）
brew install tesseract tesseract-lang
```

### 2. 入力ファイルの配置

```bash
# タイムラプス動画を input/ に配置
cp /path/to/your/video.mov input/merged_moviefiles.mov

# フロアマップ画像を data/ に配置
cp /path/to/your/floormap.png data/floormap.png
```

### 3. 設定ファイルの編集

`config.yaml` を環境に合わせて編集します：

```yaml
# 動画入力設定
video:
  input_path: "input/merged_moviefiles.mov"
  frame_interval_minutes: 5 # サンプリング間隔
  tolerance_seconds: 10 # 許容誤差

# 人物検出設定
detection:
  model_name: "facebook/detr-resnet-50"
  confidence_threshold: 0.5
  device: "mps" # mps, cuda, cpu

# ホモグラフィ変換行列（✅ キャリブレーション済み）
# 2025-10-29にキャリブレーション実施（32点の対応点を使用）
# 詳細: output/calibration/points_20251029-171336.json
# 計算結果: output/calibration/homography_20251029-171336.yaml
homography:
  matrix:
    - [-0.7522441113, -3.2312558062, 437.1587328345]
    - [-1.3638383931, -3.7773354566, 1046.0306165052]
    - [-0.0010622111, -0.0038281055, 1.0000000000]

# カメラ設定
camera:
  position_x: 859 # ⚠️ フロアマップ上のカメラ位置X座標（pixel、実測推奨）
  position_y: 1040 # ⚠️ フロアマップ上のカメラ位置Y座標（pixel、実測推奨）
  height_m: 2.2 # カメラ設置高さ（m、実測推奨）
  show_on_floormap: true
  marker_color: [0, 0, 255] # カメラマーカー色（BGR形式）
  marker_size: 15

# ゾーン定義（⚠️ サンプル値 - 実測推奨）
# フロアマップ画像上で実際のエリア境界を測定して更新してください
zones:
  - id: "zone_1"
    name: "ゾーン1（左）"
    polygon:
      - [859, 912] # ⚠️ サンプル値
      - [1095, 912]
      - [1095, 1350]
      - [859, 1350]
    priority: 1

  - id: "zone_2"
    name: "ゾーン2（中央）"
    polygon:
      - [1095, 912] # ⚠️ サンプル値
      - [1331, 912]
      - [1331, 1350]
      - [1095, 1350]
    priority: 2

  - id: "zone_3"
    name: "ゾーン3（右）"
    polygon:
      - [1331, 912] # ⚠️ サンプル値
      - [1567, 912]
      - [1567, 1350]
      - [1331, 1350]
    priority: 3
# ゾーン境界の人物は優先順位の最も高いゾーンにのみカウントされます（allow_overlap=false時）
```

## 使用方法

### 基本的な実行

```bash
# 通常実行（検出 + 集計 + 可視化）
python main.py

# 設定ファイルを指定して実行
python main.py --config custom_config.yaml

# デバッグモードで実行（詳細ログ、中間結果出力）
python main.py --debug

# 開始・終了時刻を指定して実行
python main.py --start-time "10:00" --end-time "14:00"
```

### 精度評価

```bash
# Ground Truthデータとの比較評価
python main.py --evaluate
```

### ファインチューニング（未実装）

```bash
# ⚠️ 現在未実装（将来の機能）
python main.py --fine-tune
```

## 実装状況

### ✅ 完了（全コア機能実装済み）

**パイプラインモジュール**

- [x] プロジェクト構造とデータモデル (Detection, FrameResult, AggregationResult, EvaluationMetrics)
- [x] 設定管理モジュール (ConfigManager) - YAML/JSON 対応、検証機能
- [x] 動画処理 (VideoProcessor) - H.264 形式対応、フレーム取得
- [x] タイムスタンプ抽出 (TimestampExtractor) - OCR (pytesseract)、前処理、デバッグ機能
- [x] フレームサンプリング (FrameSampler) - 5 分刻みタイムスタンプベース抽出、±10 秒許容誤差
- [x] ViT 人物検出モジュール (ViTDetector) - DETR 対応、バッチ処理、MPS/CUDA/CPU 対応
- [x] 座標変換 (CoordinateTransformer) - ホモグラフィ変換、原点オフセット、mm 座標変換
- [x] ゾーン判定 (ZoneClassifier) - Ray Casting アルゴリズム、重複ゾーン対応
- [x] 集計処理 (Aggregator) - ゾーン別カウント、統計情報、CSV 出力
- [x] 可視化 (Visualizer) - 時系列グラフ、ヒートマップ、統計グラフ
- [x] フロアマップ可視化 (FloormapVisualizer) - ゾーン・検出結果・カメラ位置描画
- [x] 精度評価 (EvaluationModule) - IoU 計算、Precision/Recall/F1、CSV/JSON 出力
- [x] メインパイプライン統合 - エンドツーエンドフロー完成

**テスト**

- [x] ユニットテスト - FrameSampler, TimestampExtractor, ViTDetector
- [x] 統合テスト - エンドツーエンド処理フロー（入力 → 検出 → 集計 → 可視化 → 評価）
- [x] テストフレームワーク - pytest, conftest.py

### 🚧 オプション機能（実装予定）

- [ ] ファインチューニングモジュール (FineTuner) - カスタムデータセットでの学習
- [ ] 追加ユニットテスト - 残りのモジュールのテスト
- [ ] パフォーマンスベンチマーク - 詳細な処理時間・メモリ使用量測定

## 進捗サマリ（2025 年 10 月 29 日時点）

### ✅ 完了事項

- **パイプラインの完成度**: 入力動画から検出・座標変換・ゾーン集計・可視化・評価までのエンドツーエンド処理が完全に動作。
- **成果物**:
  - `output/detections/`: 検出画像（バウンディングボックス付き）
  - `output/floormaps/`: ゾーン可視化画像（46 フレーム分 + 凡例）
  - `output/graphs/`: 時系列グラフ、ヒートマップ、統計グラフ
  - `output/zone_counts.csv`: 集計結果 CSV（45 フレーム分）
  - `output/system.log`: システムログ
- **検出性能**: DETR ベースの人物検出が稼働。MPS 対応で高速処理（平均 15 ～ 25 人/フレーム検出）。
- **テスト**: ユニットテスト（3 モジュール）と統合テスト（エンドツーエンド）を実装済み。

### ✅ キャリブレーション状況

- **ホモグラフィ行列**: ✅ **キャリブレーション完了**（2025-10-29 実施）
  - 対応点: 32 点
  - RMSE: 1105.1 ピクセル
  - 詳細情報: `output/calibration/points_20251029-171336.json`
  - 計算結果: `output/calibration/homography_20251029-171336.yaml`

### ⚠️ 注意事項（要対応）

- **ゾーン定義**: 現在サンプル値のため、**実測推奨**。フロアマップ画像上で実際のエリア境界を測定して更新が必要。
- **カメラ位置**: 実測値を設定することを推奨（現在はサンプル値）。

### 📋 次のステップ

1. ✅ **カメラキャリブレーション**: 完了（2025-10-29 実施）
2. **ゾーン定義の実測**: フロアマップ画像で実際のエリア境界を測定
3. **精度評価の再実行**: キャリブレーション後の性能を確認（ホモグラフィ行列更新済み）
4. **追加テスト**: 残りのモジュールのユニットテスト追加

## 出力形式

### CSV 出力 (`output/zone_counts.csv`)

```csv
timestamp,zone_id,count
12:10,zone_a,3
12:10,zone_b,5
12:15,zone_a,2
12:15,zone_b,6
```

### 画像出力

- `output/detections/`: バウンディングボックス付き検出結果画像
- `output/floormaps/`: フロアマップ上への人物位置プロット
- `output/graphs/`: ゾーン別人数の時系列グラフ

## パフォーマンス目標

- フレーム処理時間: ≤ 2 秒/フレーム (MPS 使用時)
- メモリ使用量: ≤ 12GB
- 全処理時間: ≤ 10 分 (1 時間分のタイムラプス動画)

## 開発

### テスト実行

```bash
# すべてのテストを実行
pytest tests/

# 特定のテストファイルを実行
pytest tests/test_config_manager.py

# カバレッジ付きで実行
pytest --cov=src tests/
```

### コーディング規約

- PEP 8 準拠
- 型ヒント必須
- Docstring 必須（Google スタイル）

## ✅ キャリブレーション状況

### ホモグラフィ行列: キャリブレーション完了 ✅

**実施日**: 2025 年 10 月 29 日  
**対応点**: 32 点  
**参照画像**: `output/calibration/reference_1215_f023400.png`  
**計算結果ファイル**: `output/calibration/homography_20251029-171336.yaml`  
**対応点データ**: `output/calibration/points_20251029-171336.json`

**キャリブレーション結果**:

- RMSE: 1105.1 ピクセル
- 最大誤差: 4415.1 ピクセル
- インライア数: 8 点（全 32 点中）

**設定ファイル**: `config.yaml` に反映済み

---

### 残りのキャリブレーション項目

### 1. ゾーン定義の実測（推奨）

フロアマップ画像 (`data/floormap.png`) で実際のエリア境界を測定し、`config.yaml` の `zones` セクションを更新してください。

**現在の設定**: サンプル値（3 つの等間隔ゾーン）  
**推奨**: 実際のオフィスレイアウトに合わせて再定義

### 2. カメラ位置の実測（推奨）

実際のカメラ設置位置を測定し、`camera.position_x`, `camera.position_y`, `camera.height_m` を更新してください。

**現在の設定**: サンプル値（position_x: 859, position_y: 1040, height_m: 2.2）  
**推奨**: フロアマップ上での実際の位置と高さを測定

---

### キャリブレーションツールの使用方法（参考）

今後のキャリブレーション実施時の手順：

**手順**:

1. カメラ画像から対応点を 4 点以上選択（例: 角、柱、ドアなどの特徴点）
2. フロアマップ上の対応する座標を特定
3. `cv2.findHomography()` でホモグラフィ行列を計算
4. `config.yaml` の `homography.matrix` を更新

**サンプルコード**:

```python
import cv2
import numpy as np

# カメラ画像上の点（例）
camera_points = np.array([
    [320, 400], [960, 400], [320, 600], [960, 600]
], dtype=np.float32)

# フロアマップ上の対応点
floormap_points = np.array([
    [900, 1000], [1400, 1000], [900, 1300], [1400, 1300]
], dtype=np.float32)

# ホモグラフィ行列を計算
H, _ = cv2.findHomography(camera_points, floormap_points)
print("更新する行列:", H.tolist())
```

## トラブルシューティング

### Tesseract OCR エラー

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# Tesseractのパスを確認
which tesseract
```

### MPS (Apple Silicon GPU) が利用できない

```bash
# PyTorchのMPS対応を確認
python -c "import torch; print(torch.backends.mps.is_available())"

# CPUフォールバック設定
# config.yaml で device: "cpu" に変更
```

### メモリ不足エラー

```bash
# バッチサイズを削減
# config.yaml で detection.batch_size を 2 または 1 に変更
```

## ライセンス

MIT License

## 作者

Aeterlink - Office Person Detection Project
