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
│   ├── timestamp_extractor.py  # タイムスタンプ抽出
│   ├── vit_detector.py      # ViT人物検出（実装予定）
│   ├── coordinate_transformer.py  # 座標変換（実装予定）
│   ├── zone_classifier.py   # ゾーン判定（実装予定）
│   ├── aggregator.py        # 集計処理（実装予定）
│   └── visualizer.py        # 可視化（実装予定）
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

# ホモグラフィ変換行列（カメラ較正で取得）
homography:
  matrix:
    - [1.2, 0.1, -50.0]
    - [0.05, 1.3, -30.0]
    - [0.0001, 0.0002, 1.0]

# カメラ設定
camera:
  position_x: 859 # フロアマップ上のカメラ位置X座標（pixel）
  position_y: 1040 # フロアマップ上のカメラ位置Y座標（pixel）
  height_m: 2.2 # カメラ設置高さ（m）
  show_on_floormap: true # フロアマップ上にカメラ位置を表示
  marker_color: [0, 0, 255] # カメラマーカー色（BGR形式）
  marker_size: 15 # カメラマーカーサイズ（pixel）

# ゾーン定義
zones:
  - id: "zone_1"
    name: "ゾーン1（左）"
    polygon:
      - [859, 912]
      - [1095, 912]
      - [1095, 1350]
      - [859, 1350]
    priority: 1

  - id: "zone_2"
    name: "ゾーン2（中央）"
    polygon:
      - [1095, 912]
      - [1331, 912]
      - [1331, 1350]
      - [1095, 1350]
    priority: 2

  - id: "zone_3"
    name: "ゾーン3（右）"
    polygon:
      - [1331, 912]
      - [1567, 912]
      - [1567, 1350]
      - [1331, 1350]
    priority: 3
# ゾーン境界の人物は優先順位の最も高いゾーンにのみカウントされます
```

## 使用方法

### 基本的な実行

```bash
# 通常実行（検出 + 集計 + 可視化）
python main.py

# 設定ファイルを指定して実行
python main.py --config custom_config.yaml

# デバッグモードで実行
python main.py --debug
```

### 精度評価

```bash
# Ground Truthデータとの比較評価
python main.py --evaluate
```

### ファインチューニング（オプション）

```bash
# カスタムデータセットでファインチューニング
python main.py --fine-tune
```

## 実装状況

### ✅ 完了（タスク 1-9）

- [x] プロジェクト構造とデータモデル (Detection, FrameResult, AggregationResult, EvaluationMetrics)
- [x] 設定管理モジュール (ConfigManager) - YAML/JSON 対応、検証機能
- [x] 動画処理 (VideoProcessor) - H.264 形式対応、フレーム取得
- [x] タイムスタンプ抽出 (TimestampExtractor) - OCR (pytesseract)、前処理
- [x] フレームサンプリング (FrameSampler) - 5 分刻みタイムスタンプベース抽出
- [x] ViT 人物検出モジュール (ViTDetector) - DETR 対応、バッチ処理、Attention Map
- [x] 座標変換 (CoordinateTransformer) - ホモグラフィ変換、原点オフセット対応
- [x] ゾーン判定 (ZoneClassifier) - Ray Casting アルゴリズム
- [x] 集計処理 (Aggregator) - ゾーン別カウント、統計情報、CSV 出力
- [x] 可視化 (Visualizer) - 時系列グラフ、ヒートマップ、統計グラフ
- [x] フロアマップ可視化 (FloormapVisualizer) - ゾーン・検出結果描画
- [x] 精度評価 (EvaluationModule) - IoU 計算、Precision/Recall/F1
- [x] メインパイプライン統合 - エンドツーエンドフロー完成

### 🚧 実装予定（タスク 10-12）

- [ ] ファインチューニングモジュール (FineTuner) - オプション機能
- [ ] ユニットテスト - 各モジュールのテスト
- [ ] 統合テスト - エンドツーエンド処理フロー
- [ ] パフォーマンステスト - 処理時間・メモリ使用量測定

## 進捗サマリ（2025-10-29 時点）

- **パイプラインの完成度**: タスク 1-9 が完了し、入力動画から検出・座標変換・ゾーン集計・可視化・評価までのエンドツーエンド処理が動作。
- **成果物**: `output/detections/` に検出画像、`output/floormaps/` にゾーン可視化、`output/graphs/` にグラフ類、`output/zone_counts.csv` に集計結果が生成済み。
- **検出性能**: DETR ベースの人物検出が稼働。バッチ推論と MPS 対応によりパフォーマンス確保済み（詳細なベンチマークは未実施）。
- **未対応領域**: ファインチューニング、ユニット/統合テスト、パフォーマンステストが未着手。ホモグラフィ行列は仮値のため、キャリブレーションツール整備が必要。
- **次のステップ**: カメラキャリブレーション用ツールを整備し、正確なホモグラフィ行列を取得したうえで `config.yaml` を更新。併せてテスト整備と性能計測を進める。

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
