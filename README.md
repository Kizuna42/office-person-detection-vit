# オフィス人物検出システム

Vision Transformer (ViT) ベースの物体検出モデルを使用して、オフィス内の定点カメラ映像から人物を検出し、フロアマップ上でのゾーン別人数集計を実現するバッチ処理システムです。

## 特徴

- **ViTベース検出**: DETR または ViT-Det による高精度な人物検出
- **タイムスタンプベースサンプリング**: OCR によるタイムラプス動画からの5分刻みフレーム抽出
- **ホモグラフィ変換**: カメラ座標からフロアマップ座標への射影変換
- **ゾーン別集計**: 多角形ベースのゾーン判定と人数カウント
- **MPS対応**: Apple Silicon GPU (MPS) による高速処理
- **設定駆動**: YAML設定ファイルでコード変更なしに動作をカスタマイズ

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
  frame_interval_minutes: 5  # サンプリング間隔
  tolerance_seconds: 10      # 許容誤差

# 人物検出設定
detection:
  model_name: "facebook/detr-resnet-50"
  confidence_threshold: 0.5
  device: "mps"  # mps, cuda, cpu

# ホモグラフィ変換行列（カメラ較正で取得）
homography:
  matrix:
    - [1.2, 0.1, -50.0]
    - [0.05, 1.3, -30.0]
    - [0.0001, 0.0002, 1.0]

# ゾーン定義
zones:
  - id: "zone_a"
    name: "会議室エリア"
    polygon:
      - [100, 200]
      - [300, 200]
      - [300, 400]
      - [100, 400]
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

### ✅ 完了

- [x] プロジェクト構造とデータモデル
- [x] 設定管理モジュール (ConfigManager)
- [x] 動画処理 (VideoProcessor)
- [x] タイムスタンプ抽出 (TimestampExtractor)
- [x] フレームサンプリング (FrameSampler)

### 🚧 実装予定

- [ ] ViT人物検出モジュール (ViTDetector)
- [ ] 座標変換 (CoordinateTransformer)
- [ ] ゾーン判定 (ZoneClassifier)
- [ ] 集計処理 (Aggregator)
- [ ] 可視化 (Visualizer)
- [ ] 精度評価 (EvaluationModule)
- [ ] メインパイプライン統合

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

- フレーム処理時間: ≤ 2秒/フレーム (MPS使用時)
- メモリ使用量: ≤ 12GB
- 全処理時間: ≤ 10分 (1時間分のタイムラプス動画)

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

