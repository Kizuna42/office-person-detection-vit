# オフィス人物検出システム

Vision Transformer (ViT) ベースの物体検出モデルを使用して、オフィス内の定点カメラ映像から人物を検出し、フロアマップ上でのゾーン別人数集計を実現するバッチ処理システムです。

## 特徴

- **ViT ベース検出**: DETR による高精度な人物検出
- **タイムスタンプベースサンプリング**: OCR によるタイムラプス動画からの 5 分刻みフレーム抽出
- **ホモグラフィ変換**: カメラ座標からフロアマップ座標への射影変換
- **ゾーン別集計**: 多角形ベースのゾーン判定と人数カウント
- **MPS 対応**: Apple Silicon GPU (MPS) による高速処理
- **セッション管理**: 各実行を独立したセッションとして管理
- **パイプライン処理**: 5 つのフェーズに分かれたモジュラーな処理フロー

## クイックスタート

```bash
# セットアップ
make setup
source venv/bin/activate

# 実行
python main.py
```

## セットアップ

### 1. 依存関係のインストール

```bash
# 開発環境セットアップ（仮想環境 + 依存関係 + pre-commit）
make setup-dev
source venv/bin/activate

# または基本セットアップのみ
make setup
source venv/bin/activate

# Tesseract OCR インストール（必須）
brew install tesseract tesseract-lang  # macOS
# sudo apt-get install tesseract-ocr tesseract-ocr-jpn  # Ubuntu/Debian
```

### 2. 入力ファイルの配置

```bash
# タイムラプス動画を配置
cp /path/to/video.mov input/merged_moviefiles.mov

# フロアマップ画像を配置
cp /path/to/floormap.png data/floormap.png
```

### 3. 設定ファイルの編集

`config.yaml` を編集して、動画パス、ホモグラフィ行列、ゾーン定義などを設定します。

## 使用方法

```bash
# 通常実行（全フェーズ実行）
python main.py

# デバッグモード
python main.py --debug

# タイムスタンプOCRのみ実行
python main.py --timestamps-only

# 時刻範囲指定
python main.py --start-time "10:00" --end-time "14:00"

# 精度評価
python main.py --evaluate
```

### タイムラプス運用のヒント

- 追跡を有効にする場合は `video.is_timelapse=false` にして連続フレーム、もしくは `frame_interval_minutes` を 0.03〜0.1 程度（約 2〜6 秒）まで短縮してください。
- どうしても長間隔（数分）になる場合は `tracking.enabled=false` で検出のみ運用するか、人数カウント用途に限定してください。
- 撮影条件を変えられない場合でも、`tracking.max_age` や `tracking.iou_threshold` などは `config.yaml` で即時変更できます。

## 処理パイプライン

システムは以下の 5 つのフェーズで構成されています：

1. **Phase 1: フレーム抽出** - OCR によるタイムスタンプ抽出と 5 分刻みフレーム選択
2. **Phase 2: 人物検出** - DETR モデルによる人物検出
3. **Phase 2.5: トラッキング** - トラッキングによる人物識別
4. **Phase 3: 座標変換とゾーン判定** - ホモグラフィ変換とゾーン分類
5. **Phase 4: 集計** - ゾーン別人数カウントと統計情報
6. **Phase 5: 可視化** - 時系列グラフとフロアマップ可視化

## プロジェクト構造

```
office-person-detection/
├── config.yaml              # 設定ファイル
├── main.py                  # エントリーポイント
├── src/                     # ソースコード
│   ├── pipeline/            # パイプライン処理（Phase 1-5）
│   ├── detection/           # 人物検出
│   ├── transform/           # 座標変換
│   ├── zone/                # ゾーン判定
│   ├── aggregation/         # 集計処理
│   ├── visualization/       # 可視化
│   └── ...
├── input/                   # 入力動画
├── data/                    # フロアマップ画像
└── output/                  # 出力ディレクトリ
    └── sessions/            # セッション管理（各実行ごと）
        └── YYYYMMDD_HHMMSS/
            ├── phase1_extraction/
            ├── phase2_detection/
            ├── phase3_transform/
            ├── phase4_aggregation/
            ├── phase5_visualization/
            ├── metadata.json
            └── summary.json
```

## 出力形式

セッション管理が有効な場合、各実行は独立したセッションディレクトリに出力されます：

- `phase1_extraction/`: 抽出フレームと CSV 結果
- `phase2_detection/`: 検出画像と統計情報
- `phase3_transform/`: 座標変換結果 JSON
- `phase4_aggregation/`: ゾーン別集計 CSV
- `phase5_visualization/`: グラフとフロアマップ画像
- `metadata.json`: 実行時の設定と引数
- `summary.json`: 実行サマリー

## 実装状況

### ✅ 完了

- [x] 5 フェーズパイプライン（フレーム抽出 → 検出 → 変換 → 集計 → 可視化）
- [x] セッション管理（OutputManager）
- [x] DETR ベース人物検出（MPS/CUDA/CPU 対応）
- [x] タイムスタンプ抽出（OCR、TemporalValidatorV2）
- [x] 座標変換とゾーン判定
- [x] 集計と可視化
- [x] ユニットテスト・統合テスト

### 🚧 実装予定

- [ ] ファインチューニングモジュール
- [ ] 追加ユニットテスト
- [ ] パフォーマンスベンチマーク

## 技術スタック

- **言語**: Python 3.10+
- **深層学習**: PyTorch 2.0+ (MPS 対応)
- **ViT モデル**: Hugging Face Transformers (facebook/detr-resnet-50)
- **画像処理**: OpenCV, Pillow
- **OCR**: pytesseract
- **設定管理**: PyYAML

## トラブルシューティング

### Tesseract OCR エラー

```bash
brew install tesseract tesseract-lang  # macOS
```

### MPS が利用できない

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# config.yaml で device: "cpu" に変更
```

### メモリ不足

`config.yaml` で `detection.batch_size` を 2 または 1 に変更

## 開発

### コード品質ツール

本プロジェクトでは、以下のコード品質ツールを使用しています：

- **Ruff**: 高速な Lint + Format + Import 整理ツール（flake8/black/isort の代替）
- **MyPy**: 静的型チェッカー
- **Pre-commit**: Git フック管理フレームワーク
- **Pytest**: テストフレームワーク

### セットアップ

```bash
# 開発環境セットアップ（依存関係 + pre-commit フック）
make setup-dev

# または手動でpre-commitをインストール
make precommit-install
```

### 開発コマンド

```bash
# テスト実行
make test
# または
pytest tests/

# カバレッジ付きテスト
make test-cov
# または
pytest --cov=src tests/

# Lintチェック
make lint

# コードフォーマット
make format

# フォーマットチェック（変更なし）
make format-check

# Pre-commitフックを手動実行（全ファイル）
make precommit-run
```

### CI/CD

#### GitHub Actions

プッシュや Pull Request 時に自動的に以下が実行されます：

- **Format チェック**: `ruff format --check`
- **Lint チェック**: `ruff check`
- **型チェック**: `mypy`
- **テスト実行**: `pytest`（Python 3.10, 3.11）
- **カバレッジレポート**: Codecov にアップロード

ワークフロー設定: [.github/workflows/ci.yml](.github/workflows/ci.yml)

#### Pre-commit フック

コミット前に自動的に以下が実行されます：

- Trailing whitespace 削除
- End of file fixer
- YAML/JSON/TOML チェック
- Ruff lint + format
- MyPy 型チェック

Push 前に自動的に以下が実行されます：

- 上記の pre-commit チェック
- Pytest テスト実行

#### 設定ファイル

- **Pre-commit 設定**: [.pre-commit-config.yaml](.pre-commit-config.yaml)
- **プロジェクト設定**: [pyproject.toml](pyproject.toml)（ruff, mypy, pytest 設定）

### コード品質目標

- **テストカバレッジ**: ≥ 80%
- **Lint 警告**: 0 件
- **型チェック**: MyPy 準拠（段階的導入）
- **フォーマット**: Ruff 準拠

## ライセンス

MIT License
