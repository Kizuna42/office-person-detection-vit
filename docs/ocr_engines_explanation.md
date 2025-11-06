# OCR エンジン解説: EasyOCR と PaddleOCR

## 📚 概要

本プロジェクトでは、タイムスタンプ抽出のために複数の OCR（光学文字認識）エンジンを使用できます。現在は **Tesseract** のみを使用していますが、**EasyOCR** と **PaddleOCR** も実装されており、必要に応じて有効化できます。

---

## 🔍 各 OCR エンジンの特徴

### 1. Tesseract OCR（現在使用中）

**特徴**:

- ✅ **高速**: 処理速度が速い
- ✅ **数字に強い**: タイムスタンプのような数字文字列の認識に優れている
- ✅ **軽量**: メモリ使用量が少ない
- ✅ **安定性**: 長年使われている実績のあるエンジン
- ⚠️ **精度**: 画像品質が低い場合、精度が低下する可能性

**現在の設定**:

- PSM 8（単一の単語）モードを使用
- 数字、スラッシュ、コロン、スペースのみを許可（whitelist）

**使用状況**: ✅ **現在使用中**（`config.yaml` で有効化）

---

### 2. EasyOCR

**特徴**:

- ✅ **高精度**: 深層学習ベースで、一般的に Tesseract より高精度
- ✅ **多言語対応**: 80 以上の言語をサポート
- ✅ **使いやすい**: シンプルな API
- ⚠️ **処理速度**: Tesseract よりやや遅い
- ⚠️ **メモリ使用量**: モデルが大きいため、メモリを多く使用
- ⚠️ **初回起動が遅い**: 初回実行時にモデルをダウンロード（約 1GB）

**技術詳細**:

- PyTorch ベースの深層学習モデル
- GPU 対応（CUDA、MPS は非対応）
- CPU モードでも動作可能

**使用状況**: ⚠️ **実装済みだが未使用**（`config.yaml` でコメントアウト）

---

### 3. PaddleOCR

**特徴**:

- ✅ **高精度**: 特に日本語・中国語に強い
- ✅ **多言語対応**: 日本語、中国語、英語など多数の言語をサポート
- ✅ **角度補正**: 画像の傾きを自動補正（`use_angle_cls=True`）
- ⚠️ **処理速度**: 3 つのエンジンの中で最も遅い
- ⚠️ **メモリ使用量**: 大きなモデルを使用
- ⚠️ **依存関係**: PaddlePaddle フレームワークが必要

**技術詳細**:

- PaddlePaddle フレームワークベース
- GPU 対応（CUDA、MPS は非対応）
- CPU 版が利用可能（Apple Silicon 対応）

**使用状況**: ⚠️ **実装済みだが未使用**（`config.yaml` でコメントアウト）

---

## 💻 MacBook M1 Max（メモリ 64GB）での動作可能性

### ✅ 動作可能

**M1 Max の利点**:

- **メモリ**: 64GB は十分（各エンジンは 1-2GB 程度のメモリを使用）
- **CPU 性能**: M1 Max の CPU は高性能で、CPU モードでも実用的な速度
- **統一メモリ**: GPU と CPU がメモリを共有するため、効率的

### ⚠️ 制約事項

1. **GPU 加速は利用不可**

   - EasyOCR と PaddleOCR は **MPS（Metal Performance Shaders）をサポートしていない**
   - CUDA も利用不可（NVIDIA GPU が必要）
   - **CPU モードでのみ動作**

2. **処理速度**

   - Tesseract: 高速（現在使用中）
   - EasyOCR: CPU モードで中程度の速度
   - PaddleOCR: CPU モードでやや遅い

3. **初回起動時間**
   - EasyOCR: 初回実行時にモデルをダウンロード（約 1GB、数分かかる）
   - PaddleOCR: 初回実行時にモデルをダウンロード（約 500MB）

### 📊 推奨設定（M1 Max の場合）

```yaml
# config.yaml
ocr:
  engines:
    - tesseract # 推奨: 高速で十分な精度
    # - easyocr  # オプション: より高精度が必要な場合
    # - paddleocr  # オプション: 日本語文字が含まれる場合
```

**推奨理由**:

- Tesseract は M1 Max で十分高速に動作
- タイムスタンプ（数字のみ）の認識には Tesseract で十分
- EasyOCR/PaddleOCR は CPU モードのため、速度面で劣る

---

## 🔧 インストール方法（M1 Max）

### 1. Tesseract（現在使用中）

```bash
# Homebrew でインストール
brew install tesseract tesseract-lang

# Python パッケージ（requirements.txt に含まれている）
pip install pytesseract
```

### 2. EasyOCR（オプション）

```bash
# Python パッケージのみ
pip install easyocr

# 初回実行時に自動的にモデルをダウンロード
# モデルは ~/.EasyOCR/ に保存される
```

**注意**:

- 初回実行時に約 1GB のモデルをダウンロード
- CPU モードで動作（GPU 加速なし）

### 3. PaddleOCR（オプション）

```bash
# PaddlePaddle（CPU版、Apple Silicon対応）
pip install paddlepaddle

# PaddleOCR
pip install paddleocr
```

**注意**:

- `requirements.txt` には既に含まれている（CPU 版）
- 初回実行時に約 500MB のモデルをダウンロード
- CPU モードで動作（GPU 加速なし）

---

## 🚀 使用方法

### 現在の設定（Tesseract のみ）

```yaml
# config.yaml
ocr:
  engines:
    - tesseract
  tesseract:
    config: "--psm 8 --oem 3"
    whitelist: "0123456789/:  "
```

### EasyOCR を有効化する場合

```yaml
# config.yaml
ocr:
  engines:
    - tesseract
    - easyocr # コメントアウトを解除
  tesseract:
    config: "--psm 8 --oem 3"
    whitelist: "0123456789/:  "
```

**動作**:

- 両方のエンジンで OCR を実行
- コンセンサスアルゴリズムで最も信頼性の高い結果を採用

### PaddleOCR を有効化する場合

```yaml
# config.yaml
ocr:
  engines:
    - tesseract
    - paddleocr # コメントアウトを解除
  tesseract:
    config: "--psm 8 --oem 3"
    whitelist: "0123456789/:  "
```

---

## 📊 パフォーマンス比較（予想値、M1 Max CPU モード）

| エンジン  | 処理速度        | 精度              | メモリ使用量      | 初回起動時間           |
| --------- | --------------- | ----------------- | ----------------- | ---------------------- |
| Tesseract | ⭐⭐⭐⭐⭐ 高速 | ⭐⭐⭐⭐ 良好     | ⭐⭐⭐⭐⭐ 少ない | ⭐⭐⭐⭐⭐ 即座        |
| EasyOCR   | ⭐⭐⭐ 中程度   | ⭐⭐⭐⭐⭐ 高精度 | ⭐⭐⭐ 中程度     | ⭐⭐ 数分（モデル DL） |
| PaddleOCR | ⭐⭐ やや遅い   | ⭐⭐⭐⭐⭐ 高精度 | ⭐⭐⭐ 中程度     | ⭐⭐ 数分（モデル DL） |

**注意**: 実際の性能は画像品質、文字の種類、前処理によって大きく変動します。

---

## 🎯 使い分けの指針

### Tesseract を使用すべき場合（現在の設定）

- ✅ タイムスタンプのような数字文字列
- ✅ 処理速度を重視
- ✅ メモリ使用量を抑えたい
- ✅ シンプルな設定で済ませたい

### EasyOCR を使用すべき場合

- ✅ より高精度が必要
- ✅ 複数のエンジンで結果を比較したい（コンセンサス）
- ✅ 処理速度は許容範囲内

### PaddleOCR を使用すべき場合

- ✅ 日本語文字が含まれる可能性がある
- ✅ 画像の傾きが大きい（角度補正機能）
- ✅ 最高精度が必要

---

## ⚙️ 開発環境への影響

### メモリ使用量（M1 Max 64GB の場合）

- **Tesseract**: 約 50-100MB（問題なし）
- **EasyOCR**: 約 1-2GB（問題なし、64GB あれば余裕）
- **PaddleOCR**: 約 1-2GB（問題なし、64GB あれば余裕）

**結論**: 64GB メモリがあれば、すべてのエンジンを同時に使用しても問題ありません。

### 処理時間への影響

- **Tesseract のみ**: 現在の設定、高速
- **Tesseract + EasyOCR**: 約 2-3 倍の処理時間
- **Tesseract + EasyOCR + PaddleOCR**: 約 3-5 倍の処理時間

**推奨**: 現在は Tesseract のみで十分な精度が得られているため、追加のエンジンは必要に応じて有効化することを推奨します。

---

## 🔍 実装状況

### コード内の実装

```python
# src/timestamp/ocr_engine.py
class MultiEngineOCR:
    def __init__(self, enabled_engines: List[str] = None):
        # 利用可能なエンジンを初期化
        if TESSERACT_AVAILABLE:
            self.engines["tesseract"] = self._init_tesseract()
        if EASYOCR_AVAILABLE:
            self.engines["easyocr"] = self._init_easyocr()
        if PADDLEOCR_AVAILABLE:
            self.engines["paddleocr"] = self._init_paddleocr()
```

### 現在の使用状況

- ✅ **Tesseract**: 実装済み・使用中
- ⚠️ **EasyOCR**: 実装済み・未使用（`config.yaml` でコメントアウト）
- ⚠️ **PaddleOCR**: 実装済み・未使用（`config.yaml` でコメントアウト）

---

## 📝 まとめ

### M1 Max 64GB での推奨事項

1. **現在の設定（Tesseract のみ）を維持**

   - 十分な精度と速度
   - メモリ効率が良い

2. **EasyOCR/PaddleOCR は必要に応じて有効化**

   - より高精度が必要な場合
   - 複数エンジンで結果を比較したい場合

3. **GPU 加速は利用不可**

   - すべて CPU モードで動作
   - M1 Max の CPU 性能で十分実用的

4. **メモリは問題なし**
   - 64GB あれば、すべてのエンジンを同時に使用可能

### 次のステップ

現在の Tesseract のみの設定で十分な精度が得られているため、EasyOCR や PaddleOCR を有効化する必要はありません。ただし、将来的に精度向上が必要になった場合や、複数エンジンでのコンセンサスを試したい場合は、`config.yaml` で簡単に有効化できます。
