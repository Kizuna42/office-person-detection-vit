# フレームサンプリング最適化ガイド

**最終更新**: 2025-11-07
**概要**: タイムラプス動画の圧縮率を踏まえたサンプリング間隔・探索ウィンドウのチューニング記録

---

## 変更内容

### タイムラプス動画の特性

- **時間圧縮率**: 約313.4倍
- **動画の1秒** = 実際の約313秒（約5.2分）
- **1フレーム** = 実際の約10.4秒

### サンプリング設定の変更

#### 1. 粗サンプリング間隔

**変更前**: 10秒間隔
**変更後**: 2秒間隔

**効果**:
- 実際の時間: 約52分間隔 → 約10.4分間隔
- より精密な探索が可能に

#### 2. 精密サンプリング間隔

**変更前**: 1秒間隔（30フレーム間隔）
**変更後**: 0.1秒間隔（3フレーム間隔）

**効果**:
- 実際の時間: 約5.2分間隔 → 約31秒間隔
- 目標時刻誤差を大幅に削減可能

#### 3. 探索ウィンドウ

**変更前**: 前後30秒
**変更後**: 前後60秒

**効果**:
- 実際の時間: 約2.6時間 → 約5.2時間
- より広範囲を探索し、目標時刻を見逃すリスクを低減

---

## 変更されたファイル

1. **config.yaml**
   - `coarse_interval_seconds`: 10 → 2
   - `fine_interval_seconds`: 1 → 0.1（新規追加）
   - `search_window_seconds`: 30 → 60

2. **src/video/frame_sampler.py**
   - `FineSampler`に`interval_seconds`パラメータを追加
   - デフォルト値を0.1秒に設定

3. **src/pipeline/frame_extraction_pipeline.py**
   - `fine_interval_seconds`パラメータを追加
   - デフォルト値を0.1秒に設定

4. **main.py**
   - config.yamlから`fine_interval_seconds`を読み込む処理を追加

5. **tools/phase2_verification.py**
   - config.yamlから`fine_interval_seconds`を読み込む処理を追加

6. **tests/test_frame_sampler.py**
   - テストケースを0.1秒間隔に更新

---

## 期待される効果

### 目標時刻誤差の改善

**変更前**:
- 平均誤差: 44秒
- サンプリング間隔: 1秒（実際の約5.2分）

**変更後（期待値）**:
- 平均誤差: ≤10秒（目標達成）
- サンプリング間隔: 0.1秒（実際の約31秒）

### 処理時間への影響

**粗サンプリング**:
- 変更前: 10秒間隔 → 約2,400フレームを処理
- 変更後: 2秒間隔 → 約12,000フレームを処理
- **処理時間**: 約5倍に増加（ただし、より精密な探索が可能）

**精密サンプリング**:
- 変更前: 1秒間隔、前後30秒 → 約60フレームを処理
- 変更後: 0.1秒間隔、前後60秒 → 約1,200フレームを処理
- **処理時間**: 約20倍に増加（ただし、目標時刻誤差を大幅に削減）

---

## 実装の詳細

### FineSamplerの変更

```python
# 変更前
class FineSampler:
    def __init__(self, video, search_window=30.0):
        # 1秒間隔で固定
        frame_interval = int(self.fps)  # 30フレーム間隔

# 変更後
class FineSampler:
    def __init__(self, video, search_window=60.0, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        # 指定間隔でサンプリング
        frame_interval = max(1, int(self.fps * self.interval_seconds))  # 3フレーム間隔
```

### フレーム間隔の計算

- **0.1秒間隔**: `fps * 0.1 = 30 * 0.1 = 3フレーム間隔`
- **探索範囲**: `60秒 * 30fps = 1,800フレーム`
- **サンプリング数**: `1,800 / 3 = 600フレーム`

---

## 次のステップ

1. **動作確認**
   ```bash
   python tools/phase2_verification.py --mode sample --video input/merged_moviefiles.mov
   ```

2. **精度評価**
   ```bash
   python tools/timestamp_score_evaluator.py --csv output/extracted_frames/extraction_results.csv
   ```

3. **目標時刻誤差の確認**
   - 目標: ≤10秒
   - 現在: 44秒（改善が期待される）

---

## 注意事項

- **処理時間の増加**: サンプリング間隔を狭めたため、処理時間が大幅に増加します
- **メモリ使用量**: より多くのフレームを処理するため、メモリ使用量も増加する可能性があります
- **精度と速度のトレードオフ**: 精度向上の代償として処理時間が増加しますが、目標時刻誤差の削減が優先されます

---

**実装者**: AI Assistant
**実装日時**: 2025-11-07
