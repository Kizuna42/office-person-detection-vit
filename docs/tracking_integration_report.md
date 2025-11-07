# 追跡機能パイプライン統合 - 実装完了レポート

## 実装日時
2025年11月7日

## 実装内容

### 1. TrackingPhaseの作成 ✅

**ファイル**: `src/pipeline/tracking_phase.py`

**主な機能**:
- `BasePhase`を継承した追跡フェーズクラス
- `initialize()`: Trackerと特徴量抽出用のViTDetectorを初期化
- `execute()`: 検出結果に対して追跡処理を実行
  - 特徴量抽出（DETRエンコーダーから）
  - トラッカーでID割り当て
  - フレームと検出結果をframe_numでマッチング
- `export_results()`: 追跡結果をJSON/CSV形式でエクスポート
- `get_tracks()`: 追跡結果のトラックを取得

**設定ファイル連携**:
- `tracking.enabled`: 追跡機能の有効/無効
- `tracking.max_age`, `tracking.min_hits`, `tracking.iou_threshold`
- `tracking.appearance_weight`, `tracking.motion_weight`

### 2. PipelineOrchestratorへの統合 ✅

**ファイル**: `src/pipeline/orchestrator.py`

**追加メソッド**:
- `run_tracking()`: 追跡フェーズを実行
  - 設定ファイルの`tracking.enabled`をチェック
  - 無効な場合は検出結果をそのまま返す
  - 有効な場合はTrackingPhaseを実行し、結果をエクスポート

**パフォーマンス計測**:
- `phase2.5_tracking`として計測

### 3. main.pyへの統合 ✅

**ファイル**: `main.py`

**変更内容**:
- 検出フェーズと座標変換フェーズの間に追跡フェーズを挿入
- 追跡結果を座標変換フェーズに渡す

**処理フロー**:
```
フェーズ1: フレーム抽出
  ↓
フェーズ2: 人物検出
  ↓
フェーズ2.5: オブジェクト追跡 ← 新規追加
  ↓
フェーズ3: 座標変換とゾーン判定
  ↓
フェーズ4: 集計処理
  ↓
フェーズ5: 可視化
```

### 4. モジュールエクスポートの更新 ✅

**ファイル**: `src/pipeline/__init__.py`

**変更内容**:
- `TrackingPhase`をエクスポートに追加

## 出力物

### 追跡フェーズの出力

**ディレクトリ**: `output/sessions/<session_id>/phase2.5_tracking/`

**出力ファイル**:
1. **tracks.json**: 追跡結果（JSON形式）
   ```json
   {
     "tracks": [
       {
         "track_id": 1,
         "age": 10,
         "hits": 10,
         "trajectory": [
           {"x": 100.0, "y": 200.0},
           {"x": 105.0, "y": 205.0},
           ...
         ]
       },
       ...
     ],
     "metadata": {
       "num_tracks": 5,
       "total_points": 50
     }
   }
   ```

2. **tracks.csv**: 追跡結果（CSV形式）
   - カラム: `track_id`, `frame_index`, `timestamp`, `x`, `y`, `zone_ids`, `confidence`

3. **tracking_statistics.json**: 追跡統計情報
   ```json
   {
     "total_tracks": 5,
     "total_trajectory_points": 50,
     "avg_trajectory_length": 10.0,
     "tracks": [
       {
         "track_id": 1,
         "age": 10,
         "hits": 10,
         "trajectory_length": 10
       },
       ...
     ]
   }
   ```

## 動作確認

### テスト結果 ✅

```bash
$ pytest tests/test_tracking.py -v
============================= test session starts ==============================
24 passed in 0.XXs
```

**テスト項目**:
- KalmanFilterのテスト（5件）
- SimilarityCalculatorのテスト（7件）
- Trackのテスト（4件）
- Trackerのテスト（5件）
- HungarianAlgorithmのテスト（3件）

**全てのテストが通過** ✅

### インポート確認 ✅

```bash
$ python -c "from src.pipeline.tracking_phase import TrackingPhase; print('OK')"
TrackingPhase import OK

$ python -c "from src.pipeline.orchestrator import PipelineOrchestrator; print('OK')"
PipelineOrchestrator import OK
```

### 設定ファイル確認 ✅

```bash
$ python -c "from src.config import ConfigManager; c = ConfigManager('config.yaml'); print('tracking.enabled:', c.get('tracking.enabled'))"
tracking.enabled: True
```

## 使用方法

### 1. 設定ファイルで追跡を有効化

`config.yaml`:
```yaml
tracking:
  enabled: true  # 追跡機能を有効にする
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  appearance_weight: 0.7
  motion_weight: 0.3
```

### 2. メインパイプラインを実行

```bash
python main.py --config config.yaml
```

### 3. 出力物の確認

```bash
# セッションディレクトリを確認
ls output/sessions/<session_id>/phase2.5_tracking/

# 追跡結果を確認
cat output/sessions/<session_id>/phase2.5_tracking/tracks.json
cat output/sessions/<session_id>/phase2.5_tracking/tracking_statistics.json
```

## 実装の特徴

### 1. 設定駆動
- `tracking.enabled=false`の場合、追跡処理をスキップ
- 検出結果をそのまま後続フェーズに渡す

### 2. エラーハンドリング
- 特徴量抽出に失敗した場合も処理を継続
- フレームが見つからない場合は警告を出力してスキップ

### 3. パフォーマンス計測
- 追跡フェーズの処理時間を計測
- パフォーマンスサマリーに含まれる

### 4. データエクスポート
- JSON形式: 完全な追跡データ
- CSV形式: 表形式での分析用
- 統計情報: 追跡品質の評価用

## 次のステップ

1. **実際の動画での動作確認**
   - サンプル動画での実行
   - 出力物の確認
   - 追跡精度の評価

2. **可視化の統合**
   - フロアマップ可視化に軌跡を追加
   - インタラクティブツールでの確認

3. **MOTメトリクスの評価**
   - MOTA, IDF1の計算
   - 精度レポートの生成

## 注意事項

- 追跡機能は`tracking.enabled=true`の場合のみ実行されます
- 特徴量抽出には追加の処理時間がかかります
- メモリ使用量が増加する可能性があります（特徴量の保持）
