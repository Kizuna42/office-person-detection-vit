# 実装機能の動作確認ガイド（簡易版）

## クイックスタート

### 1. すべての機能を一度にテスト

```bash
# プロジェクトルートで実行
cd /Users/kizuna/Aeterlink/yolo3

# 動作確認スクリプトを実行
python scripts/test_implementations.py
```

### 2. 個別のテストを実行

```bash
# 追跡機能のテスト
pytest tests/test_tracking.py -v

# 統合テスト
pytest tests/test_tracking_integration.py -v

# すべてのテスト
pytest tests/ -v
```

---

## 各機能の使い方

### 📍 追跡機能（Tracker）

**基本的な使い方:**

```python
from src.tracking import Tracker
from src.models.data_models import Detection
import numpy as np

# 1. トラッカーを初期化
tracker = Tracker(max_age=30, min_hits=3)

# 2. 検出結果を作成（特徴量を追加）
detection = Detection(
    bbox=(100.0, 100.0, 50.0, 100.0),
    confidence=0.9,
    class_id=1,
    class_name="person",
    camera_coords=(125.0, 200.0),
)
detection.features = np.random.rand(256).astype(np.float32)
detection.features = detection.features / np.linalg.norm(detection.features)

# 3. トラッカーを更新
tracked = tracker.update([detection])

# 4. 結果を確認
print(f"Track ID: {tracked[0].track_id}")
```

**確認ポイント:**

- ✅ `track_id` が割り当てられているか
- ✅ 複数フレームで同じ ID が維持されるか
- ✅ 軌跡が正しく記録されているか

---

### 📷 カメラキャリブレーション

**準備:**

1. チェスボード画像を 3 枚以上用意（`calibration_images/` に配置）

**実行:**

```python
from src.calibration import CameraCalibrator
from pathlib import Path

# 1. キャリブレーターを初期化
calibrator = CameraCalibrator(chessboard_size=(9, 6))

# 2. 画像パスを取得
image_paths = list(Path("calibration_images").glob("*.jpg"))

# 3. キャリブレーション実行
camera_matrix, dist_coeffs = calibrator.calibrate_from_images(image_paths)

# 4. 結果を確認・保存
print("カメラ行列:", camera_matrix)
print("歪み係数:", dist_coeffs)
```

**確認ポイント:**

- ✅ カメラ行列が正しく計算されているか（3x3 行列）
- ✅ 歪み係数が取得できているか
- ✅ 画像の歪み補正が正しく動作するか

---

### 🔄 座標変換（歪み補正付き）

**設定:**

```python
from src.transform.coordinate_transformer import CoordinateTransformer
import numpy as np

# 1. カメラ行列と歪み係数を読み込み（キャリブレーション済みの場合）
camera_matrix = np.load("camera_matrix.npy")  # または直接指定
dist_coeffs = np.load("dist_coeffs.npy")      # または直接指定

# 2. 座標変換器を初期化（歪み補正を有効化）
transformer = CoordinateTransformer(
    homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    floormap_config={
        "image_width": 1878,
        "image_height": 1369,
        "image_origin_x": 7,
        "image_origin_y": 9,
    },
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    use_distortion_correction=True,  # 歪み補正を有効化
)

# 3. 座標を変換
floor_point = transformer.transform((640.0, 360.0))
print(f"フロアマップ座標: {floor_point}")
```

**確認ポイント:**

- ✅ 座標変換が正しく動作するか
- ✅ 歪み補正の効果が確認できるか
- ✅ 原点オフセットが正しく適用されているか

---

### 📊 再投影誤差評価

**実行:**

```python
from src.calibration import ReprojectionErrorEvaluator
import numpy as np

# 1. 評価器を初期化
evaluator = ReprojectionErrorEvaluator()

# 2. 対応点とホモグラフィ行列を準備
src_points = [(100.0, 100.0), (200.0, 150.0)]
dst_points = [(50.0, 50.0), (150.0, 100.0)]
homography = np.eye(3)

# 3. 評価実行
result = evaluator.evaluate_homography(
    src_points=src_points,
    dst_points=dst_points,
    homography_matrix=homography,
)

# 4. 結果を確認
print(f"平均誤差: {result['mean_error']:.2f}px")
print(f"最大誤差: {result['max_error']:.2f}px")
```

**確認ポイント:**

- ✅ 誤差が計算されているか
- ✅ 誤差マップが生成できるか
- ✅ 精度改善の指標として使えるか

---

### 💾 軌跡データエクスポート

**CSV 形式:**

```python
from src.utils.export_utils import TrajectoryExporter
from src.tracking.track import Track
# ... トラックデータを準備 ...

exporter = TrajectoryExporter(output_dir="output/trajectories")
csv_path = exporter.export_csv(tracks, filename="trajectories.csv")
print(f"CSV出力: {csv_path}")
```

**JSON 形式:**

```python
json_path = exporter.export_json(tracks, filename="trajectories.json")
print(f"JSON出力: {json_path}")
```

**動画形式:**

```python
import cv2
floormap = cv2.imread("data/floormap.png")

video_path = exporter.export_video(
    tracks=tracks,
    floormap_image=floormap,
    filename="trajectories.mp4",
    fps=2.0,
)
print(f"動画出力: {video_path}")
```

**確認ポイント:**

- ✅ CSV/JSON ファイルが正しく生成されるか
- ✅ 動画が正しく作成されるか
- ✅ 軌跡が正しく描画されているか

---

### 🖥️ インタラクティブ可視化ツール（Streamlit）

**起動:**

```bash
# Streamlitアプリを起動
streamlit run tools/interactive_visualizer.py

# または
streamlit run tools/visualizer_app.py
```

**使い方:**

1. ブラウザが自動的に開きます（`http://localhost:8501`）
2. サイドバーで設定:
   - セッションを選択
   - フィルタを設定（軌跡表示、ID 表示など）
   - ID やゾーンでフィルタリング
3. メインエリアで確認:
   - フロアマップ上の軌跡を確認
   - フレームスライダーで時間を移動
   - 統計情報を確認

**確認ポイント:**

- ✅ アプリが起動するか
- ✅ 軌跡が正しく表示されるか
- ✅ インタラクティブな操作ができるか

---

### 🔍 目視確認ツール

**キャリブレーション結果の可視化:**

```bash
python tools/visual_inspection.py \
    --mode calibration \
    --session output/sessions/20250107_120000 \
    --output output/visualization
```

**追跡結果の可視化:**

```bash
python tools/visual_inspection.py \
    --mode tracking \
    --session output/sessions/20250107_120000 \
    --output output/visualization \
    --config config.yaml
```

**再投影誤差の可視化:**

```bash
python tools/visual_inspection.py \
    --mode reprojection \
    --session output/sessions/20250107_120000 \
    --output output/visualization \
    --config config.yaml
```

**確認ポイント:**

- ✅ 出力画像が生成されるか
- ✅ 可視化内容が正しいか
- ✅ エラーなく実行できるか

---

### 📈 MOT メトリクス評価

**実行:**

```python
from src.evaluation.mot_metrics import MOTMetrics

mot_metrics = MOTMetrics()

metrics = mot_metrics.calculate_tracking_metrics(
    ground_truth_tracks=gt_tracks,
    predicted_tracks=predicted_tracks,
    frame_count=100,
)

print(f"MOTA: {metrics['MOTA']:.3f}")
print(f"IDF1: {metrics['IDF1']:.3f}")
```

**確認ポイント:**

- ✅ メトリクスが計算されるか
- ✅ 値が 0.0-1.0 の範囲内か
- ✅ 精度評価の指標として使えるか

---

## 出力ファイルの確認

### 生成されるファイル

```
output/
├── test_export/
│   ├── test_trajectories.csv      # CSV形式の軌跡データ
│   └── test_trajectories.json     # JSON形式の軌跡データ
├── visualization/
│   ├── calibration_*.jpg          # キャリブレーション結果
│   ├── tracking_visualization.jpg # 追跡結果
│   └── reprojection_error_map.jpg # 再投影誤差マップ
└── sessions/
    └── <session_id>/
        └── tracks.json            # トラックデータ
```

### ファイル内容の確認

**CSV ファイル:**

```bash
# CSVファイルを確認
head -10 output/test_export/test_trajectories.csv

# 列: track_id, frame_index, timestamp, x, y, zone_ids, confidence
```

**JSON ファイル:**

```bash
# JSONファイルを確認
cat output/test_export/test_trajectories.json | python -m json.tool | head -30
```

---

## トラブルシューティング

### よくある問題

1. **インポートエラー**

   ```bash
   # プロジェクトルートにいるか確認
   pwd
   # /Users/kizuna/Aeterlink/yolo3 であることを確認
   ```

2. **Streamlit が起動しない**

   ```bash
   # Streamlitをインストール
   pip install streamlit

   # ポートを変更
   streamlit run tools/interactive_visualizer.py --server.port 8502
   ```

3. **テストが失敗する**

   ```bash
   # 詳細なエラー情報を表示
   pytest tests/ -v --tb=long
   ```

4. **出力ファイルが見つからない**
   ```bash
   # 出力ディレクトリを確認
   ls -la output/
   ```

---

## 次のステップ

1. **実際のデータでテスト**

   - 実際の動画データを使用
   - チェスボード画像でキャリブレーション
   - 実際の検出結果で追跡

2. **パラメータの調整**

   - `Tracker` の `max_age`, `min_hits`, `iou_threshold` を調整
   - ホモグラフィ行列を実際の値に更新
   - カメラパラメータを設定

3. **精度の改善**
   - 再投影誤差を最小化
   - MOT メトリクスを改善
   - 追跡精度を向上

詳細な説明は `docs/implementation_verification_guide.md` を参照してください。
