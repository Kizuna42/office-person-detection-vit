# 実装機能の動作確認ガイド

## 目次

1. [テストの実行](#テストの実行)
2. [追跡機能の動作確認](#追跡機能の動作確認)
3. [カメラキャリブレーションの動作確認](#カメラキャリブレーションの動作確認)
4. [座標変換・歪み補正の動作確認](#座標変換歪み補正の動作確認)
5. [再投影誤差評価の動作確認](#再投影誤差評価の動作確認)
6. [軌跡データエクスポートの動作確認](#軌跡データエクスポートの動作確認)
7. [インタラクティブ可視化ツールの使い方](#インタラクティブ可視化ツールの使い方)
8. [目視確認ツールの使い方](#目視確認ツールの使い方)
9. [MOT メトリクス評価の動作確認](#motメトリクス評価の動作確認)

---

## テストの実行

### 1. すべてのテストを実行

```bash
# プロジェクトルートで実行
cd /Users/kizuna/Aeterlink/yolo3

# すべてのテストを実行（詳細出力）
pytest tests/ -v

# 追跡関連のテストのみ実行
pytest tests/test_tracking.py tests/test_tracking_integration.py -v

# カバレッジ付きで実行
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
```

### 2. テスト結果の確認

- **成功**: `PASSED` と表示
- **失敗**: `FAILED` と表示され、エラー詳細が表示されます
- **カバレッジレポート**: `htmlcov/index.html` をブラウザで開いて確認

---

## 追跡機能の動作確認

### 1. 基本的な使い方

```python
from src.tracking import Tracker
from src.models.data_models import Detection
import numpy as np

# トラッカーを初期化
tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)

# 検出結果を作成（フレーム1）
detections_frame1 = [
    Detection(
        bbox=(100.0, 100.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 200.0),
    )
]

# 特徴量を追加（必須）
for det in detections_frame1:
    det.features = np.random.rand(256).astype(np.float32)
    det.features = det.features / np.linalg.norm(det.features)

# トラッカーを更新
tracked_detections = tracker.update(detections_frame1)

# 結果を確認
print(f"追跡された検出数: {len(tracked_detections)}")
for det in tracked_detections:
    print(f"  Track ID: {det.track_id}, 座標: {det.camera_coords}")
```

### 2. 複数フレームでの追跡確認

```python
# フレーム2の検出結果（同じ人物が少し移動）
detections_frame2 = [
    Detection(
        bbox=(105.0, 105.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(130.0, 205.0),
    )
]

# 同じ特徴量を使用（同じ人物として認識される）
detections_frame2[0].features = detections_frame1[0].features.copy()

# トラッカーを更新
tracked_detections = tracker.update(detections_frame2)

# 同じTrack IDが割り当てられているか確認
print(f"Track ID: {tracked_detections[0].track_id}")  # フレーム1と同じID
```

### 3. トラック情報の取得

```python
# すべてのトラックを取得
all_tracks = tracker.get_tracks()
print(f"総トラック数: {len(all_tracks)}")

# 確立されたトラックのみ取得
confirmed_tracks = tracker.get_confirmed_tracks()
print(f"確立されたトラック数: {len(confirmed_tracks)}")

# 各トラックの軌跡を確認
for track in confirmed_tracks:
    print(f"Track ID {track.track_id}:")
    print(f"  軌跡点数: {len(track.trajectory)}")
    print(f"  ヒット数: {track.hits}")
    print(f"  年齢: {track.age}")
```

---

## カメラキャリブレーションの動作確認

### 1. チェスボード画像の準備

```bash
# チェスボード画像を用意（例: calibration_images/ ディレクトリに配置）
mkdir -p calibration_images
# 複数のチェスボード画像を配置（最低3枚以上推奨）
```

### 2. キャリブレーション実行

```python
from src.calibration import CameraCalibrator
from pathlib import Path

# キャリブレーターを初期化（チェスボードサイズを指定）
calibrator = CameraCalibrator(chessboard_size=(9, 6))

# チェスボード画像のパスリスト
image_paths = list(Path("calibration_images").glob("*.jpg"))

# キャリブレーション実行
camera_matrix, dist_coeffs = calibrator.calibrate_from_images(image_paths)

# 結果を確認
print("カメラ行列:")
print(camera_matrix)
print("\n歪み係数:")
print(dist_coeffs)

# パラメータを保存（必要に応じて）
import numpy as np
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)
```

### 3. 画像の歪み補正確認

```python
import cv2

# 歪んだ画像を読み込み
distorted_image = cv2.imread("test_image.jpg")

# 歪み補正
undistorted_image = calibrator.undistort_image(distorted_image)

# 結果を保存
cv2.imwrite("undistorted_image.jpg", undistorted_image)

# 比較画像を作成（左右に並べて表示）
comparison = np.hstack([distorted_image, undistorted_image])
cv2.imwrite("comparison.jpg", comparison)
```

---

## 座標変換・歪み補正の動作確認

### 1. 歪み補正付き座標変換の設定

```python
from src.transform.coordinate_transformer import CoordinateTransformer
import numpy as np

# ホモグラフィ行列（例）
homography_matrix = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

# カメラ行列と歪み係数を読み込み（キャリブレーション済みの場合）
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# 座標変換器を初期化（歪み補正を有効化）
transformer = CoordinateTransformer(
    homography_matrix=homography_matrix,
    floormap_config={
        "image_width": 1878,
        "image_height": 1369,
        "image_origin_x": 7,
        "image_origin_y": 9,
        "image_x_mm_per_pixel": 28.1926406926406,
        "image_y_mm_per_pixel": 28.241430700447,
    },
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    use_distortion_correction=True,  # 歪み補正を有効化
)

# カメラ座標を変換
camera_point = (640.0, 360.0)  # カメラ座標（例）
floor_point = transformer.transform(camera_point)

print(f"カメラ座標: {camera_point}")
print(f"フロアマップ座標: {floor_point}")
```

### 2. 歪み補正の効果確認

```python
# 歪み補正ありとなしで比較
transformer_with_correction = CoordinateTransformer(
    homography_matrix=homography_matrix,
    floormap_config={...},
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    use_distortion_correction=True,
)

transformer_without_correction = CoordinateTransformer(
    homography_matrix=homography_matrix,
    floormap_config={...},
    use_distortion_correction=False,
)

# 同じカメラ座標を変換
point_with = transformer_with_correction.transform(camera_point)
point_without = transformer_without_correction.transform(camera_point)

print(f"歪み補正あり: {point_with}")
print(f"歪み補正なし: {point_without}")
print(f"差: ({point_with[0] - point_without[0]:.2f}, {point_with[1] - point_without[1]:.2f})")
```

---

## 再投影誤差評価の動作確認

### 1. ホモグラフィ変換の精度評価

```python
from src.calibration import ReprojectionErrorEvaluator
import numpy as np

# 評価器を初期化
evaluator = ReprojectionErrorEvaluator()

# 対応点の準備
src_points = [
    (100.0, 100.0),  # カメラ座標
    (200.0, 150.0),
    (300.0, 200.0),
]

dst_points = [
    (50.0, 50.0),   # フロアマップ座標
    (150.0, 100.0),
    (250.0, 150.0),
]

# ホモグラフィ行列
homography_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# 再投影誤差を評価
result = evaluator.evaluate_homography(
    src_points=src_points,
    dst_points=dst_points,
    homography_matrix=homography_matrix,
)

# 結果を確認
print("再投影誤差評価結果:")
print(f"  平均誤差: {result['mean_error']:.2f} ピクセル")
print(f"  最大誤差: {result['max_error']:.2f} ピクセル")
print(f"  最小誤差: {result['min_error']:.2f} ピクセル")
print(f"  標準偏差: {result['std_error']:.2f} ピクセル")
print(f"  各点の誤差: {result['errors']}")
```

### 2. 誤差マップの生成

```python
# 誤差マップを生成
error_map = evaluator.create_error_map(
    src_points=src_points,
    dst_points=dst_points,
    homography_matrix=homography_matrix,
    image_shape=(1369, 1878),  # (height, width)
)

# 誤差マップを可視化
import cv2
error_map_normalized = (error_map / (error_map.max() + 1e-8) * 255).astype(np.uint8)
error_colored = cv2.applyColorMap(error_map_normalized, cv2.COLORMAP_JET)
cv2.imwrite("reprojection_error_map.jpg", error_colored)
```

---

## 軌跡データエクスポートの動作確認

### 1. CSV 形式でエクスポート

```python
from src.utils.export_utils import TrajectoryExporter
from src.tracking.track import Track
from src.models.data_models import Detection
import numpy as np

# エクスポーターを初期化
exporter = TrajectoryExporter(output_dir="output/trajectories")

# トラックデータを作成（例）
tracks = []
for i in range(3):
    detection = Detection(
        bbox=(100.0 + i*10, 100.0 + i*10, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0 + i*10, 200.0 + i*10),
    )
    # 軌跡を追加（簡易版）
    track = Track(track_id=i+1, detection=detection, kalman_filter=None)
    track.trajectory = [(100.0 + i*10, 100.0 + i*10), (110.0 + i*10, 110.0 + i*10)]
    tracks.append(track)

# CSV形式でエクスポート
csv_path = exporter.export_csv(tracks, filename="trajectories.csv")
print(f"CSVファイルを出力: {csv_path}")
```

### 2. JSON 形式でエクスポート

```python
# JSON形式でエクスポート
json_path = exporter.export_json(tracks, filename="trajectories.json")
print(f"JSONファイルを出力: {json_path}")

# JSONファイルを確認
import json
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"トラック数: {data['metadata']['num_tracks']}")
    print(f"総軌跡点数: {data['metadata']['total_points']}")
```

### 3. 画像シーケンスでエクスポート

```python
import cv2

# フロアマップ画像を読み込み
floormap = cv2.imread("data/floormap.png")

# 画像シーケンスとしてエクスポート
image_paths = exporter.export_image_sequence(
    tracks=tracks,
    floormap_image=floormap,
    output_prefix="trajectory_frame",
    draw_trajectories=True,
    draw_ids=True,
)

print(f"{len(image_paths)}枚の画像を出力しました")
```

### 4. 動画形式でエクスポート

```python
# 動画形式でエクスポート
video_path = exporter.export_video(
    tracks=tracks,
    floormap_image=floormap,
    filename="trajectories.mp4",
    fps=2.0,
    draw_trajectories=True,
    draw_ids=True,
)

print(f"動画ファイルを出力: {video_path}")
```

---

## インタラクティブ可視化ツールの使い方

### 1. Streamlit アプリの起動

```bash
# プロジェクトルートで実行
cd /Users/kizuna/Aeterlink/yolo3

# Streamlitアプリを起動
streamlit run tools/visualizer_app.py

# または直接実行
streamlit run tools/interactive_visualizer.py
```

### 2. ブラウザで確認

1. ブラウザが自動的に開きます（`http://localhost:8501`）
2. サイドバーで以下を設定:

   - **セッション選択**: `output/sessions/` からセッションを選択
   - **フィルタ設定**:
     - ☑ 軌跡を表示
     - ☑ ID を表示
   - **ID フィルタ**: 表示するトラック ID を選択
   - **ゾーンフィルタ**: 表示するゾーンを選択
   - **軌跡の最大長**: スライダーで調整

3. メインエリアで以下を確認:
   - **フロアマップ可視化**: 軌跡が描画されたフロアマップ
   - **フレームスライダー**: 時間軸を移動して軌跡の変化を確認
   - **統計情報**: トラック数、総軌跡点数、トラック情報テーブル

### 3. 使い方のコツ

- **フレームスライダー**: 時間を進めて軌跡の変化を確認
- **ID フィルタ**: 特定のトラックのみ表示して詳細確認
- **軌跡の最大長**: 長い軌跡を短く表示して見やすくする

---

## 目視確認ツールの使い方

### 1. キャリブレーション結果の可視化

```bash
# キャリブレーションモードで実行
python tools/visual_inspection.py \
    --mode calibration \
    --session output/sessions/20250107_120000 \
    --output output/visualization
```

### 2. 追跡結果の可視化

```bash
# 追跡モードで実行
python tools/visual_inspection.py \
    --mode tracking \
    --session output/sessions/20250107_120000 \
    --output output/visualization \
    --config config.yaml
```

**前提条件**: `output/sessions/<session_id>/tracks.json` が存在すること

### 3. 再投影誤差の可視化

```bash
# 再投影誤差モードで実行
python tools/visual_inspection.py \
    --mode reprojection \
    --session output/sessions/20250107_120000 \
    --output output/visualization \
    --config config.yaml
```

**前提条件**: `output/sessions/<session_id>/correspondence_points.json` が存在すること

### 4. 出力ファイルの確認

```bash
# 出力ディレクトリを確認
ls -la output/visualization/

# 生成されるファイル:
# - calibration_*.jpg: キャリブレーション結果画像
# - tracking_visualization.jpg: 追跡結果画像
# - reprojection_error_map.jpg: 再投影誤差マップ
```

---

## MOT メトリクス評価の動作確認

### 1. 追跡精度の評価

```python
from src.evaluation.mot_metrics import MOTMetrics
from src.tracking.track import Track
from src.models.data_models import Detection
import numpy as np

# MOTメトリクス評価器を初期化
mot_metrics = MOTMetrics()

# Ground Truthトラック（例）
ground_truth_tracks = [
    {
        "track_id": 1,
        "trajectory": [
            {"x": 100.0, "y": 100.0},
            {"x": 105.0, "y": 105.0},
            {"x": 110.0, "y": 110.0},
        ],
    },
    {
        "track_id": 2,
        "trajectory": [
            {"x": 200.0, "y": 200.0},
            {"x": 205.0, "y": 205.0},
        ],
    },
]

# 予測トラック（例）
predicted_tracks = []
for i in range(2):
    detection = Detection(
        bbox=(100.0 + i*100, 100.0 + i*100, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0 + i*100, 200.0 + i*100),
    )
    track = Track(track_id=i+1, detection=detection, kalman_filter=None)
    track.trajectory = [
        (100.0 + i*100, 100.0 + i*100),
        (105.0 + i*100, 105.0 + i*100),
    ]
    predicted_tracks.append(track)

# メトリクスを計算
metrics = mot_metrics.calculate_tracking_metrics(
    ground_truth_tracks=ground_truth_tracks,
    predicted_tracks=predicted_tracks,
    frame_count=10,
)

# 結果を確認
print("MOTメトリクス評価結果:")
print(f"  MOTA: {metrics['MOTA']:.3f} (0.0-1.0, 高いほど良い)")
print(f"  IDF1: {metrics['IDF1']:.3f} (0.0-1.0, 高いほど良い)")
print(f"  ID Switches: {metrics['ID_Switches']}")
```

### 2. EvaluationModule 経由での評価

```python
from src.evaluation import EvaluationModule

# 評価モジュールを初期化
evaluator = EvaluationModule(
    ground_truth_path="output/labels/result_fixed.json",
    iou_threshold=0.5,
)

# 追跡精度を評価
tracking_metrics = evaluator.evaluate_tracking(
    ground_truth_tracks=ground_truth_tracks,
    predicted_tracks=predicted_tracks,
    frame_count=10,
)

print("追跡精度評価結果:")
print(tracking_metrics)
```

---

## 実用的な動作確認スクリプト

以下は、すべての機能を一度に確認するスクリプトの例です：

```python
#!/usr/bin/env python3
"""実装機能の動作確認スクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tracking():
    """追跡機能のテスト"""
    print("=" * 60)
    print("追跡機能のテスト")
    print("=" * 60)

    from src.tracking import Tracker
    from src.models.data_models import Detection
    import numpy as np

    tracker = Tracker(min_hits=1)
    detection = Detection(
        bbox=(100.0, 100.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 200.0),
    )
    detection.features = np.random.rand(256).astype(np.float32)
    detection.features = detection.features / np.linalg.norm(detection.features)

    result = tracker.update([detection])
    print(f"✓ 追跡成功: Track ID {result[0].track_id}")

def test_calibration():
    """カメラキャリブレーションのテスト"""
    print("=" * 60)
    print("カメラキャリブレーションのテスト")
    print("=" * 60)

    from src.calibration import CameraCalibrator

    calibrator = CameraCalibrator(chessboard_size=(9, 6))
    print("✓ キャリブレーター初期化成功")
    print("  注意: 実際のキャリブレーションにはチェスボード画像が必要です")

def test_coordinate_transformation():
    """座標変換のテスト"""
    print("=" * 60)
    print("座標変換のテスト")
    print("=" * 60)

    from src.transform.coordinate_transformer import CoordinateTransformer

    homography_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    transformer = CoordinateTransformer(
        homography_matrix=homography_matrix,
        floormap_config={"image_width": 1000, "image_height": 1000},
    )

    result = transformer.transform((100.0, 200.0))
    print(f"✓ 座標変換成功: (100.0, 200.0) -> {result}")

def test_reprojection_error():
    """再投影誤差評価のテスト"""
    print("=" * 60)
    print("再投影誤差評価のテスト")
    print("=" * 60)

    from src.calibration import ReprojectionErrorEvaluator
    import numpy as np

    evaluator = ReprojectionErrorEvaluator()
    result = evaluator.evaluate_homography(
        src_points=[(100.0, 100.0)],
        dst_points=[(50.0, 50.0)],
        homography_matrix=np.eye(3),
    )
    print(f"✓ 再投影誤差評価成功: 平均誤差 {result['mean_error']:.2f}px")

def test_export():
    """エクスポート機能のテスト"""
    print("=" * 60)
    print("エクスポート機能のテスト")
    print("=" * 60)

    from src.utils.export_utils import TrajectoryExporter
    from pathlib import Path

    output_dir = Path("output/test_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = TrajectoryExporter(output_dir=output_dir)
    print(f"✓ エクスポーター初期化成功: {output_dir}")

def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("実装機能の動作確認")
    print("=" * 60 + "\n")

    try:
        test_tracking()
        test_calibration()
        test_coordinate_transformation()
        test_reprojection_error()
        test_export()

        print("\n" + "=" * 60)
        print("✓ すべてのテストが成功しました！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

このスクリプトを `scripts/test_implementations.py` として保存し、実行してください：

```bash
python scripts/test_implementations.py
```

---

## トラブルシューティング

### よくある問題と解決方法

1. **インポートエラー**

   ```bash
   # プロジェクトルートで実行しているか確認
   cd /Users/kizuna/Aeterlink/yolo3

   # Pythonパスを確認
   python -c "import sys; print('\n'.join(sys.path))"
   ```

2. **Streamlit が起動しない**

   ```bash
   # Streamlitがインストールされているか確認
   pip install streamlit

   # ポートが使用中の場合、別のポートを指定
   streamlit run tools/interactive_visualizer.py --server.port 8502
   ```

3. **テストが失敗する**

   ```bash
   # 詳細なエラー情報を表示
   pytest tests/ -v --tb=long

   # 特定のテストのみ実行
   pytest tests/test_tracking.py::TestTracker::test_update_single_detection -v
   ```

4. **出力ファイルが見つからない**

   ```bash
   # 出力ディレクトリが作成されているか確認
   ls -la output/

   # セッションディレクトリを確認
   ls -la output/sessions/
   ```

---

以上が実装機能の動作確認方法です。各機能を個別にテストし、問題があれば上記のトラブルシューティングを参照してください。
