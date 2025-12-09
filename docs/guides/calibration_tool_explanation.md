# calibration_tool.py の仕組み・計算方法・参照データ解説

## 概要

`calibration_tool.py` は、カメラの外部パラメータ（高さ、角度、位置）を調整・推定するための対話型ツールです。ピンホールカメラモデルに基づいて、画像座標からフロアマップ座標への変換精度を向上させます。

---

## 1. 全体の仕組み

### 1.1 アーキテクチャ

```
calibration_tool.py
├── CalibrationTool (メインクラス)
│   ├── InteractiveCalibrator (対話的調整)
│   ├── CorrespondenceCalibrator (自動最適化)
│   ├── RayCaster (画像→World座標変換)
│   └── FloorMapTransformer (World→フロアマップ座標変換)
└── 対応点データ (correspondence_points_cam01.json)
```

### 1.2 処理フロー

```
1. 設定ファイル読み込み (config.yaml)
   ↓
2. 対応点データ読み込み (correspondence_points_cam01.json)
   ↓
3. カメラパラメータ初期化
   ├── Intrinsics (焦点距離、主点、歪み)
   └── Extrinsics (高さ、角度、位置)
   ↓
4. モード選択
   ├── 自動キャリブレーション (--auto)
   │   └── Levenberg-Marquardt最適化
   └── 対話的調整 (interactive)
       └── 手動でパラメータ調整
   ↓
5. 再投影誤差計算・表示
   ↓
6. 結果保存
```

---

## 2. 計算方法

### 2.1 座標変換の数式

#### ステップ 1: 画像座標 → World 座標（レイキャスティング）

**入力**: 画像座標 `(u, v)` [pixel]

**処理**:

1. **歪み補正**:

   ```python
   (x_n, y_n) = undistort(u, v)  # 正規化カメラ座標
   ```

2. **レイ方向の計算**:

   ```python
   # カメラ座標系でのレイ方向
   ray_camera = [x_n, y_n, 1.0]

   # World座標系への変換
   ray_world = R_inv @ ray_camera
   ```

   ここで `R_inv` は回転行列の逆行列（転置）

3. **床面との交差計算**:

   ```python
   # レイ方程式: P = C + s * ray_world
   # 床面: Z = 0
   # 交点: s = (0 - C_z) / ray_world_z

   s = -camera_height / ray_world[2]
   world_point = camera_position + s * ray_world
   ```

**出力**: World 座標 `(X, Y)` [meters]

#### ステップ 2: World 座標 → フロアマップ座標

**入力**: World 座標 `(X, Y)` [meters]

**処理**:

```python
# カメラ位置を原点とする相対座標
px = camera_x_px + X * scale_x_px_per_m
py = camera_y_px + Y * scale_y_px_per_m
```

**出力**: フロアマップ座標 `(px, py)` [pixel]

**スケール係数**:

- `scale_x_px_per_m = 1000.0 / scale_x_mm_per_px`
- `scale_y_px_per_m = 1000.0 / scale_y_mm_per_px`
- 例: `28.1926 mm/pixel` → `35.47 pixel/m`

### 2.2 自動キャリブレーション（最適化）

#### 最適化アルゴリズム: Levenberg-Marquardt 法

**目的関数**: 再投影誤差の最小化

**パラメータベクトル**:

```python
params = [height_m, pitch_deg, yaw_deg, roll_deg, cam_x_m, cam_y_m]
```

**残差関数**:

```python
def residual_function(params, image_points, world_points):
    # 1. パラメータから外部パラメータを構築
    extrinsics = params_to_extrinsics(params)

    # 2. World座標を画像座標に投影
    projected = project_points(world_points, extrinsics)

    # 3. 残差を計算
    residuals = (projected - image_points).flatten()
    return residuals
```

**最適化**:

```python
from scipy.optimize import least_squares

result = least_squares(
    fun=residual_function,
    x0=initial_params,  # 初期推定値
    args=(image_points, world_points),
    method="lm",  # Levenberg-Marquardt
    max_nfev=100,  # 最大反復回数
    ftol=1e-6,  # 収束許容誤差
)
```

**初期推定**:

- OpenCV の `cv2.solvePnP()` を使用
- 対応点から回転・並進を推定
- 失敗時はデフォルト値: `[2.2, 45.0, 0.0, 0.0, 0.0, 0.0]`

### 2.3 再投影誤差の計算

**定義**: 実際の画像座標と、World 座標から投影した画像座標の差

**計算式**:

```python
# 各対応点について
for (image_point, floormap_point) in point_pairs:
    # フロアマップ座標 → World座標
    world_point = floormap_to_world(floormap_point, camera_position_px)

    # World座標 → 画像座標に投影
    projected = project_points(world_point, extrinsics)

    # 誤差を計算
    error = ||projected - image_point||

# RMSE (Root Mean Square Error)
rmse = sqrt(mean(errors^2))
```

**目標値**: RMSE < 10 pixels

---

## 3. 参照データ

### 3.1 対応点データ形式

**ファイル**: `output/calibration/correspondence_points_cam01.json`

**構造**:

```json
{
  "camera_id": "cam01",
  "description": "対応点データ",
  "metadata": {
    "image_size": {"width": 1280, "height": 720},
    "floormap_size": {"width": 1878, "height": 1369},
    "num_line_segment_correspondences": 11,
    "reference_image": "output/latest/phase1_extraction/frames/...",
    "floormap_image": "data/floormap.png"
  },
  "line_segment_correspondences": [
    {
      "src_line": [[1140.0, 204.0], [1124.0, 400.0]],  // 画像上の垂直線分
      "dst_point": [1080.0, 1299.0]  // フロアマップ上の床面点
    },
    ...
  ]
}
```

**データの意味**:

- `src_line`: 画像上の垂直線分（人物の立ち位置を示す）
  - 上端点: `[1140.0, 204.0]`
  - 下端点（足元）: `[1124.0, 400.0]`
- `dst_point`: フロアマップ上の対応する床面点 `[1080.0, 1299.0]`

**使用される点**: 線分の下端点（足元）が使用される

```python
foot_point = line_bottom  # 線分の下端点
correspondence = (foot_point, dst_point)
```

### 3.2 設定ファイル (config.yaml)

**カメラ内部パラメータ**:

```yaml
camera_params:
  # 焦点距離 [pixel]
  focal_length_x: 1250.0
  focal_length_y: 1250.0

  # 主点 [pixel]
  center_x: 640.0 # 画像幅/2
  center_y: 360.0 # 画像高さ/2

  # 画像サイズ
  image_width: 1280
  image_height: 720

  # 歪み係数 [k1, k2, p1, p2, k3]
  dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

**カメラ外部パラメータ**:

```yaml
camera_params:
  # 高さ [m]
  height_m: 2.2

  # 角度 [度]
  pitch_deg: 45.0 # 俯角（0=水平、正=下向き）
  yaw_deg: 0.0 # 方位角（0=前方、正=右回転）
  roll_deg: 0.0 # 回転角（通常は0）

  # カメラ位置（World座標系）[m]
  camera_x_m: 0.0
  camera_y_m: 0.0

  # カメラ位置（フロアマップ座標系）[pixel]
  position_x_px: 1200.0
  position_y_px: 800.0
```

**フロアマップ設定**:

```yaml
floormap:
  image_width: 1878
  image_height: 1369
  image_x_mm_per_pixel: 28.1926406926406
  image_y_mm_per_pixel: 28.241430700447
```

**キャリブレーション設定**:

```yaml
calibration:
  correspondence_file: "output/calibration/correspondence_points_cam01.json"
```

---

## 4. 主要クラスの詳細

### 4.1 CalibrationTool

**役割**: ツールのメインクラス、対話的インターフェース

**主要メソッド**:

- `show_current_params()`: 現在のパラメータと再投影誤差を表示
- `adjust_parameter(param, delta)`: パラメータを増減
- `set_parameter(param, value)`: パラメータを直接設定
- `auto_calibrate()`: 自動キャリブレーション実行
- `test_transform(u, v)`: 座標変換をテスト
- `save_params(output_path)`: パラメータを保存

### 4.2 InteractiveCalibrator

**役割**: 対話的なパラメータ調整

**主要メソッド**:

- `adjust_parameter(param_name, delta)`: パラメータを調整
- `get_current_error(point_pairs, camera_position_px)`: 再投影誤差を計算
- `get_current_params()`: 現在のパラメータを取得

### 4.3 CorrespondenceCalibrator

**役割**: 対応点から自動でパラメータを推定

**主要メソッド**:

- `calibrate_from_correspondences()`: 対応点データからキャリブレーション
- `calibrate_from_point_pairs()`: 点-点対応からキャリブレーション
- `_residual_function()`: 最適化用の残差関数
- `_project_points()`: World 座標を画像座標に投影

### 4.4 RayCaster

**役割**: 画像座標から床面座標への変換（レイキャスティング）

**主要メソッド**:

- `image_to_floor(pixel, floor_z=0.0)`: 単一点の変換
- `batch_image_to_floor(pixels, floor_z=0.0)`: バッチ変換
- `floor_to_image(world_point)`: 逆変換（World → 画像）

**計算式**:

```python
# 1. 歪み補正
x_n, y_n = undistort(u, v)

# 2. レイ方向
ray_camera = [x_n, y_n, 1.0]
ray_world = R_inv @ ray_camera

# 3. 床面との交差
s = -camera_height / ray_world[2]
world_point = camera_position + s * ray_world
```

### 4.5 FloorMapTransformer

**役割**: World 座標からフロアマップ座標への変換

**主要メソッド**:

- `world_to_floormap(world_point)`: 単一点の変換
- `world_to_floormap_batch(world_points)`: バッチ変換
- `floormap_to_world(floormap_point)`: 逆変換

**計算式**:

```python
px = camera_x_px + world_x * scale_x_px_per_m
py = camera_y_px + world_y * scale_y_px_per_m
```

---

## 5. 使用例

### 5.1 自動キャリブレーション

```bash
# 対話モードで実行
python tools/calibration_tool.py

# 自動キャリブレーションのみ実行
python tools/calibration_tool.py --auto --output calibration_result.json
```

### 5.2 対話的調整

```bash
python tools/calibration_tool.py
```

**コマンド例**:

```
> show                    # 現在のパラメータを表示
> adj height 0.1         # 高さを0.1m増やす
> adj pitch -2.0          # 俯角を2度減らす（上向きに）
> adj yaw 5.0             # 方位角を5度右に回転
> set height 2.5          # 高さを2.5mに直接設定
> test 640 400           # 画像座標(640, 400)の変換をテスト
> auto                    # 自動キャリブレーション実行
> save result.json        # 現在のパラメータを保存
> quit                    # 終了
```

### 5.3 パラメータ調整のコツ

| 問題                       | 原因                     | 対処法                                  |
| -------------------------- | ------------------------ | --------------------------------------- |
| 変換結果が右にずれる       | 方位角(yaw)が小さい      | `adj yaw 2.0`                           |
| 変換結果が左にずれる       | 方位角(yaw)が大きい      | `adj yaw -2.0`                          |
| 変換結果が遠くに投影される | 俯角(pitch)が大きい      | `adj pitch -2.0`                        |
| 変換結果が近くに投影される | 俯角(pitch)が小さい      | `adj pitch 2.0`                         |
| 全体的にずれる             | カメラ位置が間違っている | `position_x_px`, `position_y_px` を調整 |

---

## 6. 数値計算の詳細

### 6.1 座標系の定義

**World 座標系**:

- 原点: カメラ直下の床面
- X: 右方向 [meters]
- Y: 前方（カメラが向いている方向）[meters]
- Z: 上方向 [meters]

**Camera 座標系**:

- 原点: カメラ中心
- X: 右方向
- Y: 下方向
- Z: 前方（光軸）

**Image 座標系**:

- 原点: 左上
- u: 右方向 [pixel]
- v: 下方向 [pixel]

**Floormap 座標系**:

- 原点: 左上
- px: 右方向 [pixel]
- py: 下方向 [pixel]

### 6.2 回転行列の構築

**順序**: Yaw → 基底変換 → Pitch → Roll

```python
# 基底変換: World → Camera初期状態
R_base = [[1, 0, 0],
          [0, 0, -1],
          [0, 1, 0]]

# Yaw: World Z軸周り回転
R_yaw = [[cos(yaw), -sin(yaw), 0],
         [sin(yaw),  cos(yaw), 0],
         [0,         0,        1]]

# Pitch: Camera X軸周り回転
R_pitch = [[1, 0,        0],
           [0, cos(pitch), -sin(pitch)],
           [0, sin(pitch),  cos(pitch)]]

# Roll: Camera Z軸周り回転
R_roll = [[cos(roll), -sin(roll), 0],
          [sin(roll),  cos(roll), 0],
          [0,          0,         1]]

# 合成
R = R_roll @ R_pitch @ R_base @ R_yaw
```

### 6.3 歪み補正

**放射歪み**:

```python
r^2 = x^2 + y^2
x_corrected = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)
y_corrected = y * (1 + k1*r^2 + k2*r^4 + k3*r^6)
```

**接線歪み**:

```python
x_corrected += 2*p1*x*y + p2*(r^2 + 2*x^2)
y_corrected += p1*(r^2 + 2*y^2) + 2*p2*x*y
```

---

## 7. トラブルシューティング

### 7.1 再投影誤差が大きい（> 20 pixels）

**原因**:

- 対応点の精度が低い
- カメラ位置が間違っている
- 初期推定値が悪い

**対処法**:

1. 対応点を再収集（より正確な点を選択）
2. `auto` コマンドで自動キャリブレーション
3. 手動で `position_x_px`, `position_y_px` を調整

### 7.2 最適化が収束しない

**原因**:

- 対応点が少ない（< 4 点）
- 対応点が一直線上に並んでいる
- 初期推定値が悪い

**対処法**:

1. 対応点を増やす（最低 8 点以上推奨）
2. 対応点を画像全体に均等に配置
3. 初期推定値を手動で設定

### 7.3 変換結果が NaN になる

**原因**:

- 地平線より上の点（床面と交差しない）
- カメラの後方の点

**対処法**:

- 正常な動作（変換不可な点）
- `TransformResult.is_valid == False` で判定

---

## 8. 参考資料

- **設定ファイル**: `config.yaml`
- **対応点データ**: `output/calibration/correspondence_points_cam01.json`
- **API リファレンス**: `src/transform/` モジュールの docstring
- **テスト例**: `tests/transform/test_unified_transformer.py`
- **使い方ガイド**: `docs/guides/phase3_high_precision_transform.md`

---

## まとめ

`calibration_tool.py` は以下の機能を提供します:

1. **自動キャリブレーション**: 対応点から最適なパラメータを推定
2. **対話的調整**: 手動でパラメータを微調整
3. **リアルタイム検証**: 再投影誤差をリアルタイムで表示
4. **座標変換テスト**: 特定の画像座標がどこに変換されるか確認

**目標精度**: RMSE < 10 pixels

**推奨手順**:

1. 対応点データを準備（最低 8 点以上）
2. `auto` コマンドで自動キャリブレーション
3. 必要に応じて手動で微調整
4. `save` コマンドで結果を保存
