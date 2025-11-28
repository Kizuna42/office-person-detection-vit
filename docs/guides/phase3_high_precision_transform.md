# Phase 3 高精度座標変換モジュール 使い方ガイド

## 概要

Phase 3 の座標変換システムをピンホールカメラモデルベースの高精度システムに刷新しました。このガイドでは、新しい機能の使い方とパラメータのチューニング方法を説明します。

## 主な改善点

1. **ピンホールカメラモデル**: 物理的に正確な透視変換
2. **カメラ位置パラメータ**: フロアマップ上でのカメラ位置を明示的に設定
3. **ハイブリッドキャリブレーション**: 自動推定 + 手動微調整
4. **歪み補正**: レンズ歪みを考慮した変換

---

## 1. 基本的な設定（config.yaml）

### 1.1 カメラ内部パラメータ（Intrinsics）

カメラの焦点距離、主点位置、歪み係数を設定します。

```yaml
camera_params:
  # 焦点距離（ピクセル）
  # 推定方法:
  #   fx = (画像幅 / 2) / tan(視野角 / 2)
  #   例: 1280px幅、60度視野角 → fx ≈ 1108px
  focal_length_x: 1250.0
  focal_length_y: 1250.0

  # 画像中心（通常は画像サイズの半分）
  center_x: 640.0 # 1280 / 2
  center_y: 360.0 # 720 / 2

  # 画像サイズ
  image_width: 1280
  image_height: 720

  # 歪み係数 [k1, k2, p1, p2, k3]
  # 通常のレンズ: [0, 0, 0, 0, 0]
  # 広角レンズ: 負の値（樽型歪み）
  # 魚眼レンズ: 大きい負の値
  dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

**推定方法**:

- 焦点距離が不明な場合: `calibration_tool.py --auto` で自動推定
- チェスボードキャリブレーション: `CameraCalibrator` を使用

### 1.2 カメラ外部パラメータ（Extrinsics）

カメラの位置と向きを設定します。

```yaml
camera_params:
  # カメラの高さ [メートル]
  height_m: 2.2

  # 俯角（度）: カメラが下を向く角度
  #   0度 = 水平
  #   45度 = 斜め下
  #   90度 = 真下
  pitch_deg: 45.0

  # 方位角（度）: 左右の向き
  #   0度 = 前方（Y+方向）
  #   正の値 = 右回転
  #   負の値 = 左回転
  yaw_deg: 0.0

  # 回転角（度）: カメラの回転
  #   通常は 0度
  roll_deg: 0.0

  # カメラ位置（World座標系）[メートル]
  # カメラ直下を原点とする相対位置
  camera_x_m: 0.0
  camera_y_m: 0.0
```

### 1.3 カメラ位置（フロアマップ座標系）

**重要**: フロアマップ上でカメラがどこにあるかを設定します。

```yaml
camera_params:
  # フロアマップ上のカメラ位置 [ピクセル]
  # これは実際のカメラがフロアマップ画像のどこに位置するかを表します
  position_x_px: 1200.0
  position_y_px: 800.0
```

**設定方法**:

1. フロアマップ画像を開く
2. カメラの設置位置をピクセル座標で確認
3. 値を設定

---

## 2. キャリブレーションツールの使い方

### 2.1 自動キャリブレーション

対応点ファイルから自動でパラメータを推定します。

```bash
# 対話モードで実行
python tools/calibration_tool.py

# または、自動キャリブレーションのみ実行して終了
python tools/calibration_tool.py --auto --output calibration_result.json
```

**事前準備**:

1. 対応点ファイルを用意: `output/calibration/correspondence_points_cam01.json`
2. `config.yaml` に `calibration.correspondence_file` を設定

### 2.2 対話的な手動調整

```bash
python tools/calibration_tool.py
```

**コマンド一覧**:

```
> show                    # 現在のパラメータを表示
> adj height 0.1         # 高さを0.1m増やす
> adj pitch -2.0         # 俯角を2度減らす（上向きに）
> adj yaw 5.0            # 方位角を5度右に回転
> set height 2.5         # 高さを2.5mに直接設定
> test 640 400           # 画像座標(640, 400)の変換をテスト
> auto                   # 自動キャリブレーション実行
> save result.json       # 現在のパラメータを保存
> quit                   # 終了
```

**調整のコツ**:

- `test <u> <v>` で特定の画像座標がどこに変換されるか確認
- 誤差が大きい場合、まず `auto` で自動推定
- その後、微調整で `adj` コマンドを使用

---

## 3. パラメータチューニング手順

### 3.1 ステップ 1: 初期パラメータの設定

基本的なパラメータを設定します。

```yaml
camera_params:
  height_m: 2.2 # 実測値
  pitch_deg: 45.0 # 目視で推定
  yaw_deg: 0.0 # 通常は0
  position_x_px: 1200.0 # フロアマップ上での実測値
  position_y_px: 800.0 # フロアマップ上での実測値
```

### 3.2 ステップ 2: 対応点データの準備

フロアマップ上で既知の点と、その画像上での位置を対応付けます。

```json
{
  "line_segment_correspondences": [
    {
      "src_line": [[600, 300], [600, 500]],  // 画像上の線分
      "dst_point": [1200, 1000]              // フロアマップ上の点
    },
    ...
  ]
}
```

**推奨**: 最低 4 点以上、できれば 10 点以上の対応点を用意

### 3.3 ステップ 3: 自動キャリブレーション

```bash
python tools/calibration_tool.py --auto
```

結果を確認:

- 再投影誤差（RMSE）が 10 pixels 以下が目標
- インライアー比率が 80% 以上が理想

### 3.4 ステップ 4: 手動微調整

自動キャリブレーションの結果が不十分な場合、手動で調整します。

**よくある問題と対処法**:

| 問題                       | 原因                     | 対処法                                  |
| -------------------------- | ------------------------ | --------------------------------------- |
| 変換結果が右にずれる       | 方位角(yaw)が小さい      | `adj yaw 2.0`                           |
| 変換結果が左にずれる       | 方位角(yaw)が大きい      | `adj yaw -2.0`                          |
| 変換結果が遠くに投影される | 俯角(pitch)が大きい      | `adj pitch -2.0`                        |
| 変換結果が近くに投影される | 俯角(pitch)が小さい      | `adj pitch 2.0`                         |
| 全体的にずれる             | カメラ位置が間違っている | `position_x_px`, `position_y_px` を調整 |

**調整手順**:

1. 既知の点（Ground Truth）の画像座標とフロアマップ座標を確認
2. `test <u> <v>` で変換結果を確認
3. 誤差の方向に応じてパラメータを調整
4. 誤差が 10 pixels 以下になるまで繰り返し

### 3.5 ステップ 5: 検証

実際の検出結果で精度を確認します。

```bash
# パイプラインを実行
python main.py --config config.yaml

# 結果を確認
# output/latest/phase3_transform/coordinate_transformations.json
```

---

## 4. よくある質問（FAQ）

### Q1: 焦点距離がわからない

**A**: 自動キャリブレーションを使用するか、チェスボードキャリブレーションを実行:

```python
from src.calibration import CameraCalibrator

calibrator = CameraCalibrator()
camera_matrix, dist_coeffs = calibrator.calibrate_from_images(chessboard_images)
# camera_matrixからfx, fy, cx, cyを取得
```

### Q2: 変換結果が大きくずれる

**A**: 以下を確認:

1. カメラ位置（`position_x_px`, `position_y_px`）が正しいか
2. カメラの高さ（`height_m`）が正しいか
3. 対応点データが正確か

まずは `calibration_tool.py` で自動キャリブレーションを試してください。

### Q3: 地平線上の点が変換できない

**A**: これは正常な動作です。地平線より上の点（空や遠景）は床面と交差しないため、変換できません。

`TransformResult.is_valid == False` で判定できます。

### Q4: バッチ処理でパフォーマンスを向上させたい

**A**: 自動的にバッチ処理が使用されます。`UnifiedTransformer.transform_batch()` を使用すると、NumPy のベクトル化演算で高速化されます。

---

## 5. 高度な使い方

### 5.1 プログラムからの直接使用

```python
from src.transform import (
    CameraIntrinsics,
    CameraExtrinsics,
    RayCaster,
    FloorMapTransformer,
    UnifiedTransformer,
)

# カメラパラメータを設定
intrinsics = CameraIntrinsics(
    fx=1250.0, fy=1250.0,
    cx=640.0, cy=360.0
)

extrinsics = CameraExtrinsics.from_pose(
    camera_height_m=2.2,
    pitch_deg=45.0,
    yaw_deg=0.0
)

# 変換器を作成
ray_caster = RayCaster(intrinsics, extrinsics)
floormap_transformer = FloorMapTransformer(config, (1200.0, 800.0))
transformer = UnifiedTransformer(ray_caster, floormap_transformer)

# 変換実行
result = transformer.transform_pixel((640.0, 400.0))
if result.is_valid:
    print(f"FloorMap座標: {result.floor_coords_px}")
```

### 5.2 TransformPipelineBuilder を使った柔軟な構築

```python
from src.transform import TransformPipelineBuilder

transformer = (
    TransformPipelineBuilder()
    .with_intrinsics(1250.0, 1250.0, 640.0, 360.0)
    .with_extrinsics(2.2, 45.0, 0.0)
    .with_floormap(width_px=1878, height_px=1369)
    .with_camera_position(1200.0, 800.0)
    .build()
)
```

### 5.3 歪み補正の有効化

広角レンズや魚眼レンズを使用している場合:

```yaml
camera_params:
  dist_coeffs: [-0.2, 0.1, 0.0, 0.0, 0.0] # 樽型歪みの例
```

歪み係数の取得方法:

1. OpenCV のチェスボードキャリブレーション
2. 既知の直線が画像上で曲がっている場合、手動で調整

---

## 6. トラブルシューティング

### エラー: "Not initialized"

```python
# 正しい順序
phase = TransformPhase(config, logger)
phase.initialize()  # 必ず初期化
results = phase.execute(detection_results)
```

### エラー: "カメラ位置が設定されていません"

`config.yaml` に以下を追加:

```yaml
camera_params:
  position_x_px: 1200.0
  position_y_px: 800.0
```

### 変換結果が NaN になる

- 地平線より上の点: 正常（変換不可）
- カメラの後方: 正常（変換不可）
- それ以外: パラメータ設定を確認

---

## 7. パフォーマンス最適化

### バッチ処理の利用

```python
# 単一変換
result = transformer.transform_pixel((640.0, 400.0))

# バッチ変換（推奨）
results = transformer.transform_batch([
    (640.0, 400.0),
    (320.0, 350.0),
    (960.0, 450.0),
])
```

### キャッシュの活用

`RayCaster` は逆行列を事前計算してキャッシュします。同じパラメータで複数回使用する場合は、`UnifiedTransformer` を再利用してください。

---

## 8. 参考資料

- 設定ファイル: `config.yaml`
- 対応点データ形式: `output/calibration/correspondence_points_cam01.json`
- API リファレンス: `src/transform/` モジュールの docstring
- テスト例: `tests/transform/test_unified_transformer.py`

---

## まとめ

新しい Phase 3 システムの主な使い方:

1. **基本設定**: `config.yaml` でカメラパラメータを設定
2. **自動キャリブレーション**: `calibration_tool.py --auto` で推定
3. **手動調整**: 対話ツールで微調整
4. **検証**: 実際の検出結果で精度確認

目標精度: **RMSE 10 pixels 以内**

質問や問題があれば、`tools/calibration_tool.py` の `test` コマンドで確認しながら調整してください。
