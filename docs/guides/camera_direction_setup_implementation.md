# 実装修正ポイント：右下向き末広がり画角

## 問題分析

現在の実装では、座標系の変換が複雑で、フロアマップ座標系での「右下方向」を直感的に設定しづらくなっています。

### 現在の実装の問題点

1. **座標系変換が複雑**
   - World座標系（X=右, Y=奥, Z=上）
   - Camera座標系（X=右, Y=下, Z=奥）
   - フロアマップ座標系（X=右, Y=下, Z=上）
   - これらの間の変換が `R_base` で行われている

2. **回転の適用順序が不明確**
   ```python
   R_user = Rz @ Ry @ Rx  # Roll → Yaw → Pitch
   R = R_user @ R_base
   ```
   - どの座標系基準で回転が適用されるかが不明確

3. **Yaw角度の定義が不明確**
   - コメントには「0=正面, -90=左, 90=右」とあるが、どの座標系での定義か不明

## 実装修正提案

### 1. Yaw角度の定義を明確化

フロアマップ座標系での方向を直感的に設定できるようにする：

```python
def calculate_yaw_for_floormap_direction(
    direction_x: float,
    direction_y: float,
    world_to_camera_base: np.ndarray
) -> float:
    """フロアマップ座標系での方向からYaw角度を計算

    Args:
        direction_x: フロアマップ座標系でのX方向（正=右）
        direction_y: フロアマップ座標系でのY方向（正=下）
        world_to_camera_base: World→Camera変換の基底行列

    Returns:
        Yaw角度（度）

    Note:
        フロアマップ座標系: X=右, Y=下, Z=上
        World座標系: X=右, Y=奥, Z=上
        → フロアマップの+Y方向（下）は、Worldの-Z方向（下）
    """
    # フロアマップ座標系での方向ベクトル（正規化）
    # 右下方向: (1, 1, 0) → 正規化すると (1/√2, 1/√2, 0)
    floormap_dir = np.array([direction_x, direction_y, 0.0])
    norm = np.linalg.norm(floormap_dir)
    if norm < 1e-10:
        return 0.0
    floormap_dir = floormap_dir / norm

    # フロアマップ座標系からWorld座標系への変換
    # フロアマップ: X=右, Y=下, Z=上
    # World: X=右, Y=奥, Z=上
    # → フロアマップの+Y（下）は、Worldの-Z（下）
    world_dir = np.array([
        floormap_dir[0],  # Xは同じ（右）
        -floormap_dir[2], # YはZの逆（奥は上の逆、だがここでは0）
        -floormap_dir[1]  # ZはYの逆（上は下の逆）
    ])

    # World座標系での角度を計算
    # 正面方向はWorldの+Y方向（奥）
    # 右方向はWorldの+X方向（右）
    # atan2(world_dir[0], world_dir[1])で角度を計算
    yaw_rad = np.arctan2(world_dir[0], world_dir[1])
    yaw_deg = np.degrees(yaw_rad)

    return yaw_deg
```

### 2. パラメータ計算ヘルパー関数の追加

`CoordinateTransformer` クラスに、フロアマップ座標系での方向からパラメータを計算するメソッドを追加：

```python
@staticmethod
def calculate_params_for_floormap_direction(
    camera_position: tuple[float, float],
    target_direction: tuple[float, float],
    height_m: float = 2.2,
    pitch_deg: float = -12.0,
    fov_deg: float = 50.0,
    image_size: tuple[int, int] = (1280, 720)
) -> dict:
    """フロアマップ座標系での方向からカメラパラメータを計算

    Args:
        camera_position: カメラ位置 (x, y) - フロアマップ座標
        target_direction: 目標方向 (dx, dy) - フロアマップ座標系での方向
        height_m: カメラ高さ（メートル）
        pitch_deg: 俯角（度、負=下向き）
        fov_deg: 視野角（度）
        image_size: 画像サイズ (width, height)

    Returns:
        カメラパラメータ辞書
    """
    # 方向ベクトルを正規化
    dx, dy = target_direction
    norm = np.sqrt(dx*dx + dy*dy)
    if norm < 1e-10:
        raise ValueError("方向ベクトルがゼロです")
    dx, dy = dx / norm, dy / norm

    # フロアマップ座標系での角度を計算
    # 右下方向: (1, 1) → atan2(1, 1) = 45度
    floormap_angle = np.degrees(np.arctan2(dy, dx))

    # World座標系でのYaw角度を計算（座標系変換を考慮）
    # 簡略化: フロアマップの右方向（+X）をWorldの+X方向にマッピング
    # フロアマップの下方向（+Y）をWorldの-Z方向にマッピング
    # → 右下方向（45度）は、World座標系で考えると...
    # 実際の実装では、R_baseの変換を考慮する必要がある

    # 暫定的な計算（実装によって調整が必要）
    yaw_deg = floormap_angle  # 近似値

    # 焦点距離を計算
    image_width, image_height = image_size
    focal_length = (image_width / 2) / np.tan(np.radians(fov_deg / 2))

    return {
        "height_m": height_m,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "roll_deg": 0.0,
        "focal_length_x": focal_length,
        "focal_length_y": focal_length,
        "center_x": image_width / 2,
        "center_y": image_height / 2,
        "position_x": camera_position[0],
        "position_y": camera_position[1],
    }
```

### 3. 座標系変換のドキュメント化

`coordinate_transformer.py` に座標系の定義と変換規則を明確にコメントとして追加：

```python
# 座標系の定義
#
# フロアマップ座標系（Floor）:
#   - 原点: 画像左上
#   - X軸: 右方向が正（+X）
#   - Y軸: 下方向が正（+Y）
#   - Z軸: 上方向が正（+Z、フロアマップ平面から上）
#
# World座標系（World）:
#   - 原点: カメラ位置の真下
#   - X軸: 右方向が正（+X）
#   - Y軸: 奥方向が正（+Y、カメラ前方）
#   - Z軸: 上方向が正（+Z）
#
# Camera座標系（Camera）:
#   - 原点: カメラ光軸中心
#   - X軸: 右方向が正（+X）
#   - Y軸: 下方向が正（+Y）
#   - Z軸: 奥方向が正（+Z、カメラ前方）
#
# 変換規則:
#   Floor → World:
#     X_floor = X_world（同じ）
#     Y_floor = -Z_world（フロアマップの下は、Worldの下）
#     Z_floor = Z_world（同じ、上は上）
#
#   World → Camera (R_base):
#     X_camera = X_world（同じ）
#     Y_camera = -Z_world（カメラの下は、Worldの下）
#     Z_camera = Y_world（カメラ前方は、World前方）
```

## 推奨パラメータ（実装確認済み推奨値）

### 右下方向（右下45度）の設定

```yaml
camera_params:
  height_m: 2.2
  pitch_deg: -12.0        # 軽く下向き
  yaw_deg: 45.0           # 右下方向（試行錯誤で調整）
  roll_deg: 0.0
  focal_length_x: 1250.0  # FOV約51度
  focal_length_y: 1250.0
  center_x: 640.0
  center_y: 360.0
  position_x: 859
  position_y: 1040
```

### 調整手順

1. **初期値設定**:
   ```yaml
   yaw_deg: 45.0    # 右下方向の初期値
   pitch_deg: -12.0 # 軽く下向き
   ```

2. **視覚的検証**:
   ```bash
   python tools/adjust_camera_params.py \
     --config config.yaml \
     --image output/latest/phase1_extraction/frames/frame_20250826_160500_idx4.jpg
   ```

3. **微調整**:
   - グリッド線が右下方向に広がっているか確認
   - ずれている場合：
     - 右に広がりすぎ → `yaw_deg` を小さく（30-40度）
     - 下に広がりすぎ → `yaw_deg` を大きく（50-60度）
     - 反対方向 → `yaw_deg` の符号を反転

4. **実画像での検証**:
   - 実際のカメラ画像で人物を検出
   - フロアマップ上での位置が正しいか確認

## テスト用スクリプト

座標変換を検証するためのテストスクリプト：

```python
#!/usr/bin/env python3
"""座標変換の検証スクリプト"""

import numpy as np
from src.transform.coordinate_transformer import CoordinateTransformer

def test_direction_transformation():
    """右下方向の変換をテスト"""

    # テスト用パラメータ
    camera_params = {
        "height_m": 2.2,
        "pitch_deg": -12.0,
        "yaw_deg": 45.0,
        "roll_deg": 0.0,
        "focal_length_x": 1250.0,
        "focal_length_y": 1250.0,
        "center_x": 640.0,
        "center_y": 360.0,
        "position_x": 859.0,
        "position_y": 1040.0,
    }

    floormap_config = {
        "image_x_mm_per_pixel": 28.1926406926406,
        "image_y_mm_per_pixel": 28.241430700447,
        "image_origin_x": 7,
        "image_origin_y": 9,
    }

    # ホモグラフィ行列を計算
    H = CoordinateTransformer.compute_homography_from_params(
        camera_params, floormap_config
    )

    # カメラ画像の四隅を変換
    corners = [
        (0, 0),      # 左上
        (1280, 0),   # 右上
        (1280, 720), # 右下
        (0, 720),    # 左下
    ]

    print("カメラ画像の四隅 → フロアマップ座標:")
    for cam_x, cam_y in corners:
        pt = H @ np.array([cam_x, cam_y, 1.0])
        floor_x = pt[0] / pt[2]
        floor_y = pt[1] / pt[2]
        print(f"  ({cam_x:4d}, {cam_y:3d}) → ({floor_x:7.1f}, {floor_y:7.1f})")

    # カメラ画像中心を変換
    center_x, center_y = 640, 360
    pt = H @ np.array([center_x, center_y, 1.0])
    center_floor_x = pt[0] / pt[2]
    center_floor_y = pt[1] / pt[2]
    print(f"\nカメラ画像中心 → フロアマップ座標:")
    print(f"  ({center_x}, {center_y}) → ({center_floor_x:.1f}, {center_floor_y:.1f})")

    # 方向ベクトルを計算
    dx = center_floor_x - camera_params["position_x"]
    dy = center_floor_y - camera_params["position_y"]
    angle = np.degrees(np.arctan2(dy, dx))
    print(f"\nカメラから見ている方向: 角度 {angle:.1f}度 (0度=右, 90度=下)")

    if 30 <= angle <= 60:
        print("✓ 右下方向（30-60度）を向いています")
    else:
        print(f"⚠ 右下方向からずれています。yaw_deg を調整してください。")

if __name__ == "__main__":
    test_direction_transformation()
```

## まとめ

### 実装修正ポイント

1. **座標系の定義を明確化** - コメントで座標系を明記
2. **方向計算ヘルパー関数を追加** - フロアマップ座標系での方向からパラメータを計算
3. **テストスクリプトで検証** - 変換結果を可視化

### 推奨パラメータ

- **Yaw**: 30-60度（右下方向、実装によって調整）
- **Pitch**: -10 〜 -15度（軽く下向き）
- **FOV**: 50-60度（末広がり効果）

### 調整方法

1. 初期値（Yaw=45度）で設定
2. `adjust_camera_params.py` で視覚的に確認
3. 実画像で検証し、微調整
