# フロアマップ統合ドキュメント

## 概要

このドキュメントは、オフィス人物検出システムにおけるフロアマップパラメータの統合について説明します。

## フロアマップ固定パラメータ

以下のパラメータは固定値として設定されています：

| パラメータ           | 値               | 単位     | 備考                  |
| -------------------- | ---------------- | -------- | --------------------- |
| image_width          | 1878             | pixel    | 画像幅                |
| image_height         | 1369             | pixel    | 画像高さ              |
| image_origin_x       | 7                | pixel    | 原点 X 座標オフセット |
| image_origin_y       | 9                | pixel    | 原点 Y 座標オフセット |
| image_x_mm_per_pixel | 28.1926406926406 | mm/pixel | X 軸スケール          |
| image_y_mm_per_pixel | 28.241430700447  | mm/pixel | Y 軸スケール          |

**座標系**: 画像左上が原点、右方向が X 軸、下方向が Y 軸

## 実装された機能

### 1. 設定ファイル (config.yaml)

フロアマップセクションを追加：

```yaml
floormap:
  image_path: "data/floormap.png"
  image_width: 1878
  image_height: 1369
  image_origin_x: 7
  image_origin_y: 9
  image_x_mm_per_pixel: 28.1926406926406
  image_y_mm_per_pixel: 28.241430700447
```

### 2. ConfigManager (src/config_manager.py)

- フロアマップセクションの検証機能を追加
- 必須項目チェック
- 数値範囲の検証

### 3. CoordinateTransformer (src/coordinate_transformer.py)

拡張された機能：

- **原点オフセット対応**: 画像左上からの原点位置を考慮した座標変換
- **ピクセル ⇔mm 変換**: `pixel_to_mm()`, `mm_to_pixel()` メソッド
- **範囲チェック**: `is_within_bounds()` で画像範囲内判定
- **フロアマップ情報取得**: `get_floormap_info()` でパラメータ取得

#### 使用例

```python
from src.coordinate_transformer import CoordinateTransformer
from src.config_manager import ConfigManager

# 設定を読み込み
config = ConfigManager('config.yaml')
floormap_config = config.get_section('floormap')
homography_matrix = config.get('homography.matrix')

# Transformerを初期化
transformer = CoordinateTransformer(homography_matrix, floormap_config)

# カメラ座標をフロアマップ座標に変換
camera_point = (640, 480)
floor_point = transformer.transform(camera_point)  # ピクセル単位
floor_point_mm = transformer.pixel_to_mm(floor_point)  # mm単位

# 範囲チェック
is_valid = transformer.is_within_bounds(floor_point)
```

### 4. データモデル (src/data_models.py)

`Detection` クラスに追加：

- `floor_coords_mm`: フロアマップ座標（mm 単位）

```python
detection = Detection(
    bbox=(100, 100, 50, 100),
    confidence=0.85,
    class_id=0,
    class_name='person',
    camera_coords=(125.0, 200.0),
    floor_coords=(121.0, 233.5),      # ピクセル単位
    floor_coords_mm=(3411.7, 6593.4),  # mm単位
    zone_ids=['zone_a']
)
```

### 5. FloormapVisualizer (src/floormap_visualizer.py)

新規作成された可視化モジュール：

- **ゾーン描画**: 多角形ゾーンを半透明で描画
- **検出結果描画**: 足元位置を円で表示、ゾーン別に色分け
- **フレーム情報表示**: フレーム番号、タイムスタンプ、ゾーン別カウント
- **凡例作成**: ゾーンの色と名前の対応表

#### 使用例

```python
from src.floormap_visualizer import FloormapVisualizer
from src.config_manager import ConfigManager

# 設定を読み込み
config = ConfigManager('config.yaml')
floormap_config = config.get_section('floormap')
zones = config.get('zones')

# Visualizerを初期化
visualizer = FloormapVisualizer(
    floormap_path='data/floormap.png',
    floormap_config=floormap_config,
    zones=zones
)

# フレーム結果を可視化
image = visualizer.visualize_frame(frame_result)
visualizer.save_visualization(image, 'output/floormap_frame_001.png')

# 凡例を作成
legend = visualizer.create_legend()
visualizer.save_visualization(legend, 'output/legend.png')
```

## 座標変換の流れ

1. **カメラ座標**: バウンディングボックスから足元座標を計算
2. **ホモグラフィ変換**: 3x3 行列による射影変換
3. **原点オフセット適用**: 画像左上からの原点位置を加算
4. **フロアマップ座標（ピクセル）**: 画像上の座標
5. **mm 座標変換**: スケール係数を乗算して mm 単位に変換

```
カメラ座標 (x, y)
    ↓ ホモグラフィ変換
変換後座標 (x', y')
    ↓ 原点オフセット適用
フロアマップ座標 (x' + origin_x, y' + origin_y) [pixel]
    ↓ スケール変換
フロアマップ座標 (x_mm, y_mm) [mm]
```

## ゾーン定義

ゾーンはフロアマップ上のピクセル座標で定義します：

```yaml
zones:
  - id: "zone_a"
    name: "会議室エリア"
    polygon:
      - [100, 200] # [x, y] ピクセル座標
      - [300, 200]
      - [300, 400]
      - [100, 400]
```

## 検証とテスト

すべての実装は以下の検証を通過しています：

- ✓ 設定ファイルの検証
- ✓ 座標変換の正確性
- ✓ 原点オフセットの適用
- ✓ ピクセル ⇔mm 変換
- ✓ 範囲チェック
- ✓ ゾーン分類
- ✓ データモデルとの統合

## 注意事項

1. **ホモグラフィ行列の調整**: 実際のカメラ配置に合わせて `config.yaml` の `homography.matrix` を調整する必要があります
2. **ゾーン座標**: ゾーンの多角形座標は、原点オフセット適用後のピクセル座標で指定します
3. **画像範囲**: 変換後の座標が画像範囲外になる場合があるため、`is_within_bounds()` で確認することを推奨します

## 今後の拡張

- キャリブレーションツールの実装（対応点からホモグラフィ行列を自動計算）
- リアルタイム可視化機能
- ヒートマップ生成機能
- 軌跡追跡機能

## カメラキャリブレーション手順

1. **参照フレームの抽出**

   - `python tools/export_reference_frame.py --timestamp 12:20` を実行し、`output/calibration/` にカメラ参照画像 (`reference_*.png`) とフロアマップのグリッドガイド (`floormap_grid.png`) を生成します。
   - 別の時刻・フレームを使用する場合は `--frame-number` や `--scan-interval` を調整してください。

2. **対応点の取得とホモグラフィ計算**

   - `python tools/homography_calibrator.py --reference-image output/calibration/reference_122000_f001234.png --update-config` のように実行します。
   - カメラ画像ウィンドウで対応点をクリックし、その後フロアマップウィンドウで対応点をクリックする操作を繰り返します。
     - `u`: 直前の点を取り消す
     - `c`: すべての点をクリア
     - `s` または Enter: 対応点を確定してホモグラフィを計算
     - `q` または Esc: 中断
   - 処理が完了すると `output/calibration/` に対応点 JSON (`points_*.json`) と行列を含む YAML (`homography_*.yaml`) が保存され、`--update-config` を付与した場合は自動的に `config.yaml` の `homography.matrix` が更新されます（元の設定は `config.backup/` にバックアップされます）。

3. **検証**
   - `tools/homography_calibrator.py` の標準出力に表示される RMSE や最大誤差を確認し、値が大きい場合は対応点を再取得します。
   - 更新後は `python main.py --debug` を実行して、フロアマップ上のプロット位置が期待通りになるか確認してください。
