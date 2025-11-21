# ホモグラフィ行列のベストプラクティス

## 概要

本システムでは、カメラ座標からフロアマップ座標への変換にホモグラフィ行列を使用しています。ホモグラフィ行列の取得方法には 2 つのアプローチがあります。

## ホモグラフィ行列の取得方法

### 1. 対応点ベース（推奨）

**方法**: カメラ画像とフロアマップ上の対応点を手動で収集し、`cv2.findHomography()`で計算

**設定**:

```yaml
homography:
  source: "correspondence_points" # デフォルト
  matrix:
    - [h11, h12, h13]
    - [h21, h22, h23]
    - [h31, h32, h33]
```

**手順**:

1. `tools/homography_calibrator.py`を使用して対応点を収集
2. 点対応または線分対応を選択
3. ホモグラフィ行列を計算して`config.yaml`に保存

**利点**:

- 実測データに基づくため精度が高い
- カメラの物理パラメータが不明でも使用可能
- 歪みや非線形性を自動的に補正

**欠点**:

- 対応点の収集に時間がかかる
- 対応点の精度が結果に直接影響する

**ベストプラクティス**:

- 最低 8 点以上の対応点を収集（推奨: 12-20 点）
- 画像全体に均等に分布する点を選択
- 線分対応を使用する場合は、柱の角など垂直構造物を選択
- RANSAC を使用して外れ値を除去
- RMSE が 2px 以下、最大誤差が 4px 以下を目標

### 2. カメラパラメータベース

**方法**: カメラの物理パラメータ（高さ、角度、焦点距離等）から理論的に計算

**設定**:

```yaml
homography:
  source: "camera_params" # カメラパラメータから計算

camera_params:
  height_m: 2.2 # カメラ高さ（m）
  pitch_deg: -19.0 # 俯角（度、負=下向き）
  yaw_deg: 0.0 # 方位角（度、0=右下方向）
  roll_deg: 0.0 # 回転角（度、0=水平）
  focal_length_x: 1250.0 # 焦点距離X（px）
  focal_length_y: 1250.0 # 焦点距離Y（px）
  center_x: 640.0 # 画像中心X（px）
  center_y: 360.0 # 画像中心Y（px）
  position_x: 859 # カメラ位置X（px、フロアマップ座標）
  position_y: 1040 # カメラ位置Y（px、フロアマップ座標）
```

**利点**:

- カメラの物理パラメータが分かれば自動計算可能
- パラメータ調整による微調整が容易
- 理論的な理解がしやすい

**欠点**:

- カメラパラメータの正確な測定が必要
- 歪みや非線形性を考慮しない
- 実測データとの整合性確認が必要

**ベストプラクティス**:

- `tools/adjust_camera_params.py`を使用してパラメータを調整
- 対応点ベースの結果と比較して検証
- 再投影誤差を確認してパラメータを微調整

## 現在の実装状況

### 変換フェーズでの使用

`src/pipeline/phases/transform.py`の`TransformPhase`では、以下の優先順位でホモグラフィ行列を取得します：

1. `homography.source = "camera_params"`の場合:

   - `camera_params`から`CoordinateTransformer.compute_homography_from_params()`で計算
   - カメラの物理パラメータが使用される

2. `homography.source = "correspondence_points"`の場合（デフォルト）:
   - `homography.matrix`を直接使用
   - 対応点から計算された行列が使用される

### カメラパラメータの活用

現在、以下のカメラパラメータが`config.yaml`に定義されていますが、**デフォルトでは使用されていません**：

- `camera_params.height_m`: カメラ高さ
- `camera_params.pitch_deg`: 俯角
- `camera_params.yaw_deg`: 方位角
- `camera_params.roll_deg`: 回転角
- `camera_params.focal_length_x/y`: 焦点距離
- `camera_params.center_x/y`: 画像中心
- `camera_params.position_x/y`: カメラ位置

**使用するには**:

```yaml
homography:
  source: "camera_params" # この行を追加
```

## 推奨されるワークフロー

### 初回セットアップ

1. **対応点ベースで開始**（推奨）:

   ```bash
   python tools/homography_calibrator.py \
     --reference-image output/latest/phase1_extraction/frames/frame_*.jpg \
     --config config.yaml \
     --min-points 8 \
     --update-config
   ```

2. **精度確認**:

   - RMSE < 2px、最大誤差 < 4px を目標
   - 再投影誤差を確認

3. **必要に応じてカメラパラメータを調整**:
   ```bash
   python tools/adjust_camera_params.py \
     --config config.yaml \
     --reference-image output/latest/phase1_extraction/frames/frame_*.jpg
   ```

### 精度向上のための改善

1. **対応点の追加**:

   - 画像全体に均等に分布する点を追加
   - 線分対応を使用して垂直構造物を活用

2. **RANSAC パラメータの調整**:

   ```bash
   python tools/homography_calibrator.py \
     --ransac-threshold 2.0  # より厳格な閾値
   ```

3. **カメラパラメータの微調整**:
   - `tools/adjust_camera_params.py`で視覚的に確認しながら調整
   - 対応点ベースの結果と比較

## トラブルシューティング

### 再投影誤差が大きい場合

- **対応点の精度を確認**: クリック位置が正確か確認
- **外れ値の除去**: RANSAC の閾値を調整
- **対応点の追加**: より多くの対応点を収集

### カメラパラメータベースで精度が出ない場合

- **パラメータの確認**: 高さ、角度、焦点距離が正確か確認
- **対応点ベースに切り替え**: より実測に近い結果が得られる
- **ハイブリッドアプローチ**: 対応点ベースの結果を初期値として、パラメータを微調整

## 参考資料

- [キャリブレーション・ワークフロー](calibration_workflow.md)
- [カメラ方向設定ガイド](camera_direction_setup.md)
- [座標変換・ゾーン処理ガイド](../.cursor/rules/coordinate-zone-processing.mdc)
