# ベースライン評価用データ準備ガイド

ベースライン評価を実行するために必要なデータ（Ground Truth トラックと対応点データ）の作成手順を説明します。

## 概要

ベースライン評価では、以下のデータが必要です：

1. **Ground Truth トラック**（オプション）: MOT メトリクス評価用
2. **対応点データ**（オプション）: 再投影誤差評価用

これらのデータがない場合でも、パフォーマンス測定は実行可能です。

## Ground Truth トラックの作成

### 目的

MOT メトリクス（MOTA、IDF1）を評価するために、正解のトラックデータが必要です。

### データ形式

Ground Truth トラックは、以下の JSON 形式で保存します：

```json
{
  "tracks": [
    {
      "track_id": 1,
      "trajectory": [
        { "x": 100.0, "y": 200.0, "frame": 0 },
        { "x": 105.0, "y": 205.0, "frame": 1 },
        { "x": 110.0, "y": 210.0, "frame": 2 }
      ]
    },
    {
      "track_id": 2,
      "trajectory": [
        { "x": 300.0, "y": 400.0, "frame": 0 },
        { "x": 305.0, "y": 405.0, "frame": 1 }
      ]
    }
  ]
}
```

### 作成手順

1. **動画を確認**: 評価対象の動画を再生し、人物の位置を確認します。

2. **トラックを記録**: 各人物について、フレームごとの位置（フロアマップ座標）を記録します。

   - フレーム番号は、抽出されたフレームのインデックス（0 始まり）を使用します。
   - 座標は、フロアマップ上のピクセル座標（原点オフセット適用後）を使用します。

3. **JSON ファイルに保存**: 上記の形式で JSON ファイルとして保存します。

### 推奨ツール

- **手動アノテーション**: 動画プレーヤーと座標取得ツールを使用
- **アノテーションツール**: CVAT、LabelMe 等のアノテーションツールを使用
- **可視化ツール**: `tools/interactive_visualizer.py`を使用してトラックを確認

### 注意事項

- 少なくとも 1 セッション分（複数フレーム）のデータが必要です。
- トラック ID は一貫して使用してください（同じ人物には同じ ID を割り当てます）。
- フレーム間で人物が見えない場合でも、可能な限りトラックを継続してください。

## 対応点データの作成

### 目的

再投影誤差を評価するために、カメラ画像上の点とフロアマップ上の対応点が必要です。

### データ形式

対応点データは、以下の JSON 形式で保存します：

```json
{
  "src_points": [
    [640.0, 360.0],
    [800.0, 400.0],
    [1000.0, 500.0]
  ],
  "dst_points": [
    [1000.0, 800.0],
    [1200.0, 900.0],
    [1400.0, 1100.0]
  ],
  "camera_id": "cam01",
  "description": "床面上のマーカー点"
}
```

### 作成手順

1. **マーカーを配置**: 床面上に、カメラから見える位置にマーカー（例: テープ、コーン）を配置します。

   - 少なくとも 10-20 点のマーカーを配置することを推奨します。
   - マーカーは床面上のみに配置してください（壁や天井は避けます）。

2. **カメラ画像上の座標を取得**: 各マーカーのカメラ画像上のピクセル座標を記録します。

   - 画像の左上を原点(0, 0)とします。
   - 座標は `(x, y)` 形式で記録します（x: 横方向、y: 縦方向）。

3. **フロアマップ上の座標を取得**: 各マーカーのフロアマップ上のピクセル座標を記録します。

   - フロアマップ画像の左上を原点(0, 0)とします。
   - 原点オフセットは後で適用されます。

4. **JSON ファイルに保存**: 上記の形式で JSON ファイルとして保存します。

### 推奨ツール

- **画像ビューア**: 画像を開いて座標をクリックで取得
- **OpenCV**: Python スクリプトで画像を表示し、クリックイベントで座標を取得
- **キャリブレーションツール**: `src/calibration/camera_calibrator.py`を使用

### 注意事項

- マーカーは床面上のみに配置してください。
- マーカーは画像全体に分散して配置してください（一箇所に集中しない）。
- 少なくとも 10 点以上、可能であれば 20 点以上の対応点を取得してください。
- カメラ画像とフロアマップの対応関係を正確に記録してください。

## テンプレートファイル

プロジェクトには、以下のテンプレートファイルが含まれています：

- `data/gt_tracks_template.json`: Ground Truth トラックのテンプレート
- `data/correspondence_points_cam01.json.template`: 対応点データのテンプレート

これらのテンプレートをコピーして、実際のデータで置き換えてください。

## 使用方法

### Ground Truth トラックの準備

#### 1. 自動生成

COCO形式のアノテーションデータから自動生成します：

```bash
python scripts/generate_gt_tracks.py \
    --input output/labels/result_fixed.json \
    --output data/gt_tracks_auto.json \
    --config config.yaml \
    --min-distance 50.0
```

**引数説明**:
- `--input`: COCO形式JSONファイルのパス（必須）
- `--output`: 出力ファイルパス（デフォルト: `data/gt_tracks_auto.json`）
- `--config`: 設定ファイルパス（デフォルト: `config.yaml`）
- `--min-distance`: 同一人物判定の距離閾値（ピクセル、デフォルト: 50.0）

**注意**: 自動生成されたデータは簡易的なトラッキングアルゴリズムを使用しているため、手動編集が必要です。

#### 2. 手動編集

自動生成されたトラックを手動で編集します：

```bash
python tools/edit_gt_tracks.py \
    --tracks data/gt_tracks_auto.json \
    --config config.yaml
```

**操作説明**:
- **← →**: フレーム移動
- **クリック**: トラック選択
- **ドラッグ**: ポイント移動
- **1-9**: トラックID変更
- **d**: ポイント削除
- **a**: 新しいトラック追加
- **s**: 保存
- **q**: 終了

**編集内容**:
- 個体ごとのID割り振り
- マップ上での位置対応の確認・修正
- 時系列上のトラック整合性の補正

#### 3. 評価での使用

編集済みのGround Truthトラックを使用して評価を実行：

```bash
python scripts/evaluate_baseline.py \
    --session <session_id> \
    --gt data/gt_tracks_auto.json \
    --config config.yaml
```

### 対応点データの準備

#### 1. 対応点の収集

既存の`tools/homography_calibrator.py`を使用して対応点を収集します：

```bash
python tools/homography_calibrator.py \
    --reference-image <カメラ参照画像のパス> \
    --config config.yaml \
    --camera-id cam01 \
    --output-format template \
    --min-points 10
```

**引数説明**:
- `--reference-image`: カメラ参照画像のパス（必須）
- `--config`: 設定ファイルパス（デフォルト: `config.yaml`）
- `--camera-id`: カメラID（デフォルト: `cam01`）
- `--output-format`: 出力形式（`template` または `legacy`、デフォルト: `template`）
- `--min-points`: 最小対応点数（デフォルト: 4）

**出力**: `output/calibration/correspondence_points_cam01.json`（テンプレート形式）

#### 2. 評価での使用

対応点データを使用して再投影誤差を評価：

```bash
python scripts/evaluate_baseline.py \
    --session <session_id> \
    --points output/calibration/correspondence_points_cam01.json \
    --config config.yaml
```

### 両方を使用する場合

```bash
python scripts/evaluate_baseline.py \
    --session <session_id> \
    --gt data/gt_tracks_auto.json \
    --points output/calibration/correspondence_points_cam01.json \
    --config config.yaml
```

## トラブルシューティング

### Ground Truth トラックが読み込めない

- JSON ファイルの形式を確認してください。
- フレーム番号が 0 始まりの連番になっているか確認してください。
- 座標が数値型（float）になっているか確認してください。

### 対応点データが読み込めない

- JSON ファイルの形式を確認してください。
- `src_points`と`dst_points`の数が一致しているか確認してください。
- 座標が数値型（float）のリストになっているか確認してください。

### 評価結果が期待と異なる

- データの精度を確認してください（座標の誤差、フレームの対応関係など）。
- 設定ファイル（`config.yaml`）のパラメータを確認してください。
- ログファイルを確認して、エラーや警告がないか確認してください。
