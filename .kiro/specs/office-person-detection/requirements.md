# 要件定義書

## イントロダクション

本システムは、オフィス内の定点カメラ映像から AI による人物検出・カウント・ゾーン別集計を実現し、バッチ処理で人数分布を把握するシステムです。Vision Transformer（ViT）を使用した人物検出、ホモグラフィ変換によるフロアマップへの座標変換、ゾーン判定による滞在分析を行います。開発環境は macOS（Apple M1 Max、64GB RAM、MPS GPU）を使用します。

## 用語集

- **System**: オフィス人物検出・集計システム
- **Video Input**: 入力動画ファイル（merged_moviefiles.mov, 1280×720、タイムラプス倍速録画）
- **ViT**: Vision Transformer - 画像をパッチに分割し Self-Attention 機構で処理する Transformer ベースの画像認識モデル
- **Detection**: 映像フレーム内の人物検出結果（バウンディングボックス、信頼度）
- **Patch Embedding**: 画像を固定サイズのパッチに分割し、ベクトル表現に変換する処理
- **Attention Map**: ViT の Self-Attention 層が生成する、画像領域間の関連性を示すマップ
- **Camera Coordinates**: カメラ映像上のピクセル座標系
- **Floor Map**: オフィスのフロアマップ画像（data/floormap.png、1878×1369 pixel、原点オフセット (7, 9) pixel、スケール 28.19/28.24 mm/pixel）
- **Floor Map Coordinates**: フロアマップ上のピクセル座標系（原点は画像左上、右方向が X 軸、下方向が Y 軸）
- **Homography Transform**: カメラ座標からフロアマップ座標への 3×3 射影変換行列
- **Zone**: フロアマップ上で定義された多角形エリア（ピクセル座標で指定）
- **Zone Count**: 各ゾーン内の人数カウント結果
- **Batch Processing**: 録画済み動画ファイルの一括処理モード
- **Frame Sampling**: タイムスタンプベースの 5 分刻みフレーム抽出（12:10, 12:15, 12:20 など、±10 秒許容誤差）
- **Timelapse Video**: タイムラプス撮影による倍速録画動画
- **Ground Truth Data**: 手動アノテーション結果（output/labels/result_fixed.json）
- **MPS**: Metal Performance Shaders（Apple Silicon GPU アクセラレーション）
- **Fine-tuning**: 既存モデルを特定データセットで追加学習するプロセス
- **Office Object Detection Dataset**: Kaggle データセット（Person クラス 29,200 枚含む、COCO ベース）
- **IoU**: Intersection over Union（検出精度評価指標）

## 要件

### 要件 1: 動画入力処理

**ユーザーストーリー:** システム管理者として、タイムラプス録画されたオフィスの定点カメラ映像を入力できるようにしたい。これにより、録画済みファイルから人物検出を実行できる。

#### 受入基準

1. THE System SHALL input/merged_moviefiles.mov から H.264 形式のタイムラプス動画ファイル（1280×720, 30fps）を読み込む
2. THE System SHALL 各フレームの右上に表示されたタイムスタンプを OCR または固定位置読み取りで取得する
3. THE System SHALL タイムスタンプが区切りの良い 5 分刻み（12:10:00, 12:15:00, 12:20:00 など）に最も近いフレームを抽出する（±10 秒の許容誤差）
4. THE System SHALL 抽出されたフレームのタイムスタンプを検出結果と関連付ける
5. WHEN 動画ファイルが存在しない場合、THEN THE System SHALL エラーメッセージを出力し、処理を中断する
6. WHEN タイムスタンプの読み取りに失敗した場合、THEN THE System SHALL 警告を出力し、そのフレームをスキップする
7. THE System SHALL 抽出したフレームを処理パイプラインに渡す
8. 入力動画のタイムスタンプ期間は 2025/08/26 16:04:16 ~ 2025/08/29 13:45:39 である -> 期待する出力：2025/08/26 16:05:00, 2025/08/26 16:10:00...（誤差許容含む）

### 要件 2: 人物検出（Vision Transformer）

**ユーザーストーリー:** データアナリストとして、映像フレームから人物を自動検出したい。これにより、手動カウントの負荷を削減できる。

#### 受入基準

1. THE System SHALL Vision Transformer（ViT）ベースの物体検出モデル（DETR または ViT-Det）を使用して抽出フレームから人物を検出する
2. THE System SHALL 入力画像をパッチ（16×16 または 32×32 ピクセル）に分割し、Patch Embedding を生成する
3. THE System SHALL Transformer Encoder による Self-Attention 処理を実行し、画像全体の文脈情報を抽出する
4. THE System SHALL 検出結果として、バウンディングボックス座標（x, y, width, height）と信頼度スコアを出力する
5. THE System SHALL 信頼度閾値（デフォルト 0.5）以上の検出結果のみを採用する
6. WHERE 信頼度閾値が設定ファイルで指定されている場合、THE System SHALL その値を使用する
7. WHERE MPS（Metal Performance Shaders）が利用可能な場合、THE System SHALL MPS を使用して Transformer 演算を高速化する
8. THE System SHALL Attention Map を可視化し、モデルが注目している領域を確認可能にする

### 要件 3: 精度評価・最適化

**ユーザーストーリー:** システム管理者として、人物検出の精度を評価し、最適化したい。これにより、誤検出や検出漏れを最小化できる。

#### 受入基準

1. THE System SHALL output/labels/result_fixed.json から手動アノテーション結果（80 枚）を読み込む
2. THE System SHALL 検出結果とグラウンドトゥルースを比較し、精度指標（Precision, Recall, F1-score）を計算する
3. THE System SHALL 信頼度閾値、NMS（Non-Maximum Suppression）閾値を調整可能にする
4. THE System SHALL 評価結果を CSV または JSON 形式でレポート出力する
5. THE System SHALL IoU（Intersection over Union）閾値 0.5 を使用して True Positive を判定する

### 要件 3.1: ファインチューニング（代替プラン）

**ユーザーストーリー:** システム管理者として、既存モデルの精度が不十分な場合に、オフィス環境に特化したモデルを作成したい。これにより、現場特化の高精度検出を実現できる。

#### 受入基準

1. WHERE 既存 ViT モデルの精度が目標値（F1-score 0.85）未満の場合、THE System SHALL ファインチューニングプロセスを実行可能にする
2. THE System SHALL Kaggle データセット「Office object detection」（Person クラス 29,200 枚）を学習データとして使用する
3. THE System SHALL ImageNet-21k または COCO 事前学習済み ViT 重みをベースに、オフィス環境データで追加学習を実行する
4. THE System SHALL Transformer 特有の学習戦略（Layer-wise Learning Rate Decay、Warmup）を適用する
5. THE System SHALL 学習時のハイパーパラメータ（エポック数、学習率、バッチサイズ、パッチサイズ）を設定ファイルで指定可能にする
6. THE System SHALL ファインチューニング後のモデルを保存し、検出処理で使用可能にする
7. THE System SHALL ファインチューニング前後の精度比較レポート（Precision, Recall, F1-score, Attention Map）を出力する

### 要件 4: ホモグラフィ変換とフロアマップ座標系

**ユーザーストーリー:** データアナリストとして、カメラ座標をフロアマップ座標に変換したい。これにより、data/floormap.png 上での人物位置を把握できる。

#### 受入基準

1. THE System SHALL 設定ファイルからホモグラフィ変換行列（3×3 行列）を読み込む
2. THE System SHALL 設定ファイルからフロアマップパラメータ（画像サイズ 1878×1369 pixel、原点オフセット (7, 9) pixel、スケール 28.19/28.24 mm/pixel）を読み込む
3. THE System SHALL 検出された人物のバウンディングボックス足元座標（中心下端）をカメラ座標からフロアマップ座標（ピクセル単位）に変換する
4. THE System SHALL 変換後の座標に原点オフセット（7, 9 pixel）を適用する
5. THE System SHALL フロアマップ座標（ピクセル単位）を mm 単位に変換する機能を提供する
6. THE System SHALL 変換後の座標がフロアマップ画像範囲内（0 ≤ x < 1878, 0 ≤ y < 1369）にあるか検証する
7. THE System SHALL 変換後の座標（ピクセル単位と mm 単位）を検出結果に追加する
8. WHEN ホモグラフィ変換行列が設定されていない場合、THEN THE System SHALL エラーメッセージを出力し、処理を中断する
9. WHEN フロアマップパラメータが設定されていない場合、THEN THE System SHALL エラーメッセージを出力し、処理を中断する
10. THE System SHALL data/floormap.png（1878×1369 pixel）を参照座標系として使用する

### 要件 5: ゾーン判定

**ユーザーストーリー:** データアナリストとして、フロアマップ上の特定エリア（ゾーン）内に何人いるかを知りたい。これにより、エリア別の混雑状況を把握できる。

#### 受入基準

1. THE System SHALL 設定ファイルから複数のゾーン定義（多角形の頂点座標リスト、ピクセル単位）を読み込む
2. THE System SHALL ゾーン座標がフロアマップ座標系（原点オフセット適用後）で定義されていることを前提とする
3. THE System SHALL 各人物のフロアマップ座標（ピクセル単位、原点オフセット適用後）が各ゾーン内に含まれるかを判定する（点 in 多角形アルゴリズム）
4. THE System SHALL 各人物に対して、所属するゾーン ID を割り当てる
5. WHERE 人物が複数のゾーンに重複する場合、THE System SHALL すべての該当ゾーンに人物をカウントする
6. WHERE 人物がどのゾーンにも属さない場合、THE System SHALL "未分類" として扱う

### 要件 6: ゾーン別集計

**ユーザーストーリー:** 施設管理者として、各ゾーンの人数を時系列で集計したい。これにより、混雑エリアの特定や時間帯別の利用状況を分析できる。

#### 受入基準

1. THE System SHALL 各フレームごとに、各ゾーンの人数をカウントする
2. THE System SHALL 集計結果をタイムスタンプと共に CSV 形式で出力する（列: timestamp, zone_id, count）
3. THE System SHALL 全フレームの集計結果を output ディレクトリに保存する
4. THE System SHALL 5 分間隔の集計結果を時系列データとして出力する
5. THE System SHALL 各ゾーンの平均人数、最大人数、最小人数を統計情報として出力する

### 要件 7: 出力・可視化

**ユーザーストーリー:** データアナリストとして、検出結果や集計結果を視覚的に確認したい。これにより、システムの動作確認やプレゼンテーション資料作成が容易になる。

#### 受入基準

1. THE System SHALL 検出結果をバウンディングボックス付きの画像ファイルとして output ディレクトリに保存する
2. THE System SHALL data/floormap.png（1878×1369 pixel）上に人物位置（原点オフセット適用後のピクセル座標）をプロットした画像を生成する
3. THE System SHALL フロアマップ上の人物位置を円で表示し、所属ゾーンに応じて色分けする
4. THE System SHALL 各ゾーンを半透明の多角形として色分けしてフロアマップ上に表示する
5. THE System SHALL フレーム番号、タイムスタンプ、ゾーン別人数カウントをフロアマップ画像に重畳表示する
6. THE System SHALL ゾーン別人数の時系列グラフ（PNG 形式）を生成する
7. WHERE デバッグモードが有効な場合、THE System SHALL 中間処理結果（変換座標、ゾーン判定結果、座標値）を画像に重畳表示する
8. THE System SHALL ゾーンの凡例（ゾーン名と色の対応表）を生成する

### 要件 8: 設定管理

**ユーザーストーリー:** システム管理者として、検出パラメータやゾーン定義を外部ファイルで管理したい。これにより、コード変更なしに設定を調整できる。

#### 受入基準

1. THE System SHALL YAML または JSON 形式の設定ファイルを読み込む
2. THE System SHALL 設定ファイルに以下の項目を含める: 入力動画パス、信頼度閾値、ホモグラフィ行列、ゾーン定義、出力ディレクトリ
3. WHEN 設定ファイルが存在しない場合、THEN THE System SHALL デフォルト設定を使用し、警告を出力する
4. THE System SHALL 設定ファイルの検証を行い、不正な値がある場合はエラーを出力する

### 要件 9: エラーハンドリング

**ユーザーストーリー:** システム管理者として、処理中のエラーを適切に処理したい。これにより、システムの安定性と保守性が向上する。

#### 受入基準

1. WHEN 動画ファイルの読み込みに失敗した場合、THEN THE System SHALL エラーメッセージを出力し、処理を中断する
2. WHEN ViT モデルのロードに失敗した場合、THEN THE System SHALL エラーメッセージを出力し、処理を中断する
3. WHEN フレーム処理中に例外が発生した場合、THEN THE System SHALL エラーログを記録し、次のフレームに進む
4. THE System SHALL すべてのエラーメッセージをログファイルに記録する

### 要件 10: パフォーマンス

**ユーザーストーリー:** システム管理者として、システムが効率的に動作することを確認したい。これにより、バッチ処理の実用性が保証される。

#### 受入基準

1. THE System SHALL 1280×720 解像度のフレームを 2 秒以内に処理する（MPS 使用時、ViT の Self-Attention 計算を含む）
2. THE System SHALL メモリ使用量を 12GB 以下に抑える（Transformer の Attention Matrix を考慮）
3. THE System SHALL タイムスタンプベースの 5 分刻みフレーム抽出により、実時間 1 時間分のタイムラプス動画から 12 フレームを処理する
4. WHERE MPS（Metal Performance Shaders）が利用可能な場合、THE System SHALL MPS を使用して Transformer 演算を高速化する
5. THE System SHALL Apple M1 Max 環境で全処理を 10 分以内に完了する
6. THE System SHALL バッチ推論により、複数フレームを効率的に処理する
