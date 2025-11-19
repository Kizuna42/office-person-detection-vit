## 現状サマリ

- **プロジェクト目的**
  オフィス内の定点カメラタイムラプス映像から、ViT/DETR ベースの人物検出＋DeepSORT 系トラッキングを行い、ホモグラフィによるカメラ→フロアマップ座標変換・ゾーン別人数集計・可視化・精度評価（MOT + 再投影誤差 + パフォーマンス）を一気通貫で実行するバッチ/検証用パイプライン。

- **主なデータ・アーティファクト**
  - **入力**: `input/merged_moviefiles.mov`（1280×720, 30fps, timelapse 想定）
  - **フロアマップ**: `data/floormap.png`（1878×1369, 原点 offset=(7,9), 28.19/28.24 mm/px）
  - **アノテーション**: `data/annotation_images/*.jpg`（検証用画像群）
  - **Ground Truth**:
    - 検出/トラッキング GT（COCO 風）: `output/labels/result_fixed.json`
    - トラック GT テンプレート: `data/gt_tracks_template.json`, 自動生成版: `data/gt_tracks_auto.json`
  - **キャリブレーション**:
    - 対応点: `output/calibration/correspondence_points_cam01.json`
    - ホモグラフィ: `output/calibration/homography_*.yaml`
    - 対応点テンプレ: `data/correspondence_points_cam01.json.template`
  - **最新パイプライン出力（指示された `output/latest`）**:
    - `metadata.json`: 実行 config スナップショット（video/detection/floormap/homography/tracking/calibration など）
    - `phase1_extraction/`: 抽出フレーム 4 枚＋ `extraction_results.csv`
    - `phase2_detection/`: 検出画像 4 枚＋ `detection_statistics.json`
    - `phase2.5_tracking/`: `tracks.json`, `tracks.csv`, `tracking_statistics.json`
    - `phase3_transform/`: `coordinate_transformations.json`（各検出の camera/floor/floor_mm/zone_id）
    - `phase4_aggregation/`: `zone_counts.csv`（timestamp ごとの zone_1〜3 + unclassified）
    - `phase5_visualization/`: フロアマップ画像群, 時系列/統計グラフ, `side_by_side_tracking.mp4`
    - `summary.json`: フェーズ別統計・パフォーマンス集計

- **モデル・モジュール一覧（実装済）**
  - **検出 (`src/detection/vit_detector.py`)**
    - `facebook/detr-resnet-50`（ViT 系 DETR）を使用
    - MPS / CUDA / CPU 自動切替、batch 推論対応、前処理/後処理（NMS, person フィルタ, bbox→foot point）
  - **トラッキング (`src/tracking/*`, `src/pipeline/phases/tracking.py`)**
    - DeepSORT 風 `Tracker`（Kalman Filter + Hungarian + `SimilarityCalculator`）
    - 類似度: appearance embedding（ViTDetector による feature 抽出）＋ IoU ベース motion, `appearance_weight=0.7`, `motion_weight=0.3`
    - `TrackingPhase` が detection 結果 + 元フレームから feature 抽出→追跡→`tracks.json/csv` 出力
  - **座標変換 (`src/transform/coordinate_transformer.py`)**
    - ホモグラフィ＋原点オフセット＋ mm 変換
    - バッチ変換, 足元座標抽出, 歪み補正（内部パラメータ・歪み係数を与えれば有効化）
  - **ゾーン判定 (`src/zone/zone_classifier.py`)**
    - Ray Casting による point-in-polygon 判定
    - `Detection.floor_coords` から `zone_ids` 付与
  - **可視化 (`src/visualization/*`, `src/pipeline/phases/visualization.py`)**
    - `Visualizer`: 時系列グラフ/統計グラフ
    - `FloormapVisualizer`: フロアマップ＋ゾーン＋検出＋track ID・軌跡描画
    - `SideBySideVideoExporter`: 検出画像とフロアマップ画像の並列動画生成
    - `tools/interactive_visualizer.py`: Streamlit ベースのインタラクティブ閲覧（セッション選択・トラック/ゾーン集約など）
  - **評価 (`src/evaluation/evaluation_module.py`, `src/evaluation/mot_metrics.py`, `scripts/evaluate_*`)**
    - 検出精度: Precision/Recall/F1（COCO 風 GT と IoU マッチング）
    - 追跡精度: 簡易 MOTA / IDF1 / ID Switches
    - 座標変換精度: `ReprojectionErrorEvaluator` による再投影誤差（mean/max/min/std）
    - ベースライン統合評価: `scripts/evaluate_baseline.py`（MOTA/IDF1 + 再投影誤差 + パフォーマンス）

- **最新評価指標（数値 or 未提示）**
  - **最新セッション（`summary.json` より, session_id=`20251119_132130`）**
    - 抽出フレーム数: `frames_extracted=4`, success_rate=1.0
    - 検出: `total_detections=85`, `avg_per_frame=21.25`
      - `detection_statistics.json`: confidence mean=0.851, min=0.527, max=0.992, std=0.130, median=0.891
    - 追跡: `total_tracks=21`, `total_trajectory_points=84`, `avg_trajectory_length=4.0`
    - 集計: ゾーン数=3（zone_1〜3）, `zone_counts.csv` で 4 timestamp の人数分布
      - 例: `2025/08/26 16:04:56 → zone_1=2, zone_2=11, zone_3=1, unclassified=8`
    - パフォーマンス（フェーズ別合計時間, 1 回の run）:
      - Phase1 抽出: 32.37 s（100 フレーム OCR スキャン含む）
      - Phase2 検出: 2.92 s（4 フレーム, 平均 1.31 s/batch）
      - Phase2.5 追跡: 2.29 s
      - Phase3 変換: 0.006 s
      - Phase4 集計: 0.003 s
      - Phase5 可視化: 0.628 s
  - **検出精度（Precision/Recall/F1）**
    - 実装はあるが、`output/latest` には `evaluation_report.json/csv` が見当たらず **「未提示」**。
  - **追跡精度（MOTA / IDF1 / ID_Switches）**
    - `scripts/evaluate_mot_metrics.py` + `scripts/evaluate_baseline.py` で算出可能だが、`output/latest` には `mot_metrics.json`・`baseline_metrics.json` が無く **「未提示」**。
  - **座標変換精度（再投影誤差）**
    - `scripts/evaluate_reprojection_error.py` により mean/max error を計算可能だが、`output/latest` には `reprojection_error.json` が無く **「未提示」**。
  - **全体パフォーマンス目標（`evaluate_baseline.py` 内のターゲット）**
    - MOTA ≥ 0.7, IDF1 ≥ 0.8
    - 再投影 mean error ≤ 2 px, max error ≤ 4 px
    - 処理時間 ≤ 2 s/frame
    - メモリ増加 ≤ 12 GB
    → 「目標値は定義済みだが、最新セッションでの達成状況は評価未実行」

- **既知の問題/ギャップ（高レベル）**
  - 実データでの **MOT メトリクス / 再投影誤差 / 検出 F1** が体系的に測定されていない（スクリプトは揃っているが、`baseline_metrics.json` 等が最新セッションに存在しない）。
  - **歪み補正（`calibration.use_distortion_correction`）の有効性評価**が実データで行われていない。
  - **マルチカメラ間のワールド座標整合**は設計/実装ともにほぼ未着手（現状は単一カメラ + floormap 基準）。
  - 座標変換パラメータは `config.yaml` に散在しているが、**撮影現場から見た「何をどう測って埋めるか」のスキーマ/手順が未整理**。
  - 同一オブジェクト類似度（ReID, tracklet マージ）について、`SimilarityCalculator` はあるものの、**閾値設計・評価パイプラインが未整備**。
  - ダッシュボード的な UI は `tools/interactive_visualizer.py` として存在するが、**MOT/再投影/パフォーマンス指標の統合表示や、運用向けレイアウトは未完成**。


## タスク別ギャップ分析（①〜④）

### ① オブジェクト追跡の可視化

- **現状**
  - `TrackingPhase` により DeepSORT 風トラッキングを行い、`tracks.json/csv`・`tracking_statistics.json`・ID 付き検出画像（phase2.5_tracking/images）を生成。
  - `FloormapVisualizer` + `VisualizationPhase` により、各フレームのフロアマップ上に **ゾーン塗りつぶし＋足元位置＋track ID ラベル＋ゾーン別カウント** を描画。
  - `SideBySideVideoExporter` により、検出画像とフロアマップ画像の並列動画 `side_by_side_tracking.mp4` を生成。
  - `tools/interactive_visualizer.py`（Streamlit）が、セッション選択・トラック/ゾーン平均・簡易 ID Switch 推定・MOT 評価呼び出しなどを提供。

- **主要課題**
  - 可視化・評価が **複数ツール/ファイルに分散**しており、「運用ダッシュボード」として一枚岩の UX になっていない。
  - Streamlit UI は存在するが、**最新の `baseline_metrics.json` / `mot_metrics.json` / `reprojection_error.json` との連携や、設定比較・複数セッション横比較が弱い**。
  - 現在の side-by-side 動画はオフライン可視化に留まり、**運用者がブラウザ上で任意 frame / track を drill-down する機能がない**。

- **短期（1–2 週間）推奨アクション**
  - `tools/interactive_visualizer.py` を「正規 UI」として整理し、以下を統合:
    - セッション選択（最新 / 任意 ID）
    - `summary.json`, `tracking_statistics.json`, `zone_counts.csv` の要約（人数時系列・平均在室人数・最大同時人数）
    - `mot_metrics.json`, `reprojection_error.json`, `performance_metrics.json` があれば、その数値と目標値達成状況を表示
  - side-by-side 動画と静止フロアマップを Streamlit から直接プレビュー可能にする（OpenCV → `st.image`, `st.video`）。
  - `tracks.json` 内の任意 `track_id` を選択し、その軌跡の **時間推移／ゾーン滞在履歴** をフロアマップ上でハイライト表示する機能を追加。

- **中期（1–2 ヶ月）推奨アクション**
  - ダッシュボードに複数セッション比較タブを追加し、
    - セッションごとの **MOTA/IDF1/再投影誤差/処理時間** をテーブル＋棒グラフで比較
    - 設定差分（`metadata.json` の detection/tracking/calibration 設定）を diff 表示
  - オフラインレポート生成（`generate_baseline_report.py` など）と UI を連携し、**「このセッションの PDF/markdown レポート生成」ボタン**を提供。
  - 運用向けに、「異常セッション（MOTA や再投影誤差が閾値越え）」を赤色でハイライトするヘルスチェックビューを実装。

- **長期（3 ヶ月〜）推奨アクション**
  - Web UI を CI/CD やスケジューラと連携し、**日次/週次の自動実行 → ダッシュボード自動更新** のパイプラインを構築。
  - セッションが増えた際のスケーラビリティを考慮し、メトリクスを TSDB（Prometheus, InfluxDB など）や Data Warehouse に蓄積してダッシュボード（Grafana 等）に接続。
  - 利用者ロール（開発者/運用/ビジネス）ごとにビューを分けた「本番運用ダッシュボード」として整理。


### ② 座標変換の精度改善（カメラ→ワールド / マルチカメラ）

- **現状**
  - 単一カメラについては、`CoordinateTransformer` + `ReprojectionErrorEvaluator` + `tools/homography_calibrator.py` による **カメラ→フロアマップ** ホモグラフィの推定・検証フローが実装済み。
  - フロアマップパラメータは固定（1878×1369 px, origin=(7,9), 28.19/28.24 mm/px）で統一されており、`floormap` セクションとして `config.yaml` で管理。
  - `scripts/evaluate_reprojection_error.py` が、対応点 JSON + `config.yaml` の homography から mean/max/min/std の再投影誤差を算出し、誤差マップを出力（目標: mean ≤ 2 px）。
  - `docs/plan.md` にも示されている通り、**マルチカメラ間の共通ワールド座標系設計・評価はほぼ未着手**。

- **主要課題**
  - 実運用データの対応点（camera px ↔ floormap px/mm）の収集とバージョン管理が体系化されておらず、**ホモグラフィ品質が場当たり的になりうる**。
  - `use_distortion_correction` を有効にした場合の実データ評価がなく、**内部パラメータ/歪み係数の有効性が不明**。
  - マルチカメラ前提の **ワールド座標系（3D or 2.5D）定義と、カメラごとの外部パラメータ管理/評価フローが未定義**。

- **短期（1–2 週間）推奨アクション**
  - 最低 1 カメラ（cam01）について、**床面上マーカー 10–20 点の対応点**を `data/correspondence_points_cam01.json` として整理（カメラ px, floormap px, optional mm）。
  - `scripts/evaluate_reprojection_error.py` を用いて、現行 homography に対する誤差を定量化（`mean_error`, `max_error`）し、`output/calibration/*_reprojection_error.json` を作成。
  - `calibration.use_distortion_correction` を false/true で比較し、**どの設定が mean/max error を最小化するか**を記録。
  - `CoordinateTransformer._validate_and_convert_matrix` が出す警告（条件数, det）が実データ homography でどうなっているかをログで確認し、異常があれば対応点や RANSAC 設定を見直す。

- **中期（1–2 ヶ月）推奨アクション**
  - `ReprojectionErrorEvaluator` 側で RANSAC 設定（閾値, 反復回数）を外部化し、**外れ値ロバストなホモグラフィ推定**（iterative refinement, outlier removal>5σ）を導入。
  - `CoordinateTransformer` に `source_frame` / `target_frame` の概念を明示し、
    - camera → floormap
    - floormap → world(mm)
    - （将来）camera → world 直接変換
    の API を整理。
  - マルチカメラを想定し、
    - 共通 world frame（floormap + 高さ情報 or 仮想平面）を定義
    - 各カメラに対し、homography + 外部パラメータ（姿勢, 位置）を設定ファイルとして管理
    という「座標系カタログ」を作成。

- **長期（3 ヶ月〜）推奨アクション**
  - 複数カメラから見た同一点の world 座標のズレを **統計的に評価（mean/95% CI）** し、MOT 指標（IDF1, HOTA など）との相関を分析。
  - 運用中に定期的に撮影される calibration target を用いて、**オンライン再投影誤差モニタリング**（閾値超過時にアラート + 再キャリブ要求）を実装。
  - 3D カメラモデル（ピンホールモデル + 射影）に拡張し、**高さ推定や奥行き補正を含む world 座標化**を検討。


### ③ 座標変換パラメータ洗い出しと撮影者向け手順

- **現状**
  - `config.yaml` に `floormap`, `homography`, `calibration`, `camera` セクションが存在し、それぞれ原点オフセット, スケール, 内部パラメータ, 歪み係数, カメラ位置などを保持。
  - Config 構造とバリデーションルールは `.cursor/rules/config-management.mdc` でかなり詳細に定義済み（必須キー, 3×3 行列, ゾーン polygon など）。
  - しかし、「現場の撮影者が測るべき物理量 → config のどのキーにどう反映されるか」が一枚のスキーマ/チェックリストとして整理されていない。

- **主要課題**
  - 新しいオフィス/カメラ設置ごとに **設定漏れ・誤入力（座標系の混在, 単位ミス等）が発生しやすい**。
  - 撮影者側が
    - マーカー設置
    - カメラ高さ/俯角/レンズ焦点距離
    - 床面基準線の実測値
    などをどの粒度で計測すべきか不明確。
  - `config.yaml` を直接編集する運用では、**構造エラーを起動時まで検出しづらい**。

- **短期（1–2 週間）推奨アクション**
  - 本回答で提示する **「座標変換設定スキーマ（JSON/YAML テンプレート）」** を `config/calibration_template.{yaml,json}` として追加。
  - ConfigManager に、少なくとも以下のキーを **起動時に厳格検証**するロジックを追加/強化:
    - `floormap.*` 固定パラメータ（width/height/origin/scale）
    - `homography.matrix`（3×3）, `calibration.*`（use_distortion_correction, intrinsics, distortion）
    - `camera.position_x/position_y/height_m`
  - 撮影者向けの 1 ページドキュメント（`docs/guides/camera_setup_checklist.md`）を作成し、
    「現場で必ずやること（マーカー設置・撮影アングル・記録すべき値）」をチェックリスト化。

- **中期（1–2 ヶ月）推奨アクション**
  - `tools/homography_calibrator.py` を「対話的ウィザード」として拡張し、
    `--interactive` で
    - マーカークリック
    - homography 推定
    - テンプレートへの書き込み
    まで一括自動化。
  - `metadata.json` に、**使用した calibration スナップショット（homography, distortion, floormap version 等）** を必ず記録して再現性を確保。
  - 現場別のテンプレート管理（site A, B, C ごとの標準 config）を行い、新設現場はテンプレコピー＋差分編集のみで済むようにする。

- **長期（3 ヶ月〜）推奨アクション**
  - Web UI / GUI（Streamlit や簡易フロント）で、撮影者がブラウザ操作だけで
    - マーカークリック
    - パラメータ入力（高さ, 焦点距離など）
    - config 生成
    を完結できる仕組みを提供。
  - 現場ごとの calibration 履歴を DB 管理し、**過去の設定との差分・経時劣化**を可視化。


### ④ 同一オブジェクト類似度計算（外観＋挙動）

- **現状**
  - `SimilarityCalculator` が
    - 外観 cosine similarity（L2 正規化済み embedding 前提）
    - bbox IoU ベース motion similarity
    を重み付きで統合する機能を提供、`Tracker` がそれを用いてフレーム間 association を実施。
  - `TrackingPhase` が ViTDetector を用いた feature 抽出を組み込み、DeepSORT 型の外観 embedding をすでに利用。
  - `tools/visualize_features.py` + `feature_visualizer.py` により、track embedding の t-SNE 等での可視化が可能（クラスタリング品質の定性評価）。

- **主要課題**
  - 同一人物の tracklet 分裂・再結合（同一 ID 再付与）のための **明示的な類似度スコア・閾値設計**がされていない。
  - 「同一 / 異なる人物」ラベル付きの tracklet ペアデータセットがなく、**閾値チューニングが感覚的になりやすい**。
  - マルチカメラ間の ReID（Camera ID を跨いだ同一人物判定）は未設計で、評価指標（IDF1-mc など）も未導入。

- **短期（1–2 週間）推奨アクション**
  - 既存 `tracks.json` から少なくとも 100–200 ペアの tracklet を抽出し、人手で **同一/異なるラベル付きペア** を作成。
  - 単純 baseline として、
    - track embedding の平均 or 中央値ベクトルを計算
    - cosine similarity を同一性スコア \(S_{\text{app}}\) とする
  - 軌跡特徴（平均速度, 方向ヒストグラム, 滞在ゾーンパターンなど）から motion 距離 \(D_{\text{mot}}\) を設計し、
    総合スコア \(S = w_{\text{app}} \cdot S_{\text{app}} - w_{\text{mot}} \cdot D_{\text{mot}}\) を導入（重みは仮に 0.8/0.2 から開始）。
  - 上記ラベル付きペアで ROC/PR 曲線を算出し、**F1 最大点 or TPR=0.95 の点を閾値 \(T\)** として提案。

- **中期（1–2 ヶ月）推奨アクション**
  - 外観 embedding を
    - 現行 DeepSORT の特徴
    - ViT ベース特徴（DETR backbone からの cut）
    - 既存 ReID モデル（strong baseline, TransReID 等）
    で比較し、AUC/EER/IDF1 の観点でベストな構成を選定。
  - マルチカメラ拡張を見据え、Camera ID を条件にした正規化（camera-wise norm, domain adaptation）を検証。
  - 類似度スコアをグラフとして扱い、**tracklet graph clustering による ID 統合**（graph cuts / community detection）を試験。

- **長期（3 ヶ月〜）推奨アクション**
  - 大規模 ReID データセット（社内 + 公開）で fine-tuning し、オフィスドメインに適合した ReID モデルを構築。
  - マルチカメラ tracking 評価指標（IDF1-mc, HOTA 等）を取り入れ、類似度設計の改善がシステムレベルでどの程度 ID 一貫性を向上させるか測定。


## 優先アクションプラン（上位5）

| **優先度** | **目的** | **具体手順（コマンド例 / 擬似コード）** | **必要入力/ファイル** | **所要リソース** | **定量的完了基準** |
| --- | --- | --- | --- | --- | --- |
| 1 | **ベースライン精度（MOT + 再投影 + パフォーマンス）を最新セッションで一括定量化** | 1) ベースライン実行: `python scripts/run_baseline.py --config config.yaml --tag baseline-20251119`  2) 生成された `output/sessions/<SESSION_ID>/` を確認  3) 評価統合:  (GT/対応点がある場合) `python scripts/evaluate_baseline.py --session <SESSION_ID> --config config.yaml --gt data/gt_tracks_auto.json --points output/calibration/correspondence_points_cam01.json`  4) `baseline_metrics.json` を開き、MOTA/IDF1/再投影誤差/処理時間/メモリを確認・レポート化 | - `input/merged_moviefiles.mov`  - `config.yaml`  - `data/gt_tracks_auto.json`（もしくは手動 GT）  - `output/calibration/correspondence_points_cam01.json` | 開発者 0.5〜1 人日、既存環境で可 | - `baseline_metrics.json` が生成され、少なくとも MOTA/IDF1/mean_error/time_per_frame が数値として埋まっている  - 目標: time_per_frame ≤ 2.0 s, memory_increase ≤ 12 GB（MOTA/IDF1/誤差は達成/未達を明示） |
| 2 | **座標変換（単一カメラ）の再投影誤差を定量化し、歪み補正の有無を比較** | 1) 対応点を整備: `data/correspondence_points_cam01.json` を必要に応じて更新  2) 現行設定で評価: `python scripts/evaluate_reprojection_error.py --points data/correspondence_points_cam01.json --config config.yaml --output output/calibration/reprojection_cam01_raw.json --error-map output/calibration/reprojection_cam01_raw.png`  3) `calibration.use_distortion_correction=true` にして再評価し、誤差 JSON を比較  4) 結果を `docs/architecture/floormap_integration.md` などに追記 | - 対応点 JSON  - `config.yaml`（homography, calibration セクション） | 1〜2 人日（対応点取得含む）、OpenCV/PyTorch 環境 | - 2 パターン以上の config について mean_error, max_error が JSON で取得され、**どの設定が 2 px / 4 px 目標に最も近いかが明示されている** |
| 3 | **追跡可視化 UI を「正規ダッシュボード」として整備** | 1) `tools/interactive_visualizer.py` を開発の中心とし、`summary.json`, `tracking_statistics.json`, `zone_counts.csv`, `baseline_metrics.json` を 1 画面で表示するタブを追加  2) Streamlit 起動: `streamlit run tools/interactive_visualizer.py -- --session output/sessions/<SESSION_ID>`  3) UI から `scripts/evaluate_mot_metrics.py` / `scripts/evaluate_reprojection_error.py` / `scripts/evaluate_baseline.py` を `subprocess.run` で呼び出すボタンを実装  4) サイドバイサイド動画と任意 `track_id` 軌跡のインタラクティブ表示を追加 | - `output/sessions/<SESSION_ID>/*` 各種 JSON/画像  - Python + Streamlit | 3〜4 人日（UI 開発）、ブラウザ環境 | - UI 上で **MOTA/IDF1/再投影誤差/処理時間/最大同時人数/ゾーン平均人数** が 1 画面で閲覧可能  - 任意 session/tag を選択して同様の情報が確認できる |
| 4 | **座標変換設定スキーマと撮影チェックリストの形式化** | 1) 下記テンプレートを `config/calibration_template.yaml`・`config/calibration_template.json` として追加  2) `ConfigManager.validate()` に `homography.matrix`・`floormap.*`・`camera.*`・`calibration.*` の必須チェックを追加  3) `docs/guides/camera_setup_checklist.md` を作成し、現場撮影者が収集すべき情報（マーカー位置, カメラ高さ, 画像解像度等）を列挙  4) 新現場での config 生成フロー（template → 実値 fill）の手順書を整備 | - 本テンプレート案  - `src/config/config_manager.py` | 2〜3 人日 | - `ConfigManager` 起動時に座標変換関連の必須フィールド欠落が即座に検出される  - 新現場導入時に、撮影者がチェックリストを使って config を一発で埋められることを実プロジェクトで確認 |
| 5 | **同一オブジェクト類似度スコアと閾値の初期設計・検証** | 1) 既存 `tracks.json` から tracklet を抽出し、手動で同一/異なるラベル付きペア（≥100 ペア）を `data/tracklet_pairs_labeled.json` に作成  2) 小スクリプトでペアごとに cosine similarity（embedding 平均）と単純 motion 距離を計算し、総合スコア \(S\) を算出  3) `sklearn.metrics` 等で ROC/PR 曲線・AUC を計算し、仮の閾値 \(T\) を決定  4) この閾値を用いた tracklet マージ/スプリット戦略をシミュレートし、false merge / false split 件数をログに出す  5) 必要なら `Tracker` or オフライン後処理に組み込む | - `output/sessions/*/phase2.5_tracking/tracks.json`  - ラベル付きペア JSON  - Python/NumPy/Sklearn | 4〜7 人日（アノテーション含む） | - 類似度スコアに対し **AUC ≥ 0.9（仮定目標）** が達成される or 少なくとも ROC/PR 曲線と「運用上採用する閾値 \(T\)」が決定されている  - 小規模 eval で false merge / false split の件数が数値で把握されている |


## 設定ファイル（テンプレート）

**前提**: 単一カメラのカメラ→フロアマップ座標変換に必要な最小限＋将来の歪み補正/マルチカメラ拡張を見据えた構成。

### YAML テンプレート（推奨）

```yaml
# 座標変換の最小実行可能設定スキーマ例
camera_id: "cam01"  # string: カメラ識別子

image_size:          # カメラ画像の解像度（ピクセル）
  width: 1280        # int
  height: 720        # int

floormap:            # フロアマップ画像とスケール情報
  image_path: "data/floormap.png"  # string: フロアマップ画像パス
  image_width: 1878                # int, 固定
  image_height: 1369               # int, 固定
  pixel_origin: [7, 9]             # [int, int]: 原点オフセット (x, y) [px]
  scale_mm_per_pixel:              # 実長さスケール
    x: 28.1926406926406            # float: mm/px (X)
    y: 28.241430700447             # float: mm/px (Y)

homography:          # カメラ→フロアマップの 3x3 ホモグラフィ
  matrix_3x3:        # [[float, float, float], ...] row-major
    - [1.0, 0.0, 0.0]   # 実運用ではキャリブ結果で上書き
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  source_frame: "camera"    # string: 変換元座標系
  target_frame: "floormap"  # string: 変換先座標系

distortion:          # レンズ歪み補正パラメータ（任意だが推奨）
  use_distortion_correction: false   # bool: 歪み補正を有効にするか
  camera_matrix:                   # 3x3 内部パラメータ行列
    - [fx, 0.0, cx]                # fx/fy: 焦点距離(px), cx/cy: 主点
    - [0.0, fy, cy]
    - [0.0, 0.0, 1.0]
  distortion_coeffs: [k1, k2, p1, p2, k3]  # [float,...]: 歪み係数

zones:               # ゾーン定義（フロアマップ座標, 原点オフセット適用後）
  - id: "zone_1"
    name: "左エリア"
    polygon:
      - [859, 912]
      - [1095, 912]
      - [1095, 1350]
      - [859, 1350]
  - id: "zone_2"
    name: "中央エリア"
    polygon:
      - [1095, 912]
      - [1331, 912]
      - [1331, 1350]
      - [1095, 1350]
```

### JSON テンプレート（コメント付き擬似ドキュメント）

```json
{
  "__comment": "Minimal coordinate transform config. Values are examples; replace with actual calibration.",

  "camera_id": "cam01",

  "image_size": {
    "width": 1280,
    "height": 720
  },

  "floormap": {
    "image_path": "data/floormap.png",
    "image_width": 1878,
    "image_height": 1369,
    "pixel_origin": { "x": 7, "y": 9 },
    "scale_mm_per_pixel": {
      "x": 28.1926406926406,
      "y": 28.241430700447
    }
  },

  "homography": {
    "matrix_3x3": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "source_frame": "camera",
    "target_frame": "floormap"
  },

  "distortion": {
    "use_distortion_correction": false,
    "camera_matrix": [
      ["fx", 0.0, "cx"],
      [0.0, "fy", "cy"],
      [0.0, 0.0, 1.0]
    ],
    "distortion_coeffs": ["k1", "k2", "p1", "p2", "k3"]
  },

  "zones": [
    {
      "id": "zone_1",
      "name": "左エリア",
      "polygon": [[859, 912], [1095, 912], [1095, 1350], [859, 1350]]
    },
    {
      "id": "zone_2",
      "name": "中央エリア",
      "polygon": [[1095, 912], [1331, 912], [1331, 1350], [1095, 1350]]
    }
  ]
}
```

- **キーの型・意味（要点）**
  - `camera_id`: `str` — カメラ識別子。マルチカメラ時は `cam01`, `cam02` 等。
  - `image_size.width/height`: `int` — カメラ画像サイズ [px]。
  - `floormap.image_path`: `str` — フロアマップ画像パス。
  - `floormap.image_width/height`: `int` — フロアマップ画像サイズ [px]（本プロジェクトでは固定）。
  - `floormap.pixel_origin`: `int[2]` or `{x:int,y:int}` — 原点オフセット [px]（CoordinateTransformer 内で + される）。
  - `floormap.scale_mm_per_pixel.x/y`: `float` — ピクセル→mm のスケール。
  - `homography.matrix_3x3`: `float[3][3]` — カメラ px → フロアマップ px の射影変換行列。
  - `distortion.*`: 内部パラメータ・歪み係数。`use_distortion_correction=true` のときに `CoordinateTransformer` が利用。
  - `zones[].polygon`: `float[2][]` — 原点オフセット適用後のフロアマップ座標系での多角形頂点。


## 評価指標と検証手順

- **検出精度**
  - **メトリクス**
    - Precision \(P = \frac{TP}{TP+FP}\)
    - Recall \(R = \frac{TP}{TP+FN}\)
    - F1-score \(F1 = \frac{2PR}{P+R}\)
  - **計算方法**
    - `EvaluationModule` が COCO 風 GT（`output/labels/result_fixed.json`）と検出結果（Detection.bbox）を IoU≥`iou_threshold` でマッチングして算出。
    - コマンド例（擬似）: main パイプライン実行後に `run_evaluation()` を呼び、`evaluation_report.json/csv` を生成。
  - **推奨合格基準（仮定）**
    - 静的カメラ＋オフィス環境前提で **F1 ≥ 0.85**, Recall ≥ 0.9 を目標（誤検出より取りこぼし抑制を優先する場合は Recall を重めに設定）。

- **追跡精度（MOT メトリクス）**
  - **メトリクス**
    - MOTA \(= 1 - \frac{FN + FP + IDSW}{\#GT\_objects}\)
    - IDF1 \(= \frac{2 \cdot IDTP}{IDTP + IDFP + IDFN}\)
    - ID Switches（IDSW）: ID 付け替え回数
  - **計算方法**
    - `scripts/evaluate_mot_metrics.py --gt <gt_tracks.json> --tracks <session>/phase2.5_tracking/tracks.json --frames <N> --output mot_metrics.json`
    - 内部で `MOTMetrics.calculate_tracking_metrics()` を呼び、簡易マッチング（位置距離 < 50px, ID ベース）で MOTA/IDF1/IDSW を算出。
  - **推奨合格基準**
    - 既存コードのターゲット: **MOTA ≥ 0.7, IDF1 ≥ 0.8**（`evaluate_mot_metrics.py` / `evaluate_baseline.py` より）。

- **座標変換精度（再投影誤差）**
  - **メトリクス**
    - 各対応点 \(i\) についての再投影誤差 \(e_i = \|\hat{y}_i - y_i\|_2\) [px]
    - mean_error \(= \frac{1}{N}\sum_i e_i\)
    - max_error, min_error, std_error
  - **計算方法**
    - `scripts/evaluate_reprojection_error.py --points data/correspondence_points_cam01.json --config config.yaml --output reprojection_error.json --error-map reprojection_error.png`
    - `ReprojectionErrorEvaluator.evaluate_homography()` が上記統計値を出力。
  - **推奨合格基準**
    - 既存ターゲット: **mean_error ≤ 2 px, max_error ≤ 4 px**。これ以上の場合、対応点見直し or homography 再推定が必要。

- **同一オブジェクト類似度**
  - **メトリクス**
    - 類似度スコア \(S\) に対する ROC/AUC, PR/AUC, EER（Equal Error Rate）
    - 運用上の閾値 \(T\) での Precision/Recall/F1
    - False merge / False split の件数（異なる人物を同一と誤る / 同一人物を分割する）
  - **計算方法（提案）**
    - ラベル付き tracklet ペア\((i,j)\) に対し \(S_{ij}\) を算出（appearance cosine + motion 距離）。
    - `sklearn.metrics.roc_curve`, `precision_recall_curve` 等で曲線・AUC を算出。
    - 閾値 \(T\) を sweep して false merge / false split をカウント。
  - **推奨合格基準（仮定）**
    - AUC ≥ 0.9（同一/異なるの識別がほぼ確実）。
    - 運用上の閾値 \(T\) で、false merge 率（同一化ミス）が 1〜2% 以下、かつ false split 率が 10〜15% 以下程度を目安。

- **パフォーマンス**
  - **メトリクス**
    - time_per_frame_seconds（平均処理時間/フレーム, `measure_performance.py` より）
    - total_time_seconds（セッション全体）
    - memory_increase_mb, memory_peak_mb
  - **計算方法**
    - `scripts/measure_performance.py --video input/merged_moviefiles.mov --config config.yaml --output performance_metrics.json`
    - または `scripts/evaluate_baseline.py` 経由で取得。
  - **推奨合格基準**
    - 既存ターゲット: **time_per_frame ≤ 2.0 s, memory_increase ≤ 12,288 MB (12 GB)**。

- **検証実験手順（例）**
  - **データ分割**
    - 代表的な時間帯/混雑度（少人数〜多人数）を含む 3〜5 セッションを選定。
    - うち 1–2 セッションをパラメータチューニング（開発用）、残りを評価用とする。
  - **統計検定（設定 A vs B 比較時, 仮定）**
    - 各セッションでの MOTA/IDF1/mean_error/time_per_frame を指標とし、ペアワイズで
      - サンプル数が十分（≥30 frame 相当）なら **対応のある t 検定**
      - ノンパラメトリックを好む場合は **Wilcoxon signed-rank test**
      を適用して差の有意性 (p<0.05) を確認。
    - 同一人物類似度については、AUC/PR-AUC の 95% 信頼区間を bootstrap で推定。


## 必要インプットチェックリスト

- **既にリポジトリ内で確認できる入力（入手済）**
  - **タイムラプス動画**: `input/merged_moviefiles.mov` — パイプライン実行の主要入力。
  - **フロアマップ画像**: `data/floormap.png` — coordinate transform / 可視化の基底。
  - **フロアマップ固定パラメータ**: `config.yaml` の `floormap` セクション、および `.cursor/rules/coordinate-zone-processing.mdc` — image size, origin, scale。
  - **座標変換パラメータ**
    - ホモグラフィ: `config.yaml` の `homography.matrix`, `output/calibration/homography_*.yaml`
    - カメラ内部/歪み（デフォルト値含む）: `config.yaml` の `calibration.intrinsics/distortion`
  - **GT ラベル**
    - 検出/人数 GT: `output/labels/result_fixed.json`
    - トラック GT テンプレ/自動生成: `data/gt_tracks_template.json`, `data/gt_tracks_auto.json`
  - **キャリブレーション対応点**: `output/calibration/correspondence_points_cam01.json`
  - **最新パイプライン出力**: `output/latest/*`, `output/sessions/20251119_132130/*`
  - **評価/ベースラインスクリプト**: `scripts/evaluate_baseline.py`, `scripts/evaluate_mot_metrics.py`, `scripts/evaluate_reprojection_error.py`, `scripts/measure_performance.py`

- **未入手 or 状態不明（要確認・多くは入手可能）**
  - **マルチカメラ用データ**
    - 他カメラ（cam02, cam03, ...）の動画・homography・intrinsics・対応点
      → 現状ファイル構造上は単一カメラ前提。**入手可否: 環境依存（別撮影で入手可能と推定, 要確認）**。
  - **ラベル付き tracklet 類似度ペア**
    - 「同一人物/別人物」のラベル付き tracklet ペアデータセット
      → 現在は存在しない。**入手可否: アノテーション作業により作成可能（工数次第）**。
  - **実運用セッション多数分の baseline_metrics**
    - `output/sessions/*/baseline_metrics.json` が一部 or 未生成の可能性。
      → `scripts/evaluate_baseline.py` によりいつでも生成可能。**入手可**。
  - **実測されたカメラ物理パラメータ**
    - 正確なカメラ高さ, 俯角, レンズ焦点距離・歪み係数（現状はデフォルト or 仮定値）
      → 測定すれば取得可能。**入手可だが現場作業が必要**。
  - **複数現場（異なるオフィスレイアウト）での calibration/config**
    - 他ロケーションの floormap, zones, homography 群
      → 将来拡張に向けて必要。**現時点では不明（要確認）**。


## リスクと緩和策

- **データ偏り・汎化性**
  - **リスク**: 特定時間帯/混雑パターンに偏ったデータのみで評価すると、他条件（夜間, 別曜日）での精度が保証されない。
  - **緩和策**: 時間帯・曜日・混雑度で層別した複数セッションを評価セットに含める。`baseline_metrics.json` を時系列で蓄積し、指標のドリフトを監視。

- **キャリブレーション誤差**
  - **リスク**: 対応点の誤クリックやマーカー配置の誤りで再投影誤差が大きくなり、ゾーン判定や人数集計に系統的なバイアスが生じる。
  - **緩和策**: `evaluate_reprojection_error.py` を必ず走らせ、mean/max error を閾値ベースで判定。誤差マップを可視化し、特定エリアで誤差が集中していないか確認。高誤差時には対応点を見直し、RANSAC パラメータを調整。

- **スケール・単位の不整合**
  - **リスク**: mm/px スケールや origin offset を誤設定すると、距離ベース指標（移動距離, 滞在距離）が現実と大きく乖離。
  - **緩和策**: フロアマップ上で既知長さ（例: 柱間 3m）を mm/px から逆算し、実測と比較する簡易チェックを導入。ConfigManager で floormap パラメータを固定値として検証。

- **トラッキング不安定性・ID スイッチ**
  - **リスク**: occlusion や高密度時に ID スイッチが頻発し、ゾーン滞在時間・一人あたり滞在軌跡が信頼できなくなる。
  - **緩和策**: DeepSORT パラメータ（appearance_weight, motion_weight, iou_threshold, max_age, min_hits）を網羅的にスイープし、MOTA/IDF1/IDSW を中心に最適点を探す。ReID embedding の高精度モデル導入を検討。

- **同一性判定閾値の誤設定**
  - **リスク**: 類似度閾値が緩すぎると異なる人物を同一扱い（false merge）、厳しすぎると同一人物トラックが分断（false split）。
  - **緩和策**: ラベル付きペアで ROC/PR を計測し、用途（混雑度, 必要な ID 一貫性レベル）に応じた閾値を明示的に選択。定期的な再評価をルール化。

- **運用制約（Tesseract/MPS 依存, リソース制約）**
  - **リスク**: OCR エンジンや MPS が利用不可な環境で性能低下 or 動作不全。
  - **緩和策**: `config.yaml` で device=`cpu`・Tesseract オフ時の fallback 設定を整備。`scripts/measure_performance.py` で CPU ケースの baseline を測り、許容範囲を定義。

- **設定管理ミス**
  - **リスク**: `config.yaml` におけるキー名 typo / 型不一致 / セクション抜けにより、実行時まで問題に気づかない。
  - **緩和策**: ConfigManager のバリデーションを強化し、座標変換・tracking・evaluation に必須なキーをすべてチェック。CI で `python scripts/comprehensive_test.py` を実行し、config エラーを早期検知。


## 追加提案（任意）

- **ツールチェーン/オペレーション**
  - `Makefile` に `make baseline`（`run_baseline.py` + `evaluate_baseline.py` を連鎖実行）のターゲットを追加し、「1 コマンドで baseline 指標更新」を実現。
  - CI に `scripts/evaluate_baseline.py` を optional step として組み込み、マージ前に主要メトリクスが一定水準を下回った場合に警告を出す。

- **アルゴリズム/実装パターン**
  - トラッキング評価には、将来的に **HOTA (Higher Order Tracking Accuracy)** 等の指標を導入し、ID 一貫性と空間精度の両方を同時にモニタリング。
  - 類似度計算には、既存 DeepSORT embedding に加え、軽量な ViT ベース ReID モデル（例: fast-reid, strong baseline）をバックエンドとして差し替え可能な抽象化レイヤを追加。
  - マルチカメラ拡張時には、カメラごとに homography + 外部パラメータを持つ「CameraConfig」データクラスを導入し、MLOps 観点で config のバージョニング（タグ/日付管理）を行う。

- **優先度付き技術オプション**
  - **高優先度**: baseline 評価の自動化（scripts + dashboard 連携）、座標変換スキーマ/チェックリスト整備。
  - **中優先度**: ReID/類似度モジュールの拡張と評価パイプライン構築、追跡パラメータの系統的チューニング。
  - **低〜中優先度**: マルチカメラ world 座標系設計、3D 情報の導入、外部モニタリング（Prometheus/Grafana 連携）など。

これらを順に進めることで、現状の「機能は揃っているが評価・可視化・設定スキーマが分散している状態」から、**一貫したベースライン評価 + 運用ダッシュボード + 再現性の高い calibration/config 管理**に段階的に移行できます。
