## 現状サマリ

- **プロジェクト目的**

  - オフィス内の定点カメラ映像（タイムラプス, 約 1280×720）から人物を **検出 (ViT/DETR)** → **追跡 (DeepSORT 系)** → **カメラ座標 → フロアマップ座標変換 (Homography)** → **ゾーン別人数集計** → **可視化・評価** を行うバッチ処理パイプラインを構築し、定量的な人数・行動指標を提供する。

- **データ・アーティファクト（主要なもの）**

  - **入力データ**
    - `input/merged_moviefiles.mov`: 本番想定のタイムラプス動画（複数セッションを merge 済み）。
  - **静的リソース**
    - `data/floormap.png`: 固定フロアマップ（1878×1369 px, 原点オフセット (7,9), mm/px スケール固定）。
    - `config.yaml`: 全フェーズ共通の設定（tracking, calibration, zones 等を含む）。
  - **スクリプト群**
    - `scripts/evaluate_reprojection_error.py`: 再投影誤差の測定と誤差マップ生成。
    - `scripts/evaluate_mot_metrics.py`: MOTA/IDF1 等の MOT メトリクス評価。
    - `scripts/visualize_features.py`: トラック特徴量の t-SNE 可視化とクラスタリング評価。
    - `scripts/measure_performance.py`: 各フェーズの処理時間・メモリ使用量の測定。
    - `scripts/comprehensive_test.py`: テスト＋メトリクスの総合評価ランナー。
  - **可視化・UI**
    - `output/latest/phase5_visualization/side_by_side_tracking.mp4`: 原動画＋検出枠 vs フロアマップ＋軌跡のサイドバイサイド動画。
    - `output/latest/phase5_visualization/floormaps/*.png`: ゾーン別人数・軌跡描画済みフロアマップ静止画。
    - `tools/visual_inspection.py`: `tracking` / `reprojection` モードで検証用動画・誤差マップ生成。
    - `tools/visualizer_app.py`: Streamlit ベースの可視化 UI（task_followup_plan 上の記載）。
    - `tools/interactive_visualizer.py`: エクスポート機能・統計パネル付きの拡張 Streamlit UI（実装完了レポートで拡張済み）。

- **モデル・アルゴリズム一覧**

  - **検出 (Detection)**
    - ViT ベース物体検出: Hugging Face の `facebook/detr-resnet-50` or `microsoft/vit-det` 系（`src/detection/vit_detector.py`）。
  - **追跡 (Tracking)**
    - DeepSORT 系トラッカー (`src/tracking/`): 外観特徴量＋ Kalman Filter ＋ Hungarian algorithm によるアサイン。
    - パラメータ: `appearance_weight`, `motion_weight`, `iou_threshold`, `max_age`, `min_hits` など。
  - **座標変換 (Coordinate Transform)**
    - カメラ → フロアマップのホモグラフィ (`src/transform/coordinate_transformer.py`)。
    - キャリブレーションツール (`src/calibration/camera_calibrator.py`) と再投影誤差評価 (`src/calibration/reprojection_error.py`)。
  - **ゾーン判定 (Zone Classification)**
    - ポリゴン内判定 (Ray Casting) を用いたゾーン分類 (`src/zone/zone_classifier.py`)。
  - **評価 (Evaluation)**
    - MOT メトリクス (`src/evaluation/mot_metrics.py`)。
    - 再投影誤差評価 (`ReprojectionErrorEvaluator`)。
    - 特徴量クラスタリング・可視化 (`src/utils/feature_visualizer.py`)。

- **最新評価指標（2025-11-14 時点）**

  - **テスト**
    - 総テスト: **443 件**（2025 年 11 月 14 日時点）
    - **全体カバレッジ**: 測定が必要（`make test TEST_MODE=coverage` で確認可能）
    - 高カバレッジモジュール: pipeline phases 93–100%, `vit_detector` 76%, `coordinate_transformer` 84%, `zone_classifier` 86% など。
  - **MOT メトリクス（テスト用サンプル）**
    - `scripts/evaluate_mot_metrics.py` を `data/test_*_tracks.json` で実行:
      - **MOTA = 1.0**, **IDF1 = 1.0**, **ID Switches = 0**（小規模テスト, 実データの性能を代表しない）。
    - 実セッションの MOTA/IDF1: **未提示**（未計測または未記録）。
  - **再投影誤差（テスト用サンプル）**
    - `scripts/evaluate_reprojection_error.py` + `data/test_correspondence_points.json`:
      - 平均誤差 ≈ **790 px**, 最大誤差 ≈ **2582 px**（明示的に「テストデータのため」, 実システムを評価する値ではない）。
      - 目標: **平均誤差 ≤ 2 px, 最大誤差 ≤ 4 px**（設計上のターゲット）。
    - 実セッションでの再投影誤差: **未提示**。
  - **パフォーマンス**
    - `scripts/measure_performance.py` は実装済み（処理時間 ≤ 2 秒/フレーム, メモリ ≤ 12GB が目標）。
    - 実セッションでの測定結果: **未提示**。

- **既知の問題点・ギャップ**
  - **実データ上のトラッキング・座標変換精度が未定量化**
    - 実セッションの MOTA/IDF1, 再投影誤差, パフォーマンス結果がレポートとして残っていない。
  - **テストカバレッジが目標 80% に未達**（現状 ≈76%, 特に `evaluation_module`, 一部 utils が低カバレッジ）。
  - **可視化系のインターフェースが分散**
    - `visualizer_app.py` と `interactive_visualizer.py` の役割重複・命名不整合。
  - **同一オブジェクト類似度・マルチカメラ整合性の設計が未成熟**
    - DeepSORT の appearance embedding は存在するが、ReID/類似度を軸にした **アルゴリズム設計・閾値設計・評価** が明示されていない。
  - **座標変換パラメータの「運用スキーマ」や撮影者向け手順が断片的**
    - `config.yaml` にパラメータは存在するが、「新設置現場での一連のセットアップ手順」と紐づいたスキーマ・手順書が不足。

## タスク別ギャップ分析（①〜④）

### ① オブジェクト追跡の可視化

- **現状**

  - サイドバイサイド動画 (`side_by_side_tracking.mp4`)、フロアマップ静止画、`visual_inspection.py`（tracking モード）、Streamlit アプリ（`visualizer_app.py` / `interactive_visualizer.py`）が存在。
  - Streamlit アプリには ID/ゾーンフィルタ、エクスポート機能、統計パネル（トラック数・平均軌跡長など）が実装済み。
  - ただし **「運用ダッシュボード」レベルの集約ビュー（期間集計, KPI トレンド, アラート）は未設計**。

- **主要課題**

  - ビューとコードが二重化（`visualizer_app.py` vs `interactive_visualizer.py`）しており、将来的な保守コストが高い。
  - 可視化の目的が **「デバッグ・目視確認」中心**であり、運用者が日常的に見るべき KPI が整理されていない。
  - セッション横断の比較（パラメータ A vs B, 日付別トレンド）が UI/ツールレベルでサポートされていない。

- **短期（1–2 週間）推奨アクション**

  - **単一の「正規 UI モジュール」を決める**（例: `tools/interactive_visualizer.py` に統合し、`visualizer_app.py` を deprecate）。
  - UI の表示項目を以下に整理:
    - フレーム単位ビュー（既存）。
    - セッション集約ビュー: 総滞在人数, ゾーン別平均人数, 最大同時在室人数。
    - トラック品質指標: セッション毎の推定 ID Switch 数（推計でも可）。
  - UI から **評価スクリプトを起動するためのボタン**（例: 「MOT メトリクス計算」）を追加し、結果 JSON を UI にインライン表示。

- **中期（1–2 ヶ月）推奨アクション**

  - **セッション比較機能**:
    - 複数セッションの summary.json/mot_metrics.json を読み込み、MOTA/IDF1/平均人数を一覧表示。
    - 追跡パラメータセット（appearance_weight, motion_weight 等）ごとに色分け表示。
  - **ゾーン別ヒートマップ・滞在時間分布**の可視化を追加（`floormap_visualizer` を拡張）。
  - `scripts/measure_performance.py` の結果を UI 上で閲覧可能にし、「1 フレーム平均処理時間」「メモリピーク」をパラメータセット別に比較。

- **長期（3 ヶ月〜）推奨アクション**
  - Streamlit アプリを **運用ダッシュボード**として整理:
    - 日次・週次で自動生成されるレポートの一覧。
    - KPI 閾値（例: MOTA < 0.7, 平均誤差 > 2 px）を下回るセッションをハイライト。
  - 必要に応じて Grafana/Prometheus 等と連携し、処理状況・パフォーマンスのモニタリングに拡張。

### ② 座標変換の精度改善（単一カメラ & 複数カメラ）

- **現状**

  - 単一カメラ前提でのホモグラフィ変換は実装済み。
  - キャリブレーションツール (`homography_calibrator.py`) と再投影誤差評価スクリプト (`evaluate_reprojection_error.py`) があり、「平均 ≤2 px, 最大 ≤4 px」を目標としている。
  - テスト用の対応点では大きな誤差（平均 ≈790 px）が出ているが、これはスケール・データが実運用を代表していないと明記。
  - マルチカメラ整合（共通ワールド座標系への統一）については **要件は暗示されているが実装・評価はほぼ未着手**。

- **主要課題**

  - 実運用データの対応点（カメラ画像上の座標 vs フロアマップ上の正解座標）の収集・管理が不足。
  - `calibration.use_distortion_correction` を有効にした場合の **実データ評価が行われていない**。
  - マルチカメラの場合の **ワールド座標系の定義・整合方法（例: floorplan 基準 vs 3D 座標系）が未定義**。

- **短期（1–2 週間）推奨アクション**

  - 最低 1 カメラについて、**10–20 点程度の高精度対応点**を取得（床面上のマーカー推奨）。
  - 対応点を `data/correspondence_points_<camera_id>.json` として保存し、以下を実行:

    ```bash
    python scripts/evaluate_reprojection_error.py \
        --points data/correspondence_points_cam01.json \
        --config config.yaml \
        --output output/cam01_reprojection_error.json \
        --error-map output/cam01_error_map.png \
        --image-shape 1369 1878
    ```

  - `use_distortion_correction = false / true` 両ケースで誤差を比較し、2 px/4 px 目標との乖離を定量的に記録。

- **中期（1–2 ヶ月）推奨アクション**

  - **外れ値ロバストなキャリブレーション**:
    - RANSAC 閾値・反復回数のチューニング、および外れ点自動検出（誤差 > 5σ など）による iterated refinement を実装。
  - **座標系の明示化と正規化**:
    - `CoordinateTransformer` に source/target frame を明示（camera/floormap/world）。
    - 必要に応じて「中間正規化座標系」（P4）の導入検証。
  - マルチカメラを想定した **共通ワールド座標系**の設計（例: フロア図面座標, 単位 mm）と、カメラごとの外部パラメータ管理。

- **長期（3 ヶ月〜）推奨アクション**
  - **マルチカメラ統合評価**:
    - 異なるカメラから見た同一点のワールド座標のズレを統計的に評価。
    - 上記ズレを MOTA/IDF1 に対する影響として分析。
  - 座標変換の **オンライン監視**:
    - 運用中に定期撮影される calibration target を用い、自動で誤差を再計測し、閾値を超えたら再キャリブレーションを要求。

### ③ 座標変換のパラメータ洗い出しと撮影者向け手順

- **現状**

  - `config.yaml` に `calibration.use_distortion_correction` などのフラグがあり、task_followup_plan でも P3/P4 として調整対象に挙げられている。
  - ただし「座標変換に必要なパラメータ一覧」と「現場撮影者がどの値をどう測るか」が一貫したスキーマで整理されていない。
  - カメラ設置・撮影手順は暗黙のノウハウに依存しており、「チェックリスト化」されていない。

- **主要課題**

  - 新規設置現場ごとに **設定漏れ・誤入力が起きやすい**。
  - 撮影者が収集すべき情報（カメラ高さ, 俯角, 基準点配置など）が不明確で、キャリブレーション品質がばらつく。
  - `config.yaml` を直接編集する運用では、キー名や型のミスを防ぎにくい。

- **短期（1–2 週間）推奨アクション**

  - 本回答の **座標変換設定スキーマ（JSON/YAML）** をベースに、`config/calibration_schema.md` 的なドキュメントを作成。
  - 撮影者向け **チェックリスト** を定義:
    - フロアマップ画像パス。
    - カメラ ID と設置位置メモ。
    - 少なくとも 10 点のマーカー（床上）の camera pixel 座標と floormap pixel 座標。
    - カメラ画像解像度（width, height）。
  - `ConfigManager` にスキーマ検証ロジックを追加し、必須キーの欠落・型不一致を起動時に検知。

- **中期（1–2 ヶ月）推奨アクション**

  - **CLI ベースのキャリブレーションウィザード**:
    - `python tools/homography_calibrator.py --interactive` でマーカークリック・設定生成まで行う。
  - `session-tag` と連動した **設定スナップショット**（使用した homography, distortion, scale 等）を `metadata.json` に記録し、再現性を確保。
  - 撮影手順書に、カメラ高さ・俯角・レンズ焦点距離等の推奨レンジを明記。

- **長期（3 ヶ月〜）推奨アクション**
  - Web UI または簡易 GUI（例: Streamlit ベース）でキャリブレーションと config 生成を一体化。
  - 異なる現場間での **テンプレート化**（オフィスレイアウトごとのプリセット config）を行い、現場ごとの差分だけを記入させる運用に移行。

### ④ 同一オブジェクト類似度計算（外観＋挙動）

- **現状**

  - DeepSORT による appearance embedding は導入済み（トラック ID 付与に利用）。
  - `scripts/visualize_features.py` + `src/utils/feature_visualizer.py` により、トラック特徴量の t-SNE 可視化・クラスタリング評価が可能。
  - しかし **同一オブジェクト類似度を明示的に扱うモジュール（スコアリング・閾値・評価指標）は設計されていない**。

- **主要課題**

  - 同一人物のトラックが分断された場合（tracklet 分裂）に、**再結合のロジックが弱い / 未定義**。
  - マルチカメラ間での同一人物判定（ReID）が設計されていない。
  - 閾値設計が経験則ベースになりやすく、**再現性のある評価・チューニングフローがない**。

- **短期（1–2 週間）推奨アクション**

  - **同一オブジェクト類似度モジュールの設計**:
    - appearance similarity: track ごとの平均/中央値 embedding を計算し、cosine similarity をスコアとする。
    - motion similarity: 速度ベクトル・方向ヒストグラム・位置分布など簡易な軌跡特徴をまとめ、L2 距離 or DTW 距離を定義。
    - 総合スコア `S = w_app * S_app - w_motion * D_motion` を導入（符号に注意）。
  - 小規模でもよいので **ラベル付きペアデータ**（同一人物/別人物 tracklet ペア）を作成し、ROC 曲線・PR 曲線から閾値を決める。
  - `evaluate_mot_metrics.py` に、**cluster-based IDF1** や false-merge/false-split 数を出す拡張案を検討。

- **中期（1–2 ヶ月）推奨アクション**

  - 外観特徴量として、**既存 DeepSORT embedding vs ViT 特徴量 vs 専用 ReID モデル**（e.g. strong baseline, TransReID）の比較実験。
  - マルチカメラシナリオを想定し、カメラ ID を含む feature normalization（camera-wise batch norm, domain adaptation）を検討。
  - 同一性判定アルゴリズムを **Graph-based Association**（tracklet をノード, 類似度をエッジとする）に拡張。

- **長期（3 ヶ月〜）推奨アクション**
  - 大規模な ReID データセット（市場の公開データ＋社内データ）で **fine-tuning** し、ドメイン適応済み embedding を構築。
  - マルチカメラ tracking の評価指標（IDF1, IDF1-mc, HOTA 等）を導入し、同一オブジェクト類似度の改善をシステムレベルで評価。

## 優先アクションプラン（上位 5）

| 優先度 | 目的                                                                  | 具体手順（コマンド例 / 擬似コード）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 必要入力/ファイル                                                           | 所要リソース                   | 定量的完了基準                                                                                                                                  |
| ------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | 実データ上のベースライン精度（MOT + 再投影 + パフォーマンス）を定量化 | 1) 実セッションでパイプライン実行: `python main.py --config config.yaml --session-tag baseline-<date>` 2) MOT メトリクス計測: `python scripts/evaluate_mot_metrics.py --gt <gt_tracks.json> --tracks <session>/phase2.5_tracking/tracks.json --frames <N> --output <session>/mot_metrics.json` 3) 再投影誤差: `python scripts/evaluate_reprojection_error.py --points data/correspondence_points_cam01.json --config config.yaml --output <session>/reprojection_error.json --error-map <session>/reprojection_error.png` 4) パフォーマンス: `python scripts/measure_performance.py --video input/merged_moviefiles.mov --config config.yaml --output <session>/performance_metrics.json` | - 実セッション動画 - GT トラック（少なくとも 1 セッション分） - 対応点 JSON | 1–2 人日, MPS/GPU or CPU       | - 実セッション 1 本以上について **MOTA, IDF1, 再投影平均・最大誤差, fps, メモリピーク** が JSON で出力され、レポートに整理されている            |
| 2      | DeepSORT + 座標変換のパラメータチューニング（P1–P4 の実験設計）       | 擬似コード: 1) パラメータグリッド定義（例: appearance*weight∈{0.5,0.65,0.8}, motion_weight=1-aw, iou_threshold∈{0.3,0.35,0.4}） 2) 各設定ごとに: `python main.py --config config_tuning*<k>.yaml --session-tag tuning-<k>`3) 実行後に`python scripts/comprehensive_test.py --session <session_id> --metrics` を実行し、`summary.json` に MOTA, IDF1, 再投影誤差, 処理時間を集約 4) 結果を表形式で比較                                                                                                                                                                                                                                                                                     | - 複数の config バリアント - 1–2 本の代表セッション                         | 3–5 人日, スクリプト実行環境   | - 少なくとも 5–10 通りのパラメータセットについて **MOTA/IDF1/誤差/速度** を比較し、「推奨設定」を 1–2 パターン定義                              |
| 3      | 追跡可視化 UI を「正規ダッシュボード」として整備                      | 1) `tools/visualizer_app.py` の機能を `tools/interactive_visualizer.py` に統合し、前者をラッパー or 非推奨化 2) UI に「セッション選択」「summary.json/mot_metrics.json の数値表示」「パフォーマンス結果表示」を追加 3) 起動: `streamlit run tools/interactive_visualizer.py -- --session latest` 4) UI から評価スクリプトを呼び出すボタン（内部で `subprocess.run([...])`）を実装                                                                                                                                                                                                                                                                                                         | - `output/sessions/*` の summary/mot_metrics JSON - Python/Streamlit        | 3 人日前後, フロント寄りスキル | - UI から任意セッションの **人数時系列・MOT 指標・処理時間** が一画面で確認可能 - `visualizer_app.py` の機能が重複なく統合されている            |
| 4      | 座標変換設定スキーマと撮影手順を形式化                                | 1) 本回答の YAML/JSON スキーマを `config/calibration_template.*` としてプロジェクトに反映 2) `ConfigManager` に必須キー検証を追加（camera_id, image_size, floormap, homography, distortion） 3) 撮影者向けチェックリストを docs に追加 4) 新現場で template → 実データへの置き換え手順をドキュメント化                                                                                                                                                                                                                                                                                                                                                                                    | - 本テンプレート - `ConfigManager` 実装                                     | 2–3 人日                       | - 新規現場セットアップにおいて **config のバリデーションエラーが 0** - 撮影者がチェックリストに従うことで、対応点と config が一貫して取得される |
| 5      | 同一オブジェクト類似度スコア・閾値の初期設計と評価                    | 1) ラベル付き tracklet ペア（同一/異なる）を少なくとも 100–200 ペア作成 2) 既存 DeepSORT embedding を用いて cosine similarity を計算 3) motion feature（平均速度, 方向ヒストグラム等）を追加し、総合スコアを定義 4) Python スクリプトで ROC/PR 曲線を算出し、F1 最大 or TPR=0.95 の点を閾値として採用 5) `evaluate_mot_metrics.py` を拡張 or 別スクリプトで false-merge/false-split を測定                                                                                                                                                                                                                                                                                                | - トラック JSON + ラベル付きペア情報 - 既存 embedding 出力                  | 4–7 人日, 分析スキル           | - 類似度スコアに対して **再現可能な閾値（例: S≥0.8）** が定義され、少なくとも小規模データで precision/recall の定量結果が得られている           |

## 設定ファイル（テンプレート）

### YAML テンプレート（推奨）

```yaml
# 座標変換の最小実行可能設定スキーマ（例）
camera_id: "cam01" # string: カメラの一意な ID

image_size: # カメラ画像の解像度（ピクセル）
  width: 1280 # int
  height: 720 # int

floormap: # フロアマップ画像とスケール情報
  image_path: "data/floormap.png" # string: フロアマップ画像パス
  pixel_origin: [7, 9] # [int, int]: フロアマップ上の原点オフセット (px)
  scale_mm_per_pixel: # 実長さスケール
    x: 28.1926406926406 # float: mm/px (X 軸)
    y: 28.241430700447 # float: mm/px (Y 軸)

homography: # カメラ→フロアマップの 3x3 ホモグラフィ行列
  matrix_3x3: # [[float, float, float], ...] row-major
    - [1.0, 0.0, 0.0] # 実運用ではキャリブレーション結果で上書き
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  source_frame: "camera" # string: 変換元座標系
  target_frame: "floormap" # string: 変換先座標系

distortion: # レンズ歪み補正パラメータ（任意だが推奨）
  use_distortion_correction: true # bool: 歪み補正を有効化するか
  camera_matrix: # 3x3 内部パラメータ行列
    - [fx, 0.0, cx] # fx, fy, cx, cy は実測値で置換
    - [0.0, fy, cy]
    - [0.0, 0.0, 1.0]
  distortion_coeffs: [k1, k2, p1, p2, k3] # [float, ...]: 歪み係数
```

### JSON テンプレート（コメントは擬似的に付与）

```json
{
  "__comment": "Minimal coordinate transform config. Values are examples and must be replaced.",

  "camera_id": "cam01",

  "image_size": {
    "width": 1280,
    "height": 720
  },

  "floormap": {
    "image_path": "data/floormap.png",
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
    "use_distortion_correction": true,
    "camera_matrix": [
      ["fx", 0.0, "cx"],
      [0.0, "fy", "cy"],
      [0.0, 0.0, 1.0]
    ],
    "distortion_coeffs": ["k1", "k2", "p1", "p2", "k3"]
  }
}
```

- **型・意味（要点）**
  - `camera_id`: `str` — カメラ識別子。
  - `image_size.width/height`: `int` — 生画像のピクセルサイズ。
  - `floormap.image_path`: `str` — フロアマップ画像への相対/絶対パス。
  - `floormap.pixel_origin`: `[int,int]` — フロアマップ画像上のワールド原点（px）。
  - `floormap.scale_mm_per_pixel.x/y`: `float` — mm/px スケール。
  - `homography.matrix_3x3`: `List[List[float]]` — camera→floormap の射影変換。
  - `distortion.*`: 内部パラメータと歪み係数（OpenCV 互換形式）。

## 評価指標と検証手順

- **必須メトリクスと計算式**

  - **MOTA (Multiple Object Tracking Accuracy)**
    \[
    \mathrm{MOTA} = 1 - \frac{\sum_t (\mathrm{FN}\_t + \mathrm{FP}\_t + \mathrm{IDSW}\_t)}{\sum_t \mathrm{GT}\_t}
    \]
    - 目標値: **MOTA ≥ 0.7**（task_followup_plan の設定）。
  - **IDF1 (Identity F1-score)**
    \[
    \mathrm{IDF1} = \frac{2 \cdot \mathrm{IDTP}}{2 \cdot \mathrm{IDTP} + \mathrm{IDFP} + \mathrm{IDFN}}
    \]
    - 目標値: **IDF1 ≥ 0.8**。
  - **再投影誤差 (Reprojection Error)**
    - 各対応点 \(i\) について:
      \[
      e*i = \left\lVert p^{\mathrm{proj}}\_i - p^{\mathrm{gt}}\_i \right\rVert_2
      \]
      \[
      \overline{e} = \frac{1}{N}\sum_i e_i,\quad e*{\max} = \max_i e_i
      \]
    - 目標値: **平均誤差 \(\overline{e} ≤ 2\) px、最大誤差 \(e\_{\max} ≤ 4\) px**。
  - **同一オブジェクト類似度メトリクス**
    - 類似度スコア \(S\)（cosine similarity）に対し、二値分類（同一/別）として:
      - Precision / Recall / F1, ROC-AUC。
    - 推奨: **F1 ≥ 0.9（小規模ラベルセットでのターゲット, 仮定）**。
  - **パフォーマンス**
    - 1 フレームあたり処理時間 \(T*{\mathrm{frame}}\)、最大メモリ使用量 \(M*{\max}\)。
    - 目標値: **\(T\_{\mathrm{frame}} ≤ 2\) 秒**, **\(M\_{\max} ≤ 12\) GB**。

- **検証実験の手順（例）**
  - **Tracking/MOT**
    1. 代表的なセッション（昼/夜, 混雑/非混雑など）を 3–5 本選定。
    2. 全フレームまたはサンプリングされたフレームに対して Ground Truth トラック（ID 付き）を作成。
    3. 各パラメータセットで pipeline を実行し、`evaluate_mot_metrics.py` で MOTA/IDF1 を計算。
    4. ベースライン vs 新パラメータの MOTA/IDF1 の差について、**ペア t 検定 or Wilcoxon 符号付順位検定**（セッション数が十分な場合）で有意差を評価（ここはラベル数に依存, 仮定）。
  - **座標変換**
    1. カメラごとに 10–20 点の対応点を取得。
    2. `evaluate_reprojection_error.py` を実行し、\(\overline{e}\), \(e\_{\max}\) の分布を取得。
    3. パラメータ変更前後で誤差の差を比較し、必要ならブートストラップで信頼区間を推定（仮定: 対応点数 ≥ 20）。
  - **同一オブジェクト類似度**
    1. tracklet ペア（同一/別）をラベル付け（例: 100–200 ペア）。
    2. 各ペアについて appearance/motion similarity を計算。
    3. ROC/PR 曲線を作成し、運用上許容可能な False Merge/False Split 比率から閾値を決定。
  - **パフォーマンス**
    1. `scripts/measure_performance.py` を代表セッションに対して実行。
    2. 平均フレーム時間, 95%ile, 最大値を算出し、目標 ≤2 秒/フレームとの比較。
    3. 将来的には CI に組み込み、閾値を超えた場合に警告。

## 必要インプットチェックリスト

| 項目                              | 説明                                            | 状態                                                       | 入手可否/備考                                             |
| --------------------------------- | ----------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| 実セッション動画                  | 本番想定の timelapse 動画（複数日, 複数シーン） | 一部入手済 (`input/merged_moviefiles.mov`)                 | 追加セッションがある場合は取得推奨                        |
| フロアマップ画像                  | フロアレイアウト画像                            | 入手済 (`data/floormap.png`)                               | 固定パラメータあり（サイズ, 原点, スケール）              |
| Ground Truth トラック（実データ） | MOTA/IDF1 計測用の ID 付き GT                   | **未入手 / 規模不明**                                      | 最低 1–2 セッション分の整備が必要                         |
| 対応点（実現場）                  | カメラ画像座標 vs フロアマップ座標              | テスト用 (`data/test_correspondence_points.json`) のみ明示 | 実データ版 `data/correspondence_points_camXX.json` が必要 |
| カメラ内部・外部パラメータ        | fx, fy, cx, cy, 歪み係数, カメラ高さ等          | **未入手（config 内部に暗黙か不明）**                      | キャリブレーション時に測定・記録推奨                      |
| Multi-camera データ               | 複数カメラからの同一空間映像                    | 不明（コード上は単一カメラ前提が多い）                     | マルチカメラ対応を行う場合は新規収集が必要                |
| ReID 用ラベル付きペア             | 同一/異なる人物 tracklet ペア                   | **未入手**                                                 | 100–200 ペア程度から開始推奨                              |
| 運用要件/KPI                      | ビジネス側の KPI（混雑閾値, アラート条件など）  | **未入手**                                                 | ダッシュボード設計の前提として必要                        |
| インフラ情報                      | 実運用環境の CPU/GPU, ストレージ, CI/CD         | 一部 docs のみ, 詳細不明                                   | 性能・デプロイ要件を詰める際に必要                        |

- これら「未入手」のうち、**Ground Truth トラック**と**実対応点**が短期タスクで最重要です。
  これらが揃えば、上記アクションプラン 1,2 の精度評価を即時に進められます。

## リスクと緩和策

- **データ偏り**

  - リスク: 昼間のみ/特定曜日のみのデータに偏ると、夜間やイベント時に検出・追跡精度が劣化。
  - 緩和策:
    - セッション選定時に「時間帯 × 曜日 × 混雑度」の分布を意識し、代表セッションを複数用意。
    - 低照度・逆光等の条件での性能を別途確認し、必要に応じてモデル再学習や前処理（ガンマ補正等）を追加。

- **キャリブレーション誤差**

  - リスク: 対応点の誤指定や床面以外の点を混在させることで、ホモグラフィが不安定になりゾーン判定を誤る。
  - 緩和策:
    - マーカーは **床面上のみ**に限定し、10–20 点以上を撮影。
    - RANSAC・外れ値除去を導入し、`error_map` を定期確認。
    - キャリブレーションごとに `mean_error, max_error` を記録し、閾値超過時は再キャリブレーション必須。

- **スケール・深度問題**

  - リスク: mm/px スケールや原点オフセットが誤っていると、距離・面積系の指標（滞在距離, 人流量）が誤る。
  - 緩和策:
    - 実測距離（例: 廊下の幅, 会議室の長辺）と floormap 上のピクセル距離からスケールを再検証。
    - 人物の高さや非平面構造の影響が大きい箇所については、近似的な補正係数をゾーン単位で導入する。

- **追跡モデルのドリフト**

  - リスク: 環境変化（家具配置, ライティング, カメラ更新）により DeepSORT embedding の分布が変化し、ID スイッチが増加。
  - 緩和策:
    - 定期的に `visualize_features.py` で embedding 分布を可視化し、クラスタ構造変化を監視。
    - MOTA/IDF1 を定期計測し、閾値を下回った場合に ReID モデルの再学習 or パラメータ再チューニングをトリガ。

- **運用制約（リソース・人手）**
  - リスク: 高解像度・長時間動画の処理がリソース不足で滞る、評価やラベリングに割ける時間が限られる。
  - 緩和策:
    - `measure_performance.py` によるプロファイル結果をもとに、バッチサイズ/フレーム間引き比率を調整。
    - ラベリングは短時間セグメント（例: 各セッション 5 分間）に集中し、高品質な評価用データセットを少数用意。
    - CI には「短い代表セッション」「軽量設定」を用いてテストを走らせ、フル評価はバッチジョブに分離。

## 追加提案（任意）

- **ツールチェーン/MLOps**

  - **高優先度**: `mlflow` あるいは `Weights & Biases` を導入し、`main.py` 実行ごとに **パラメータ・メトリクス・アーティファクト**（MOT 指標, 再投影誤差, 可視化画像）を自動ログ。
  - **中優先度**: Hydra や OmegaConf を用いた設定管理に移行し、「ベース config + 差分 config」で実験を管理。
  - **中優先度**: GitHub Actions で `make test`, `scripts/comprehensive_test.py --metrics` を定期実行し、カバレッジと主要メトリクスの変動を監視。

- **アルゴリズム**

  - **高優先度**: 同一オブジェクト類似度には、まず既存 DeepSORT embedding による cosine similarity + motion feature の二段階フィルタを導入（実装コストが低く効果が高い）。
  - **中優先度**: 追跡アルゴリズムとして ByteTrack/OC-SORT などの最新 tracker との比較検証を行い、IDF1/MOTA の改善余地を検証。
  - **中〜長期**: マルチカメラ ReID では TransReID 等の ViT ベース ReID モデルの導入を検討し、ドメイン適応（fine-tuning）を行う。

- **実装パターン**
  - **高優先度**: `CoordinateTransformer`, `ZoneClassifier`, `ReID/Similarity` を interface（抽象基底クラス）と実装クラスに分離し、将来のモデル差し替えを容易にする（Open/Closed 原則）。
  - **中優先度**: config スキーマを `pydantic` / `dataclasses` で型安全に表現し、起動時に検証＆エラーを fail-fast にする。

以上をベースに、まずは **「実データ上のベースライン定量化」→「追跡/座標変換パラメータの系統的チューニング」→「UI/ダッシュボードの正規化」→「同一オブジェクト類似度モジュールの導入」** という順で進めることを推奨します。
