# 技術導入 PoC キット（M1 Max / CPU 前提）

本ドキュメントは **遮蔽・照明変化へのロバスト性向上** を主目的とした PoC の実行手順と、導入メリット/デメリット、期待効果をまとめたものです。GPU は使わず **M1 Max 64GB のローカル CPU** 想定です。

## 0. 現状サマリ（コードベース確認）
- パイプライン: `main.py` → `PipelineOrchestrator` が `01_extraction` → `02_detection`(ViT) → `03_tracking`(DeepSORT ベース) → `04_transform`(ホモグラフィ/区分的アフィン) → `05_aggregation` → `06_visualization` を実行。
- 追跡: `src/tracking/tracker.py` + `SimilarityCalculator` + Hungarian + Kalman。外観特徴は `DetectionPhase` の ViT で抽出。
- 座標変換: 既定は `HomographyTransformer` (`src/transform/homography.py`)。`piecewise_affine.py` で高精度補正あり。評価器は `src/evaluation/transform_evaluator.py`。
- 出力: セッション単位で `output/sessions/YYYYMMDD_*` 以下に各フェーズ成果物と `summary.json` を保存。

## 1. PoC 共通方針
- 対象データ: 遮蔽・照明変化を含む動画 2–3 本を代表ケースに採用。既存セッション出力を再利用可。
- 計測指標: `fps`（追跡区間）、`MOTA/MOTP`（追跡）、`再投影誤差 RMSE / Max`（座標変換）。加えて「準備/実行にかかる手間・時間」をメモ。
- 低負荷設定: 解像度を一段ダウンスケール、フレーム間引き（例: 2〜3 フレームに 1 回）で CPU 負荷を抑制。

## 2. 追跡強化: PyTracking (TaMOs/RTS)
- 期待効果: 遮蔽中のリロック性能向上、動きの滑らかさ改善。
- デメリット: 依存追加・モデルサイズ増、CPU 推論はやや重い。
- 導入手順（PoC）:
  1. 依存導入（ローカルのみ）
     ```bash
     pip install pytracking==0.5 torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cpu
     ```
  2. 既存検出結果を入力し PyTracking を呼び出す簡易ラッパーを `tools/poc_tracking_pytracking.py` に追加済み（PyTracking 未導入時は DeepSORT ベースラインのみ出力）。
     - 入力: 画像ディレクトリと検出 JSON（frame_num, bbox[x,y,w,h], score）。
     - 出力: `tracks_pytracking.csv/json`（track_id 付与済み）＋fps ログ。
  3. 比較: 既存 DeepSORT の `tracks.csv` と MOTA/MOTP を並べる。
- 軽量化ヒント: 256〜320px 辺長へのリサイズ、フレーム間引き、`--max-frames` 指定で短尺試験。

## 3. 座標変換ロバスト化: Deep Homography Estimation
- 期待効果: 照明変化・局所歪みに対する安定化、再投影誤差の低下。
- デメリット: モデル依存追加、推論コスト増（ただし Tiny/Small モデルなら CPU でも現実的）。
- 導入手順（PoC）:
  1. 依存導入（必要時のみ）
     ```bash
     pip install kornia==0.7.3 torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu
     ```
  2. `tools/poc_deep_homography.py` で、既存ホモグラフィと簡易 DeepHomography (Kornia) を比較。
     - 入力: 対応点 JSON (`data/correspondence_points_cam01.json` 相当)
     - 出力: RMSE/Max エラーと推論時間の比較表。
  3. 誤差が閾値を下回る設定（学習有無/反復回数）をメモ。

## 4. 実行管理: Dagster PoC
- 期待効果: 各フェーズの入出力を Asset として可視化し、中間データを再利用。
- デメリット: 依存追加・UI 起動のオーバーヘッド。
- 手順:
  1. 依存（ローカルのみ）: `pip install dagster dagit`.
  2. `tools/dagster_poc.py` を起動:
     ```bash
     dagit -m tools.dagster_poc
     ```
     - Asset: `extraction_frames`, `detections`, `tracks`, `transforms`, `aggregations`.
     - 実体は既存 `PipelineOrchestrator` 呼び出しを薄くラップしただけなので、CPU 環境で小さな動画を指定して動作確認。
  3. 成果: UI スクリーンショットとランタイム計測。

## 5. 再現性: DVC（ローカルキャッシュのみ）
- 期待効果: 重データの差分管理とキャッシュ再利用。
- デメリット: `.dvc/cache` 容量増、初回セットアップ手間。
- 手順:
  1. `pip install dvc`.
  2. ルートに `dvc.yaml` を追加済み（ステージ `pipeline-run`）。
     ```bash
     dvc repro
     ```
     で `python main.py --config config/calibration_template.yaml` を実行し、`output/` をキャッシュ対象に。
  3. `.dvcignore` にログ/画像を追加するなどでキャッシュ量を制御（必要に応じて調整）。

## 6. 品質保証: Great Expectations
- 期待効果: 異常データ（null, 範囲外）の早期検知。
- デメリット: 依存追加、初回スイート作成の手間。
- 手順:
  1. `pip install great-expectations`.
  2. `tools/gx_validate.py` で `output/summary.json` と `data/gt_tracks_auto.json` を簡易チェック。
     - 成功/失敗を標準出力に要約。
     - Dagster から呼び出す op も同ファイル内に用意。

## 7. 自動評価: CML (GitHub Actions)
- 期待効果: PR/手動トリガーで精度変化を Markdown レポート化。
- デメリット: CI 時間増。GPU なし前提のためテストを軽量に設定。
- 手順:
  - `.github/workflows/cml-poc.yml` を追加済み（`workflow_dispatch` 手動起動）。
  - 実行内容: 依存インストール → 軽量 pytest (`-k transform`) → `metrics.md` を生成 → アーティファクト保存（コメント送信は任意で `CML_SEND_COMMENT` true 時のみ）。

## 8. 優先度と判断基準
- まずは PyTracking / Deep Homography の **精度差分と fps** を測り、目標 (例: MOTA +3pt, 再投影 RMSE -20%) を満たすか確認。
- Dagster/DVC/GX/CML は「運用コスト < 可視化・再現性メリット」かを、セットアップ時間と実行時間で評価。
- 導入を進める場合は、設定の永続化（`config/`）と CI トリガー種別（push/pr/manual）の整理を次ステップとする。
