# システム全体概要

最終更新: 2025年11月14日

本ドキュメントは、オフィス人物検出システムの全体像と主要コンポーネントの整理された一覧です。詳細な実装レポートや履歴資料は `docs/archive/` 以下に保管しています。

## 現在のプロジェクト状態

- **テスト数**: 443件（2025年11月14日時点）
- **テストカバレッジ**: 測定が必要（`make test TEST_MODE=coverage` で確認可能）
- **主要機能**: すべて実装済み
  - ✅ フレーム抽出（OCR タイムスタンプ）
  - ✅ 人物検出（ViT/DETR）
  - ✅ 追跡（DeepSORT）
  - ✅ 座標変換（ホモグラフィ）
  - ✅ ゾーン判定・集計
  - ✅ 可視化

---

## パイプラインの流れ

1. **Phase1: フレーム抽出**
   - `src/pipeline/frame_extraction_pipeline.py`
   - OCR でタイムスタンプを読み取り、5 分刻みのフレームを抽出
2. **Phase2: 検出**
   - `src/pipeline/phases/detection.py`
   - ViT ベースの人物検出 (`src/detection/vit_detector.py`)
3. **Phase2.5: 追跡（オプション）**
   - `src/pipeline/phases/tracking.py`
   - DeepSORT ベースの ID 付与と外観特徴量抽出
4. **Phase3: 座標変換**
   - `src/pipeline/phases/transform.py`
   - ホモグラフィによるカメラ座標 → フロアマップ座標変換
5. **Phase4: 集計**
   - `src/pipeline/phases/aggregation.py`
   - ゾーン別人数カウントと統計値算出
6. **Phase5: 可視化**
   - `src/pipeline/phases/visualization.py`
   - タイムシリーズグラフとフロアマップ生成

---

## 主要ディレクトリと責務

- `src/config/`
  設定ファイル読み込みと検証 (`ConfigManager`)
- `src/timestamp/`
  OCR・バリデーション・ROI 抽出などのタイムスタンプ処理
- `src/video/`
  動画入出力とサンプリング (`VideoProcessor`, `CoarseSampler`, `FineSampler`)
- `src/detection/`
  ViT ベース検出器、前処理、後処理
- `src/tracking/`
  追跡アルゴリズム、特徴量生成、トラック管理
- `src/transform/`
  ホモグラフィ行列の適用と座標変換
- `src/aggregation/`
  集計ロジックと CSV エクスポート
- `src/visualization/`
  グラフ生成、フロアマップ描画、特徴量可視化
- `src/pipeline/`
  フェーズオーケストレーション (`PipelineOrchestrator`, `phases/`)
- `src/utils/`
  ログ・出力管理・統計・PyTorch/MPS ユーティリティ

---

## 実行・運用のエントリポイント

- `main.py`
  CLI で `--config`, `--start-time`, `--timestamps-only`, `--evaluate` などを指定可能
- `scripts/`
  精度評価、再投影誤差測定、特徴量可視化などのユーティリティスクリプト群
- `Makefile`
  `make test`, `make lint`, `make format`, `make run` を定義

---

## 推奨ワークフロー

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python main.py --config config.yaml`
4. 出力は `output/` 直下またはセッションディレクトリに生成

詳細な手順は `docs/QUICK_START.md` を参照してください。

---

## 関連ドキュメント

- `docs/architecture/`
  出力構造とフロアマップ統合の詳細
- `docs/guides/`
  出力整理ガイド・OCR エンジンの選定指針
- `docs/operations/ci_cd.md`
  CI/CD と運用手順のサマリー
- `docs/references/`
  サンプリング設定やタイムラプス処理の詳細ノート
- `docs/archive/`
  過去の詳細レポートと検証履歴（必要な場合のみ参照）
