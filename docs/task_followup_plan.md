<!-- 追跡・座標変換改善タスク フォローアップ計画 -->

# タスクフォローアップ計画（タスク 1–4 実装後）

## 1. 目的

- タスク 1 ～ 4 で実装済みの成果物を目視確認し、フィードバックループを確立する。
- 追跡・座標変換・可視化の精度を高めるため、パラメータ最適化や追加検証を継続的に行う。
- 品質指標（テストカバレッジ、精度指標、パフォーマンス計測）を基準値まで引き上げ、運用体制を整える。

---

## 2. 直近アクション（目視確認）

| ID  | 内容                                                                                                      | 担当     | 目的・期待成果                                 | 参考コマンド/パス                                                                            | 状態   |
| --- | --------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------- | ------ |
| V1  | `output/latest/phase5_visualization/side_by_side_tracking.mp4` を確認し、ID 一貫性・軌跡描画をレビュー    | ユーザー | DeepSORT 統合の最終確認、誤追跡の洗い出し      | `/Users/kizuna/Aeterlink/yolo3/output/latest/phase5_visualization/side_by_side_tracking.mp4` | 未着手 |
| V2  | 可視化フロアマップ（`output/latest/phase5_visualization/floormaps/`）で軌跡とゾーン表示の重なり具合を確認 | ユーザー | ゾーン境界近辺での誤差確認、色分け可読性の評価 | `/Users/kizuna/Aeterlink/yolo3/output/latest/phase5_visualization/floormaps/`                | 未着手 |
| V3  | Streamlit アプリ `tools/visualizer_app.py` を起動し、ID フィルタ・ゾーンフィルタ操作感をチェック          | ユーザー | UI/UX 評価、リアルタイム更新のレスポンス把握   | `streamlit run tools/visualizer_app.py`                                                      | 未着手 |
| V4  | `tools/visual_inspection.py --mode tracking` で軌跡確認動画を生成し、アニメーションの滑らかさを確認       | ユーザー | `TrajectoryExporter` の動画生成結果検証        | `python tools/visual_inspection.py --mode tracking --session <session_id>`                   | 未着手 |
| V5  | `tools/visual_inspection.py --mode reprojection` で再投影誤差マップを生成し、歪み補正有無の差分を比較     | ユーザー | ホモグラフィ精度の体感把握                     | `python tools/visual_inspection.py --mode reprojection --session <session_id>`               | 未着手 |

> フィードバックは `docs/implementation_evaluation_report.md` または新規ノートに追記して共有する。

### 2.1 目視タスク詳細手順

#### V1: サイドバイサイド動画で追跡挙動を把握

1. Finder で `output/latest/phase5_visualization/side_by_side_tracking.mp4` を開き（QuickTime 再生で可）、左右の表示が同期しているかを確認。
2. 観察ポイント
   - 左（原動画＋検出枠）と右（フロアマップ＋軌跡）がフレーム単位で同期。
   - 同一人物に付く `track_id` が途切れない（ID スイッチ/欠落/再割り当ての有無）。
   - 軌跡線が急激にジャンプ・消失しない（Kalman Filter とアサイン結果を推測）。
3. 気づきはタイムスタンプと合わせてメモし、後続パラメータ調整の材料にする。

#### V2: フロアマップ静止画でゾーン精度を確認

1. `output/latest/phase5_visualization/floormaps/` 配下の PNG を閲覧（例: `floormap_2025/08/26 161455.png`）。
2. 観察ポイント
   - ゾーン境界付近で足元位置が外れていないか。
   - `track_id` ごとの色が識別しやすいか（過度に似た色が出ていないか）。
   - 左上表示（Frame/Time）とゾーン別人数が期待値と合致するか。
3. フレーム連番を並べると ID 継続性やゾーン遷移の傾向が把握しやすい。

#### V3: Streamlit UI でインタラクティブ検証

1. ルートで仮想環境を有効化後、以下を実行:
   ```bash
   streamlit run tools/visualizer_app.py -- --session latest
   ```
   - `--session` で任意のセッション ID を指定可能。未指定時は最新セッション。
2. ブラウザ（http://localhost:8501）が開いたら、次を確認:
   - 時間スライダー操作時の描画遅延や崩れ。
   - 「ID フィルタ」「ゾーンフィルタ」で表示対象が正しく絞り込めるか。
   - 統計情報パネルの数値が画面に描画されている結果と一致するか。
3. UI からエクスポートしたファイルは `output/latest/interactive_exports/` に保存される。

#### V4: `visual_inspection.py`（tracking モード）で軌跡動画を生成

1. セッション ID を確認（例: 最新は `output/latest` シンボリックリンクを参照）。
   ```bash
   ls -lrt output | tail
   ```
2. 以下で専用軌跡動画を生成:
   ```bash
   python tools/visual_inspection.py --mode tracking --session <session_id> --output-dir output/inspection/tracking
   ```
3. 出力: `output/inspection/tracking/trajectories.mp4`
4. 観察ポイント
   - 最新フレームほど濃く描かれる透明度グラデーションが適切か。
   - 停滞人物で軌跡が過密になり過ぎないか。
   - ID 再割当時の軌跡分断状況。

#### V5: `visual_inspection.py`（reprojection モード）で誤差マップを生成

1. 歪み補正有無を比較する場合は `--compare-distortion` オプションを使用:
   ```bash
   python tools/visual_inspection.py --mode reprojection --session <session_id> --output-dir output/inspection/reprojection --compare-distortion
   ```
2. 出力
   - 誤差マップ: `output/inspection/reprojection/error_map.png`
   - 集計: `output/inspection/reprojection/summary.json`（平均/最大誤差など）
3. 観察ポイント
   - 平均誤差 ≤ 2px、最大誤差 ≤ 4px を満たすか。
   - 高誤差領域が特定の位置に偏っていないか（対応点追加・ホモグラフィ再調整の判断材料）。

---

## 3. パラメータ最適化タスク

| ID  | 領域             | 作業内容                                                                               | アプローチ                                                                                | 成果物                              | 優先度 |
| --- | ---------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------- | ------ |
| P1  | 追跡（DeepSORT） | `tracking.appearance_weight`, `tracking.motion_weight`, `iou_threshold` のチューニング | - サンプルセッションで複数設定を試行<br>- MOT メトリクス計測と併用                        | 調整結果サマリ（設定値・MOTA/IDF1） | 高     |
| P2  | 追跡             | `max_age`, `min_hits` の見直し                                                         | - ID 切替頻度（APS：Assignments per second）と比較<br>- 夜間・人流減少時の挙動を確認      | 設定見直し案                        | 中     |
| P3  | 座標変換         | `calibration.use_distortion_correction` を有効化した際の再投影誤差評価                 | - `tools/homography_calibrator.py` で RANSAC 阈値を変更<br>- 歪みパラメータ有無で誤差比較 | 誤差計測レポート                    | 高     |
| P4  | 座標変換         | 多段階変換（正規化座標ステップ）の実装検討                                             | - `CoordinateTransformer` に正規化処理を追加<br>- 座標変換後の誤差を再評価                | 実装方針メモ／PR                    | 中     |
| P5  | 可視化           | 軌跡透明度 (`alpha`) と `max_length` の調整                                            | - 視認性アンケート（UX 観点）<br>- 長時間セッションでの重なり具合確認                     | 推奨設定値                          | 低     |

---

### 3.1 パラメータ調整の進め方（共通フロー）

1. `config.yaml` の該当セクションを書き換える。
   ```yaml
   tracking:
     appearance_weight: 0.65
     motion_weight: 0.35
     iou_threshold: 0.35
   calibration:
     use_distortion_correction: true
   ```
   - `config/` 以下で別ファイルを作りたい場合は `python main.py --config config/custom.yaml` とする。
2. 新設定でパイプラインを実行。
   ```bash
   python main.py --config config.yaml --session-tag tuning-20250128-01
   ```
   - `--session-tag` を付けると `output/tuning-20250128-01_YYYYMMDDHHMMSS/` に成果物がまとまる。
3. 実行完了後の成果物を確認。
   - `output/<session>/metadata.json` … 実行設定の記録。
   - `output/<session>/summary.json` … ゾーン別人数や統計値。
   - `output/<session>/phase5_visualization/` … 目視対象の動画・静止画。
4. 評価スクリプトを併用して定量評価。
   ```bash
   python scripts/comprehensive_test.py --session <session_id> --metrics
   python tools/visual_inspection.py --mode reprojection --session <session_id>
   ```
   - MOTA / IDF1、再投影誤差などをメモに残す。
5. 比較ログを残す。
   - `git diff config.yaml` をコピーしてノートに貼る。
   - テンプレート例:
     ```
     trial: tuning-20250128-01
     params: appearance_weight=0.65 / motion_weight=0.35 / iou_threshold=0.35
     observation:
       - 13:05 frame -> ID2とID4が交換
       - 13:20 frame -> 停滞人物のID維持成功
     next action:
       - motion_weightを0.30に再調整し再実行
     ```

---

## 4. 品質向上タスク（未完了・要対応）

### 4.1 テストカバレッジ向上

- **現状**: 総合カバレッジ 75%（目標 80%）
- **課題モジュール**:
  - キャリブレーション: `src/calibration/camera_calibrator.py`, `src/calibration/reprojection_error.py`
  - エクスポート: `src/utils/export_utils.py`
- **アクション**:
  1. テストケース追加
     - `tests/test_camera_calibrator.py`: 既存だがケース不足 → 画像モックの追加
     - `tests/test_export_utils.py`: 動画出力のモック（`cv2.VideoWriter`）で分岐網羅
  2. `pytest --cov=src --cov-report=html` で改善状況を可視化
  3. カバレッジレポートを `output/coverage/` に保存し共有

### 4.2 精度指標の運用化

- `src/evaluation/mot_metrics.py` を用い、以下を定期計測
  - MOTA ≥ 0.7、IDF1 ≥ 0.8 を目標値として設定
  - `scripts/comprehensive_test.py --metrics` の実行結果を記録
- 再投影誤差（`ReprojectionErrorEvaluator`）をキャリブレーション毎に記録
  - 目標: 平均誤差 ≤ 2px、最大誤差 ≤ 4px
  - 結果は `output/evaluations/<session>/` へ保存

### 4.3 パフォーマンス計測

- `src/utils/performance_monitor.py` を活用し、代表セッションで以下を測定:
  - フレーム処理時間（目標 ≤ 2 秒／フレーム）
  - メモリ使用量（目標 ≤ 12 GB）
- 計測スクリプト（例: `scripts/measure_performance.py --session latest`）を整備し、CI のパフォーマンスジョブで自動実行できるよう改善

---

## 5. ドキュメント・運用整備

| 項目 | 内容                                                                                | 期限目安                          |
| ---- | ----------------------------------------------------------------------------------- | --------------------------------- |
| D1   | `docs/implementation_evaluation_report.md` に目視確認結果・パラメータ調整結果を追記 | フィードバック取得後 3 営業日以内 |
| D2   | `docs/final_implementation_report.md` を最新状況へ更新（進捗率 87%→ 調整後反映）    | P タスク完了のタイミング          |
| D3   | `tasks.md`（または本ドキュメント）を定期更新し、未完了タスク欄を最新化              | 週次更新                          |

---

## 6. タスク管理サマリ（次ステップ）

1. 目視確認（V1–V5）を実施し、結果をレポートへ記録
2. DeepSORT およびホモグラフィ関連パラメータの調整（P1–P4）
3. テスト増強・精度／パフォーマンス計測の運用化（セクション 4）
4. 結果をドキュメントへ反映し、次回計画に備える

---

## 7. 参考リソース

- 追跡関連コード: `src/tracking/`
- 座標変換・キャリブレーション: `src/transform/`, `src/calibration/`
- 可視化: `src/visualization/floormap_visualizer.py`, `tools/visualizer_app.py`
- エクスポート: `src/utils/export_utils.py`
- CI/CD 構成: `.github/workflows/ci.yml`

上記を基に、目視確認とパラメータチューニングを進めつつ、品質保証タスクを順次完了させる。
