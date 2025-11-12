<!-- 追跡・座標変換改善タスク フォローアップ計画 -->
# タスクフォローアップ計画（タスク1–4実装後）

## 1. 目的

- タスク1～4で実装済みの成果物を目視確認し、フィードバックループを確立する。
- 追跡・座標変換・可視化の精度を高めるため、パラメータ最適化や追加検証を継続的に行う。
- 品質指標（テストカバレッジ、精度指標、パフォーマンス計測）を基準値まで引き上げ、運用体制を整える。

---

## 2. 直近アクション（目視確認）

| ID | 内容 | 担当 | 目的・期待成果 | 参考コマンド/パス | 状態 |
|----|------|------|----------------|--------------------|------|
| V1 | `output/latest/phase5_visualization/side_by_side_tracking.mp4` を確認し、ID一貫性・軌跡描画をレビュー | ユーザー | DeepSORT統合の最終確認、誤追跡の洗い出し | `/Users/kizuna/Aeterlink/yolo3/output/latest/phase5_visualization/side_by_side_tracking.mp4` | 未着手 |
| V2 | 可視化フロアマップ（`output/latest/phase5_visualization/floormaps/`）で軌跡とゾーン表示の重なり具合を確認 | ユーザー | ゾーン境界近辺での誤差確認、色分け可読性の評価 | `/Users/kizuna/Aeterlink/yolo3/output/latest/phase5_visualization/floormaps/` | 未着手 |
| V3 | Streamlitアプリ `tools/visualizer_app.py` を起動し、IDフィルタ・ゾーンフィルタ操作感をチェック | ユーザー | UI/UX評価、リアルタイム更新のレスポンス把握 | `streamlit run tools/visualizer_app.py` | 未着手 |
| V4 | `tools/visual_inspection.py --mode tracking` で軌跡確認動画を生成し、アニメーションの滑らかさを確認 | ユーザー | `TrajectoryExporter` の動画生成結果検証 | `python tools/visual_inspection.py --mode tracking --session <session_id>` | 未着手 |
| V5 | `tools/visual_inspection.py --mode reprojection` で再投影誤差マップを生成し、歪み補正有無の差分を比較 | ユーザー | ホモグラフィ精度の体感把握 | `python tools/visual_inspection.py --mode reprojection --session <session_id>` | 未着手 |

> フィードバックは `docs/implementation_evaluation_report.md` または新規ノートに追記して共有する。

---

## 3. パラメータ最適化タスク

| ID | 領域 | 作業内容 | アプローチ | 成果物 | 優先度 |
|----|------|----------|------------|--------|--------|
| P1 | 追跡（DeepSORT） | `tracking.appearance_weight`, `tracking.motion_weight`, `iou_threshold` のチューニング | - サンプルセッションで複数設定を試行<br>- MOTメトリクス計測と併用 | 調整結果サマリ（設定値・MOTA/IDF1） | 高 |
| P2 | 追跡 | `max_age`, `min_hits` の見直し | - ID切替頻度（APS：Assignments per second）と比較<br>- 夜間・人流減少時の挙動を確認 | 設定見直し案 | 中 |
| P3 | 座標変換 | `calibration.use_distortion_correction` を有効化した際の再投影誤差評価 | - `tools/homography_calibrator.py` で RANSAC 阈値を変更<br>- 歪みパラメータ有無で誤差比較 | 誤差計測レポート | 高 |
| P4 | 座標変換 | 多段階変換（正規化座標ステップ）の実装検討 | - `CoordinateTransformer` に正規化処理を追加<br>- 座標変換後の誤差を再評価 | 実装方針メモ／PR | 中 |
| P5 | 可視化 | 軌跡透明度 (`alpha`) と `max_length` の調整 | - 視認性アンケート（UX観点）<br>- 長時間セッションでの重なり具合確認 | 推奨設定値 | 低 |

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
- 計測スクリプト（例: `scripts/measure_performance.py --session latest`）を整備し、CIのパフォーマンスジョブで自動実行できるよう改善

---

## 5. ドキュメント・運用整備

| 項目 | 内容 | 期限目安 |
|------|------|---------|
| D1 | `docs/implementation_evaluation_report.md` に目視確認結果・パラメータ調整結果を追記 | フィードバック取得後3営業日以内 |
| D2 | `docs/final_implementation_report.md` を最新状況へ更新（進捗率87%→調整後反映） | Pタスク完了のタイミング |
| D3 | `tasks.md`（または本ドキュメント）を定期更新し、未完了タスク欄を最新化 | 週次更新 |

---

## 6. タスク管理サマリ（次ステップ）

1. 目視確認（V1–V5）を実施し、結果をレポートへ記録
2. DeepSORTおよびホモグラフィ関連パラメータの調整（P1–P4）
3. テスト増強・精度／パフォーマンス計測の運用化（セクション4）
4. 結果をドキュメントへ反映し、次回計画に備える

---

## 7. 参考リソース

- 追跡関連コード: `src/tracking/`
- 座標変換・キャリブレーション: `src/transform/`, `src/calibration/`
- 可視化: `src/visualization/floormap_visualizer.py`, `tools/visualizer_app.py`
- エクスポート: `src/utils/export_utils.py`
- CI/CD構成: `.github/workflows/ci.yml`

上記を基に、目視確認とパラメータチューニングを進めつつ、品質保証タスクを順次完了させる。
