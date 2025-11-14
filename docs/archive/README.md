# アーカイブドキュメント

このディレクトリには、過去の実装レポート、評価レポート、実行結果などの履歴資料が保管されています。

> **重要**: アーカイブ内のドキュメントは過去の実装状況を記録したものであり、現状のプロジェクト状態とは異なる場合があります。最新の情報は `docs/README.md` を参照してください。

## ドキュメント一覧

### 実装完了レポート

- **`final_implementation_report.md`** (2025-11-07)
  - 推奨事項実装完了レポート（最終版）
  - スクリプト実行結果、テストカバレッジ改善結果を含む

- **`recommendations_implementation_complete_report.md`** (2025-11-07)
  - 推奨事項実装完了レポート（スクリプト実行結果）

- **`script_execution_results.md`** (2025-11-07)
  - スクリプト実行結果のまとめ（`final_implementation_report.md` と重複）

### テスト評価レポート

- **`test_evaluation_report_final.md`** (2025-11-07)
  - テストスクリプト評価レポート（最終版）
  - テスト数: 282件（当時）、現在は443件

- **`test_coverage_improvement_report.md`** (2025-11-07)
  - テスト拡充とカバレッジ向上レポート
  - カバレッジ: 88%（当時）

- **`bugfix_and_coverage_report.md`** (2025-11-07)
  - 検出画像保存エラー修正 & テストカバレッジ測定レポート

### 実装状況レポート

- **`implementation_evaluation_report.md`** (2025-11-07)
  - 実装状況と効果測定レポート（詳細版）

- **`implementation_status_report.md`** (2025-11-07)
  - 実装状況評価レポート（簡易版）
  - 詳細は `implementation_evaluation_report.md` を参照

- **`tracking_integration_report.md`** (2025-11-07)
  - 追跡機能パイプライン統合レポート

- **`phase1-3_complete.md`**
  - Phase 1-3 実装完了レポート（現在はPhase 5まで完了）

### 機能統合・改善レポート

- **`improved_features_integration.md`**
  - 改善機能の統合とテストガイド（過去の統合ガイド）

- **`ml_engineer_action_plan.md`**
  - 機械学習エンジニア向けアクション計画（過去の計画）

### CI/CD関連

- **`ci_cd_summary.md`**
  - CI/CD改善サマリー（過去の改善状況）

- **`ci_cd_improvements.md`**
  - CI/CD改善提案レポート（過去の提案）

- **`ci_troubleshooting.md`**
  - CI/CDトラブルシューティングガイド（過去のガイド）

### 検証・確認ガイド

- **`implementation_verification_guide.md`**
  - 実装機能の動作確認ガイド（過去のガイド）
  - 最新は `docs/QUICK_START.md` を参照

- **`visual_inspection_checklist.md`**
  - 目視確認チェックリスト（過去のチェックリスト）

### OCR関連

- **`ocr_improvement/ocr_accuracy_evaluation_report.md`**
  - OCR精度評価レポート

## 現在のプロジェクト状態（2025-11-14時点）

- **テスト数**: 443件
- **テストカバレッジ**: 測定が必要（`make test TEST_MODE=coverage` で確認可能）
- **主要機能**: すべて実装済み

最新の情報は `docs/overview/system_overview.md` を参照してください。
