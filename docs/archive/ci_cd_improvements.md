# CI/CD 改善提案レポート

## 📋 概要

本ドキュメントは、現在の GitHub Actions 設定を評価し、ベストプラクティスに基づいた改善提案をまとめたものです。

## 🔍 現在の設定の評価

### ✅ 良い点

1. **ジョブの分離**: Format、Lint、型チェック、テストが適切に分離されている
2. **マトリックス戦略**: 複数 Python バージョンでのテスト実行
3. **カバレッジレポート**: Codecov へのアップロードと HTML レポート生成
4. **Pre-commit チェック**: コード品質の早期検出
5. **システム依存関係**: Tesseract OCR のインストール

### ⚠️ 改善が必要な点

1. **カバレッジ閾値の不整合**

   - CI: `--cov-fail-under=80` (80%必須)
   - `pyproject.toml`: `--cov-fail-under=0` (チェック無効)
   - **問題**: 設定が矛盾しており、CI で失敗する可能性

2. **依存関係キャッシュの不完全性**

   - pip キャッシュのみで、venv 全体をキャッシュしていない
   - **影響**: 毎回依存関係を再インストールするため、CI 時間が長い

3. **macOS ランナーの不在**

   - プロジェクトは MPS（Apple Silicon GPU）対応を重視
   - **問題**: macOS 環境での動作確認ができない

4. **セキュリティチェックの欠如**

   - 依存関係の脆弱性スキャンがない
   - **リスク**: セキュリティホールの早期発見ができない

5. **PR コメント機能の欠如**

   - カバレッジやテスト結果が PR に表示されない
   - **影響**: レビューアが CI 結果を確認しにくい

6. **パフォーマンステストの欠如**

   - プロジェクトのパフォーマンス目標（≤2 秒/フレーム）を CI で検証していない
   - **影響**: パフォーマンス劣化の早期発見ができない

7. **タイムアウト設定の欠如**

   - ジョブが無限に実行される可能性
   - **影響**: CI リソースの無駄遣い

8. **並列実行の最適化不足**
   - 依存関係のインストールが重複している
   - **影響**: CI 時間の無駄

## 🚀 改善提案

### 1. カバレッジ閾値の統一

**推奨**: 段階的にカバレッジを上げる戦略

```yaml
# CI設定
--cov-fail-under=75 # まず75%を目標（現状76%）
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "--cov-fail-under=75",  # CIと統一
]
```

**理由**:

- 現状のカバレッジ 76%を維持しつつ、段階的に 80%を目指す
- CI とローカル設定を統一して一貫性を保つ

### 2. 依存関係キャッシュの改善

**推奨**: pip キャッシュ + venv キャッシュの併用

```yaml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pipenv
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

**効果**: CI 時間を 30-50%短縮可能

### 3. macOS ランナーの追加

**推奨**: マトリックス戦略に macOS を追加

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11"]
    os: [ubuntu-latest, macos-latest]
```

**理由**:

- MPS 対応の動作確認が可能
- Apple Silicon 環境での互換性検証

### 4. セキュリティチェックの追加

**推奨**: Safety + Bandit の導入

```yaml
- name: Run safety check
  run: |
    pip install safety
    safety check --file requirements.txt

- name: Run bandit
  run: |
    pip install bandit[toml]
    bandit -r src/
```

**効果**:

- 依存関係の脆弱性を早期発見
- コード内のセキュリティ問題を検出

### 5. PR コメント機能の追加

**推奨**: python-coverage-comment-action の使用

```yaml
- name: Comment PR with coverage
  uses: py-cov-action/python-coverage-comment-action@v3
  with:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    MINIMUM_GREEN: 80
    MINIMUM_ORANGE: 60
```

**効果**: PR にカバレッジ情報が自動表示される

### 6. パフォーマンステストの追加

**推奨**: メインブランチでのみ実行

```yaml
performance:
  if: github.ref == 'refs/heads/main'
  steps:
    - name: Run performance benchmarks
      run: python scripts/measure_performance.py --ci-mode
```

**効果**: パフォーマンス劣化の早期発見

### 7. タイムアウト設定の追加

**推奨**: ジョブごとに適切なタイムアウトを設定

```yaml
jobs:
  test:
    timeout-minutes: 30 # テストは30分でタイムアウト
  format:
    timeout-minutes: 5 # Formatチェックは5分でタイムアウト
```

**効果**: 無限ループやハングの防止

### 8. スローテストの分離

**推奨**: マーカーでスローテストを分離

```yaml
- name: Run tests (fast)
  run: pytest tests/ -v -m "not slow"

- name: Run slow tests
  run: pytest tests/ -v -m "slow"
  if: github.event_name == 'pull_request'
```

**効果**:

- PR 時の CI 時間短縮
- メインブランチでは全テスト実行

## 📊 改善後の CI 構成

### ジョブ構成

1. **コード品質チェック**（並列実行、5-10 分）

   - Format Check
   - Lint Check
   - Type Check
   - Pre-commit Hooks

2. **テスト実行**（並列実行、15-30 分）

   - Ubuntu + Python 3.10
   - Ubuntu + Python 3.11
   - macOS + Python 3.11

3. **スローテスト**（条件付き実行、30-60 分）

   - 統合テストなど

4. **セキュリティチェック**（並列実行、5-10 分）

   - Safety（依存関係）
   - Bandit（コード）

5. **パフォーマンステスト**（メインブランチのみ、20-30 分）
   - ベンチマーク実行

### 推定 CI 時間

- **PR 時**: 約 20-30 分（スローテスト除外）
- **メインブランチ**: 約 40-60 分（全テスト実行）

## 🔧 PR 開発工程の改善提案

### 1. PR テンプレートの作成

`.github/pull_request_template.md`を作成：

```markdown
## 変更内容

<!-- 何を実装/修正したか -->

## 変更理由

<!-- なぜこの変更が必要か -->

## テスト方法

<!-- どのようにテストしたか -->

## チェックリスト

- [ ] テストが全て通る
- [ ] Lint エラーがない
- [ ] ドキュメントを更新
- [ ] カバレッジが 75%以上
```

### 2. ブランチ保護ルールの推奨

GitHub リポジトリ設定で以下を推奨：

- **必須ステータスチェック**:

  - Format Check
  - Lint Check
  - Type Check
  - Test (Python 3.10, Ubuntu)
  - Test (Python 3.11, Ubuntu)

- **レビュー要件**: 1 名以上の承認

- **マージ前のチェック**:
  - 最新のコミットが CI を通過していること

### 3. 自動レビューアアサイン

`.github/CODEOWNERS`を作成：

```
# コアモジュール
/src/pipeline/ @team-lead
/src/detection/ @ml-engineer
/src/tracking/ @ml-engineer

# テスト
/tests/ @qa-engineer

# 設定
/config.yaml @team-lead
```

### 4. 依存関係の更新ワークフロー

`.github/workflows/dependabot.yml`を作成（オプション）：

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## 📈 メトリクスとモニタリング

### 推奨メトリクス

1. **CI 成功率**: ≥95%
2. **平均 CI 時間**: ≤30 分（PR 時）
3. **テストカバレッジ**: ≥75%（段階的に 80%へ）
4. **Lint 警告**: 0 件
5. **型チェックエラー**: 0 件（段階的導入）

### モニタリング方法

- GitHub Actions の Insights タブで CI 時間と成功率を確認
- Codecov でカバレッジトレンドを追跡
- 週次で CI メトリクスをレビュー

## 🎯 実装優先順位

### Phase 1: 緊急（今すぐ実装）

1. ✅ カバレッジ閾値の統一
2. ✅ タイムアウト設定の追加
3. ✅ 依存関係キャッシュの改善

### Phase 2: 重要（1 週間以内）

4. ✅ macOS ランナーの追加
5. ✅ PR コメント機能の追加
6. ✅ セキュリティチェックの追加

### Phase 3: 推奨（2 週間以内）

7. ✅ スローテストの分離
8. ✅ パフォーマンステストの追加
9. ✅ PR テンプレートの作成

## 📝 まとめ

現在の CI 設定は基本的な機能は実装されていますが、以下の改善により、より堅牢で効率的な CI/CD パイプラインを構築できます：

1. **一貫性**: 設定ファイル間の不整合を解消
2. **効率性**: キャッシュと並列実行の最適化
3. **信頼性**: macOS 対応とセキュリティチェック
4. **可視性**: PR コメントとメトリクス監視
5. **保守性**: タイムアウトとエラーハンドリング

これらの改善により、開発効率とコード品質の両方が向上します。
