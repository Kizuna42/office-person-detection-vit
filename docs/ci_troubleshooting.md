# CIトラブルシューティングガイド

## 🔍 よくあるCI失敗の原因と対処法

### 1. Format Check失敗

**症状**: `ruff format --check`が失敗する

**原因**: コードがRuffのフォーマット規約に従っていない

**対処法**:
```bash
# ローカルでフォーマットを実行
make format
# または
ruff format .

# 変更をコミット
git add .
git commit -m "style: コードフォーマットを修正"
```

### 2. Lint Check失敗

**症状**: `ruff check`が失敗する

**原因**: Lintエラーがある

**対処法**:
```bash
# ローカルでLintチェック
make lint

# 自動修正可能なエラーを修正
ruff check . --fix

# 変更をコミット
git add .
git commit -m "fix: Lintエラーを修正"
```

### 3. Type Check失敗

**症状**: `mypy`が失敗する

**原因**: 型エラーがある

**対処法**:
```bash
# ローカルで型チェック
mypy src/ --ignore-missing-imports

# エラーを確認して修正
# 必要に応じて型ヒントを追加
```

### 4. Pre-commit失敗

**症状**: `pre-commit run --all-files`が失敗する

**原因**: Pre-commitフックが失敗している

**対処法**:
```bash
# ローカルでPre-commitを実行
make precommit-run

# エラーを確認して修正
# 自動修正される場合は、変更をコミット
```

### 5. テスト失敗

**症状**: `pytest`が失敗する

**原因**:
- テストが失敗している
- カバレッジが75%未満

**対処法**:
```bash
# ローカルでテストを実行
make test

# カバレッジ付きで実行
make test-cov

# 失敗したテストを確認
pytest tests/ -v -k "test_name"

# カバレッジが75%未満の場合、テストを追加
```

### 6. macOSでのTesseractインストール失敗

**症状**: macOSランナーで`brew install`が失敗する

**原因**: Homebrewがインストールされていない、または権限の問題

**対処法**:
- CI設定でHomebrewのインストールを自動化済み
- ローカルで確認する場合:
```bash
# Homebrewがインストールされているか確認
command -v brew

# インストールされていない場合
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Tesseractをインストール
brew install tesseract tesseract-lang
```

### 7. カバレッジコメントが表示されない

**症状**: PRにカバレッジコメントが表示されない

**原因**:
- `python-coverage-comment-action`の設定問題
- `coverage.xml`が生成されていない

**対処法**:
- CI設定で`continue-on-error: true`を設定済み（コメント失敗でCIを止めない）
- `coverage.xml`が生成されているか確認

### 8. セキュリティチェックの警告

**症状**: SafetyやBanditで警告が出る

**原因**: 依存関係の脆弱性やセキュリティ問題

**対処法**:
```bash
# ローカルで確認
pip install safety bandit[toml]
safety check --file requirements.txt
bandit -r src/

# 脆弱性が見つかった場合、依存関係を更新
pip install --upgrade package_name
```

## 🔧 CI設定の確認方法

### GitHub Actionsのログを確認

1. GitHubリポジトリの「Actions」タブを開く
2. 失敗したワークフローを選択
3. 失敗したジョブをクリック
4. エラーメッセージを確認

### ローカルでCIコマンドを実行

```bash
# Formatチェック
ruff format --check .

# Lintチェック
ruff check .

# 型チェック
mypy src/ --ignore-missing-imports

# Pre-commit
pre-commit run --all-files

# テスト
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75 -m "not slow"
```

## 📊 CI時間の最適化

### 現在のCI時間

- **PR時**: 約20-30分（スローテスト除外）
- **メインブランチ**: 約40-60分（全テスト実行）

### 時間短縮のヒント

1. **スローテストの分離**: PR時は`-m "not slow"`で除外
2. **キャッシュの活用**: pipキャッシュで依存関係インストール時間を短縮
3. **並列実行**: ジョブを並列実行して全体時間を短縮

## 🚨 緊急時の対処

### CIが完全に失敗する場合

1. **一時的にCIをスキップ**（非推奨）:
```bash
git commit -m "fix: ... [skip ci]"
```

2. **問題のあるジョブを一時的に無効化**:
`.github/workflows/ci.yml`で該当ジョブをコメントアウト

3. **緊急修正をマージ**:
- ブランチ保護ルールを一時的に無効化（管理者権限が必要）
- 修正後に再度有効化

### カバレッジが急に下がった場合

1. カバレッジレポートを確認（`htmlcov/index.html`）
2. カバレッジが下がったファイルを特定
3. テストを追加してカバレッジを回復

## 📝 チェックリスト

PR作成前に以下を確認：

- [ ] `make format`でフォーマットチェック
- [ ] `make lint`でLintチェック
- [ ] `mypy src/`で型チェック
- [ ] `make precommit-run`でPre-commitチェック
- [ ] `make test-cov`でテストとカバレッジ確認
- [ ] カバレッジが75%以上であることを確認

## 🔗 関連ドキュメント

- CI/CD改善提案: `docs/ci_cd_improvements.md`
- CI設定: `.github/workflows/ci.yml`
- テストガイドライン: `.cursor/rules/testing-standards.mdc`
