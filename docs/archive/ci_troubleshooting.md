# CI/CD トラブルシューティングガイド

> **注意**: このガイドは過去のものです。最新のCI/CD情報は `docs/operations/ci_cd.md` を参照してください。

## 📋 概要

このドキュメントは、GitHub ActionsのCI/CDパイプラインで発生する可能性のある問題とその解決方法をまとめたものです。

## 🔍 よくある問題と解決方法

### 1. Format Check が失敗する

**症状**: `ruff format --check`が失敗する

**原因**:
- コードがRuffのフォーマット規約に従っていない

**解決方法**:

```bash
# ローカルでフォーマットを実行
make format
# または
ruff format .

# 変更をコミット
git add .
git commit -m "style: コードフォーマットを修正"
git push
```

### 2. Lint Check が失敗する

**症状**: `ruff check`が失敗する

**原因**:
- Lintエラーが存在する

**解決方法**:

```bash
# ローカルでLintチェックを実行
make lint
# または
ruff check .

# 自動修正可能なエラーを修正
ruff check . --fix

# 変更をコミット
git add .
git commit -m "fix(lint): Lintエラーを修正"
git push
```

### 3. Type Check が失敗する

**症状**: `mypy src/`が失敗する

**原因**:
- 型アノテーションの問題

**解決方法**:

```bash
# ローカルで型チェックを実行
mypy src/ --ignore-missing-imports

# エラーを確認して修正
# 必要に応じて型アノテーションを追加
```

**注意**: 既存コードの型エラーは段階的に対応するため、`--ignore-missing-imports`を使用しています。

### 4. Pre-commit Check が失敗する

**症状**: `pre-commit run --all-files`が失敗する

**原因**:
- Pre-commitフックのエラー

**解決方法**:

```bash
# ローカルでPre-commitを実行
make precommit-run
# または
pre-commit run --all-files

# 自動修正可能なエラーを修正
pre-commit run --all-files --hook-stage commit

# 変更をコミット
git add .
git commit -m "fix: pre-commitフックのエラーを修正"
git push
```

### 5. Test が失敗する

**症状**: `pytest tests/`が失敗する

**原因**:
- テストが失敗している
- カバレッジが75%未満

**解決方法**:

```bash
# ローカルでテストを実行
make test
# または
pytest tests/ -v

# カバレッジ付きで実行
make test-cov
# または
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75

# 失敗したテストを確認して修正
# カバレッジが不足している場合はテストを追加
```

### 6. macOSランナーでのテストが失敗する

**症状**: macOSランナーでのテストが失敗する

**原因**:
- Homebrewがインストールされていない
- Tesseract OCRがインストールされていない

**解決方法**:

CI設定で自動的にHomebrewとTesseractをインストールするようになっていますが、問題が発生する場合は：

```yaml
# .github/workflows/ci.yml の macOS依存関係インストール部分を確認
- name: Install system dependencies (macOS)
  if: matrix.os == 'macos-latest'
  run: |
    # Homebrewがインストールされているか確認
    if ! command -v brew &> /dev/null; then
      echo "Homebrew is not installed. Installing..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install tesseract tesseract-lang
```

### 7. カバレッジが75%未満で失敗する

**症状**: `--cov-fail-under=75`でテストが失敗する

**原因**:
- テストカバレッジが75%未満

**解決方法**:

```bash
# カバレッジレポートを確認
make test-cov
# htmlcov/index.html を開いて、カバレッジが低いモジュールを確認

# テストを追加してカバレッジを上げる
# 特に以下のモジュールに注意：
# - 新しく追加したコード
# - エラーハンドリング部分
# - エッジケース
```

### 8. Security Check が警告を出す

**症状**: SafetyやBanditが警告を出す

**原因**:
- 依存関係の脆弱性
- コード内のセキュリティ問題

**解決方法**:

```bash
# Safetyチェックを実行
pip install safety
safety check --file requirements.txt

# Banditチェックを実行
pip install bandit[toml]
bandit -r src/

# 警告を確認して対応
# 重大な脆弱性の場合は依存関係を更新
```

**注意**: 現在のCI設定では、セキュリティチェックの警告でCIを止めない設定になっています（`|| true`）。重大な問題がある場合は手動で対応してください。

### 9. CI Status ジョブが失敗する

**症状**: `ci-status`ジョブが失敗する

**原因**:
- 他のジョブが失敗している

**解決方法**:

1. GitHub Actionsのログを確認
2. 失敗したジョブを特定
3. 上記の解決方法に従って修正

### 10. PRコメントが表示されない

**症状**: カバレッジコメントがPRに表示されない

**原因**:
- `GITHUB_TOKEN`の権限不足
- アクションの設定問題

**解決方法**:

1. GitHubリポジトリの設定を確認
   - Settings > Actions > General > Workflow permissions
   - "Read and write permissions"を選択

2. CI設定を確認
   ```yaml
   - name: Comment PR with coverage
     uses: py-cov-action/python-coverage-comment-action@v3
     with:
       GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
   ```

## 🔧 ローカルでの事前チェック

PRを送る前に、以下のコマンドでローカルでチェックすることを推奨します：

```bash
# 1. フォーマットチェック
make format-check

# 2. Lintチェック
make lint

# 3. 型チェック
mypy src/ --ignore-missing-imports

# 4. Pre-commitチェック
make precommit-run

# 5. テスト実行（カバレッジ付き）
make test-cov
```

すべてのチェックが通れば、CIも通過する可能性が高いです。

## 📊 CI時間の最適化

### スローテストの除外

PR時はスローテストを除外してCI時間を短縮しています：

```bash
# PR時（CI）
pytest tests/ -v -m "not slow"

# メインブランチ（全テスト）
pytest tests/ -v
```

### マーカーの使用

テストにマーカーを付けることで、スローテストを識別できます：

```python
import pytest

@pytest.mark.slow
def test_slow_integration():
    # 時間がかかるテスト
    pass

@pytest.mark.integration
def test_integration():
    # 統合テスト
    pass
```

## 🚨 緊急時の対応

### CIを一時的にスキップする

緊急時は、コミットメッセージに`[skip ci]`を追加することでCIをスキップできます：

```bash
git commit -m "fix: 緊急修正 [skip ci]"
```

**注意**: この方法は緊急時のみ使用し、通常は使用しないでください。

### 特定のジョブをスキップする

特定のジョブのみをスキップする場合は、コミットメッセージに特定のタグを追加：

```bash
git commit -m "fix: 修正 [skip format] [skip lint]"
```

## 📝 チェックリスト

PRを送る前に以下を確認：

- [ ] `make format-check`が通る
- [ ] `make lint`が通る
- [ ] `mypy src/ --ignore-missing-imports`が通る（警告は許容）
- [ ] `make precommit-run`が通る
- [ ] `make test-cov`が通る（カバレッジ75%以上）
- [ ] コミットメッセージが規約に従っている
- [ ] ドキュメントを更新（必要に応じて）

## 🔗 関連ドキュメント

- CI/CD改善提案: `docs/ci_cd_improvements.md`
- CI/CDサマリー: `docs/ci_cd_summary.md`
- CI設定: `.github/workflows/ci.yml`
- Pre-commit設定: `.pre-commit-config.yaml`
