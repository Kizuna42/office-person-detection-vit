# CI/CD と運用サマリー

このドキュメントは、CI/CD・ローカル検証・トラブルシューティングに関する要点のみをまとめた運用ガイドです。詳細な経緯やログは `docs/archive/` に退避しています。

---

## パイプライン概要

### GitHub Actions CI

3つの並列ジョブで構成:

1. **lint**: Ruff による静的解析とフォーマットチェック
2. **type-check**: mypy による型チェック
3. **test**: pytest + カバレッジ（Python 3.10, 3.11 マトリックス）

```yaml
# .github/workflows/ci.yml
jobs:
  lint:        # ruff check + ruff format --check
  type-check:  # mypy src/
  test:        # pytest --cov=src (Python 3.10, 3.11)
```

### ローカルツール

- **Lint**: `ruff check .` + `mypy src/`
- **フォーマット**: `ruff format .`
- **テスト**: `pytest --cov=src`（カバレッジ 70% 以上を要求）
- **Makefile**: `make lint`, `make format`, `make format-check`, `make test`, `make test-cov`

### pre-commit フック

コミット時:
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-json, check-toml
- ruff check --fix, ruff format
- mypy src/

プッシュ時:
- pytest

---

## ローカル検証チェックリスト

1. `pip install -r requirements.txt`
2. `make lint` （静的解析を通過させる）
3. `make test` または `make test-cov` （ユニットテスト + カバレッジ計測）
4. `make format-check` （フォーマットチェック）
5. セッション管理有効時の出力確認
   - 最新結果は `output/latest/` シンボリックリンクで参照可能

---

## 代表的なトラブルと対処

- **OCR 推論が失敗する / 精度が低い**
  - ROI 設定 (`config.yaml` → `timestamp.extraction.roi`) を見直す
  - `docs/references/frame_sampling.md` の調整例を参照
- **MPS デバイスで PyTorch がクラッシュする**
  - `src/utils/torch_utils.setup_mps_compatibility()` が呼ばれているか確認
  - 環境変数 `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定して CPU フォールバック
- **CI で依存関係が解決できない**
  - `requirements.txt` を更新し、`pip install --upgrade pip` を実行
  - pip キャッシュは `actions/setup-python` の `cache: "pip"` で自動管理
- **出力ディレクトリの肥大化**
  - `docs/guides/output_cleanup.md` のクリーンアップ手順を適用
  - `make clean` で不要セッションを削除

詳細なベストプラクティスと過去の検証ログ:
`docs/archive/ci_cd_improvements.md`, `docs/archive/ci_cd_summary.md`, `docs/archive/ci_troubleshooting.md`

---

## 運用Tips

- Pull Request は `feature/*`, `fix/*` ブランチで作成し、`develop` へマージ
- コミット前に `pre-commit` を実行: `make precommit-run`
- 開発環境セットアップ: `make setup-dev` (pre-commit フック含む)
- 出力先セッションを共有する際は `output/latest/metadata.json` を添付するとレビューが容易
