# CI/CD と運用サマリー

このドキュメントは、CI/CD・ローカル検証・トラブルシューティングに関する要点のみをまとめた運用ガイドです。詳細な経緯やログは `docs/archive/` に退避しています。

---

## パイプライン概要

- **テスト**: `pytest -v --cov=src`（カバレッジ 80% 以上が目標）
- **Lint**: `flake8 src/ tests/` と `mypy src/`
- **フォーマット**: `black src/ tests/` と `isort src/ tests/`
- **Makefile**: `make test`, `make lint`, `make format`, `make run`
- **CI (想定)**: GitHub Actions の `python -m pip install -r requirements.txt` → lint → test の直列実行

---

## ローカル検証チェックリスト

1. `pip install -r requirements.txt`
2. `make lint` （静的解析を通過させる）
3. `make test` （ユニットテスト + カバレッジ計測）
4. 主要スクリプトのスポットチェック
   - `scripts/evaluate_reprojection_error.py`
   - `scripts/evaluate_mot_metrics.py`
   - `scripts/visualize_features.py`
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
  - キャッシュ削除後に再実行（GitHub Actions なら `actions/cache` のキー変更）
- **出力ディレクトリの肥大化**
  - `docs/guides/output_cleanup.md` のクリーンアップ手順を適用
  - `tools/cleanup_output.py` で不要セッションを削除

詳細なベストプラクティスと過去の検証ログ:
`docs/archive/ci_cd_improvements.md`, `docs/archive/ci_cd_summary.md`, `docs/archive/ci_troubleshooting.md`

---

## 運用Tips

- Pull Request は `feature/*`, `fix/*` ブランチで作成し、`develop` へマージ
- コミット前に `pre-commit` を実行する場合は `.cursor/rules/config-management.mdc` に従う
- 出力先セッションを共有する際は `output/latest/metadata.json` を添付するとレビューが容易
