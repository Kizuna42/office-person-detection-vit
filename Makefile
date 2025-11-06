# オフィス人物検出システム - Makefile
# Office Person Detection System - Makefile

# Python実行コマンド（venvが存在する場合は優先、python3の存在確認を先に実行）
PYTHON := $(shell if command -v python3 >/dev/null 2>&1; then \
		if [ -f venv/bin/python ]; then \
			echo "venv/bin/python"; \
		else \
			echo "python3"; \
		fi; \
	else \
		if [ -f venv/bin/python ]; then \
			echo "venv/bin/python"; \
		else \
			echo "python"; \
		fi; \
	fi)

# 設定ファイル（デフォルト）
CONFIG := config.yaml

# デフォルトターゲット
.DEFAULT_GOAL := help

# ========================================
# ワークフロー実行コマンド
# ========================================

.PHONY: run
run: ## 通常実行（メインパイプライン）
	@echo "=========================================="
	@echo "ワークフロー実行: 通常モード"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG)

.PHONY: run-debug
run-debug: ## デバッグモードで実行（詳細ログ、中間結果出力）
	@echo "=========================================="
	@echo "ワークフロー実行: デバッグモード"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG) --debug

.PHONY: run-eval
run-eval: ## 評価モードで実行（Ground Truthとの比較）
	@echo "=========================================="
	@echo "ワークフロー実行: 評価モード"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG) --evaluate

.PHONY: run-time
run-time: ## 時刻指定で実行（例: make run-time START_TIME="2025/08/26 16:05:00" END_TIME="2025/08/26 16:15:00"）
	@# 注意: 日付/時刻のパースは main.py 側で堅牢に処理されることを想定
	@# シェル依存の処理を避けるため、引数はそのまま main.py に渡す
	@if [ -z "$(START_TIME)" ] || [ -z "$(END_TIME)" ]; then \
		echo "エラー: START_TIMEとEND_TIMEを指定してください"; \
		echo "例: make run-time START_TIME=\"2025/08/26 16:05:00\" END_TIME=\"2025/08/26 16:15:00\""; \
		echo "または: make run-time START_TIME=\"16:05:00\" END_TIME=\"16:15:00\""; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "ワークフロー実行: 時刻指定モード"
	@echo "開始時刻: $(START_TIME)"
	@echo "終了時刻: $(END_TIME)"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG) --start-time "$(START_TIME)" --end-time "$(END_TIME)"

.PHONY: run-timestamps
run-timestamps: ## タイムスタンプOCRのみ実行（5分刻みフレーム抽出+OCR、CSV+オーバーレイ画像出力）
	@echo "=========================================="
	@echo "ワークフロー実行: タイムスタンプOCRモード"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG) --timestamps-only --debug

# ========================================
# クリーンアップコマンド
# ========================================

.PHONY: clean
clean: ## outputディレクトリ内の生成ファイルを削除（labels/result_fixed.jsonとcalibration/は保持）
	@echo "=========================================="
	@echo "outputディレクトリをクリーンアップ中..."
	@echo "=========================================="
	@if [ -d output ]; then \
		find output -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.csv" -o -name "*.json" -o -name "*.log" \) \
			! -path "*/labels/result_fixed.json" \
			! -path "*/calibration/*" \
			-exec rm -f {} + 2>/dev/null || true; \
		find output -type d -empty -delete 2>/dev/null || true; \
		echo "✓ outputディレクトリをクリーンアップしました"; \
	else \
		echo "✓ outputディレクトリが存在しません"; \
	fi

.PHONY: clean-all
clean-all: clean clean-cache ## output + Pythonキャッシュを削除

.PHONY: clean-cache
clean-cache: ## Pythonキャッシュ（__pycache__、*.pyc）を削除（注意: CIや他ツールのキャッシュも削除対象）
	@echo "=========================================="
	@echo "Pythonキャッシュをクリーンアップ中..."
	@echo "注意: CIや他ツールのキャッシュも削除対象です"
	@echo "=========================================="
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Pythonキャッシュをクリーンアップしました"

# ========================================
# テストコマンド
# ========================================

.PHONY: test
test: ## 全テストを実行
	@echo "=========================================="
	@echo "テスト実行中..."
	@echo "=========================================="
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-cov
test-cov: ## カバレッジ付きでテスト実行
	@echo "=========================================="
	@echo "カバレッジ付きテスト実行中..."
	@echo "=========================================="
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term -v
	@echo ""
	@echo "カバレッジレポート: htmlcov/index.html を開いてください"

.PHONY: test-verbose
test-verbose: ## 詳細出力付きでテスト実行
	@echo "=========================================="
	@echo "詳細出力付きテスト実行中..."
	@echo "=========================================="
	$(PYTHON) -m pytest tests/ -vv -s

# ========================================
# セットアップコマンド
# ========================================

.PHONY: install
install: ## 依存関係をインストール
	@echo "=========================================="
	@echo "依存関係をインストール中..."
	@echo "=========================================="
	pip install -r requirements.txt
	@echo "✓ 依存関係のインストールが完了しました"

.PHONY: help
help: ## 利用可能なコマンド一覧を表示
	@echo "=========================================="
	@echo "オフィス人物検出システム - Makefile"
	@echo "=========================================="
	@echo ""
	@echo "利用可能なコマンド:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / { \
		command = $$1; \
		description = $$2; \
		gsub(/^[ \t]+|[ \t]+$$/, "", command); \
		gsub(/^[ \t]+|[ \t]+$$/, "", description); \
		printf "  \033[36m%-20s\033[0m %s\n", command, description \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "例:"
	@echo "  make run                    # 通常実行"
	@echo "  make run-debug              # デバッグモード"
	@echo "  make run-timestamps         # タイムスタンプOCRのみ実行"
	@echo "  make run-time START_TIME=\"2025/08/26 16:05:00\" END_TIME=\"2025/08/26 16:15:00\"  # 時刻指定（日時形式）"
	@echo "  make run-time START_TIME=\"16:05:00\" END_TIME=\"16:15:00\"  # 時刻指定（時刻のみ）"
	@echo "  make clean                  # outputクリーンアップ"
	@echo "  make test                   # テスト実行"
	@echo ""

# ========================================
# その他の便利コマンド
# ========================================

.PHONY: lint
lint: ## Lintチェック（flake8、black、mypy）
	@echo "=========================================="
	@echo "Lintチェック実行中..."
	@echo "=========================================="
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "✓ flake8チェック中..."; \
		flake8 src/ tests/ --max-line-length=120 --ignore=E203 --exclude=venv,__pycache__ || true; \
	else \
		echo "⚠ flake8がインストールされていません（スキップ）"; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		echo "✓ blackチェック中..."; \
		black --check src/ tests/ || true; \
	else \
		echo "⚠ blackがインストールされていません（スキップ）"; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "✓ mypyチェック中..."; \
		mypy src/ --ignore-missing-imports || true; \
	else \
		echo "⚠ mypyがインストールされていません（スキップ）"; \
	fi

.PHONY: format
format: ## コードフォーマット（black、isort）
	@echo "=========================================="
	@echo "コードフォーマット実行中..."
	@echo "=========================================="
	@if command -v black >/dev/null 2>&1; then \
		echo "✓ blackでフォーマット中..."; \
		black src/ tests/; \
	else \
		echo "⚠ blackがインストールされていません（スキップ）"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		echo "✓ isortでインポート整理中..."; \
		isort src/ tests/; \
	else \
		echo "⚠ isortがインストールされていません（スキップ）"; \
	fi
	@echo "✓ フォーマットが完了しました"

.PHONY: pre-commit
pre-commit: format lint  ## commit前に実行するチェック（format + lint）
	@echo "✓ pre-commit checks passed"

.PHONY: pre-push
pre-push: format lint test  ## push前に実行する総合チェック（format + lint + test）
	@echo "✓ pre-push checks passed"

.PHONY: precommit-install
precommit-install: ## Gitフックをインストール（pre-commitとpre-push）
	@echo "=========================================="
	@echo "Gitフックをインストール中..."
	@echo "=========================================="
	@if [ ! -d .git/hooks ]; then \
		echo "エラー: .git/hooks ディレクトリが見つかりません。Gitリポジトリで実行してください。"; \
		exit 1; \
	fi
	@echo '#!/bin/bash' > .git/hooks/pre-commit
	@echo 'set -e' >> .git/hooks/pre-commit
	@echo 'echo "Running pre-commit checks..."' >> .git/hooks/pre-commit
	@echo 'make pre-commit' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo '#!/bin/bash' > .git/hooks/pre-push
	@echo 'set -e' >> .git/hooks/pre-push
	@echo 'echo "Running pre-push checks..."' >> .git/hooks/pre-push
	@echo 'make pre-push' >> .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "✓ Gitフックをインストールしました"
	@echo "  - commit前に自動的に make pre-commit が実行されます（format + lint）"
	@echo "  - push前に自動的に make pre-push が実行されます（format + lint + test）"

