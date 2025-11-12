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

.PHONY: run-eval
run-eval: ## 評価モードで実行（Ground Truthとの比較）
	@echo "=========================================="
	@echo "ワークフロー実行: 評価モード"
	@echo "=========================================="
	$(PYTHON) main.py --config $(CONFIG) --evaluate

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
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
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

.PHONY: setup
setup: setup-venv setup-deps setup-check ## 一括セットアップ（仮想環境作成 + 依存関係インストール + 確認）
	@echo "=========================================="
	@echo "✓ セットアップが完了しました！"
	@echo "=========================================="
	@echo ""
	@echo "次のステップ:"
	@echo "  1. 仮想環境を有効化: source venv/bin/activate"
	@echo "  2. システム依存関係をインストール: make setup-system"
	@echo "  3. 実行: make run"
	@echo ""

.PHONY: setup-venv
setup-venv: ## 仮想環境を作成（既に存在する場合はスキップ）
	@echo "=========================================="
	@echo "仮想環境を作成中..."
	@echo "=========================================="
	@if [ -d venv ]; then \
		echo "✓ 仮想環境は既に存在します（スキップ）"; \
	else \
		echo "仮想環境を作成中..."; \
		$(PYTHON) -m venv venv; \
		echo "✓ 仮想環境を作成しました"; \
	fi
	@echo ""
	@echo "仮想環境を有効化するには:"
	@echo "  source venv/bin/activate  # macOS/Linux"
	@echo "  venv\\Scripts\\activate     # Windows"

.PHONY: setup-deps
setup-deps: ## Python依存関係をインストール（pip upgrade + requirements.txt）
	@echo "=========================================="
	@echo "Python依存関係をインストール中..."
	@echo "=========================================="
	@echo "pipをアップグレード中..."
	@$(PYTHON) -m pip install --upgrade pip
	@echo "依存関係をインストール中..."
	@echo "（初回インストールには時間がかかります。特にPaddleOCR、EasyOCRはモデルをダウンロードします）"
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Python依存関係のインストールが完了しました"

.PHONY: setup-system
setup-system: ## システム依存関係のインストール確認とガイド表示
	@echo "=========================================="
	@echo "システム依存関係の確認"
	@echo "=========================================="
	@echo ""
	@echo "【Tesseract OCR】"
	@if command -v tesseract >/dev/null 2>&1; then \
		echo "✓ Tesseract OCR がインストールされています"; \
		tesseract --version | head -1; \
	else \
		echo "❌ Tesseract OCR がインストールされていません"; \
		echo ""; \
		echo "インストール方法:"; \
		echo "  macOS:    brew install tesseract tesseract-lang"; \
		echo "  Ubuntu:   sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-jpn"; \
		echo "  Debian:   sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-jpn"; \
	fi
	@echo ""
	@echo "【PaddleOCR / EasyOCR】"
	@echo "  Pythonパッケージとしてインストール済み（requirements.txtに含まれています）"
	@echo "  初回実行時にモデルを自動ダウンロードします"
	@echo ""

.PHONY: setup-check
setup-check: ## インストール状況を確認（依存関係確認スクリプト実行）
	@echo "=========================================="
	@echo "インストール状況を確認中..."
	@echo "=========================================="
	@if [ -f scripts/check_dependencies.py ]; then \
		$(PYTHON) scripts/check_dependencies.py; \
	else \
		echo "⚠ 依存関係確認スクリプトが見つかりません"; \
		echo "手動で確認してください: python -c \"import torch; print('PyTorch:', torch.__version__)\""; \
	fi

.PHONY: install
install: setup-deps ## 依存関係をインストール（setup-depsのエイリアス）
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
	@echo "  make setup                  # 一括セットアップ（推奨）"
	@echo "  make setup-venv             # 仮想環境作成"
	@echo "  make setup-deps             # 依存関係インストール"
	@echo "  make setup-system           # システム依存関係確認"
	@echo "  make setup-check            # インストール確認"
	@echo ""
	@echo "  make run                    # 通常実行"
	@echo "  make run-timestamps         # タイムスタンプOCRのみ実行"
	@echo "  make clean                  # outputクリーンアップ"
	@echo "  make test                   # テスト実行"
	@echo "  make lint                   # Lintチェック（ruff + mypy）"
	@echo "  make format                 # コードフォーマット（ruff）"
	@echo "  make format-check           # フォーマットチェック（変更なし）"
	@echo "  make precommit-install      # Pre-commitフレームワークのセットアップ"
	@echo "  make precommit-run          # Pre-commitフックを手動実行"
	@echo ""

# ========================================
# その他の便利コマンド
# ========================================

.PHONY: lint
lint: ## Lintチェック（ruff + mypy）
	@echo "=========================================="
	@echo "Lintチェック実行中..."
	@echo "=========================================="
	@if command -v ruff >/dev/null 2>&1; then \
		echo "✓ ruffチェック中..."; \
		ruff check .; \
	else \
		echo "⚠ ruffがインストールされていません"; \
		echo "  インストール: pip install ruff"; \
		exit 1; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "✓ mypyチェック中..."; \
		mypy src/ --ignore-missing-imports || true; \
	else \
		echo "⚠ mypyがインストールされていません（スキップ）"; \
	fi
	@echo "✓ Lintチェックが完了しました"

.PHONY: format
format: ## コードフォーマット（ruff format + ruff check --fix）
	@echo "=========================================="
	@echo "コードフォーマット実行中..."
	@echo "=========================================="
	@if command -v ruff >/dev/null 2>&1; then \
		echo "✓ ruff format実行中..."; \
		ruff format .; \
		echo "✓ ruff check --fix実行中..."; \
		ruff check . --fix; \
	else \
		echo "⚠ ruffがインストールされていません"; \
		echo "  インストール: pip install ruff"; \
		exit 1; \
	fi
	@echo "✓ フォーマットが完了しました"

.PHONY: format-check
format-check: ## フォーマットチェック（変更なし）
	@echo "=========================================="
	@echo "フォーマットチェック実行中..."
	@echo "=========================================="
	@if command -v ruff >/dev/null 2>&1; then \
		echo "✓ ruff format --check実行中..."; \
		ruff format --check .; \
	else \
		echo "⚠ ruffがインストールされていません"; \
		echo "  インストール: pip install ruff"; \
		exit 1; \
	fi
	@echo "✓ フォーマットチェックが完了しました"

.PHONY: precommit-install
precommit-install: ## Pre-commitフレームワークのGitフックをインストール
	@echo "=========================================="
	@echo "Pre-commitフレームワークをセットアップ中..."
	@echo "=========================================="
	@if [ ! -d .git ]; then \
		echo "エラー: .git ディレクトリが見つかりません。Gitリポジトリで実行してください。"; \
		exit 1; \
	fi
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "✓ pre-commitをインストール中..."; \
		pre-commit install; \
		pre-commit install --hook-type pre-push; \
		echo ""; \
		echo "✓ Pre-commitフレームワークのセットアップが完了しました"; \
		echo "  - commit前に自動的にフックが実行されます"; \
		echo "  - push前に自動的にテストが実行されます"; \
		echo ""; \
		echo "手動実行: pre-commit run --all-files"; \
	else \
		echo "⚠ pre-commitがインストールされていません"; \
		echo "  インストール: pip install pre-commit"; \
		exit 1; \
	fi

.PHONY: precommit-update
precommit-update: ## Pre-commitフレームワークのフックを更新
	@echo "=========================================="
	@echo "Pre-commitフレームワークを更新中..."
	@echo "=========================================="
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit autoupdate; \
		echo "✓ Pre-commitフレームワークを更新しました"; \
	else \
		echo "⚠ pre-commitがインストールされていません"; \
		echo "  インストール: pip install pre-commit"; \
		exit 1; \
	fi

.PHONY: precommit-run
precommit-run: ## Pre-commitフックを手動実行（全ファイル）
	@echo "=========================================="
	@echo "Pre-commitフックを手動実行中..."
	@echo "=========================================="
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "⚠ pre-commitがインストールされていません"; \
		echo "  インストール: pip install pre-commit"; \
		exit 1; \
	fi
