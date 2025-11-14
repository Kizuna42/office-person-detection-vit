# オフィス人物検出システム - Makefile
# Office Person Detection System - Makefile

# ========================================
# 変数定義
# ========================================

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

# 仮想環境設定
VENV_DIR := venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PY := $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,$(PYTHON))
VENV_ACTIVATE := $(VENV_BIN)/activate

# 実行モード・設定
TEST_MODE ?= default
TEST_PARALLEL ?= auto
CONFIG := config.yaml
REQUIREMENTS := requirements.txt

# ディレクトリ・ファイルパス
OUTPUT_DIR := output
TESTS_DIR := tests
SRC_DIR := src
SCRIPTS_DIR := scripts

# ツールコマンド
RUFF := $(shell command -v ruff 2>/dev/null || echo "")
MYPY := $(shell command -v mypy 2>/dev/null || echo "")

# カラー出力（ANSIエスケープシーケンス）
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_CYAN := \033[36m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_RED := \033[31m

# デフォルトターゲット
.DEFAULT_GOAL := help

# ========================================
# ユーティリティ関数
# ========================================

# セクション区切りを表示
define print_section
	echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"
	echo "$(COLOR_BOLD)$(1)$(COLOR_RESET)"
	echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"
endef

# 成功メッセージを表示
define print_success
	echo "$(COLOR_GREEN)✓$(COLOR_RESET) $(1)"
endef

# 警告メッセージを表示
define print_warning
	echo "$(COLOR_YELLOW)⚠$(COLOR_RESET) $(1)"
endef

# エラーメッセージを表示
define print_error
	echo "$(COLOR_RED)❌$(COLOR_RESET) $(1)"
endef

# コマンドの存在確認
define check_command
	@if ! command -v $(1) >/dev/null 2>&1; then \
		$(call print_error,"$(1)がインストールされていません"); \
		echo "  インストール: $(2)"; \
		exit 1; \
	fi
endef

# ========================================
# ワークフロー実行コマンド
# ========================================

.PHONY: run
run: ## 通常実行（メインパイプライン）
	$(call print_section,"ワークフロー実行: 通常モード")
	@$(PYTHON) main.py --config $(CONFIG)

# ========================================
# クリーンアップコマンド
# ========================================

.PHONY: clean
clean: ## outputディレクトリ内の生成ファイルを削除（labels/result_fixed.json、calibration/、shared/は保持）
	@$(call print_section,"outputディレクトリをクリーンアップ中...")
	@if [ -d $(OUTPUT_DIR) ]; then \
		echo "生成ファイルを削除中..."; \
		find $(OUTPUT_DIR) -type f \( \
			-name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o \
			-name "*.bmp" -o -name "*.tiff" -o -name "*.webp" -o \
			-name "*.mov" -o -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o \
			-name "*.webm" -o -name "*.flv" -o -name "*.wmv" -o \
			-name "*.csv" -o -name "*.json" -o -name "*.log" -o -name "*.md" -o \
			-name "*.tmp" -o -name "*.temp" -o -name "*.swp" -o -name "*.swo" -o \
			-name "*~" -o -name "._*" \
		\) \
			! -path "*/labels/result_fixed.json" \
			! -path "*/calibration/*" \
			! -path "*/shared/*" \
			-exec rm -f {} + 2>/dev/null || true; \
		echo "シンボリックリンクを削除中..."; \
		find $(OUTPUT_DIR) -type l -name "latest" -delete 2>/dev/null || true; \
		echo "セッションディレクトリを削除中..."; \
		if [ -d $(OUTPUT_DIR)/sessions ]; then \
			find $(OUTPUT_DIR)/sessions -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true; \
		fi; \
		echo "空ディレクトリを削除中..."; \
		find $(OUTPUT_DIR) -type d -empty -delete 2>/dev/null || true; \
		$(call print_success,"outputディレクトリをクリーンアップしました"); \
	else \
		$(call print_success,"outputディレクトリが存在しません"); \
	fi

# ========================================
# テストコマンド
# ========================================

.PHONY: test
test: ## テストを実行（TEST_MODE=coverage|verbose|fast、TEST_PARALLEL=auto|no|N を指定可能）
	@set -e; \
	PARALLEL_OPTS=""; \
	if [ "$(TEST_PARALLEL)" != "no" ]; then \
		if $(VENV_PY) -c "import pytest_xdist" 2>/dev/null || $(PYTHON) -c "import pytest_xdist" 2>/dev/null; then \
			if [ "$(TEST_PARALLEL)" = "auto" ]; then \
				PARALLEL_OPTS="-n auto"; \
			else \
				PARALLEL_OPTS="-n $(TEST_PARALLEL)"; \
			fi; \
		fi; \
	fi; \
	if [ "$(TEST_MODE)" = "coverage" ]; then \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)カバレッジ付きテストを実行中...$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term -v $$PARALLEL_OPTS; \
	elif [ "$(TEST_MODE)" = "verbose" ]; then \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)詳細出力付きテストを実行中...$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -vv -s $$PARALLEL_OPTS; \
	elif [ "$(TEST_MODE)" = "fast" ]; then \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)高速テストを実行中（並列実行）...$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -v -m "not slow" $$PARALLEL_OPTS; \
	else \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)テストを実行中...$(COLOR_RESET)"; \
		echo "$(COLOR_BOLD)==========================================$(COLOR_RESET)"; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -v $$PARALLEL_OPTS; \
	fi

# ========================================
# セットアップコマンド
# ========================================

.PHONY: setup
setup: ## 開発環境を一括初期化（仮想環境 + 依存関係 + 動作確認）
	$(call print_section,"開発環境セットアップを開始します")
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "仮想環境を作成中..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(call print_success,"仮想環境を作成しました"); \
	else \
		$(call print_success,"仮想環境は既に存在します"); \
	fi
	@echo ""
	@SETUP_PY="$(VENV_BIN)/python"; \
	if [ ! -x "$$SETUP_PY" ]; then \
		$(call print_warning,"仮想環境が見つからないためホストPythonを使用します: $(PYTHON)"); \
		SETUP_PY="$(PYTHON)"; \
	fi; \
	echo "pipの状態を確認中..."; \
	if ! "$$SETUP_PY" -m pip --version >/dev/null 2>&1; then \
		$(call print_warning,"pipが破損している可能性があります。仮想環境を再作成します..."); \
		rm -rf $(VENV_DIR); \
		$(PYTHON) -m venv $(VENV_DIR); \
		SETUP_PY="$(VENV_BIN)/python"; \
		$(call print_success,"仮想環境を再作成しました"); \
	fi; \
	echo "pipをアップグレード中..."; \
	if ! "$$SETUP_PY" -m pip install --upgrade pip >/dev/null 2>&1; then \
		$(call print_warning,"pipのアップグレードに失敗しました。get-pip.pyで再インストールを試みます..."); \
		curl -sSL https://bootstrap.pypa.io/get-pip.py | "$$SETUP_PY" || \
		($(call print_error,"pipの再インストールに失敗しました。仮想環境を再作成します..."); \
		 rm -rf $(VENV_DIR); \
		 $(PYTHON) -m venv $(VENV_DIR); \
		 SETUP_PY="$(VENV_BIN)/python"; \
		 "$$SETUP_PY" -m pip install --upgrade pip); \
	fi; \
	echo "依存関係をインストール中..."; \
	echo "（初回はモデルダウンロードのため時間がかかる場合があります）"; \
	"$$SETUP_PY" -m pip install -r $(REQUIREMENTS); \
	echo ""; \
	echo "システム依存関係を確認しています..."; \
	if command -v tesseract >/dev/null 2>&1; then \
		$(call print_success,"Tesseract OCR が利用可能です: $$(tesseract --version | head -1)"); \
	else \
		$(call print_error,"Tesseract OCR が見つかりません"); \
		echo "   例: brew install tesseract tesseract-lang"; \
	fi; \
	echo ""; \
	if [ -f $(SCRIPTS_DIR)/check_dependencies.py ]; then \
		echo "Python依存関係を検証しています..."; \
		"$$SETUP_PY" $(SCRIPTS_DIR)/check_dependencies.py; \
	else \
		$(call print_warning,"$(SCRIPTS_DIR)/check_dependencies.py が見つからないためスキップしました"); \
	fi; \
	echo ""; \
	$(call print_section,"セットアップが完了しました！"); \
	echo ""; \
	echo "次のステップ:"; \
	echo "  1. 仮想環境を有効化: source $(VENV_ACTIVATE)"; \
	echo "  2. 実行: make run"; \
	echo ""

.PHONY: help
help: ## 利用可能なコマンド一覧を表示
	$(call print_section,"オフィス人物検出システム - Makefile")
	@echo ""
	@echo "利用可能なコマンド:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / { \
		command = $$1; \
		description = $$2; \
		gsub(/^[ \t]+|[ \t]+$$/, "", command); \
		gsub(/^[ \t]+|[ \t]+$$/, "", description); \
		printf "  $(COLOR_CYAN)%-25s$(COLOR_RESET) %s\n", command, description \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "使用例:"
	@echo "  $(COLOR_CYAN)make setup$(COLOR_RESET)                  # 開発環境の初期化"
	@echo "  $(COLOR_CYAN)make test$(COLOR_RESET)                   # テスト実行"
	@echo "  $(COLOR_CYAN)make test TEST_MODE=coverage$(COLOR_RESET)  # カバレッジ付きテスト"
	@echo "  $(COLOR_CYAN)make run$(COLOR_RESET)                    # 通常実行"
	@echo "  $(COLOR_CYAN)make clean$(COLOR_RESET)                  # outputクリーンアップ"
	@echo "  $(COLOR_CYAN)make lint$(COLOR_RESET)                   # Lintチェック"
	@echo "  $(COLOR_CYAN)make format$(COLOR_RESET)                # コードフォーマット"
	@echo ""

# ========================================
# コード品質コマンド
# ========================================

.PHONY: lint
lint: ## Lintチェック（ruff + mypy）
	$(call print_section,"Lintチェック実行中...")
	@if [ -z "$(RUFF)" ]; then \
		$(call print_error,"ruffがインストールされていません"); \
		echo "  インストール: pip install ruff"; \
		exit 1; \
	fi
	@$(call print_success,"ruffチェック中...")
	@ruff check .
	@if [ -n "$(MYPY)" ]; then \
		$(call print_success,"mypyチェック中..."); \
		mypy $(SRC_DIR)/ --ignore-missing-imports || true; \
	else \
		$(call print_warning,"mypyがインストールされていません（スキップ）"); \
	fi
	@$(call print_success,"Lintチェックが完了しました")

.PHONY: format
format: ## コードフォーマット（ruff format + ruff check --fix）
	$(call print_section,"コードフォーマット実行中...")
	@if [ -z "$(RUFF)" ]; then \
		$(call print_error,"ruffがインストールされていません"); \
		echo "  インストール: pip install ruff"; \
		exit 1; \
	fi
	@$(call print_success,"ruff format実行中...")
	@ruff format .
	@$(call print_success,"ruff check --fix実行中...")
	@ruff check . --fix
	@$(call print_success,"フォーマットが完了しました")
