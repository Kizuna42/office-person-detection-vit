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

# カラー出力（ANSIエスケープシーケンス） - ターミナル対応チェック付き
# NO_COLOR環境変数が設定されている、またはターミナルがカラー非対応の場合は無効化
NO_COLOR_ENV := $(shell echo $$NO_COLOR)
TERM_SUPPORT := $(shell if [ -t 1 ] && [ -z "$(NO_COLOR_ENV)" ]; then echo "yes"; else echo "no"; fi)

ifeq ($(TERM_SUPPORT),yes)
	COLOR_RESET := \033[0m
	COLOR_BOLD := \033[1m
	COLOR_DIM := \033[2m
	COLOR_CYAN := \033[36m
	COLOR_GREEN := \033[32m
	COLOR_YELLOW := \033[33m
	COLOR_RED := \033[31m
	COLOR_BLUE := \033[34m
	COLOR_MAGENTA := \033[35m
	ICON_SUCCESS := ✓
	ICON_WARNING := ⚠
	ICON_ERROR := ✗
	ICON_INFO := ℹ
	ICON_ARROW := →
else
	COLOR_RESET :=
	COLOR_BOLD :=
	COLOR_DIM :=
	COLOR_CYAN :=
	COLOR_GREEN :=
	COLOR_YELLOW :=
	COLOR_RED :=
	COLOR_BLUE :=
	COLOR_MAGENTA :=
	ICON_SUCCESS := [OK]
	ICON_WARNING := [WARN]
	ICON_ERROR := [ERROR]
	ICON_INFO := [INFO]
	ICON_ARROW := ->
endif

# デフォルトターゲット
.DEFAULT_GOAL := help

# ========================================
# ユーティリティ関数
# ========================================

# セクション区切りを表示（改善版）
define print_section
	echo ""; \
	echo "$(COLOR_BOLD)$(COLOR_CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(COLOR_RESET)"; \
	echo "$(COLOR_BOLD)$(COLOR_CYAN)  $(1)$(COLOR_RESET)"; \
	echo "$(COLOR_BOLD)$(COLOR_CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(COLOR_RESET)"; \
	echo ""
endef

# サブセクション表示
define print_subsection
	echo "$(COLOR_BOLD)$(COLOR_BLUE)$(ICON_ARROW)$(COLOR_RESET) $(COLOR_BOLD)$(1)$(COLOR_RESET)"
endef

# 成功メッセージを表示（改善版）
define print_success
	echo "$(COLOR_GREEN)$(ICON_SUCCESS)$(COLOR_RESET) $(COLOR_GREEN)$(1)$(COLOR_RESET)"
endef

# 警告メッセージを表示（改善版）
define print_warning
	echo "$(COLOR_YELLOW)$(ICON_WARNING)$(COLOR_RESET) $(COLOR_YELLOW)$(1)$(COLOR_RESET)" >&2
endef

# エラーメッセージを表示（改善版）
define print_error
	echo "$(COLOR_RED)$(ICON_ERROR)$(COLOR_RESET) $(COLOR_BOLD)$(COLOR_RED)$(1)$(COLOR_RESET)" >&2
endef

# 情報メッセージを表示
define print_info
	echo "$(COLOR_CYAN)$(ICON_INFO)$(COLOR_RESET) $(COLOR_DIM)$(1)$(COLOR_RESET)"
endef

# 進捗メッセージを表示
define print_progress
	echo "$(COLOR_BLUE)$(ICON_ARROW)$(COLOR_RESET) $(COLOR_DIM)$(1)$(COLOR_RESET)..."
endef

# ステップ表示（番号付き）
define print_step
	echo ""; \
	echo "$(COLOR_BOLD)$(COLOR_MAGENTA)[$(1)/$(2)]$(COLOR_RESET) $(COLOR_BOLD)$(3)$(COLOR_RESET)"
endef

# コマンドの存在確認
define check_command
	@if ! command -v $(1) >/dev/null 2>&1; then \
		$(call print_error,"$(1)がインストールされていません"); \
		echo "  $(COLOR_DIM)インストール: $(2)$(COLOR_RESET)"; \
		exit 1; \
	fi
endef

# ========================================
# ワークフロー実行コマンド
# ========================================

.PHONY: run
run: ## 通常実行（メインパイプライン）
	$(call print_section,"ワークフロー実行: 通常モード")
	@$(call print_info,"設定ファイル: $(CONFIG)")
	@$(call print_info,"Python: $(PYTHON)")
	@echo ""
	@$(call print_progress,"パイプラインを実行中")
	@$(PYTHON) main.py --config $(CONFIG) || \
		($(call print_error,"パイプライン実行に失敗しました"); exit 1)
	@echo ""
	@$(call print_success,"パイプライン実行が完了しました")

.PHONY: baseline
baseline: ## ベースライン実行と評価を連鎖実行（run_baseline.py + evaluate_baseline.py）
	@set -e; \
	$(call print_section,"ベースライン実行と評価"); \
	$(call print_info,"設定ファイル: $(CONFIG)"); \
	$(if $(TAG),$(call print_info,"タグ: $(TAG)")); \
	$(if $(GT),$(call print_info,"Ground Truth: $(GT)")); \
	$(if $(POINTS),$(call print_info,"対応点: $(POINTS)")); \
	echo ""; \
	$(call print_step,1,3,"パイプライン実行"); \
	$(call print_progress,"パイプラインを実行中"); \
	$(PYTHON) $(SCRIPTS_DIR)/run_baseline.py --config $(CONFIG) $(if $(TAG),--tag $(TAG)) || \
		($(call print_error,"パイプライン実行に失敗しました"); exit 1); \
	$(call print_success,"パイプライン実行が完了しました"); \
	echo ""; \
	$(call print_step,2,3,"セッションID取得"); \
	$(call print_progress,"セッションIDを取得中"); \
	SESSION_ID=""; \
	if [ -L $(OUTPUT_DIR)/latest ] && [ -e $(OUTPUT_DIR)/latest ]; then \
		SESSION_ID=$$(basename $$(readlink $(OUTPUT_DIR)/latest)) || true; \
	fi; \
	if [ -z "$$SESSION_ID" ] && [ -d $(OUTPUT_DIR)/sessions ]; then \
		SESSION_ID=$$(ls -td $(OUTPUT_DIR)/sessions/*/ 2>/dev/null | head -1 | xargs -n1 basename) || true; \
	fi; \
	if [ -z "$$SESSION_ID" ]; then \
		$(call print_error,"セッションIDを取得できませんでした"); \
		echo "  $(COLOR_DIM)output/latest シンボリックリンクまたは output/sessions/ ディレクトリを確認してください$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	echo "  $(COLOR_DIM)セッションID: $(COLOR_CYAN)$$SESSION_ID$(COLOR_RESET)"; \
	$(call print_success,"セッションIDを取得しました"); \
	echo ""; \
	$(call print_step,3,3,"評価実行"); \
	$(call print_progress,"評価を実行中"); \
	$(PYTHON) $(SCRIPTS_DIR)/evaluate_baseline.py --session $$SESSION_ID --config $(CONFIG) \
		$(if $(GT),--gt $(GT)) $(if $(POINTS),--points $(POINTS)) || \
		($(call print_error,"評価実行に失敗しました"); exit 1); \
	$(call print_success,"評価実行が完了しました"); \
	echo ""; \
	$(call print_section,"ベースライン実行と評価が完了しました"); \
	echo "$(COLOR_BOLD)セッションID:$(COLOR_RESET) $(COLOR_CYAN)$$SESSION_ID$(COLOR_RESET)"; \
	echo "$(COLOR_BOLD)評価結果:$(COLOR_RESET) $(COLOR_DIM)$(OUTPUT_DIR)/sessions/$$SESSION_ID/baseline_metrics.json$(COLOR_RESET)"; \
	echo ""

# ========================================
# クリーンアップコマンド
# ========================================

.PHONY: clean
clean: ## outputディレクトリ内の生成ファイルを削除（labels/result_fixed.json、calibration/、shared/は保持）
	@$(call print_section,"outputディレクトリをクリーンアップ中")
	@if [ -d $(OUTPUT_DIR) ]; then \
		$(call print_progress,"生成ファイルを削除中"); \
		FILE_COUNT=$$(find $(OUTPUT_DIR) -type f \( \
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
			! -path "*/shared/*" 2>/dev/null | wc -l | tr -d ' '); \
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
		if [ "$$FILE_COUNT" -gt 0 ]; then \
			echo "  $(COLOR_DIM)削除したファイル数: $$FILE_COUNT$(COLOR_RESET)"; \
		fi; \
		$(call print_progress,"シンボリックリンクを削除中"); \
		LINK_COUNT=$$(find $(OUTPUT_DIR) -type l -name "latest" 2>/dev/null | wc -l | tr -d ' '); \
		find $(OUTPUT_DIR) -type l -name "latest" -delete 2>/dev/null || true; \
		if [ "$$LINK_COUNT" -gt 0 ]; then \
			echo "  $(COLOR_DIM)削除したシンボリックリンク数: $$LINK_COUNT$(COLOR_RESET)"; \
		fi; \
		$(call print_progress,"セッションディレクトリを削除中"); \
		if [ -d $(OUTPUT_DIR)/sessions ]; then \
			SESSION_COUNT=$$(find $(OUTPUT_DIR)/sessions -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '); \
			find $(OUTPUT_DIR)/sessions -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true; \
			if [ "$$SESSION_COUNT" -gt 0 ]; then \
				echo "  $(COLOR_DIM)削除したセッション数: $$SESSION_COUNT$(COLOR_RESET)"; \
			fi; \
		fi; \
		$(call print_progress,"空ディレクトリを削除中"); \
		find $(OUTPUT_DIR) -type d -empty -delete 2>/dev/null || true; \
		echo ""; \
		$(call print_success,"outputディレクトリをクリーンアップしました"); \
	else \
		$(call print_info,"outputディレクトリが存在しません（スキップ）"); \
	fi

# ========================================
# テストコマンド
# ========================================

.PHONY: test
test: ## テストを実行（TEST_MODE=coverage|verbose|fast、TEST_PARALLEL=auto|no|N を指定可能）
	@set -e; \
	PARALLEL_OPTS=""; \
	PARALLEL_INFO=""; \
	if [ "$(TEST_PARALLEL)" != "no" ]; then \
		if $(VENV_PY) -c "import pytest_xdist" 2>/dev/null || $(PYTHON) -c "import pytest_xdist" 2>/dev/null; then \
			if [ "$(TEST_PARALLEL)" = "auto" ]; then \
				PARALLEL_OPTS="-n auto"; \
				PARALLEL_INFO="（並列: 自動）"; \
			else \
				PARALLEL_OPTS="-n $(TEST_PARALLEL)"; \
				PARALLEL_INFO="（並列: $(TEST_PARALLEL)プロセス）"; \
			fi; \
		fi; \
	fi; \
	if [ "$(TEST_MODE)" = "coverage" ]; then \
		$(call print_section,"カバレッジ付きテスト実行"); \
		$(call print_info,"テストディレクトリ: $(TESTS_DIR)")$(if $$PARALLEL_INFO,$(call print_info,"並列実行: $$PARALLEL_INFO")); \
		echo ""; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term -v $$PARALLEL_OPTS; \
	elif [ "$(TEST_MODE)" = "verbose" ]; then \
		$(call print_section,"詳細出力付きテスト実行"); \
		$(call print_info,"テストディレクトリ: $(TESTS_DIR)")$(if $$PARALLEL_INFO,$(call print_info,"並列実行: $$PARALLEL_INFO")); \
		echo ""; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -vv -s $$PARALLEL_OPTS; \
	elif [ "$(TEST_MODE)" = "fast" ]; then \
		$(call print_section,"高速テスト実行"); \
		$(call print_info,"テストディレクトリ: $(TESTS_DIR)")$(if $$PARALLEL_INFO,$(call print_info,"並列実行: $$PARALLEL_INFO")); \
		$(call print_info,"スキップ: slowマーカー付きテスト"); \
		echo ""; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -v -m "not slow" $$PARALLEL_OPTS; \
	else \
		$(call print_section,"テスト実行"); \
		$(call print_info,"テストディレクトリ: $(TESTS_DIR)")$(if $$PARALLEL_INFO,$(call print_info,"並列実行: $$PARALLEL_INFO")); \
		echo ""; \
		$(VENV_PY) -m pytest $(TESTS_DIR)/ -v $$PARALLEL_OPTS; \
	fi

# ========================================
# セットアップコマンド
# ========================================

.PHONY: setup
setup: ## 開発環境を一括初期化（仮想環境 + 依存関係 + 動作確認）
	@$(call print_section,"開発環境セットアップ")
	@$(call print_info,"Python: $(PYTHON)")
	@$(call print_info,"仮想環境: $(VENV_DIR)")
	@echo ""
	@$(call print_step,1,5,"仮想環境の確認・作成")
	@if [ ! -d $(VENV_DIR) ]; then \
		$(call print_progress,"仮想環境を作成中"); \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(call print_success,"仮想環境を作成しました"); \
	else \
		$(call print_info,"仮想環境は既に存在します（スキップ）"); \
	fi
	@echo ""
	@SETUP_PY="$(VENV_BIN)/python"; \
	if [ ! -x "$$SETUP_PY" ]; then \
		$(call print_warning,"仮想環境が見つからないためホストPythonを使用します: $(PYTHON)"); \
		SETUP_PY="$(PYTHON)"; \
	fi; \
	$(call print_step,2,5,"pipの確認・アップグレード"); \
	$(call print_progress,"pipの状態を確認中"); \
	if ! "$$SETUP_PY" -m pip --version >/dev/null 2>&1; then \
		$(call print_warning,"pipが破損している可能性があります。仮想環境を再作成します..."); \
		rm -rf $(VENV_DIR); \
		$(PYTHON) -m venv $(VENV_DIR); \
		SETUP_PY="$(VENV_BIN)/python"; \
		$(call print_success,"仮想環境を再作成しました"); \
	fi; \
	$(call print_progress,"pipをアップグレード中"); \
	if ! "$$SETUP_PY" -m pip install --upgrade pip >/dev/null 2>&1; then \
		$(call print_warning,"pipのアップグレードに失敗しました。get-pip.pyで再インストールを試みます..."); \
		curl -sSL https://bootstrap.pypa.io/get-pip.py | "$$SETUP_PY" || \
		($(call print_error,"pipの再インストールに失敗しました。仮想環境を再作成します..."); \
		 rm -rf $(VENV_DIR); \
		 $(PYTHON) -m venv $(VENV_DIR); \
		 SETUP_PY="$(VENV_BIN)/python"; \
		 "$$SETUP_PY" -m pip install --upgrade pip); \
	fi; \
	$(call print_success,"pipの準備が完了しました"); \
	echo ""; \
	$(call print_step,3,5,"依存関係のインストール"); \
	$(call print_progress,"依存関係をインストール中"); \
	$(call print_info,"初回はモデルダウンロードのため時間がかかる場合があります"); \
	"$$SETUP_PY" -m pip install -r $(REQUIREMENTS) || \
		($(call print_error,"依存関係のインストールに失敗しました"); exit 1); \
	$(call print_success,"依存関係のインストールが完了しました"); \
	echo ""; \
	$(call print_step,4,5,"システム依存関係の確認"); \
	if command -v tesseract >/dev/null 2>&1; then \
		TESSERACT_VERSION=$$(tesseract --version | head -1); \
		$(call print_success,"Tesseract OCR が利用可能です: $$TESSERACT_VERSION"); \
	else \
		$(call print_error,"Tesseract OCR が見つかりません"); \
		echo "  $(COLOR_DIM)インストール例: brew install tesseract tesseract-lang$(COLOR_RESET)"; \
	fi; \
	echo ""; \
	$(call print_step,5,5,"Python依存関係の検証"); \
	if [ -f $(SCRIPTS_DIR)/check_dependencies.py ]; then \
		$(call print_progress,"Python依存関係を検証中"); \
		"$$SETUP_PY" $(SCRIPTS_DIR)/check_dependencies.py || \
			($(call print_warning,"依存関係の検証で警告がありました"); true); \
	else \
		$(call print_warning,"$(SCRIPTS_DIR)/check_dependencies.py が見つからないためスキップしました"); \
	fi; \
	echo ""; \
	$(call print_section,"セットアップが完了しました"); \
	echo ""; \
	echo "$(COLOR_BOLD)次のステップ:$(COLOR_RESET)"; \
	echo "  $(COLOR_CYAN)1.$(COLOR_RESET) 仮想環境を有効化: $(COLOR_DIM)source $(VENV_ACTIVATE)$(COLOR_RESET)"; \
	echo "  $(COLOR_CYAN)2.$(COLOR_RESET) 実行: $(COLOR_DIM)make run$(COLOR_RESET)"; \
	echo ""

.PHONY: help
help: ## 利用可能なコマンド一覧を表示
	@$(call print_section,"オフィス人物検出システム - Makefile")
	@echo ""
	@echo "$(COLOR_BOLD)利用可能なコマンド:$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / { \
		command = $$1; \
		description = $$2; \
		gsub(/^[ \t]+|[ \t]+$$/, "", command); \
		gsub(/^[ \t]+|[ \t]+$$/, "", description); \
		printf "  $(COLOR_CYAN)$(ICON_ARROW)$(COLOR_RESET) $(COLOR_BOLD)$(COLOR_CYAN)%-20s$(COLOR_RESET) %s\n", command, description \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(COLOR_BOLD)使用例:$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make setup$(COLOR_RESET)                  $(COLOR_DIM)# 開発環境の初期化$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make setup-dev$(COLOR_RESET)              $(COLOR_DIM)# 開発環境の初期化（pre-commit含む）$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make test$(COLOR_RESET)                   $(COLOR_DIM)# テスト実行$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make test-cov$(COLOR_RESET)               $(COLOR_DIM)# カバレッジ付きテスト$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make test TEST_MODE=coverage$(COLOR_RESET)  $(COLOR_DIM)# カバレッジ付きテスト（別形式）$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make run$(COLOR_RESET)                    $(COLOR_DIM)# 通常実行$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make baseline$(COLOR_RESET)               $(COLOR_DIM)# ベースライン実行と評価$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make baseline GT=data/gt_tracks_auto.json$(COLOR_RESET)  $(COLOR_DIM)# オプション指定$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make clean$(COLOR_RESET)                  $(COLOR_DIM)# outputクリーンアップ$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make lint$(COLOR_RESET)                   $(COLOR_DIM)# Lintチェック$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make format$(COLOR_RESET)                 $(COLOR_DIM)# コードフォーマット$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make format-check$(COLOR_RESET)           $(COLOR_DIM)# フォーマットチェック$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make precommit-run$(COLOR_RESET)          $(COLOR_DIM)# pre-commit実行（全ファイル）$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make ship$(COLOR_RESET)                   $(COLOR_DIM)# 最強コマンド: format→lint→test→git add→commit→push$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)make ship COMMIT_MSG=\"fix: bug修正\"$(COLOR_RESET)  $(COLOR_DIM)# カスタムコミットメッセージ付き$(COLOR_RESET)"
	@echo ""

# ========================================
# コード品質コマンド
# ========================================

# Ruff/Mypyコマンド（venv優先、フォールバック）
RUFF_CMD := $(shell if [ -f $(VENV_BIN)/ruff ]; then echo "$(VENV_BIN)/ruff"; elif command -v ruff >/dev/null 2>&1; then echo "ruff"; else echo ""; fi)
MYPY_CMD := $(shell if [ -f $(VENV_BIN)/mypy ]; then echo "$(VENV_BIN)/mypy"; elif command -v mypy >/dev/null 2>&1; then echo "mypy"; else echo ""; fi)

.PHONY: lint
lint: ## Lintチェック（ruff + mypy）
	@set -e; \
	$(call print_section,"Lintチェック"); \
	if [ -z "$(RUFF_CMD)" ]; then \
		$(call print_error,"ruffがインストールされていません"); \
		echo "  $(COLOR_DIM)インストール: pip install ruff$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	$(call print_subsection,"ruffチェック"); \
	$(call print_progress,"ruffでコードをチェック中"); \
	$(RUFF_CMD) check . || exit 1; \
	$(call print_success,"ruffチェックが完了しました"); \
	echo ""; \
	if [ -n "$(MYPY_CMD)" ]; then \
		$(call print_subsection,"mypyチェック"); \
		$(call print_progress,"mypyで型チェック中"); \
		$(MYPY_CMD) $(SRC_DIR)/ --ignore-missing-imports || \
		($(call print_warning,"mypyチェックで警告がありました（続行）"); true); \
		$(call print_success,"mypyチェックが完了しました"); \
	else \
		$(call print_warning,"mypyがインストールされていません（スキップ）"); \
		echo "  $(COLOR_DIM)インストール: pip install mypy$(COLOR_RESET)"; \
	fi; \
	echo ""; \
	$(call print_success,"Lintチェックが完了しました")

.PHONY: format
format: ## コードフォーマット（ruff format + ruff check --fix）
	@set -e; \
	$(call print_section,"コードフォーマット"); \
	if [ -z "$(RUFF_CMD)" ]; then \
		$(call print_error,"ruffがインストールされていません"); \
		echo "  $(COLOR_DIM)インストール: pip install ruff$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	$(call print_step,1,3,"ruff format実行"); \
	$(call print_progress,"コードをフォーマット中"); \
	$(RUFF_CMD) format . || exit 1; \
	$(call print_success,"フォーマットが完了しました"); \
	echo ""; \
	$(call print_step,2,3,"ruff check --fix実行"); \
	$(call print_progress,"自動修正可能な問題を修正中"); \
	$(RUFF_CMD) check . --fix --unsafe-fixes || exit 1; \
	$(call print_success,"自動修正が完了しました"); \
	echo ""; \
	$(call print_step,3,3,"残りのエラー確認"); \
	$(call print_progress,"残りのエラーがないか確認中"); \
	$(RUFF_CMD) check . || \
	($(call print_warning,"修正できないエラーが残っています。手動で修正してください。"); exit 1); \
	$(call print_success,"フォーマットが完了しました"); \
	echo ""

.PHONY: format-check
format-check: ## フォーマットチェック（変更なし）
	@set -e; \
	$(call print_section,"フォーマットチェック"); \
	if [ -z "$(RUFF_CMD)" ]; then \
		$(call print_error,"ruffがインストールされていません"); \
		echo "  $(COLOR_DIM)インストール: pip install ruff$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	$(call print_progress,"フォーマットをチェック中"); \
	$(RUFF_CMD) format --check . || \
		($(call print_error,"フォーマットの問題があります。make format を実行してください。"); exit 1); \
	$(call print_success,"フォーマットチェックが完了しました")

.PHONY: test-cov
test-cov: ## カバレッジ付きテスト（詳細レポート）
	@$(call print_section,"カバレッジ付きテスト実行")
	@$(call print_info,"テストディレクトリ: $(TESTS_DIR)")
	@$(call print_info,"ソースディレクトリ: $(SRC_DIR)")
	@echo ""
	@$(VENV_PY) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html -v
	@echo ""
	@$(call print_success,"カバレッジレポートを生成しました: htmlcov/index.html")

# ========================================
# 開発環境コマンド
# ========================================

.PHONY: setup-dev
setup-dev: setup ## 開発環境セットアップ（pre-commit 含む）
	@$(call print_section,"開発環境セットアップ（拡張）")
	@$(call print_step,1,2,"pre-commitのインストール")
	@$(call print_progress,"pre-commitをインストール中")
	@$(VENV_PY) -m pip install pre-commit
	@$(call print_success,"pre-commitをインストールしました")
	@echo ""
	@$(call print_step,2,2,"pre-commitフックの設定")
	@$(call print_progress,"Gitフックを設定中")
	@$(VENV_BIN)/pre-commit install
	@$(VENV_BIN)/pre-commit install --hook-type pre-push
	@$(call print_success,"pre-commitフックを設定しました")
	@echo ""
	@$(call print_section,"開発環境セットアップが完了しました")

.PHONY: precommit-install
precommit-install: ## pre-commit フックをインストール
	@$(call print_section,"pre-commitフックのインストール")
	@$(call print_progress,"pre-commitフックをインストール中")
	@pre-commit install
	@pre-commit install --hook-type pre-push
	@$(call print_success,"pre-commitフックをインストールしました")

.PHONY: precommit-run
precommit-run: ## pre-commit を全ファイルに実行
	@$(call print_section,"pre-commit実行（全ファイル）")
	@$(call print_progress,"pre-commitを実行中")
	@pre-commit run --all-files
	@$(call print_success,"pre-commitが完了しました")

# ========================================
# 最強コマンド（format + lint + test + git add + commit + push）
# ========================================

COMMIT_MSG ?= "chore: format, lint, and test"

.PHONY: ship
ship: ## 最強コマンド: format → lint → test → git add → commit → push
	@set -euo pipefail; \
	$(call print_section,"🚀 最強コマンド実行"); \
	echo ""; \
	$(call print_step,1,6,"コードフォーマット"); \
	if ! $(MAKE) format; then \
		$(call print_error,"フォーマットに失敗しました"); \
		exit 1; \
	fi; \
	echo ""; \
	$(call print_step,2,6,"Lintチェック"); \
	if ! $(MAKE) lint; then \
		$(call print_error,"Lintチェックに失敗しました"); \
		exit 1; \
	fi; \
	echo ""; \
	$(call print_step,3,6,"テスト実行"); \
	if ! $(MAKE) test; then \
		$(call print_error,"テストに失敗しました"); \
		exit 1; \
	fi; \
	echo ""; \
	$(call print_step,4,6,"Gitステージング"); \
	$(call print_progress,"変更をステージング中"); \
	if ! git add .; then \
		$(call print_error,"git addに失敗しました"); \
		exit 1; \
	fi; \
	$(call print_success,"変更をステージングしました"); \
	echo ""; \
	$(call print_step,5,6,"Gitコミット"); \
	$(call print_progress,"コミット中: $(COMMIT_MSG)"); \
	if git diff --cached --quiet; then \
		$(call print_warning,"コミットする変更がありません（スキップ）"); \
	else \
		if ! git commit -m $(COMMIT_MSG); then \
			$(call print_error,"コミットに失敗しました"); \
			exit 1; \
		fi; \
		$(call print_success,"コミットが完了しました"); \
	fi; \
	echo ""; \
	$(call print_step,6,6,"Gitプッシュ"); \
	$(call print_progress,"リモートにプッシュ中"); \
	if ! git push; then \
		$(call print_error,"プッシュに失敗しました"); \
		exit 1; \
	fi; \
	$(call print_success,"プッシュが完了しました"); \
	echo ""; \
	$(call print_section,"✨ 最強コマンドが完了しました"); \
	echo ""

# ========================================
# 依存関係管理コマンド
# ========================================

.PHONY: sync-requirements
sync-requirements: ## pyproject.toml から requirements.txt を生成
	@$(call print_section,"依存関係の同期")
	@$(call print_progress,"pip-tools をインストール中")
	@$(VENV_PY) -m pip install pip-tools
	@$(call print_progress,"requirements.txt を生成中")
	@$(VENV_BIN)/pip-compile pyproject.toml -o requirements.txt --resolver=backtracking
	@$(call print_success,"requirements.txt を生成しました")
	@echo ""
	@$(call print_info,"開発依存関係も含める場合:")
	@echo "  $(COLOR_DIM)pip-compile pyproject.toml --extra dev -o requirements-dev.txt$(COLOR_RESET)"
