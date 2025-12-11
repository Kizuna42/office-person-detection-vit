# オフィス人物検出システム - Makefile
# シンプルで保守しやすい make ターゲット群

SHELL := bash
.ONESHELL:
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV_DIR ?= venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PY := $(VENV_BIN)/python
RUN_PY = $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(PYTHON))
PIP = $(RUN_PY) -m pip
PYTEST = $(RUN_PY) -m pytest
RUFF = $(RUN_PY) -m ruff
MYPY = $(RUN_PY) -m mypy

CONFIG ?= config.yaml
OUTPUT_DIR ?= output
TESTS_DIR ?= tests
SRC_DIR ?= src
SCRIPTS_DIR ?= scripts
REQUIREMENTS ?= requirements.txt
COMMIT_MSG ?= "chore: format lint test"

export PYTHONPATH := $(PWD)
export PIP_DISABLE_PIP_VERSION_CHECK=1

TEST_MODE ?=
TEST_PARALLEL ?=
PYTEST_OPTS ?= -v
ifeq ($(TEST_MODE),coverage)
  PYTEST_OPTS := --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html -v
endif
ifeq ($(TEST_MODE),fast)
  PYTEST_OPTS := -v -m "not slow"
endif
ifeq ($(TEST_MODE),verbose)
  PYTEST_OPTS := -vv -s
endif
ifneq ($(TEST_PARALLEL),)
  PYTEST_OPTS += -n $(TEST_PARALLEL)
endif

.PHONY: help
help: ## 利用可能なコマンドを表示
	@echo "オフィス人物検出システム - Makefile"
	@echo ""
	@echo "ターゲット一覧:"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "使用例:"
	@echo "  make setup"
	@echo "  make test TEST_MODE=coverage"
	@echo "  make run CONFIG=config.yaml"

.PHONY: venv
venv: ## 仮想環境を作成しpipを最新化
	@if [ ! -x "$(VENV_PY)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@$(VENV_PY) -m pip install -U pip

.PHONY: deps
deps: venv ## 依存関係をインストール
	@$(PIP) install -r $(REQUIREMENTS)

.PHONY: deps-dev
deps-dev: deps ## pre-commit など開発用依存を追加
	@$(PIP) install pre-commit
	@$(VENV_BIN)/pre-commit install
	@$(VENV_BIN)/pre-commit install --hook-type pre-push

.PHONY: setup
setup: deps ## 開発環境初期化（venv + 依存）
	@$(RUN_PY) -m pip list >/dev/null

.PHONY: run
run: venv ## 通常実行（メインパイプライン）
	@$(RUN_PY) main.py --config $(CONFIG)

.PHONY: baseline
baseline: venv ## ベースライン実行と評価
	@set -e
	$(RUN_PY) $(SCRIPTS_DIR)/run_baseline.py --config $(CONFIG) $(if $(TAG),--tag $(TAG))
	session_id=""
	if [ -L $(OUTPUT_DIR)/latest ] && [ -e $(OUTPUT_DIR)/latest ]; then
		session_id=$$(basename $$(readlink $(OUTPUT_DIR)/latest))
	elif [ -d $(OUTPUT_DIR)/sessions ]; then
		session_id=$$(ls -td $(OUTPUT_DIR)/sessions/*/ 2>/dev/null | head -1 | xargs -n1 basename)
	fi
	test -n "$$session_id"
	$(RUN_PY) $(SCRIPTS_DIR)/evaluate_baseline.py --session $$session_id --config $(CONFIG) $(if $(GT),--gt $(GT)) $(if $(POINTS),--points $(POINTS))
	@echo "session: $$session_id"

.PHONY: lint
lint: venv ## ruff + mypy
	@$(RUFF) check $(SRC_DIR) $(TESTS_DIR)
	@if $(MYPY) --version >/dev/null 2>&1; then \
		$(MYPY) $(SRC_DIR) --ignore-missing-imports; \
	else \
		echo "mypy が未インストールです。make deps を実行してください。"; \
	fi

.PHONY: format
format: venv ## コードフォーマットと自動修正
	@$(RUFF) format $(SRC_DIR) $(TESTS_DIR)
	@$(RUFF) check $(SRC_DIR) $(TESTS_DIR) --fix --unsafe-fixes

.PHONY: format-check
format-check: venv ## フォーマットが崩れていないか確認
	@$(RUFF) format --check $(SRC_DIR) $(TESTS_DIR)

.PHONY: test
test: venv ## テスト実行（TEST_MODE, TEST_PARALLELを指定可）
	@$(PYTEST) $(TESTS_DIR) $(PYTEST_OPTS)

.PHONY: test-cov
test-cov: ## カバレッジ付きテスト（HTML出力）
	@$(MAKE) test TEST_MODE=coverage

.PHONY: precommit-run
precommit-run: venv ## pre-commit を全ファイルに実行
	@$(VENV_BIN)/pre-commit run --all-files

.PHONY: clean
clean: ## output ディレクトリを安全にクリーンアップ
	@python - <<'PY'
from pathlib import Path

root = Path("$(OUTPUT_DIR)")
preserve_files = {root / "labels" / "result_fixed.json"}
preserve_dirs = {root / "calibration", root / "shared"}
if not root.exists():
    print("skip: output directory not found")
    raise SystemExit

deleted = 0
for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
    if any(parent in preserve_dirs for parent in path.parents):
        continue
    if path in preserve_files:
        continue
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted += 1
        elif path.is_dir():
            path.rmdir()
    except Exception as exc:
        print(f"skip {path}: {exc}")

for child in root.iterdir():
    if child in preserve_dirs:
        continue
    if child.is_dir() and not any(child.rglob("*")):
        child.rmdir()
print(f"deleted files: {deleted}")
PY

.PHONY: sync-requirements
sync-requirements: venv ## pyproject.toml から requirements.txt を生成
	@$(PIP) install pip-tools
	@$(VENV_BIN)/pip-compile pyproject.toml -o requirements.txt --resolver=backtracking

.PHONY: ci
ci: format-check lint test ## CI 相当の一括実行
	@echo "ci OK"

.PHONY: ship
ship: ## format→lint→test→git add→commit→push
	@set -euo pipefail
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
	git add .
	if git diff --cached --quiet; then \
		echo "no staged changes; skip commit"; \
	else \
		git commit -m $(COMMIT_MSG); \
	fi
	git push
