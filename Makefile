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
DASHBOARD_APP ?= src/visualization/dashboard_app.py
DASHBOARD_HOST ?= 0.0.0.0
DASHBOARD_PORT ?= 8501
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

.PHONY: dashboard
dashboard: venv ## Streamlit ダッシュボードを起動
	@$(RUN_PY) -m streamlit run $(DASHBOARD_APP) --server.address=$(DASHBOARD_HOST) --server.port=$(DASHBOARD_PORT)

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

# ベンチマーク関連変数
BENCHMARK_GT ?= output/ground_truth/gt_tracking.json
BENCHMARK_PRED ?= output/latest/03_tracking/tracks.csv
BENCHMARK_OUT ?= output/benchmark
GT_DIR ?= output/ground_truth
CVAT_CSV ?= output/ground_truth/gt_tracking_fixed.csv

.PHONY: gt-prepare
gt-prepare: venv ## パイプライン出力からGT初期データを生成
	@$(RUN_PY) tools/convert_to_gold_gt.py \
		--input output/latest/04_transform/coordinate_transformations.json \
		--output $(GT_DIR)/gt_tracking.json \
		--mot-output $(GT_DIR)/gt_tracking.csv
	@echo "GT生成完了: $(GT_DIR)/gt_tracking.json"
	@echo "MOT CSV: $(GT_DIR)/gt_tracking.csv (CVAT用)"

.PHONY: gt-from-cvat
gt-from-cvat: venv ## CVATからエクスポートしたCSVをGold GTに変換
	@$(RUN_PY) tools/convert_mot_to_gold.py \
		--input $(CVAT_CSV) \
		--output $(GT_DIR)/gt_tracking.json
	@echo "CVAT CSVからGold GTに変換完了"

.PHONY: benchmark-tracking
benchmark-tracking: venv ## トラッキング精度を評価（make benchmark-tracking）
	@$(RUN_PY) -m src.benchmark \
		--gt $(BENCHMARK_GT) \
		--pred $(BENCHMARK_PRED) \
		-o $(BENCHMARK_OUT) \
		--gt-format gold \
		--report

.PHONY: benchmark-tracking-sparse
benchmark-tracking-sparse: venv ## 疎サンプリングモードで評価（5分間隔）
	@$(RUN_PY) -m src.benchmark \
		--gt $(BENCHMARK_GT) \
		--pred $(BENCHMARK_PRED) \
		-o $(BENCHMARK_OUT) \
		--gt-format gold \
		--sparse \
		--report

# 検出ベンチマーク変数
DETECTION_GT ?= output/labels/result_fixed.json
DETECTION_PRED ?= output/latest/04_transform/coordinate_transformations.json

.PHONY: benchmark-detection
benchmark-detection: venv ## 検出精度を評価（make benchmark-detection）
	@$(RUN_PY) -m src.benchmark.detection_runner \
		--gt $(DETECTION_GT) \
		--pred $(DETECTION_PRED) \
		-o $(BENCHMARK_OUT) \
		--report

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
	@$(RUN_PY) -c "from pathlib import Path; \
root = Path('$(OUTPUT_DIR)'); \
preserve_files = {root / 'labels' / 'result_fixed.json'}; \
preserve_dirs = {root / 'calibration', root / 'shared'}; \
\
if not root.exists(): \
    print('skip: output directory not found'); \
    raise SystemExit; \
deleted = 0; \
paths = sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True); \
for path in paths: \
    if any(parent in preserve_dirs for parent in path.parents): \
        continue; \
    if path in preserve_files: \
        continue; \
    try: \
        if path.is_file() or path.is_symlink(): \
            path.unlink(); \
            deleted += 1; \
        elif path.is_dir(): \
            path.rmdir(); \
    except Exception as exc: \
        print(f'skip {path}: {exc}'); \
for child in root.iterdir(): \
    if child in preserve_dirs: \
        continue; \
    if child.is_dir() and not any(child.rglob('*')): \
        child.rmdir(); \
print(f'deleted files: {deleted}')"

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
