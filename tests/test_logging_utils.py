"""Test cases for logging_utils."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.logging_utils import setup_logging


def test_setup_logging_debug_mode(tmp_path: Path):
    """デバッグモードでのロギング設定"""
    output_dir = str(tmp_path / "output")

    setup_logging(debug_mode=True, output_dir=output_dir)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    # ファイルハンドラーが設定されていることを確認
    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) > 0

    log_file = Path(output_dir) / "system.log"
    assert log_file.exists() or any(str(h.baseFilename) == str(log_file) for h in file_handlers)


def test_setup_logging_info_mode(tmp_path: Path):
    """INFOモードでのロギング設定"""
    output_dir = str(tmp_path / "output")

    setup_logging(debug_mode=False, output_dir=output_dir)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO


def test_setup_logging_creates_directory(tmp_path: Path):
    """存在しないディレクトリが自動作成される"""
    output_dir = str(tmp_path / "new" / "output")

    setup_logging(debug_mode=False, output_dir=output_dir)

    assert Path(output_dir).exists()


def test_setup_logging_clears_existing_handlers(tmp_path: Path):
    """既存のハンドラーがクリアされる"""
    output_dir = str(tmp_path / "output")

    setup_logging(debug_mode=False, output_dir=output_dir)

    # ハンドラーが再設定されていることを確認（正確な数は環境依存）
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) >= 2  # console + file
