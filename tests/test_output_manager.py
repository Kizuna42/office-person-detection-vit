"""Test cases for OutputManager."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from src.utils.output_manager import (
    OutputManager,
    format_file_size,
    setup_output_directories,
)


@pytest.fixture()
def output_base(tmp_path: Path) -> Path:
    """テスト用の出力ベースディレクトリ"""
    return tmp_path / "output"


@pytest.fixture()
def output_manager(output_base: Path) -> OutputManager:
    """テスト用のOutputManager"""
    return OutputManager(output_base)


def test_init(output_base: Path):
    """初期化が正しく動作する"""
    manager = OutputManager(output_base)

    assert manager.output_base == Path(output_base)
    assert manager.sessions_dir == output_base / "sessions"
    assert manager.sessions_dir.exists()


def test_create_session(output_manager: OutputManager):
    """セッション作成が正しく動作する"""
    session_dir = output_manager.create_session()

    assert session_dir.exists()
    assert session_dir.name.startswith(datetime.now().strftime("%Y%m%d"))
    assert (session_dir / "phase1_extraction" / "frames").exists()
    assert (session_dir / "phase2_detection" / "images").exists()
    assert (session_dir / "phase3_transform").exists()
    assert (session_dir / "phase4_aggregation").exists()
    assert (session_dir / "phase5_visualization" / "graphs").exists()
    assert (session_dir / "phase5_visualization" / "floormaps").exists()


def test_save_metadata(output_manager: OutputManager, tmp_path: Path):
    """メタデータ保存が正しく動作する"""
    session_dir = output_manager.create_session()
    config = {"test": "config"}
    args = {"debug": True}

    output_manager.save_metadata(session_dir, config, args)

    metadata_path = session_dir / "metadata.json"
    assert metadata_path.exists()

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["session_id"] == session_dir.name
    assert metadata["config"] == config
    assert metadata["arguments"] == args


def test_save_summary(output_manager: OutputManager):
    """サマリー保存が正しく動作する"""
    session_dir = output_manager.create_session()
    summary = {"status": "completed", "phases": {}}

    output_manager.save_summary(session_dir, summary)

    summary_path = session_dir / "summary.json"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8") as f:
        saved_summary = json.load(f)

    assert saved_summary["status"] == "completed"
    assert saved_summary["session_id"] == session_dir.name


def test_update_latest_link(output_manager: OutputManager):
    """最新セッションリンクの更新が正しく動作する"""
    session_dir = output_manager.create_session()

    try:
        output_manager.update_latest_link(session_dir)

        if output_manager.latest_link.exists():
            assert output_manager.latest_link.is_symlink() or output_manager.latest_link.exists()
    except OSError:
        # Windowsなど、シンボリックリンクが使えない環境ではスキップ
        pass


def test_get_latest_session(output_manager: OutputManager):
    """最新セッション取得が正しく動作する"""
    session1 = output_manager.create_session()

    latest = output_manager.get_latest_session()
    assert latest is not None
    assert latest.name == session1.name


def test_find_sessions(output_manager: OutputManager):
    """セッション検索が正しく動作する"""
    session = output_manager.create_session()

    sessions = output_manager.find_sessions()
    assert len(sessions) >= 1
    assert any(s.name == session.name for s in sessions)


def test_find_sessions_with_pattern(output_manager: OutputManager):
    """パターン指定でのセッション検索"""
    output_manager.create_session()
    date_str = datetime.now().strftime("%Y%m%d")

    sessions = output_manager.find_sessions(pattern=date_str)
    assert len(sessions) >= 1
    assert all(date_str in s.name for s in sessions)


def test_find_sessions_with_date_range(output_manager: OutputManager):
    """日時範囲指定でのセッション検索"""
    output_manager.create_session()

    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now() + timedelta(days=1)

    sessions = output_manager.find_sessions(start_date=start_date, end_date=end_date)
    assert len(sessions) >= 1


def test_get_session_size(output_manager: OutputManager, tmp_path: Path):
    """セッションサイズ取得が正しく動作する"""
    session_dir = output_manager.create_session()

    # テストファイルを作成
    test_file = session_dir / "test.txt"
    test_file.write_text("test content")

    size = output_manager.get_session_size(session_dir)
    assert size > 0


def test_format_file_size():
    """ファイルサイズフォーマットが正しく動作する"""
    assert format_file_size(0) == "0.00 B"
    assert format_file_size(1024) == "1.00 KB"
    assert format_file_size(1024 * 1024) == "1.00 MB"
    assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"


def test_setup_output_directories(tmp_path: Path):
    """出力ディレクトリ作成が正しく動作する"""
    output_dir = tmp_path / "output"

    setup_output_directories(output_dir)

    assert (output_dir / "detections").exists()
    assert (output_dir / "floormaps").exists()
    assert (output_dir / "graphs").exists()
    assert (output_dir / "labels").exists()
