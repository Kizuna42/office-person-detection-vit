"""Unit tests for checkpoint manager module."""

from __future__ import annotations

import json

import pytest

from src.utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """CheckpointManagerのテスト"""

    @pytest.fixture
    def session_dir(self, tmp_path):
        """テスト用セッションディレクトリ"""
        session_dir = tmp_path / "test_session"
        session_dir.mkdir()
        return session_dir

    @pytest.fixture
    def manager(self, session_dir):
        """テスト用CheckpointManager"""
        return CheckpointManager(session_dir)

    def test_init(self, session_dir):
        """初期化テスト"""
        manager = CheckpointManager(session_dir)
        assert manager.session_dir == session_dir
        assert manager.checkpoint_path == session_dir / "pipeline_checkpoint.json"

    def test_save_checkpoint_basic(self, manager, session_dir):
        """基本的なチェックポイント保存テスト"""
        manager.save_checkpoint("01_extraction", status="completed")

        assert manager.checkpoint_path.exists()
        with open(manager.checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "phases" in data
        assert "01_extraction" in data["phases"]
        assert data["phases"]["01_extraction"]["status"] == "completed"
        assert data["last_phase"] == "01_extraction"

    def test_save_checkpoint_with_data(self, manager):
        """データ付きチェックポイント保存テスト"""
        checkpoint_data = {"frames_processed": 100, "detections": 50}
        manager.save_checkpoint("02_detection", data=checkpoint_data, status="completed")

        with open(manager.checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["phases"]["02_detection"]["data"]["frames_processed"] == 100
        assert data["phases"]["02_detection"]["data"]["detections"] == 50

    def test_save_checkpoint_multiple_phases(self, manager):
        """複数フェーズのチェックポイント保存テスト"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="completed")
        manager.save_checkpoint("03_tracking", status="in_progress")

        with open(manager.checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        # 初期化時に6フェーズ作成 + 3フェーズ更新 = 6フェーズ
        assert len(data["phases"]) == 6
        assert data["phases"]["01_extraction"]["status"] == "completed"
        assert data["phases"]["02_detection"]["status"] == "completed"
        assert data["phases"]["03_tracking"]["status"] == "in_progress"

    def test_load_checkpoint_existing(self, manager):
        """既存チェックポイントの読み込みテスト"""
        checkpoint_data = {"frames_processed": 100}
        manager.save_checkpoint("01_extraction", data=checkpoint_data, status="completed")

        loaded = manager.load_checkpoint("01_extraction")
        assert loaded is not None
        assert loaded["frames_processed"] == 100

    def test_load_checkpoint_nonexistent(self, manager):
        """存在しないチェックポイントの読み込みテスト"""
        loaded = manager.load_checkpoint("nonexistent_phase")
        assert loaded is None

    def test_load_checkpoint_incomplete(self, manager):
        """未完了フェーズのチェックポイント読み込みテスト"""
        manager.save_checkpoint("01_extraction", status="in_progress")

        loaded = manager.load_checkpoint("01_extraction")
        assert loaded is None  # in_progressのフェーズはNoneを返す

    def test_is_phase_completed_true(self, manager):
        """フェーズ完了確認テスト（完了）"""
        manager.save_checkpoint("01_extraction", status="completed")

        assert manager.is_phase_completed("01_extraction") is True

    def test_is_phase_completed_false(self, manager):
        """フェーズ完了確認テスト（未完了）"""
        manager.save_checkpoint("01_extraction", status="in_progress")

        assert manager.is_phase_completed("01_extraction") is False

    def test_is_phase_completed_nonexistent(self, manager):
        """フェーズ完了確認テスト（存在しない）"""
        assert manager.is_phase_completed("nonexistent_phase") is False

    def test_get_last_completed_phase_none(self, manager):
        """最後の完了フェーズ取得テスト（なし）"""
        assert manager.get_last_completed_phase() is None

    def test_get_last_completed_phase_single(self, manager):
        """最後の完了フェーズ取得テスト（単一）"""
        manager.save_checkpoint("01_extraction", status="completed")

        assert manager.get_last_completed_phase() == "01_extraction"

    def test_get_last_completed_phase_multiple(self, manager):
        """最後の完了フェーズ取得テスト（複数）"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="completed")
        manager.save_checkpoint("03_tracking", status="completed")

        assert manager.get_last_completed_phase() == "03_tracking"

    def test_get_last_completed_phase_with_incomplete(self, manager):
        """最後の完了フェーズ取得テスト（未完了あり）"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="completed")
        manager.save_checkpoint("03_tracking", status="in_progress")

        assert manager.get_last_completed_phase() == "02_detection"

    def test_get_resumable_phase_none(self, manager):
        """再開可能フェーズ取得テスト（なし）"""
        assert manager.get_resumable_phase() is None

    def test_get_resumable_phase_from_phase1(self, manager):
        """再開可能フェーズ取得テスト（01_extractionから）"""
        manager.save_checkpoint("01_extraction", status="completed")

        assert manager.get_resumable_phase() == "02_detection"

    def test_get_resumable_phase_from_phase2(self, manager):
        """再開可能フェーズ取得テスト（02_detectionから）"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="completed")

        assert manager.get_resumable_phase() == "03_tracking"

    def test_get_resumable_phase_all_complete(self, manager):
        """再開可能フェーズ取得テスト（全完了）"""
        phases = [
            "01_extraction",
            "02_detection",
            "03_tracking",
            "04_transform",
            "05_aggregation",
            "06_visualization",
        ]
        for phase in phases:
            manager.save_checkpoint(phase, status="completed")

        assert manager.get_resumable_phase() is None

    def test_clear_checkpoint(self, manager):
        """チェックポイントクリアテスト"""
        manager.save_checkpoint("01_extraction", status="completed")
        assert manager.checkpoint_path.exists()

        manager.clear_checkpoint()
        assert not manager.checkpoint_path.exists()

    def test_clear_checkpoint_nonexistent(self, manager):
        """存在しないチェックポイントのクリアテスト"""
        # エラーにならないことを確認
        manager.clear_checkpoint()

    def test_get_summary_empty(self, manager, session_dir):
        """サマリー取得テスト（空）"""
        summary = manager.get_summary()

        assert summary["session_dir"] == str(session_dir)
        assert summary["last_updated"] is None
        assert summary["last_phase"] is None
        assert summary["completed_phases"] == []
        assert summary["resumable_from"] is None

    def test_get_summary_with_phases(self, manager, session_dir):
        """サマリー取得テスト（フェーズあり）"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="completed")

        summary = manager.get_summary()

        assert summary["session_dir"] == str(session_dir)
        assert summary["last_updated"] is not None
        assert summary["last_phase"] == "02_detection"
        assert "01_extraction" in summary["completed_phases"]
        assert "02_detection" in summary["completed_phases"]
        assert summary["resumable_from"] == "03_tracking"

    def test_checkpoint_persistence(self, session_dir):
        """チェックポイント永続化テスト"""
        # 最初のマネージャでチェックポイントを保存
        manager1 = CheckpointManager(session_dir)
        manager1.save_checkpoint("01_extraction", data={"test": "data"}, status="completed")

        # 新しいマネージャで読み込み
        manager2 = CheckpointManager(session_dir)
        loaded = manager2.load_checkpoint("01_extraction")

        assert loaded is not None
        assert loaded["test"] == "data"

    def test_corrupted_checkpoint_file(self, manager, session_dir):
        """破損したチェックポイントファイルの処理テスト"""
        # 破損したJSONを書き込み
        with open(manager.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        # エラーにならず、新しいチェックポイントが作成されることを確認
        summary = manager.get_summary()
        assert summary["completed_phases"] == []

    def test_save_checkpoint_creates_directory(self, tmp_path):
        """ディレクトリ自動作成テスト"""
        new_session_dir = tmp_path / "new_session" / "nested"
        manager = CheckpointManager(new_session_dir)

        manager.save_checkpoint("01_extraction", status="completed")

        assert new_session_dir.exists()
        assert manager.checkpoint_path.exists()

    def test_status_types(self, manager):
        """ステータスタイプのテスト"""
        manager.save_checkpoint("01_extraction", status="completed")
        manager.save_checkpoint("02_detection", status="in_progress")
        manager.save_checkpoint("03_tracking", status="failed")

        with open(manager.checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["phases"]["01_extraction"]["status"] == "completed"
        assert data["phases"]["02_detection"]["status"] == "in_progress"
        assert data["phases"]["03_tracking"]["status"] == "failed"

    def test_timestamp_updated(self, manager):
        """タイムスタンプ更新テスト"""
        manager.save_checkpoint("01_extraction", status="completed")

        with open(manager.checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["phases"]["01_extraction"]["timestamp"] is not None
        assert data["last_updated"] is not None
        assert data["created_at"] is not None
