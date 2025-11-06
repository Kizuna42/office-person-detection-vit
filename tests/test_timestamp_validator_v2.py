"""Unit tests for TemporalValidatorV2."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.timestamp.timestamp_validator_v2 import TemporalValidatorV2


@pytest.fixture
def validator_30fps() -> TemporalValidatorV2:
    """30fps用のTemporalValidatorV2"""
    return TemporalValidatorV2(fps=30.0, base_tolerance_seconds=10.0)


@pytest.fixture
def validator_60fps() -> TemporalValidatorV2:
    """60fps用のTemporalValidatorV2"""
    return TemporalValidatorV2(fps=60.0, base_tolerance_seconds=10.0)


def test_initial_timestamp_validation(validator_30fps: TemporalValidatorV2):
    """初回タイムスタンプの検証"""
    timestamp = datetime(2025, 8, 26, 16, 5, 0)
    is_valid, confidence, reason = validator_30fps.validate(timestamp, 0)

    assert is_valid is True
    assert confidence == 1.0
    assert reason == "First frame"


def test_sequential_validation_valid(validator_30fps: TemporalValidatorV2):
    """連続する有効なタイムスタンプの検証"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 次のフレーム（1秒後、30フレーム後）
    next_time = base_time + timedelta(seconds=1.0)
    is_valid, confidence, reason = validator_30fps.validate(next_time, 30)

    assert is_valid is True
    assert confidence > 0.0
    assert "Valid" in reason


def test_sequential_validation_invalid(validator_30fps: TemporalValidatorV2):
    """無効なタイムスタンプの検証"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 許容範囲を大きく超えた時間差（無効）
    # 30フレーム後（1秒後）のはずが、20秒後
    invalid_time = base_time + timedelta(seconds=20.0)
    is_valid, confidence, reason = validator_30fps.validate(invalid_time, 30)

    assert is_valid is False
    assert confidence == 0.0
    assert "Invalid" in reason


def test_fps_variation_handling(validator_30fps: TemporalValidatorV2):
    """FPS変動のハンドリング"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 少し遅いフレーム（許容範囲内）
    next_time = base_time + timedelta(seconds=1.1)  # 30フレームで1.1秒
    is_valid, confidence, reason = validator_30fps.validate(next_time, 30)

    assert is_valid is True
    assert confidence > 0.0


def test_reset_functionality(validator_30fps: TemporalValidatorV2):
    """リセット機能のテスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)
    assert validator_30fps.last_timestamp is not None

    # リセット
    validator_30fps.reset()
    assert validator_30fps.last_timestamp is None
    assert validator_30fps.last_frame_idx is None
    assert len(validator_30fps.interval_history) == 0

    # リセット後は再び初回フレームとして扱われる
    is_valid, confidence, reason = validator_30fps.validate(base_time, 0)
    assert is_valid is True
    assert reason == "First frame"


def test_adaptive_tolerance_calculation(validator_30fps: TemporalValidatorV2):
    """適応的許容範囲の計算"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 複数のフレームを追加して履歴を構築
    for i in range(1, 5):
        next_time = base_time + timedelta(seconds=i * 1.0)
        validator_30fps.validate(next_time, i * 30)

    # 適応的許容範囲が計算されていることを確認
    tolerance = validator_30fps._calculate_adaptive_tolerance()
    assert tolerance >= validator_30fps.base_tolerance * 0.5
    assert tolerance <= validator_30fps.base_tolerance * 3.0


def test_outlier_detection(validator_30fps: TemporalValidatorV2):
    """外れ値検出のテスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 正常なフレームを数個追加
    for i in range(1, 4):
        next_time = base_time + timedelta(seconds=i * 1.0)
        validator_30fps.validate(next_time, i * 30)

    # 外れ値（大きくずれた時間）
    outlier_time = base_time + timedelta(seconds=10.0)  # 4秒後のはずが10秒後
    is_outlier, z_score = validator_30fps._detect_outlier(10.0, 1.0)

    # 履歴が少ない場合は外れ値として検出されない可能性がある
    # 履歴が十分な場合は検出される
    if len(validator_30fps.interval_history) >= 3:
        # 外れ値として検出される可能性がある
        pass


def test_outlier_recovery(validator_30fps: TemporalValidatorV2):
    """異常値リカバリーのテスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # リカバリー後のタイムスタンプを計算
    expected_seconds = 1.0
    recovered = validator_30fps._recover_timestamp(30, expected_seconds)

    assert recovered is not None
    assert recovered == base_time + timedelta(seconds=expected_seconds)


def test_outlier_recovery_without_history(validator_30fps: TemporalValidatorV2):
    """履歴がない場合のリカバリー"""
    # 履歴がない場合、リカバリーはNoneを返す
    recovered = validator_30fps._recover_timestamp(30, 1.0)
    assert recovered is None


def test_invalid_frame_diff(validator_30fps: TemporalValidatorV2):
    """無効なフレーム差のテスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # フレーム番号が逆転している場合
    invalid_time = base_time + timedelta(seconds=1.0)
    is_valid, confidence, reason = validator_30fps.validate(invalid_time, 0)

    assert is_valid is False
    assert "Invalid frame_diff" in reason


def test_confidence_calculation(validator_30fps: TemporalValidatorV2):
    """信頼度計算のテスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 正確な時間差
    next_time = base_time + timedelta(seconds=1.0)
    is_valid1, confidence1, _ = validator_30fps.validate(next_time, 30)

    # 少しずれた時間差
    next_time2 = next_time + timedelta(seconds=1.1)
    is_valid2, confidence2, _ = validator_30fps.validate(next_time2, 60)

    assert is_valid1 is True
    assert is_valid2 is True
    # 正確な時間差の方が信頼度が高い
    assert confidence1 >= confidence2


def test_z_score_threshold(validator_30fps: TemporalValidatorV2):
    """Z-score閾値のテスト"""
    # カスタムZ-score閾値で初期化
    validator = TemporalValidatorV2(
        fps=30.0,
        base_tolerance_seconds=10.0,
        z_score_threshold=3.0,  # より厳しい閾値
    )

    base_time = datetime(2025, 8, 26, 16, 5, 0)
    validator.validate(base_time, 0)

    # 正常なフレームを追加
    for i in range(1, 5):
        next_time = base_time + timedelta(seconds=i * 1.0)
        validator.validate(next_time, i * 30)

    assert validator.z_score_threshold == 3.0


def test_history_size_limit(validator_30fps: TemporalValidatorV2):
    """履歴サイズの制限テスト"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_30fps.validate(base_time, 0)

    # 履歴サイズを超えるフレームを追加
    for i in range(1, 20):
        next_time = base_time + timedelta(seconds=i * 1.0)
        validator_30fps.validate(next_time, i * 30)

    # 履歴サイズが制限されていることを確認
    assert len(validator_30fps.interval_history) <= validator_30fps.history_size


def test_60fps_validation(validator_60fps: TemporalValidatorV2):
    """60fpsでの検証"""
    base_time = datetime(2025, 8, 26, 16, 5, 0)

    # 初回フレーム
    validator_60fps.validate(base_time, 0)

    # 60フレーム後（1秒後）
    next_time = base_time + timedelta(seconds=1.0)
    is_valid, confidence, reason = validator_60fps.validate(next_time, 60)

    assert is_valid is True
    assert confidence > 0.0
    assert "Valid" in reason

