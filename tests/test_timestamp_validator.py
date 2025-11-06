"""Unit tests for TemporalValidator."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.timestamp.timestamp_validator import TemporalValidator


@pytest.fixture
def validator_30fps() -> TemporalValidator:
    """30fpsのTemporalValidator"""
    return TemporalValidator(fps=30.0)


@pytest.fixture
def validator_60fps() -> TemporalValidator:
    """60fpsのTemporalValidator"""
    return TemporalValidator(fps=60.0)


def test_initial_timestamp_validation(validator_30fps: TemporalValidator):
    """時系列整合性チェックのテスト（初回タイムスタンプ）"""
    timestamp = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx = 0
    
    is_valid, confidence, reason = validator_30fps.validate(timestamp, frame_idx)
    
    # 初回は常に受け入れられる
    assert is_valid is True
    assert confidence == 1.0
    assert "Initial" in reason


def test_sequential_validation_valid(validator_30fps: TemporalValidator):
    """時系列整合性チェックのテスト（有効な連続フレーム）"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 1秒後（30フレーム後）
    timestamp2 = datetime(2025, 8, 26, 16, 7, 46)
    frame_idx2 = 30
    
    is_valid, confidence, reason = validator_30fps.validate(timestamp2, frame_idx2)
    
    assert is_valid is True
    assert confidence > 0.5
    assert "Valid" in reason


def test_sequential_validation_invalid(validator_30fps: TemporalValidator):
    """時系列整合性チェックのテスト（無効な連続フレーム）"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 大きく外れたタイムスタンプ（1時間後だが30フレーム後）
    timestamp2 = datetime(2025, 8, 26, 17, 7, 45)
    frame_idx2 = 30
    
    is_valid, confidence, reason = validator_30fps.validate(timestamp2, frame_idx2)
    
    assert is_valid is False
    assert confidence == 0.0
    assert "Invalid" in reason


def test_fps_variation_handling(validator_30fps: TemporalValidator, validator_60fps: TemporalValidator):
    """フレームレート変動への対応テスト"""
    # 30fpsでの検証
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 1秒後（30フレーム後）
    timestamp2 = datetime(2025, 8, 26, 16, 7, 46)
    frame_idx2 = 30
    is_valid_30, _, _ = validator_30fps.validate(timestamp2, frame_idx2)
    
    # 60fpsでの検証
    validator_60fps.validate(timestamp1, frame_idx1)
    # 1秒後（60フレーム後）
    frame_idx2_60 = 60
    is_valid_60, _, _ = validator_60fps.validate(timestamp2, frame_idx2_60)
    
    # 両方とも有効であることを確認
    assert is_valid_30 is True
    assert is_valid_60 is True


def test_reset_functionality(validator_30fps: TemporalValidator):
    """リセット機能のテスト"""
    # いくつかのタイムスタンプを検証
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    timestamp2 = datetime(2025, 8, 26, 16, 7, 46)
    frame_idx2 = 30
    validator_30fps.validate(timestamp2, frame_idx2)
    
    # リセット
    validator_30fps.reset()
    
    # リセット後、再度初回タイムスタンプとして検証
    timestamp3 = datetime(2025, 8, 26, 16, 8, 0)
    frame_idx3 = 100
    
    is_valid, confidence, reason = validator_30fps.validate(timestamp3, frame_idx3)
    
    # リセット後は初回として扱われる
    assert is_valid is True
    assert "Initial" in reason


def test_tolerance_calculation(validator_30fps: TemporalValidator):
    """許容範囲の計算テスト"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 期待値から±20%以内のタイムスタンプ
    # 1秒後（30フレーム）の期待値に対して、0.8秒後（24フレーム）のタイムスタンプ
    timestamp2 = datetime(2025, 8, 26, 16, 7, 45, 800000)  # 0.8秒後
    frame_idx2 = 30
    
    is_valid, confidence, _ = validator_30fps.validate(timestamp2, frame_idx2)
    
    # ±20%の許容範囲内なので有効
    assert is_valid is True


def test_confidence_calculation(validator_30fps: TemporalValidator):
    """信頼度計算のテスト"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 完全に一致する場合
    timestamp2 = datetime(2025, 8, 26, 16, 7, 46)
    frame_idx2 = 30
    _, conf_exact, _ = validator_30fps.validate(timestamp2, frame_idx2)
    
    # 少しずれている場合
    validator_30fps.reset()
    validator_30fps.validate(timestamp1, frame_idx1)
    timestamp3 = datetime(2025, 8, 26, 16, 7, 45, 500000)  # 0.5秒後
    _, conf_offset, _ = validator_30fps.validate(timestamp3, frame_idx2)
    
    # 完全一致の方が信頼度が高い
    assert conf_exact >= conf_offset


def test_backward_timestamp(validator_30fps: TemporalValidator):
    """過去のタイムスタンプのテスト"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 過去のタイムスタンプ（無効）
    timestamp2 = datetime(2025, 8, 26, 16, 7, 44)
    frame_idx2 = 30
    
    is_valid, confidence, reason = validator_30fps.validate(timestamp2, frame_idx2)
    
    assert is_valid is False
    assert confidence == 0.0


def test_large_frame_gap(validator_30fps: TemporalValidator):
    """大きなフレーム間隔のテスト"""
    # 最初のフレーム
    timestamp1 = datetime(2025, 8, 26, 16, 7, 45)
    frame_idx1 = 0
    validator_30fps.validate(timestamp1, frame_idx1)
    
    # 10秒後（300フレーム後）
    timestamp2 = datetime(2025, 8, 26, 16, 7, 55)
    frame_idx2 = 300
    
    is_valid, confidence, _ = validator_30fps.validate(timestamp2, frame_idx2)
    
    # 大きな間隔でも許容範囲内なら有効
    assert is_valid is True
    assert confidence > 0.0

