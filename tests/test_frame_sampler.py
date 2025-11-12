"""Unit tests for frame samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.video.frame_sampler import AdaptiveSampler, CoarseSampler, FineSampler

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_video_path(tmp_path: Path) -> Path:
    """モック動画ファイルパス"""
    return tmp_path / "test_video.mov"


@pytest.fixture
def mock_video_capture():
    """モックVideoCapture"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 9000,  # 5分間（30fps × 300秒）
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    # フレーム読み込みのモック
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)

    return mock_cap


@patch("cv2.VideoCapture")
def test_coarse_sampler_interval(mock_video_capture_class, mock_video_path: Path, mock_video_capture):
    """CoarseSamplerのフレーム間隔テスト"""
    mock_video_capture_class.return_value = mock_video_capture

    sampler = CoarseSampler(str(mock_video_path), interval_seconds=10.0)

    # サンプリング実行
    frames = list(sampler.sample())

    # 間隔が正しいことを確認（10秒 = 300フレーム）
    if len(frames) >= 2:
        frame_idx1, _ = frames[0]
        frame_idx2, _ = frames[1]
        interval = frame_idx2 - frame_idx1

        # 10秒間隔（300フレーム）であることを確認
        assert interval == 300


@patch("cv2.VideoCapture")
def test_coarse_sampler_fps_calculation(mock_video_capture_class, mock_video_path: Path, mock_video_capture):
    """CoarseSamplerのFPS計算テスト"""
    mock_video_capture_class.return_value = mock_video_capture

    sampler = CoarseSampler(str(mock_video_path), interval_seconds=10.0)
    sampler._ensure_opened()

    # FPSが正しく取得されていることを確認
    assert sampler.fps == 30.0
    assert sampler.interval_frames == 300  # 30fps × 10秒


def test_fine_sampler_search_window(mock_video_capture):
    """FineSamplerの探索範囲テスト"""
    sampler = FineSampler(mock_video_capture, search_window=30.0)

    # FPSを設定
    sampler.fps = 30.0

    # 探索範囲の計算
    approx_frame_idx = 1000
    frames = list(sampler.sample_around_target(approx_frame_idx))

    # 探索範囲が正しいことを確認（±30秒 = ±900フレーム）
    if frames:
        frame_indices = [idx for idx, _ in frames]
        min_idx = min(frame_indices)
        max_idx = max(frame_indices)

        # 探索範囲が±30秒以内であることを確認
        assert min_idx >= approx_frame_idx - 900
        assert max_idx <= approx_frame_idx + 900


def test_fine_sampler_interval(mock_video_capture):
    """FineSamplerのサンプリング間隔テスト"""
    sampler = FineSampler(mock_video_capture, search_window=30.0, interval_seconds=0.1)
    sampler.fps = 30.0

    approx_frame_idx = 1000
    frames = list(sampler.sample_around_target(approx_frame_idx))

    # 0.1秒間隔（3フレーム）でサンプリングされていることを確認
    if len(frames) >= 2:
        frame_indices = [idx for idx, _ in frames]
        intervals = [frame_indices[i + 1] - frame_indices[i] for i in range(len(frame_indices) - 1)]

        # すべての間隔が3フレーム（0.1秒）であることを確認
        # 0.1秒間隔 = fps * 0.1 = 30 * 0.1 = 3フレーム間隔
        for interval in intervals:
            assert interval == 3


def test_fine_sampler_boundary_check(mock_video_capture):
    """FineSamplerの境界チェックテスト"""
    sampler = FineSampler(mock_video_capture, search_window=30.0)
    sampler.fps = 30.0

    # 動画の先頭付近
    approx_frame_idx = 100
    frames = list(sampler.sample_around_target(approx_frame_idx))

    if frames:
        frame_indices = [idx for idx, _ in frames]
        # すべてのフレームが0以上であることを確認
        assert all(idx >= 0 for idx in frame_indices)

    # 動画の末尾付近
    mock_video_capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 1000,
    }.get(prop, 0)

    approx_frame_idx = 950
    frames = list(sampler.sample_around_target(approx_frame_idx))

    if frames:
        frame_indices = [idx for idx, _ in frames]
        # すべてのフレームが総フレーム数未満であることを確認
        assert all(idx < 1000 for idx in frame_indices)


def test_adaptive_sampler_high_confidence():
    """AdaptiveSamplerの間隔調整ロジックテスト（高信頼度）"""
    sampler = AdaptiveSampler(base_interval=10.0, min_interval=1.0, max_interval=30.0)

    # 高信頼度（0.9以上）の場合、間隔が広がる
    interval = sampler.adjust_interval(0.95)

    assert interval > sampler.base_interval
    assert interval <= sampler.max_interval


def test_adaptive_sampler_low_confidence():
    """AdaptiveSamplerの間隔調整ロジックテスト（低信頼度）"""
    sampler = AdaptiveSampler(base_interval=10.0, min_interval=1.0, max_interval=30.0)

    # 低信頼度（0.5未満）の場合、間隔が狭まる
    interval = sampler.adjust_interval(0.3)

    assert interval == sampler.min_interval


def test_adaptive_sampler_normal_confidence():
    """AdaptiveSamplerの間隔調整ロジックテスト（通常信頼度）"""
    sampler = AdaptiveSampler(base_interval=10.0, min_interval=1.0, max_interval=30.0)

    # 通常信頼度（0.5-0.9）の場合、ベース間隔を使用
    interval = sampler.adjust_interval(0.7)

    assert interval == sampler.base_interval


def test_adaptive_sampler_boundary_values():
    """AdaptiveSamplerの境界値テスト"""
    sampler = AdaptiveSampler(base_interval=10.0, min_interval=1.0, max_interval=30.0)

    # 境界値のテスト
    interval_high = sampler.adjust_interval(0.9)  # >= 0.9なので間隔が広がる
    interval_low = sampler.adjust_interval(0.4)  # < 0.5なのでmin_interval
    interval_normal = sampler.adjust_interval(0.7)  # 0.5 <= x < 0.9なのでbase_interval
    interval_boundary = sampler.adjust_interval(0.5)  # 0.5は境界値（< 0.5ではない）なのでbase_interval

    assert interval_high > sampler.base_interval
    assert interval_low == sampler.min_interval
    assert interval_normal == sampler.base_interval
    assert interval_boundary == sampler.base_interval  # 0.5は< 0.5ではないのでbase_interval


def test_coarse_sampler_close():
    """CoarseSamplerのリソース解放テスト"""
    sampler = CoarseSampler("dummy_path.mov", interval_seconds=10.0)
    sampler.video = MagicMock()

    sampler.close()

    assert sampler.video is None


def test_fine_sampler_fps_validation(mock_video_capture):
    """FineSamplerのFPS検証テスト"""
    sampler = FineSampler(mock_video_capture, search_window=30.0)

    # FPSが0以下の場合のエラーハンドリング
    # getメソッドをモックして、FPS取得時に0.0を返すようにする
    def mock_get(prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return 0

    mock_video_capture.get = mock_get
    sampler.fps = None  # リセット

    with pytest.raises(RuntimeError):
        sampler._ensure_fps()
