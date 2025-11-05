from datetime import datetime

from src.frame_sampler import FrameSampler


def test_find_target_timestamps_basic():
    sampler = FrameSampler(interval_minutes=5)
    result = sampler.find_target_timestamps("12:00", "12:20")
    assert result == ["12:05", "12:10", "12:15", "12:20"]


def test_find_closest_frame_within_tolerance():
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=60)
    frames = {10: "12:05", 50: "12:06"}
    assert sampler.find_closest_frame("12:05", frames) == 10


def test_find_closest_frame_outside_tolerance():
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=5)
    frames = {10: "12:05", 20: "12:15"}
    assert sampler.find_closest_frame("12:00", frames) is None


def test_find_closest_frame_progressive_search():
    """漸増探索のテスト"""
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=10)

    # ±10秒以内にないが、±15秒以内にあるフレーム
    target = datetime(2025, 8, 26, 16, 10, 0)  # 16:10:00
    frames = {
        10: datetime(2025, 8, 26, 16, 10, 12),  # 16:10:12 (12秒差、±10秒外、±15秒内)
        20: datetime(2025, 8, 26, 16, 10, 25),  # 16:10:25 (25秒差)
    }

    # 漸増探索でフレーム10が見つかるはず
    result = sampler.find_closest_frame(target, frames)
    assert result == 10  # ±15秒でマッチ


def test_find_closest_frame_progressive_search_30s():
    """±30秒までの漸増探索のテスト"""
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=10)

    target = datetime(2025, 8, 26, 16, 10, 0)
    frames = {
        10: datetime(2025, 8, 26, 16, 10, 25),  # 25秒差（±15秒外、±30秒内）
    }

    result = sampler.find_closest_frame(target, frames)
    assert result == 10  # ±30秒でマッチ


def test_find_closest_frame_handles_midnight_wrap():
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=120)
    frames = {100: "23:58", 200: "00:02"}
    assert sampler.find_closest_frame("00:00", frames) == 200
