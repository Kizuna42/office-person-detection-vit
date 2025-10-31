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


def test_find_closest_frame_handles_midnight_wrap():
    sampler = FrameSampler(interval_minutes=5, tolerance_seconds=120)
    frames = {100: "23:58", 200: "00:02"}
    assert sampler.find_closest_frame("00:00", frames) == 200


