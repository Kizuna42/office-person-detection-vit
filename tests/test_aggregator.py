"""Unit tests for Aggregator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pytest

from src.aggregator import Aggregator
from src.data_models import Detection


def _make_detection(zone_ids: Optional[Sequence[str]] = None) -> Detection:
    return Detection(
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(5.0, 10.0),
        zone_ids=list(zone_ids) if zone_ids is not None else [],
    )


def test_get_zone_counts_counts_all_zones():
    """ゾーンごとに人数をカウントし、未分類は `unclassified` で集計する。"""

    aggregator = Aggregator()
    detections = [
        _make_detection(["zone_a"]),
        _make_detection(["zone_a", "zone_b"]),
        _make_detection([]),
    ]

    counts = aggregator.get_zone_counts(detections)

    assert counts["zone_a"] == 2
    assert counts["zone_b"] == 1
    assert counts["unclassified"] == 1


def test_aggregate_frame_stores_results():
    """`aggregate_frame` で結果が `results` と `_zone_data` に保存される。"""

    aggregator = Aggregator()
    detections = [_make_detection(["zone_a"]), _make_detection(["zone_b"])]

    zone_counts = aggregator.aggregate_frame("12:00", detections)

    assert zone_counts == {"zone_a": 1, "zone_b": 1}
    assert len(aggregator.results) == 2
    assert aggregator._zone_data["zone_a"] == [1]
    assert aggregator._zone_data["zone_b"] == [1]


def test_export_csv(tmp_path: Path):
    """集計結果をCSVに出力できる。"""

    aggregator = Aggregator()
    detections = [_make_detection(["zone_a"])]
    aggregator.aggregate_frame("12:05", detections)

    output_path = tmp_path / "result.csv"
    aggregator.export_csv(str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "timestamp,zone_id,count" in content
    assert "12:05,zone_a,1" in content


def test_get_statistics_returns_summary():
    """ゾーンの平均・最大・最小・フレーム数を返す。"""

    aggregator = Aggregator()
    aggregator.aggregate_frame("12:00", [_make_detection(["zone_a"])] * 2)
    aggregator.aggregate_frame("12:05", [_make_detection(["zone_a"])] * 3)

    stats = aggregator.get_statistics()

    assert stats["zone_a"]["average"] == pytest.approx(2.5)
    assert stats["zone_a"]["max"] == 3
    assert stats["zone_a"]["min"] == 2
    assert stats["zone_a"]["total_frames"] == 2


def test_clear_resets_state():
    """`clear` で内部状態をリセットする。"""

    aggregator = Aggregator()
    aggregator.aggregate_frame("12:00", [_make_detection(["zone_a"])])
    aggregator.clear()

    assert aggregator.results == []
    assert len(aggregator._zone_data) == 0


