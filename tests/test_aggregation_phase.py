"""Unit tests for AggregationPhase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from src.aggregation import Aggregator
from src.config import ConfigManager
from src.models import Detection, FrameResult
from src.pipeline.phases import AggregationPhase

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_config(tmp_path: Path) -> ConfigManager:
    """テスト用のConfigManager"""
    config = ConfigManager("nonexistent_config.yaml")
    config.set(
        "zones",
        [
            {
                "id": "zone_a",
                "name": "Zone A",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "priority": 1,
            },
            {
                "id": "zone_b",
                "name": "Zone B",
                "polygon": [[100, 0], [200, 0], [200, 100], [100, 100]],
                "priority": 2,
            },
        ],
    )
    return config


@pytest.fixture
def sample_logger():
    """テスト用のロガー"""
    logger = logging.getLogger("test_aggregation_phase")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_detections() -> list[Detection]:
    """テスト用の検出結果"""
    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
            floor_coords=(50.0, 50.0),
            zone_ids=["zone_a"],
        ),
        Detection(
            bbox=(200.0, 300.0, 60.0, 120.0),
            confidence=0.8,
            class_id=1,
            class_name="person",
            camera_coords=(230.0, 420.0),
            floor_coords=(150.0, 50.0),
            zone_ids=["zone_b"],
        ),
    ]


@pytest.fixture
def sample_frame_results(sample_detections) -> list[FrameResult]:
    """テスト用のFrameResultリスト"""
    return [
        FrameResult(
            frame_number=0,
            timestamp="2025/08/26 16:05:00",
            detections=sample_detections,
            zone_counts={},
        ),
        FrameResult(
            frame_number=1,
            timestamp="2025/08/26 16:10:00",
            detections=sample_detections[:1],
            zone_counts={},
        ),
        FrameResult(
            frame_number=2,
            timestamp="2025/08/26 16:15:00",
            detections=[],
            zone_counts={},
        ),
    ]


def test_execute_success(sample_config, sample_logger, sample_frame_results, tmp_path):
    """executeが正しく動作する"""
    phase = AggregationPhase(sample_config, sample_logger)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    aggregator = phase.execute(sample_frame_results, output_path)

    assert isinstance(aggregator, Aggregator)
    assert (output_path / "zone_counts.csv").exists()

    # ゾーンカウントが正しく設定されていることを確認
    assert sample_frame_results[0].zone_counts["zone_a"] == 1
    assert sample_frame_results[0].zone_counts["zone_b"] == 1
    assert sample_frame_results[1].zone_counts["zone_a"] == 1
    assert sample_frame_results[2].zone_counts == {}


def test_execute_empty_results(sample_config, sample_logger, tmp_path):
    """空の結果でexecuteを呼ぶ"""
    phase = AggregationPhase(sample_config, sample_logger)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    aggregator = phase.execute([], output_path)

    assert isinstance(aggregator, Aggregator)
    assert (output_path / "zone_counts.csv").exists()


def test_execute_with_no_zones(sample_config, sample_logger, sample_frame_results, tmp_path):
    """ゾーン定義がない場合"""
    sample_config.set("zones", [])

    phase = AggregationPhase(sample_config, sample_logger)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    aggregator = phase.execute(sample_frame_results, output_path)

    assert isinstance(aggregator, Aggregator)
    assert (output_path / "zone_counts.csv").exists()


def test_execute_csv_output_format(sample_config, sample_logger, sample_frame_results, tmp_path):
    """CSV出力フォーマットが正しい"""
    phase = AggregationPhase(sample_config, sample_logger)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    phase.execute(sample_frame_results, output_path)

    csv_path = output_path / "zone_counts.csv"
    assert csv_path.exists()

    import csv

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    # CSVフォーマット: timestamp, zone_a, zone_b, unclassified
    assert "timestamp" in rows[0]
    assert "zone_a" in rows[0]
    assert "zone_b" in rows[0]
    # zone_aのデータが存在することを確認
    assert any(int(row["zone_a"]) > 0 for row in rows)


def test_execute_statistics(sample_config, sample_logger, sample_frame_results, tmp_path):
    """統計情報が正しく計算される"""
    phase = AggregationPhase(sample_config, sample_logger)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    aggregator = phase.execute(sample_frame_results, output_path)

    statistics = aggregator.get_statistics()
    assert "zone_a" in statistics
    assert "zone_b" in statistics
    assert statistics["zone_a"]["average"] > 0
    assert statistics["zone_b"]["average"] > 0
