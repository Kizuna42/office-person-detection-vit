"""Unit tests for ZoneClassifier."""

from __future__ import annotations

import pytest

from src.zone import ZoneClassifier


@pytest.fixture
def sample_zones():
    return [
        {
            "id": "zone_a",
            "name": "Zone A",
            "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "priority": 2,
        },
        {
            "id": "zone_b",
            "name": "Zone B",
            "polygon": [(50, 50), (150, 50), (150, 150), (50, 150)],
            "priority": 1,
        },
    ]


def test_invalid_zones_definition():
    """ゾーン定義がリストでない場合は ValueError。"""

    with pytest.raises(ValueError, match=r".*リスト.*"):
        ZoneClassifier({})


def test_classify_inside_multiple_zones(sample_zones):
    """デフォルトでは重複ゾーンをすべて返す。"""

    classifier = ZoneClassifier(sample_zones)
    zone_ids = classifier.classify((60, 60))
    assert set(zone_ids) == {"zone_a", "zone_b"}


def test_classify_without_overlap_returns_highest_priority(sample_zones):
    """allow_overlap=False の場合は優先順位が最も高いゾーンを返す。"""

    classifier = ZoneClassifier(sample_zones, allow_overlap=False)
    zone_ids = classifier.classify((60, 60))
    assert zone_ids == ["zone_b"]  # priority=1 が最優先


def test_classify_outside_returns_empty(sample_zones):
    """すべてのゾーン外なら空リスト。"""

    classifier = ZoneClassifier(sample_zones)
    assert classifier.classify((200, 200)) == []


def test_classify_with_unclassified(sample_zones):
    """`classify_with_unclassified` はゾーン外で `unclassified` を返す。"""

    classifier = ZoneClassifier(sample_zones)
    assert classifier.classify_with_unclassified((200, 200)) == ["unclassified"]


def test_get_zone_info(sample_zones):
    """`get_zone_info` で詳細情報を取得できる。"""

    classifier = ZoneClassifier(sample_zones)
    zone_info = classifier.get_zone_info("zone_a")
    assert zone_info is not None
    assert zone_info["name"] == "Zone A"
    assert classifier.get_zone_count() == 2
    assert set(classifier.get_all_zone_ids()) == {"zone_a", "zone_b"}
