from datetime import datetime

import numpy as np
import pytesseract

from src.timestamp_extractor import TimestampExtractor


def test_parse_timestamp_variants():
    extractor = TimestampExtractor()
    assert extractor.parse_timestamp("2023/04/01 12:34") == "2023/04/01 12:34:00"
    assert extractor.parse_timestamp("20230401123456") == "2023/04/01 12:34:56"
    assert extractor.parse_timestamp("  2023-04-01   07:05  ") == "2023/04/01 07:05:00"
    assert extractor.parse_timestamp("invalid") is None


def test_parse_strict_regex():
    """厳密な正規表現抽出のテスト"""
    extractor = TimestampExtractor()

    # 厳密なパターン
    assert extractor._parse_strict_regex("2023/04/01 12:34:56") == "2023/04/01 12:34:56"
    assert (
        extractor._parse_strict_regex("foo 2025/08/26 16:04:16 bar")
        == "2025/08/26 16:04:16"
    )

    # スペース欠落の補正
    assert extractor._parse_strict_regex("2025/08/270:13:31") == "2025/08/27 00:13:31"
    assert extractor._parse_strict_regex("2025/08/27 0:13:31") == "2025/08/27 00:13:31"

    # 1桁時のゼロ埋め
    assert extractor._parse_strict_regex("2025/08/27 5:13:31") == "2025/08/27 05:13:31"

    # 失敗ケース
    assert extractor._parse_strict_regex("invalid") is None
    assert extractor._parse_strict_regex("2023/04/01") is None  # 時刻なし


def test_parse_timestamp_space_missing():
    """スペース欠落補正のテスト"""
    extractor = TimestampExtractor()

    # スペース欠落（2025/08/270:13:31形式）
    result = extractor.parse_timestamp("2025/08/270:13:31")
    assert result is not None
    assert result.startswith("2025/08/27")


def test_parse_timestamp_single_digit_hour():
    """1桁時のゼロ埋めテスト"""
    extractor = TimestampExtractor()

    # 1桁時
    result = extractor.parse_timestamp("2025/08/27 5:13:31")
    assert result == "2025/08/27 05:13:31"

    result = extractor.parse_timestamp("2025/08/27 0:13:31")
    assert result == "2025/08/27 00:13:31"


def test_parse_timestamp_year_corruption():
    """年の桁化け補正テスト"""
    extractor = TimestampExtractor()

    # 最初のタイムスタンプを設定（履歴として使用）
    extractor._last_timestamp = datetime(2025, 8, 26, 16, 4, 16)

    # 年の桁化け（0257など）
    result = extractor.parse_timestamp("0257/08/26 16:04:16")
    assert result is not None
    # 履歴から補正されるはず
    assert "2025" in result or result is None  # 補正が適用されるか、または破棄される


def test_get_last_corrections():
    """補正ログ取得のテスト"""
    extractor = TimestampExtractor()

    # 補正ログは空から始まる
    assert extractor.get_last_corrections() == []

    # タイムスタンプをパースすると補正ログが記録される可能性がある
    extractor._last_timestamp = datetime(2025, 8, 26, 16, 4, 16)
    extractor.parse_timestamp("2025/08/26 16:04:16")
    # 補正が行われた場合のみログが記録される


def test_extract_saves_debug_outputs(tmp_path, monkeypatch):
    extractor = TimestampExtractor(roi=(0, 0, 10, 10))
    extractor.enable_debug(tmp_path)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[0:10, 0:10] = 255

    monkeypatch.setattr(
        pytesseract,
        "image_to_string",
        lambda image, config=None: "2023/04/01 12:34:56",
    )
    monkeypatch.setattr(
        pytesseract,
        "image_to_data",
        lambda image, config=None, output_type=None: {
            "text": ["2023/04/01", "12:34:56"],
            "conf": ["90", "95"],
        },
    )

    timestamp = extractor.extract(frame, frame_index=5)

    assert timestamp == "2023/04/01 12:34:56"
    assert (tmp_path / "frame_000005_roi.png").exists()
    assert (tmp_path / "frame_000005_preprocessed.png").exists()
    assert (tmp_path / "frame_000005_overlay.png").exists()
