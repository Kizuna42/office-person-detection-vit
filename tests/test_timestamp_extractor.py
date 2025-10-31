import numpy as np
import pytest
import pytesseract

from src.timestamp_extractor import TimestampExtractor


def test_parse_timestamp_variants():
    extractor = TimestampExtractor()
    assert extractor.parse_timestamp("2023/04/01 12:34") == "2023/04/01 12:34:00"
    assert extractor.parse_timestamp("20230401123456") == "2023/04/01 12:34:56"
    assert extractor.parse_timestamp("  2023-04-01   07:05  ") == "2023/04/01 07:05:00"
    assert extractor.parse_timestamp("invalid") is None


def test_extract_saves_debug_outputs(tmp_path, monkeypatch):
    extractor = TimestampExtractor(roi=(0, 0, 10, 10))
    extractor.enable_debug(tmp_path)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[0:10, 0:10] = 255

    monkeypatch.setattr(
        pytesseract,
        "image_to_string",
        lambda image, config=None: "2023/04/01 12:34",
    )

    timestamp = extractor.extract(frame, frame_index=5)

    assert timestamp == "2023/04/01 12:34:00"
    assert (tmp_path / "frame_000005_roi.png").exists()
    assert (tmp_path / "frame_000005_preprocessed.png").exists()
    assert (tmp_path / "frame_000005_overlay.png").exists()


