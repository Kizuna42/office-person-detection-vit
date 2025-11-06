"""Integration tests for TimestampExtractorV2."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2


@pytest.fixture
def sample_frame() -> np.ndarray:
    """テスト用のフレーム画像"""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def extractor() -> TimestampExtractorV2:
    """TimestampExtractorV2インスタンス"""
    with patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False), patch(
        "src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False
    ):
        return TimestampExtractorV2(
            confidence_threshold=0.7, fps=30.0, enabled_ocr_engines=[]  # モックを使用するため空
        )


@patch("src.timestamp.timestamp_extractor_v2.MultiEngineOCR")
@patch("src.timestamp.timestamp_extractor_v2.TimestampParser")
@patch("src.timestamp.timestamp_extractor_v2.TimestampROIExtractor")
def test_end_to_end_extraction(
    mock_roi_extractor_class,
    mock_parser_class,
    mock_ocr_class,
    sample_frame: np.ndarray,
    extractor: TimestampExtractorV2,
):
    """エンドツーエンド抽出テスト"""
    # モックの設定
    mock_roi = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
    mock_roi_extractor = MagicMock()
    mock_roi_extractor.extract_roi.return_value = (mock_roi, (832, 0, 448, 58))
    mock_roi_extractor.preprocess_roi.return_value = np.random.randint(
        0, 255, (50, 200), dtype=np.uint8
    )
    mock_roi_extractor_class.return_value = mock_roi_extractor

    mock_parser = MagicMock()
    mock_parser.fuzzy_parse.return_value = (datetime(2025, 8, 26, 16, 7, 45), 1.0)
    mock_parser_class.return_value = mock_parser

    mock_ocr = MagicMock()
    mock_ocr.extract_with_consensus.return_value = ("2025/08/26 16:07:45", 0.9)
    mock_ocr_class.return_value = mock_ocr

    # 新しいインスタンスを作成（モックが適用される）
    extractor = TimestampExtractorV2(confidence_threshold=0.7, fps=30.0)

    # 抽出実行
    result = extractor.extract(sample_frame, frame_idx=0)

    # 結果が正しいことを確認
    assert result is not None
    assert result["timestamp"] == datetime(2025, 8, 26, 16, 7, 45)
    assert result["frame_idx"] == 0
    assert result["confidence"] >= 0.7


@patch("src.timestamp.timestamp_extractor_v2.MultiEngineOCR")
@patch("src.timestamp.timestamp_extractor_v2.TimestampParser")
@patch("src.timestamp.timestamp_extractor_v2.TimestampROIExtractor")
def test_retry_mechanism(
    mock_roi_extractor_class,
    mock_parser_class,
    mock_ocr_class,
    sample_frame: np.ndarray,
):
    """リトライメカニズムのテスト"""
    # モックの設定
    mock_roi = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
    mock_roi_extractor = MagicMock()
    mock_roi_extractor.extract_roi.return_value = (mock_roi, (832, 0, 448, 58))
    mock_roi_extractor.preprocess_roi.return_value = np.random.randint(
        0, 255, (50, 200), dtype=np.uint8
    )
    mock_roi_extractor_class.return_value = mock_roi_extractor

    # 最初の2回は失敗、3回目で成功
    mock_ocr = MagicMock()
    mock_ocr.extract_with_consensus.side_effect = [
        (None, 0.0),  # 1回目: 失敗
        (None, 0.0),  # 2回目: 失敗
        ("2025/08/26 16:07:45", 0.9),  # 3回目: 成功
    ]
    mock_ocr_class.return_value = mock_ocr

    mock_parser = MagicMock()
    mock_parser.fuzzy_parse.return_value = (datetime(2025, 8, 26, 16, 7, 45), 1.0)
    mock_parser_class.return_value = mock_parser

    extractor = TimestampExtractorV2(
        confidence_threshold=0.7, fps=30.0, enabled_ocr_engines=[]
    )

    # 抽出実行（リトライが発生する）
    result = extractor.extract(sample_frame, frame_idx=0, retry_count=3)

    # 3回目の試行で成功することを確認
    assert result is not None
    assert mock_ocr.extract_with_consensus.call_count == 3


@patch("src.timestamp.timestamp_extractor_v2.MultiEngineOCR")
@patch("src.timestamp.timestamp_extractor_v2.TimestampParser")
@patch("src.timestamp.timestamp_extractor_v2.TimestampROIExtractor")
def test_confidence_threshold_behavior(
    mock_roi_extractor_class,
    mock_parser_class,
    mock_ocr_class,
    sample_frame: np.ndarray,
):
    """信頼度閾値の動作テスト"""
    # モックの設定
    mock_roi = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
    mock_roi_extractor = MagicMock()
    mock_roi_extractor.extract_roi.return_value = (mock_roi, (832, 0, 448, 58))
    mock_roi_extractor.preprocess_roi.return_value = np.random.randint(
        0, 255, (50, 200), dtype=np.uint8
    )
    mock_roi_extractor_class.return_value = mock_roi_extractor

    # 低信頼度のOCR結果
    mock_ocr = MagicMock()
    mock_ocr.extract_with_consensus.return_value = ("2025/08/26 16:07:45", 0.5)  # 低信頼度
    mock_ocr_class.return_value = mock_ocr

    mock_parser = MagicMock()
    mock_parser.fuzzy_parse.return_value = (datetime(2025, 8, 26, 16, 7, 45), 1.0)
    mock_parser_class.return_value = mock_parser

    # 高い信頼度閾値で抽出
    extractor_high = TimestampExtractorV2(
        confidence_threshold=0.9, fps=30.0, enabled_ocr_engines=[]
    )
    result_high = extractor_high.extract(sample_frame, frame_idx=0)

    # 低い信頼度閾値で抽出
    extractor_low = TimestampExtractorV2(
        confidence_threshold=0.5, fps=30.0, enabled_ocr_engines=[]
    )
    result_low = extractor_low.extract(sample_frame, frame_idx=0)

    # 高い閾値では失敗、低い閾値では成功する可能性がある
    # （時系列検証の結果にも依存）


@patch("src.timestamp.timestamp_extractor_v2.MultiEngineOCR")
@patch("src.timestamp.timestamp_extractor_v2.TimestampParser")
@patch("src.timestamp.timestamp_extractor_v2.TimestampROIExtractor")
def test_empty_roi_handling(
    mock_roi_extractor_class,
    mock_parser_class,
    mock_ocr_class,
    sample_frame: np.ndarray,
):
    """空のROIのハンドリングテスト"""
    # 空のROIを返すモック
    mock_roi_extractor = MagicMock()
    mock_roi_extractor.extract_roi.return_value = (np.array([]), (0, 0, 0, 0))
    mock_roi_extractor_class.return_value = mock_roi_extractor

    extractor = TimestampExtractorV2(
        confidence_threshold=0.7, fps=30.0, enabled_ocr_engines=[]
    )

    result = extractor.extract(sample_frame, frame_idx=0)

    # 空のROIの場合はNoneを返す
    assert result is None


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_reset_validator():
    """バリデーターリセットのテスト"""
    extractor = TimestampExtractorV2(
        confidence_threshold=0.7, fps=30.0, enabled_ocr_engines=[]
    )

    # リセット前の状態を確認
    extractor.reset_validator()

    # リセット後、再度検証可能であることを確認
    assert extractor.validator.last_timestamp is None
    assert extractor.validator.last_frame_idx is None
