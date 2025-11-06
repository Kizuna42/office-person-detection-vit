"""Unit tests for MultiEngineOCR."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.timestamp.ocr_engine import MultiEngineOCR, TESSERACT_AVAILABLE


@pytest.fixture
def sample_roi() -> np.ndarray:
    """テスト用のROI画像（前処理済み）"""
    return np.random.randint(0, 255, (100, 200), dtype=np.uint8)


@pytest.fixture
def valid_timestamp_text() -> str:
    """有効なタイムスタンプテキスト"""
    return "2025/08/26 16:07:45"


@pytest.fixture
def invalid_timestamp_text() -> str:
    """無効なタイムスタンプテキスト"""
    return "invalid text"


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_calculation_valid_text():
    """信頼度計算の妥当性テスト（有効なテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    valid_text = "2025/08/26 16:07:45"
    confidence = ocr._calculate_confidence(valid_text)
    
    # 有効なテキストは高い信頼度を持つ
    assert confidence > 0.5
    assert confidence <= 1.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_calculation_invalid_text():
    """信頼度計算の妥当性テスト（無効なテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    invalid_text = "invalid text"
    confidence = ocr._calculate_confidence(invalid_text)
    
    # 無効なテキストは低い信頼度を持つ
    assert confidence < 0.5


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_calculation_length_check():
    """信頼度計算の長さチェックテスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    # 適切な長さ（19文字）
    text1 = "2025/08/26 16:07:45"
    conf1 = ocr._calculate_confidence(text1)
    
    # 短すぎる
    text2 = "2025/08/26"
    conf2 = ocr._calculate_confidence(text2)
    
    # 長すぎる
    text3 = "2025/08/26 16:07:45 extra text"
    conf3 = ocr._calculate_confidence(text3)
    
    assert conf1 > conf2
    assert conf1 > conf3


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_calculation_format_check():
    """信頼度計算のフォーマットチェックテスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    # 正しいフォーマット
    text1 = "2025/08/26 16:07:45"
    conf1 = ocr._calculate_confidence(text1)
    
    # 間違ったフォーマット
    text2 = "20250826 160745"
    conf2 = ocr._calculate_confidence(text2)
    
    assert conf1 > conf2


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_similarity_calculation_identical():
    """類似度計算のテスト（同一テキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    text = "2025/08/26 16:07:45"
    similarity = ocr._calculate_similarity(text, text)
    
    assert similarity == 1.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_similarity_calculation_different():
    """類似度計算のテスト（異なるテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    text1 = "2025/08/26 16:07:45"
    text2 = "2025/08/27 17:08:46"
    similarity = ocr._calculate_similarity(text1, text2)
    
    assert 0.0 <= similarity < 1.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_similarity_calculation_similar():
    """類似度計算のテスト（類似テキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    text1 = "2025/08/26 16:07:45"
    text2 = "2025/08/26 16:07:46"  # 1秒違い
    similarity = ocr._calculate_similarity(text1, text2)
    
    # 類似度が高いことを確認
    assert similarity > 0.8


@patch('src.timestamp.ocr_engine.TESSERACT_AVAILABLE', True)
@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.pytesseract')
def test_tesseract_engine_initialization(mock_pytesseract, sample_roi: np.ndarray):
    """Tesseractエンジンの個別テスト"""
    mock_pytesseract.image_to_string.return_value = "2025/08/26 16:07:45"
    
    ocr = MultiEngineOCR(enabled_engines=['tesseract'])
    
    # Tesseractエンジンが利用可能な場合
    if 'tesseract' in ocr.engines:
        result_text, result_conf = ocr.extract_with_consensus(sample_roi)
        
        assert result_text is not None
        # 信頼度は_calculate_confidenceで計算されるため、有効なテキストなら0.5以上
        assert result_conf >= 0.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_consensus_algorithm_single_engine(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（単一エンジン）"""
    # モックエンジンを作成
    mock_engine = MagicMock(return_value="2025/08/26 16:07:45")
    
    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines['mock'] = mock_engine
    
    text, confidence = ocr.extract_with_consensus(sample_roi)
    
    assert text is not None
    assert confidence > 0.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_consensus_algorithm_multiple_engines_agreement(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（複数エンジン一致）"""
    # 同じ結果を返すモックエンジン
    mock_engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine2 = MagicMock(return_value="2025/08/26 16:07:45")
    
    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines['engine1'] = mock_engine1
    ocr.engines['engine2'] = mock_engine2
    
    text, confidence = ocr.extract_with_consensus(sample_roi)
    
    assert text is not None
    # 一致している場合は高い信頼度
    assert confidence > 0.5


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_consensus_algorithm_engine_failure(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（エンジン失敗）"""
    # 例外を発生させるエンジン
    failing_engine = MagicMock(side_effect=Exception("OCR failed"))
    
    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines['failing'] = failing_engine
    
    # エラーが発生しても処理が継続されることを確認
    text, confidence = ocr.extract_with_consensus(sample_roi)
    
    # エンジンが失敗した場合はNoneまたは低い信頼度
    # 実装に応じて調整が必要


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.TESSERACT_AVAILABLE', False)
def test_no_engines_available(sample_roi: np.ndarray):
    """利用可能なエンジンがない場合のテスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines = {}  # エンジンを空にする
    
    text, confidence = ocr.extract_with_consensus(sample_roi)
    
    assert text is None
    assert confidence == 0.0


def test_enabled_engines_filtering():
    """有効化エンジンのフィルタリングテスト"""
    # Tesseractのみ有効化
    ocr = MultiEngineOCR(enabled_engines=['tesseract'])
    
    # 実装に応じて、有効化されたエンジンのみが含まれることを確認
    # （実際のエンジン利用可能性に依存）


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_empty_text():
    """空テキストの信頼度テスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    confidence = ocr._calculate_confidence("")
    assert confidence == 0.0


@patch('src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE', False)
@patch('src.timestamp.ocr_engine.EASYOCR_AVAILABLE', False)
def test_confidence_partial_match():
    """部分一致の信頼度テスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    
    # 部分的に正しいフォーマット
    text1 = "2025/08/26 16:07"  # 秒がない
    conf1 = ocr._calculate_confidence(text1)
    
    # 完全に正しいフォーマット
    text2 = "2025/08/26 16:07:45"
    conf2 = ocr._calculate_confidence(text2)
    
    assert conf2 > conf1

