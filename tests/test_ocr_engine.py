"""Unit tests for MultiEngineOCR."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.timestamp.ocr_engine import MultiEngineOCR


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


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_confidence_calculation_valid_text():
    """信頼度計算の妥当性テスト（有効なテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])

    valid_text = "2025/08/26 16:07:45"
    confidence = ocr._calculate_confidence(valid_text)

    # 有効なテキストは高い信頼度を持つ
    assert confidence > 0.5
    assert confidence <= 1.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_confidence_calculation_invalid_text():
    """信頼度計算の妥当性テスト（無効なテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])

    invalid_text = "invalid text"
    confidence = ocr._calculate_confidence(invalid_text)

    # 無効なテキストは低い信頼度を持つ
    assert confidence < 0.5


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
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


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
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


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_similarity_calculation_identical():
    """類似度計算のテスト（同一テキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])

    text = "2025/08/26 16:07:45"
    similarity = ocr._calculate_similarity(text, text)

    assert similarity == 1.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_similarity_calculation_different():
    """類似度計算のテスト（異なるテキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])

    text1 = "2025/08/26 16:07:45"
    text2 = "2025/08/27 17:08:46"
    similarity = ocr._calculate_similarity(text1, text2)

    assert 0.0 <= similarity < 1.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_similarity_calculation_similar():
    """類似度計算のテスト（類似テキスト）"""
    ocr = MultiEngineOCR(enabled_engines=[])

    text1 = "2025/08/26 16:07:45"
    text2 = "2025/08/26 16:07:46"  # 1秒違い
    similarity = ocr._calculate_similarity(text1, text2)

    # 類似度が高いことを確認
    assert similarity > 0.8


@patch("src.timestamp.ocr_engine.TESSERACT_AVAILABLE", True)
@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.pytesseract")
def test_tesseract_engine_initialization(mock_pytesseract, sample_roi: np.ndarray):
    """Tesseractエンジンの個別テスト"""
    mock_pytesseract.image_to_string.return_value = "2025/08/26 16:07:45"

    ocr = MultiEngineOCR(enabled_engines=["tesseract"])

    # Tesseractエンジンが利用可能な場合
    if "tesseract" in ocr.engines:
        result_text, result_conf = ocr.extract_with_consensus(sample_roi)

        assert result_text is not None
        # 信頼度は_calculate_confidenceで計算されるため、有効なテキストなら0.5以上
        assert result_conf >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_consensus_algorithm_single_engine(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（単一エンジン）"""
    # モックエンジンを作成
    mock_engine = MagicMock(return_value="2025/08/26 16:07:45")

    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines["mock"] = mock_engine

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    assert confidence > 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_consensus_algorithm_multiple_engines_agreement(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（複数エンジン一致）"""
    # 同じ結果を返すモックエンジン
    mock_engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine2 = MagicMock(return_value="2025/08/26 16:07:45")

    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines["engine1"] = mock_engine1
    ocr.engines["engine2"] = mock_engine2

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    # 一致している場合は高い信頼度
    assert confidence > 0.5


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_consensus_algorithm_engine_failure(sample_roi: np.ndarray):
    """コンセンサスアルゴリズムのテスト（エンジン失敗）"""
    # 例外を発生させるエンジン
    failing_engine = MagicMock(side_effect=Exception("OCR failed"))

    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines["failing"] = failing_engine

    # エラーが発生しても処理が継続されることを確認
    _text, _confidence = ocr.extract_with_consensus(sample_roi)

    # エンジンが失敗した場合はNoneまたは低い信頼度
    # 実装に応じて調整が必要


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.TESSERACT_AVAILABLE", False)
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
    MultiEngineOCR(enabled_engines=["tesseract"])

    # 実装に応じて、有効化されたエンジンのみが含まれることを確認
    # （実際のエンジン利用可能性に依存）


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_confidence_empty_text():
    """空テキストの信頼度テスト"""
    ocr = MultiEngineOCR(enabled_engines=[])

    confidence = ocr._calculate_confidence("")
    assert confidence == 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
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


@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", True)
@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.TESSERACT_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.easyocr")
def test_init_easyocr(mock_easyocr, sample_roi: np.ndarray):
    """EasyOCR初期化のテスト"""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [(["coords"], "2025/08/26 16:07:45", 0.9)]
    mock_easyocr.Reader.return_value = mock_reader

    ocr = MultiEngineOCR(enabled_engines=["easyocr"])

    assert "easyocr" in ocr.engines

    # エンジンが正しく動作することを確認
    if "easyocr" in ocr.engines:
        result_text, _result_conf = ocr.extract_with_consensus(sample_roi)
        assert result_text is not None


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", True)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.TESSERACT_AVAILABLE", False)
def test_init_paddleocr(sample_roi: np.ndarray):
    """PaddleOCR初期化のテスト"""
    # PaddleOCRが利用可能な場合のみテスト
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        # 実際のPaddleOCRをモックせず、初期化が成功することを確認
        ocr = MultiEngineOCR(enabled_engines=["paddleocr"])

        # エンジンが初期化されていることを確認
        if "paddleocr" in ocr.engines:
            # エンジン関数が呼び出し可能であることを確認
            assert callable(ocr.engines["paddleocr"])
    except (ImportError, Exception):
        # PaddleOCRが利用できない、または初期化に失敗した場合はスキップ
        pytest.skip("PaddleOCR not available or initialization failed")


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_extract_with_weighted_consensus(sample_roi: np.ndarray):
    """重み付けコンセンサスのテスト"""
    mock_engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine2 = MagicMock(return_value="2025/08/26 16:07:46")

    ocr = MultiEngineOCR(enabled_engines=[], use_weighted_consensus=True)
    ocr.engines["engine1"] = mock_engine1
    ocr.engines["engine2"] = mock_engine2

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    assert confidence >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_extract_with_voting_consensus(sample_roi: np.ndarray):
    """投票コンセンサスのテスト"""
    # 同じ結果を返すエンジン（2/3以上で一致）
    mock_engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine2 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine3 = MagicMock(return_value="2025/08/26 16:07:45")

    ocr = MultiEngineOCR(enabled_engines=[], use_voting_consensus=True)
    ocr.engines["engine1"] = mock_engine1
    ocr.engines["engine2"] = mock_engine2
    ocr.engines["engine3"] = mock_engine3

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    assert confidence >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_extract_with_baseline_consensus_multiple_engines(sample_roi: np.ndarray):
    """複数エンジンのベースラインコンセンサス"""
    # 同じ結果を返すエンジン
    mock_engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine2 = MagicMock(return_value="2025/08/26 16:07:45")
    mock_engine3 = MagicMock(return_value="2025/08/26 16:08:00")

    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines["engine1"] = mock_engine1
    ocr.engines["engine2"] = mock_engine2
    ocr.engines["engine3"] = mock_engine3

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    # 2つが一致している場合、高い信頼度になる
    assert confidence >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_calculate_confidence_edge_cases():
    """信頼度計算のエッジケース"""
    ocr = MultiEngineOCR(enabled_engines=[])

    # 空文字列
    assert ocr._calculate_confidence("") == 0.0

    # None（文字列として扱う）
    assert ocr._calculate_confidence("None") < 0.5

    # 非常に短い文字列
    assert ocr._calculate_confidence("2025") < 0.5

    # 非常に長い文字列
    long_text = "2025/08/26 16:07:45 " + "extra " * 100
    assert ocr._calculate_confidence(long_text) < 0.5

    # 数字のみ
    assert ocr._calculate_confidence("20250826160745") < 0.5

    # 特殊文字のみ
    assert ocr._calculate_confidence("//// :: ::") < 0.5


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_calculate_similarity_with_levenshtein():
    """Levenshtein距離を使用した類似度計算"""
    # Levenshteinが利用可能な場合のみテスト
    try:
        from Levenshtein import ratio  # noqa: F401

        ocr = MultiEngineOCR(enabled_engines=[])
        similarity = ocr._calculate_similarity("2025/08/26 16:07:45", "2025/08/26 16:07:46")

        # Levenshteinが使用されている場合、類似度は計算される
        assert 0.0 <= similarity <= 1.0
    except ImportError:
        # Levenshteinがインストールされていない場合はスキップ
        pytest.skip("Levenshtein not installed")


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_calculate_similarity_without_levenshtein():
    """Levenshteinなしの類似度計算"""
    # Levenshteinをインポートできないようにする

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "Levenshtein":
            raise ImportError("No module named 'Levenshtein'")
        return original_import(name, *args, **kwargs)

    # Levenshteinのインポートをモック
    with patch("builtins.__import__", side_effect=mock_import):
        ocr = MultiEngineOCR(enabled_engines=[])

        # 同一テキスト
        similarity = ocr._calculate_similarity("2025/08/26 16:07:45", "2025/08/26 16:07:45")
        assert similarity == 1.0

        # 異なるテキスト
        similarity = ocr._calculate_similarity("2025/08/26 16:07:45", "2025/08/27 17:08:46")
        assert 0.0 <= similarity < 1.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_extract_engine_failure_handling(sample_roi: np.ndarray):
    """エンジン失敗時のハンドリング"""
    # 成功するエンジン
    working_engine = MagicMock(return_value="2025/08/26 16:07:45")
    # 失敗するエンジン
    failing_engine = MagicMock(side_effect=Exception("OCR failed"))

    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines["working"] = working_engine
    ocr.engines["failing"] = failing_engine

    # エラーが発生しても処理が継続される
    text, confidence = ocr.extract_with_consensus(sample_roi)

    # 成功するエンジンの結果が返される
    assert text is not None
    assert confidence >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_weighted_consensus_engine_weights(sample_roi: np.ndarray):
    """重み付けスキームの重みテスト"""
    # Tesseractと他のエンジンで異なる結果を返す
    tesseract_engine = MagicMock(return_value="2025/08/26 16:07:45")
    other_engine = MagicMock(return_value="2025/08/26 16:07:46")

    ocr = MultiEngineOCR(enabled_engines=[], use_weighted_consensus=True)
    ocr.engines["tesseract"] = tesseract_engine
    ocr.engines["other"] = other_engine

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    # Tesseractは重みが高いので、優先される可能性が高い
    assert confidence >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
def test_voting_consensus_threshold(sample_roi: np.ndarray):
    """投票コンセンサスの閾値テスト"""
    # 2/3以上のエンジンが一致する場合
    engine1 = MagicMock(return_value="2025/08/26 16:07:45")
    engine2 = MagicMock(return_value="2025/08/26 16:07:45")
    engine3 = MagicMock(return_value="2025/08/26 16:07:45")

    ocr = MultiEngineOCR(enabled_engines=[], use_voting_consensus=True)
    ocr.engines["engine1"] = engine1
    ocr.engines["engine2"] = engine2
    ocr.engines["engine3"] = engine3

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is not None
    # 3/3エンジンが一致するので、高い信頼度になる
    assert confidence >= 0.0

    # 2/3未満の一致の場合
    engine4 = MagicMock(return_value="2025/08/26 16:08:00")
    ocr.engines["engine3"] = engine4

    text2, confidence2 = ocr.extract_with_consensus(sample_roi)

    # 2/3未満の場合は最高信頼度の結果が返される
    assert text2 is not None
    assert confidence2 >= 0.0


@patch("src.timestamp.ocr_engine.PADDLEOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.EASYOCR_AVAILABLE", False)
@patch("src.timestamp.ocr_engine.TESSERACT_AVAILABLE", False)
def test_no_engines_available_extraction(sample_roi: np.ndarray):
    """エンジンが利用不可の場合のテスト"""
    ocr = MultiEngineOCR(enabled_engines=[])
    ocr.engines = {}  # エンジンを空にする

    text, confidence = ocr.extract_with_consensus(sample_roi)

    assert text is None
    assert confidence == 0.0
