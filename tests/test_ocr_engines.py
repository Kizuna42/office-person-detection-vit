"""Unit tests for OCR engines module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.timestamp.ocr_engines import (
    EASYOCR_AVAILABLE,
    PADDLEOCR_AVAILABLE,
    run_easyocr,
    run_ocr,
    run_paddleocr,
    run_tesseract,
)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """テスト用のグレースケール画像"""

    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """テスト用のBGR画像"""

    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@patch("src.timestamp.ocr_engines.pytesseract")
def test_run_tesseract_success(mock_pytesseract, sample_grayscale_image: np.ndarray):
    """Tesseract OCRが正常に実行できる。"""

    mock_pytesseract.image_to_string.return_value = "2023/04/01 12:34:56"
    mock_pytesseract.image_to_data.return_value = {
        "text": ["2023/04/01", "12:34:56"],
        "conf": ["90", "95"],
    }

    text, confidence = run_tesseract(sample_grayscale_image)

    assert text == "2023/04/01 12:34:56"
    assert confidence == pytest.approx(0.925)  # (90 + 95) / 2 / 100


@patch("src.timestamp.ocr_engines.pytesseract")
def test_run_tesseract_empty_text(mock_pytesseract, sample_grayscale_image: np.ndarray):
    """テキストが空の場合は None が返される。"""

    mock_pytesseract.image_to_string.return_value = ""
    mock_pytesseract.image_to_data.return_value = {
        "text": [],
        "conf": [],
    }

    text, confidence = run_tesseract(sample_grayscale_image)

    assert text is None
    assert confidence == 0.0


@patch("src.timestamp.ocr_engines.pytesseract")
def test_run_tesseract_with_params(mock_pytesseract, sample_grayscale_image: np.ndarray):
    """パラメータを指定してTesseract OCRを実行できる。"""

    mock_pytesseract.image_to_string.return_value = "12:34"
    mock_pytesseract.image_to_data.return_value = {
        "text": ["12:34"],
        "conf": ["80"],
    }

    text, confidence = run_tesseract(
        sample_grayscale_image,
        psm=8,
        whitelist="0123456789:",
        lang="eng",
        oem=3,
    )

    mock_pytesseract.image_to_string.assert_called_once()
    config_str = mock_pytesseract.image_to_string.call_args[1]["config"]
    assert "--psm 8" in config_str
    assert "tessedit_char_whitelist=0123456789:" in config_str


@patch("src.timestamp.ocr_engines.pytesseract")
def test_run_tesseract_exception(mock_pytesseract, sample_grayscale_image: np.ndarray):
    """例外が発生した場合は (None, 0.0) が返される。"""

    mock_pytesseract.image_to_string.side_effect = Exception("Tesseract error")

    text, confidence = run_tesseract(sample_grayscale_image)

    assert text is None
    assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_paddleocr_instance")
def test_run_paddleocr_success(mock_get_instance, sample_bgr_image: np.ndarray):
    """PaddleOCRが正常に実行できる。"""

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], ("2023/04/01 12:34:56", 0.95)),
        ]
    ]
    mock_get_instance.return_value = mock_ocr

    if PADDLEOCR_AVAILABLE:
        text, confidence = run_paddleocr(sample_bgr_image)

        assert text == "2023/04/01 12:34:56"
        assert confidence == pytest.approx(0.95)
    else:
        text, confidence = run_paddleocr(sample_bgr_image)
        assert text is None
        assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_paddleocr_instance")
def test_run_paddleocr_grayscale(mock_get_instance, sample_grayscale_image: np.ndarray):
    """グレースケール画像がBGRに変換される。"""

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], ("test", 0.9)),
        ]
    ]
    mock_get_instance.return_value = mock_ocr

    if PADDLEOCR_AVAILABLE:
        run_paddleocr(sample_grayscale_image)
        # BGR画像が渡されることを確認
        call_args = mock_ocr.ocr.call_args[0][0]
        assert len(call_args.shape) == 3
        assert call_args.shape[2] == 3


@patch("src.timestamp.ocr_engines._get_paddleocr_instance")
def test_run_paddleocr_empty_result(mock_get_instance, sample_bgr_image: np.ndarray):
    """結果が空の場合は None が返される。"""

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [[]]
    mock_get_instance.return_value = mock_ocr

    if PADDLEOCR_AVAILABLE:
        text, confidence = run_paddleocr(sample_bgr_image)

        assert text is None
        assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_paddleocr_instance")
def test_run_paddleocr_not_available(mock_get_instance, sample_bgr_image: np.ndarray):
    """PaddleOCRが利用できない場合は (None, 0.0) が返される。"""

    mock_get_instance.return_value = None

    text, confidence = run_paddleocr(sample_bgr_image)

    assert text is None
    assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_easyocr_reader")
def test_run_easyocr_success(mock_get_reader, sample_bgr_image: np.ndarray):
    """EasyOCRが正常に実行できる。"""

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 20], [0, 20]], "2023/04/01 12:34:56", 0.92),
    ]
    mock_get_reader.return_value = mock_reader

    if EASYOCR_AVAILABLE:
        text, confidence = run_easyocr(sample_bgr_image)

        assert text == "2023/04/01 12:34:56"
        assert confidence == pytest.approx(0.92)
    else:
        text, confidence = run_easyocr(sample_bgr_image)
        assert text is None
        assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_easyocr_reader")
def test_run_easyocr_grayscale(mock_get_reader, sample_grayscale_image: np.ndarray):
    """グレースケール画像がBGRに変換される。"""

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 20], [0, 20]], "test", 0.9),
    ]
    mock_get_reader.return_value = mock_reader

    if EASYOCR_AVAILABLE:
        run_easyocr(sample_grayscale_image)
        # BGR画像が渡されることを確認
        call_args = mock_reader.readtext.call_args[0][0]
        assert len(call_args.shape) == 3
        assert call_args.shape[2] == 3


@patch("src.timestamp.ocr_engines._get_easyocr_reader")
def test_run_easyocr_empty_result(mock_get_reader, sample_bgr_image: np.ndarray):
    """結果が空の場合は None が返される。"""

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []
    mock_get_reader.return_value = mock_reader

    if EASYOCR_AVAILABLE:
        text, confidence = run_easyocr(sample_bgr_image)

        assert text is None
        assert confidence == 0.0


@patch("src.timestamp.ocr_engines._get_easyocr_reader")
def test_run_easyocr_not_available(mock_get_reader, sample_bgr_image: np.ndarray):
    """EasyOCRが利用できない場合は (None, 0.0) が返される。"""

    mock_get_reader.return_value = None

    text, confidence = run_easyocr(sample_bgr_image)

    assert text is None
    assert confidence == 0.0


@patch("src.timestamp.ocr_engines.run_tesseract")
def test_run_ocr_tesseract(mock_run_tesseract, sample_grayscale_image: np.ndarray):
    """run_ocr で Tesseract を実行できる。"""

    mock_run_tesseract.return_value = ("2023/04/01 12:34:56", 0.9)

    text, confidence = run_ocr(sample_grayscale_image, engine="tesseract", psm=7)

    mock_run_tesseract.assert_called_once()
    assert text == "2023/04/01 12:34:56"
    assert confidence == 0.9


@patch("src.timestamp.ocr_engines.run_paddleocr")
def test_run_ocr_paddleocr(mock_run_paddleocr, sample_grayscale_image: np.ndarray):
    """run_ocr で PaddleOCR を実行できる。"""

    mock_run_paddleocr.return_value = ("2023/04/01 12:34:56", 0.92)

    text, confidence = run_ocr(sample_grayscale_image, engine="paddleocr")

    mock_run_paddleocr.assert_called_once()
    assert text == "2023/04/01 12:34:56"
    assert confidence == 0.92


@patch("src.timestamp.ocr_engines.run_easyocr")
def test_run_ocr_easyocr(mock_run_easyocr, sample_grayscale_image: np.ndarray):
    """run_ocr で EasyOCR を実行できる。"""

    mock_run_easyocr.return_value = ("2023/04/01 12:34:56", 0.91)

    text, confidence = run_ocr(sample_grayscale_image, engine="easyocr")

    mock_run_easyocr.assert_called_once()
    assert text == "2023/04/01 12:34:56"
    assert confidence == 0.91


@patch("src.timestamp.ocr_engines.run_tesseract")
def test_run_ocr_invalid_engine(mock_run_tesseract, sample_grayscale_image: np.ndarray):
    """不正なエンジン名の場合は Tesseract が使用される。"""

    mock_run_tesseract.return_value = ("2023/04/01 12:34:56", 0.9)

    text, confidence = run_ocr(sample_grayscale_image, engine="invalid_engine")

    mock_run_tesseract.assert_called_once()
    assert text == "2023/04/01 12:34:56"
