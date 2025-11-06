"""OCRエンジン共通インターフェースモジュール

Tesseract、PaddleOCR、EasyOCRを統一的に扱うためのラッパー。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract

# MPS互換性設定を適用（警告抑制）
try:
    from src.utils.torch_utils import setup_mps_compatibility

    setup_mps_compatibility()
except ImportError:
    # インポートに失敗した場合はスキップ（循環インポートを避ける）
    pass

logger = logging.getLogger(__name__)

# PaddleOCRとEasyOCRはオプショナル
try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCRがインストールされていません。pip install paddleocr でインストールできます。")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCRがインストールされていません。pip install easyocr でインストールできます。")


# グローバルインスタンス（遅延初期化）
# None: 未初期化, False: 初期化失敗（再試行しない）, object: 初期化成功
_paddleocr_instance: Optional[object] = None
_easyocr_reader: Optional[object] = None


def _get_paddleocr_instance():
    """PaddleOCRインスタンスを取得（遅延初期化）"""
    global _paddleocr_instance
    if not PADDLEOCR_AVAILABLE:
        return None
    # 初期化失敗済みの場合は再試行しない
    if _paddleocr_instance is False:
        return None
    if _paddleocr_instance is None:
        try:
            # PaddleOCRのバージョンによって引数が異なる可能性があるため、
            # まず最小限の引数で試行
            try:
                _paddleocr_instance = PaddleOCR(lang="en")
            except (TypeError, ValueError):
                # 引数エラーの場合、use_angle_clsを追加して再試行
                try:
                    _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
                except (TypeError, ValueError) as e2:
                    # それでも失敗した場合は警告を出してFalseに設定（再試行を防ぐ）
                    logger.warning(
                        f"PaddleOCRの初期化に失敗（引数エラー）: {e2}。" "以降のフレームではPaddleOCRをスキップします。"
                    )
                    _paddleocr_instance = False
                    return None
        except Exception as e:
            # 初期化失敗時はFalseに設定して再試行を防ぐ
            logger.warning(f"PaddleOCRの初期化に失敗: {e}。" "以降のフレームではPaddleOCRをスキップします。")
            _paddleocr_instance = False
            return None
    return _paddleocr_instance


def _get_easyocr_reader():
    """EasyOCRリーダーを取得（遅延初期化）"""
    global _easyocr_reader
    if not EASYOCR_AVAILABLE:
        return None
    # 初期化失敗済みの場合は再試行しない
    if _easyocr_reader is False:
        return None
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            # 初期化失敗時はFalseに設定して再試行を防ぐ
            logger.warning(f"EasyOCRの初期化に失敗: {e}。" "以降のフレームではEasyOCRをスキップします。")
            _easyocr_reader = False
            return None
    return _easyocr_reader


def run_tesseract(
    image: np.ndarray,
    psm: int = 7,
    whitelist: str = "0123456789/:",
    lang: str = "eng",
    oem: int = 3,
    dpi: int = 300,
) -> Tuple[Optional[str], float]:
    """Tesseract OCRを実行

    Args:
        image: 入力画像（グレースケールまたは二値画像）
        psm: PSMモード（Page Segmentation Mode）
        whitelist: 認識する文字のホワイトリスト
        lang: 言語コード（例: "eng", "jpn+eng"）
        oem: OCR Engine Mode
        dpi: DPI設定（解像度の指定）

    Returns:
        (認識テキスト, 平均信頼度) のタプル。失敗時は (None, 0.0)
    """
    try:
        config_str = f"--psm {psm} --oem {oem} --dpi {dpi}"
        if whitelist:
            config_str += f" -c tessedit_char_whitelist={whitelist}"
        if lang:
            config_str += f" -l {lang}"

        # テキスト取得
        text = pytesseract.image_to_string(image, config=config_str).strip()

        # 信頼度取得
        data = pytesseract.image_to_data(
            image, config=config_str, output_type=pytesseract.Output.DICT
        )

        confidences = [int(c) for c in data["conf"] if c != "-1"]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text if text else None, avg_confidence / 100.0

    except Exception as e:
        logger.error(f"Tesseract OCRエラー: {e}")
        return None, 0.0


def run_paddleocr(image: np.ndarray) -> Tuple[Optional[str], float]:
    """PaddleOCRを実行

    Args:
        image: 入力画像（グレースケールまたはBGR）

    Returns:
        (認識テキスト, 平均信頼度) のタプル。失敗時は (None, 0.0)
    """
    if not PADDLEOCR_AVAILABLE:
        logger.warning("PaddleOCRが利用できません")
        return None, 0.0

    try:
        ocr = _get_paddleocr_instance()
        if ocr is None:
            return None, 0.0

        # PaddleOCRはBGR画像を期待
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        result = ocr.ocr(image_bgr, cls=True)

        if not result or not result[0]:
            return None, 0.0

        # テキストと信頼度を結合
        texts = []
        confidences = []
        for line in result[0]:
            if line:
                text, conf = line[1]
                texts.append(text)
                confidences.append(conf)

        if not texts:
            return None, 0.0

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    except Exception as e:
        logger.error(f"PaddleOCRエラー: {e}")
        return None, 0.0


def run_easyocr(image: np.ndarray) -> Tuple[Optional[str], float]:
    """EasyOCRを実行

    Args:
        image: 入力画像（グレースケールまたはBGR）

    Returns:
        (認識テキスト, 平均信頼度) のタプル。失敗時は (None, 0.0)
    """
    if not EASYOCR_AVAILABLE:
        logger.warning("EasyOCRが利用できません")
        return None, 0.0

    try:
        reader = _get_easyocr_reader()
        if reader is None:
            return None, 0.0

        # EasyOCRはBGR画像を期待
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        results = reader.readtext(image_bgr)

        if not results:
            return None, 0.0

        # テキストと信頼度を結合
        texts = []
        confidences = []
        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        if not texts:
            return None, 0.0

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    except Exception as e:
        logger.error(f"EasyOCRエラー: {e}")
        return None, 0.0


def run_ocr(
    image: np.ndarray, engine: str = "tesseract", **kwargs
) -> Tuple[Optional[str], float]:
    """統一OCRインターフェース

    Args:
        image: 入力画像
        engine: OCRエンジン名 ("tesseract", "paddleocr", "easyocr")
        **kwargs: エンジン固有のパラメータ

    Returns:
        (認識テキスト, 平均信頼度) のタプル
    """
    if engine == "tesseract":
        psm = kwargs.get("psm", 7)
        whitelist = kwargs.get("whitelist", "0123456789/:")
        lang = kwargs.get("lang", "eng")
        oem = kwargs.get("oem", 3)
        return run_tesseract(image, psm=psm, whitelist=whitelist, lang=lang, oem=oem)
    elif engine == "paddleocr":
        return run_paddleocr(image)
    elif engine == "easyocr":
        return run_easyocr(image)
    else:
        logger.warning(f"未知のOCRエンジン: {engine}。Tesseractを使用します。")
        return run_tesseract(image)
