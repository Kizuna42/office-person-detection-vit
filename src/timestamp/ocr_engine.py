"""Multi-engine OCR wrapper for timestamp extraction."""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# OCRエンジンの利用可能性をチェック
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract is not available")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    logger.debug("easyocr is not available")

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    logger.debug("paddleocr is not available")


class MultiEngineOCR:
    """複数のOCRエンジンを使用して信頼性を向上

    複数のOCRエンジン（Tesseract, EasyOCR, PaddleOCR）を統合し、
    コンセンサスアルゴリズムで最も信頼性の高い結果を返します。
    """

    def __init__(self, enabled_engines: List[str] = None):
        """MultiEngineOCRを初期化

        Args:
            enabled_engines: 有効にするエンジンのリスト
                            Noneの場合は利用可能な全エンジンを使用
        """
        self.engines: Dict[str, callable] = {}
        self.enabled_engines = enabled_engines or []

        # 利用可能なエンジンを初期化
        if TESSERACT_AVAILABLE and (
            not enabled_engines or "tesseract" in enabled_engines
        ):
            self.engines["tesseract"] = self._init_tesseract()

        if EASYOCR_AVAILABLE and (not enabled_engines or "easyocr" in enabled_engines):
            self.engines["easyocr"] = self._init_easyocr()

        if PADDLEOCR_AVAILABLE and (
            not enabled_engines or "paddleocr" in enabled_engines
        ):
            self.engines["paddleocr"] = self._init_paddleocr()

        if not self.engines:
            logger.warning(
                "No OCR engines available. Please install at least one OCR engine."
            )

    def _init_tesseract(self) -> callable:
        """Tesseract: 高速、数字に強い"""
        import pytesseract

        # PSM 8: 単一の単語（最適化テストの結果、PSM 8が最も正確）
        config = r"--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789/: "

        def tesseract_func(img: np.ndarray) -> str:
            return pytesseract.image_to_string(img, config=config)

        return tesseract_func

    def _init_easyocr(self) -> callable:
        """EasyOCR: 高精度、やや遅い"""
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)  # GPU利用は環境に応じて調整

        def easyocr_func(img: np.ndarray) -> str:
            results = reader.readtext(img)
            return " ".join([r[1] for r in results])

        return easyocr_func

    def _init_paddleocr(self) -> callable:
        """PaddleOCR: 中国語カメラでも対応"""
        from paddleocr import PaddleOCR

        # 新しいバージョンではuse_gpuの代わりにdeviceを使用
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="japan", device="cpu")
        except (ValueError, TypeError):
            # 古いバージョンとの互換性
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang="japan", use_gpu=False)
            except Exception:
                # パラメータなしで初期化を試みる
                ocr = PaddleOCR(lang="japan")

        def paddleocr_func(img: np.ndarray) -> str:
            result = ocr.ocr(img, cls=True)
            if result and result[0]:
                return " ".join([r[1][0] for r in result[0]])
            return ""

        return paddleocr_func

    def extract_with_consensus(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """複数エンジンの結果を統合（コンセンサスアルゴリズム）

        Args:
            roi: 前処理済みROI画像

        Returns:
            (抽出テキスト, 信頼度) のタプル
            失敗した場合は (None, 0.0)
        """
        if not self.engines:
            logger.error("No OCR engines available")
            return None, 0.0

        results: List[Dict[str, any]] = []

        for engine_name, engine_func in self.engines.items():
            try:
                text = engine_func(roi)
                confidence = self._calculate_confidence(text)
                results.append(
                    {
                        "engine": engine_name,
                        "text": text.strip(),
                        "confidence": confidence,
                    }
                )
            except Exception as e:
                logger.error(f"{engine_name} failed: {e}")

        if not results:
            return None, 0.0

        # 信頼度でソート
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # 上位2つが類似していれば高信頼度で採用
        if len(results) >= 2:
            top1, top2 = results[0], results[1]
            similarity = self._calculate_similarity(top1["text"], top2["text"])

            if similarity > 0.8:
                avg_confidence = (top1["confidence"] + top2["confidence"]) / 2
                logger.debug(
                    f"Consensus reached: {top1['text']} (similarity={similarity:.2f})"
                )
                return top1["text"], avg_confidence

        # 最高信頼度の結果を返す
        best = results[0]
        logger.debug(
            f"Best result from {best['engine']}: {best['text']} (confidence={best['confidence']:.2f})"
        )
        return best["text"], best["confidence"]

    def _calculate_confidence(self, text: str) -> float:
        """テキストの妥当性から信頼度を計算

        Args:
            text: OCR結果テキスト

        Returns:
            信頼度（0.0-1.0）
        """
        if not text:
            return 0.0

        score = 0.0

        # 長さチェック（期待: "2025/08/26 16:07:45" = 19文字）
        text_len = len(text.strip())
        if 17 <= text_len <= 21:
            score += 0.3

        # フォーマットチェック（正規表現）
        pattern = r"^\d{4}[/-]\d{2}[/-]\d{2}\s+\d{2}:\d{2}:\d{2}$"
        if re.match(pattern, text.strip()):
            score += 0.5

        # 数字とスラッシュ・コロンの割合
        valid_chars = sum(c.isdigit() or c in "/: -" for c in text)
        if len(text) > 0:
            score += 0.2 * (valid_chars / len(text))

        return min(score, 1.0)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Levenshtein距離ベースの類似度

        Args:
            text1: テキスト1
            text2: テキスト2

        Returns:
            類似度（0.0-1.0）
        """
        try:
            from Levenshtein import ratio

            return ratio(text1, text2)
        except ImportError:
            # Levenshteinがインストールされていない場合は簡易版
            logger.warning("python-Levenshtein not installed, using simple similarity")
            if text1 == text2:
                return 1.0
            # 簡易的な類似度計算
            common_chars = sum(c1 == c2 for c1, c2 in zip(text1, text2))
            max_len = max(len(text1), len(text2))
            return common_chars / max_len if max_len > 0 else 0.0
