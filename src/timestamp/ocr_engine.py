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
    import pytesseract  # noqa: F401

    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract is not available")

try:
    import easyocr  # noqa: F401

    EASYOCR_AVAILABLE = True
except ImportError:
    logger.debug("easyocr is not available")

try:
    from paddleocr import PaddleOCR  # noqa: F401

    PADDLEOCR_AVAILABLE = True
except ImportError:
    logger.debug("paddleocr is not available")


class MultiEngineOCR:
    """複数のOCRエンジンを使用して信頼性を向上

    複数のOCRエンジン（Tesseract, EasyOCR, PaddleOCR）を統合し、
    コンセンサスアルゴリズムで最も信頼性の高い結果を返します。
    """

    def __init__(
        self,
        enabled_engines: list[str] = None,
        use_weighted_consensus: bool = False,
        use_voting_consensus: bool = False,
    ):
        """MultiEngineOCRを初期化

        Args:
            enabled_engines: 有効にするエンジンのリスト
                            Noneの場合は利用可能な全エンジンを使用
            use_weighted_consensus: 重み付けスキームを使用するか（デフォルト: False）
            use_voting_consensus: 投票ロジックを使用するか（デフォルト: False）
        """
        self.engines: dict[str, callable] = {}
        self.enabled_engines = enabled_engines or []
        self.use_weighted_consensus = use_weighted_consensus
        self.use_voting_consensus = use_voting_consensus

        # 利用可能なエンジンを初期化
        if TESSERACT_AVAILABLE and (not enabled_engines or "tesseract" in enabled_engines):
            self.engines["tesseract"] = self._init_tesseract()

        if EASYOCR_AVAILABLE and (not enabled_engines or "easyocr" in enabled_engines):
            self.engines["easyocr"] = self._init_easyocr()

        if PADDLEOCR_AVAILABLE and (not enabled_engines or "paddleocr" in enabled_engines):
            self.engines["paddleocr"] = self._init_paddleocr()

        if not self.engines:
            logger.warning("No OCR engines available. Please install at least one OCR engine.")

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

    def extract_with_consensus(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        """複数エンジンの結果を統合（コンセンサスアルゴリズム）

        改善されたアルゴリズム（重み付けスキーム、投票ロジック）をサポート

        Args:
            roi: 前処理済みROI画像

        Returns:
            (抽出テキスト, 信頼度) のタプル
            失敗した場合は (None, 0.0)
        """
        if not self.engines:
            logger.error("No OCR engines available")
            return None, 0.0

        # 投票ロジックが有効な場合は優先
        if self.use_voting_consensus and len(self.engines) > 1:
            return self._extract_with_voting(roi)

        # 重み付けスキームが有効な場合
        if self.use_weighted_consensus:
            return self._extract_with_weighted_consensus(roi)

        # デフォルトのコンセンサスアルゴリズム
        return self._extract_with_baseline_consensus(roi)

    def _extract_with_baseline_consensus(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        """ベースラインのコンセンサスアルゴリズム（既存実装）

        Args:
            roi: 前処理済みROI画像

        Returns:
            (抽出テキスト, 信頼度) のタプル
        """
        results: list[dict[str, any]] = []

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
                logger.debug(f"Consensus reached: {top1['text']} (similarity={similarity:.2f})")
                return top1["text"], avg_confidence

        # 最高信頼度の結果を返す
        best = results[0]
        logger.debug(f"Best result from {best['engine']}: {best['text']} (confidence={best['confidence']:.2f})")
        return best["text"], best["confidence"]

    def _extract_with_weighted_consensus(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        """重み付けスキームによるコンセンサス

        Args:
            roi: 前処理済みROI画像

        Returns:
            (抽出テキスト, 信頼度) のタプル
        """
        results: list[dict[str, any]] = []

        for engine_name, engine_func in self.engines.items():
            try:
                text = engine_func(roi)
                confidence = self._calculate_confidence(text)

                # エンジン別の重み（Tesseractを優先）
                weight = 1.0 if engine_name == "tesseract" else 0.8
                weighted_confidence = confidence * weight

                results.append(
                    {
                        "engine": engine_name,
                        "text": text.strip(),
                        "confidence": confidence,
                        "weighted_confidence": weighted_confidence,
                    }
                )
            except Exception as e:
                logger.debug(f"{engine_name} failed: {e}")

        if not results:
            return None, 0.0

        # 重み付け信頼度でソート
        results.sort(key=lambda x: x["weighted_confidence"], reverse=True)

        # エンジン間の一致度を評価
        if len(results) >= 2:
            top1, top2 = results[0], results[1]
            similarity = self._calculate_similarity(top1["text"], top2["text"])

            # 一致度が高い場合は信頼度を向上
            if similarity > 0.8:
                avg_confidence = (top1["weighted_confidence"] + top2["weighted_confidence"]) / 2
                avg_confidence = min(avg_confidence * 1.1, 1.0)  # 10%向上
                logger.debug(
                    f"Weighted consensus reached: {top1['text']} (similarity={similarity:.2f}, "
                    f"weighted_confidence={avg_confidence:.2f})"
                )
                return top1["text"], avg_confidence

        best = results[0]
        logger.debug(
            f"Best weighted result from {best['engine']}: {best['text']} "
            f"(weighted_confidence={best['weighted_confidence']:.2f})"
        )
        return best["text"], best["weighted_confidence"]

    def _extract_with_voting(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        """投票ロジックによるコンセンサス

        Args:
            roi: 前処理済みROI画像

        Returns:
            (抽出テキスト, 信頼度) のタプル
        """
        results: list[dict[str, any]] = []

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
                logger.debug(f"{engine_name} failed: {e}")

        if not results:
            return None, 0.0

        # テキストごとに投票
        text_votes: dict[str, list[float]] = {}
        for r in results:
            text = r["text"]
            if text not in text_votes:
                text_votes[text] = []
            text_votes[text].append(r["confidence"])

        # 2/3以上のエンジンが一致したテキストを採用
        threshold = len(self.engines) * 2 / 3
        for text, confidences in text_votes.items():
            if len(confidences) >= threshold:
                avg_confidence = sum(confidences) / len(confidences)
                logger.debug(
                    f"Voting consensus reached: {text} "
                    f"({len(confidences)}/{len(self.engines)} engines, "
                    f"confidence={avg_confidence:.2f})"
                )
                return text, avg_confidence

        # 2/3一致がない場合は最高信頼度を返す
        results.sort(key=lambda x: x["confidence"], reverse=True)
        best = results[0]
        logger.debug(
            f"No voting consensus, using best result from {best['engine']}: "
            f"{best['text']} (confidence={best['confidence']:.2f})"
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
            common_chars = sum(c1 == c2 for c1, c2 in zip(text1, text2, strict=False))
            max_len = max(len(text1), len(text2))
            return common_chars / max_len if max_len > 0 else 0.0
