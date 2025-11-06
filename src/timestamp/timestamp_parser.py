"""Timestamp parsing from OCR text results."""

import logging
import re
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TimestampParser:
    """OCR結果を datetime オブジェクトに変換

    OCRエンジンから取得したテキストを解析し、
    タイムスタンプ形式の文字列をdatetimeオブジェクトに変換します。
    誤認識を考慮した柔軟なパース機能も提供します。
    """

    def __init__(self):
        """TimestampParserを初期化"""
        # 複数のパターンに対応
        self.patterns = [
            r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})',  # メイン: 2025/08/26 16:07:45
            r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})',  # ハイフン: 2025-08-26 16:07:45
            r'(\d{4})年(\d{2})月(\d{2})日\s+(\d{2}):(\d{2}):(\d{2})',  # 日本語: 2025年08月26日 16:07:45
        ]

    def parse(self, ocr_text: str) -> Tuple[Optional[datetime], float]:
        """OCR結果をdatetimeに変換

        Args:
            ocr_text: OCRエンジンから取得したテキスト

        Returns:
            (datetimeオブジェクト, 信頼度) のタプル
            パースに失敗した場合は (None, 0.0)
        """
        if not ocr_text or not ocr_text.strip():
            return None, 0.0

        for pattern in self.patterns:
            match = re.search(pattern, ocr_text)
            if match:
                groups = match.groups()
                try:
                    dt = datetime(
                        int(groups[0]),  # year
                        int(groups[1]),  # month
                        int(groups[2]),  # day
                        int(groups[3]),  # hour
                        int(groups[4]),  # minute
                        int(groups[5])   # second
                    )
                    return dt, 1.0  # 成功時は信頼度1.0
                except ValueError as e:
                    logger.warning(f"Invalid datetime: {groups}, {e}")

        return None, 0.0

    def fuzzy_parse(self, ocr_text: str) -> Tuple[Optional[datetime], float]:
        """OCR誤認識を考慮した柔軟なパース

        OCRエンジンがよく誤認識する文字を修正してからパースを試みます。

        Args:
            ocr_text: OCRエンジンから取得したテキスト

        Returns:
            (datetimeオブジェクト, 信頼度) のタプル
            パースに失敗した場合は (None, 0.0)
        """
        if not ocr_text or not ocr_text.strip():
            return None, 0.0

        # よくある誤認識を修正
        corrections = {
            'O': '0', 'o': '0',  # O -> 0
            'l': '1', 'I': '1',  # l,I -> 1
            'S': '5', 's': '5',  # S -> 5
            'B': '8',            # B -> 8
            'Z': '2',            # Z -> 2
            'G': '6',            # G -> 6
        }

        corrected = ocr_text
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)

        return self.parse(corrected)

