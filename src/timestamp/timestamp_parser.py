"""Timestamp parsing from OCR text results."""

from datetime import datetime
import logging
import re

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
            r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})",  # メイン: 2025/08/26 16:07:45
            r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})",  # ハイフン: 2025-08-26 16:07:45
            r"(\d{4})年(\d{2})月(\d{2})日\s+(\d{2}):(\d{2}):(\d{2})",  # 日本語: 2025年08月26日 16:07:45
        ]

    def parse(self, ocr_text: str) -> tuple[datetime | None, float]:
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
                        int(groups[5]),  # second
                    )
                    return dt, 1.0  # 成功時は信頼度1.0
                except ValueError as e:
                    logger.warning(f"Invalid datetime: {groups}, {e}")

        return None, 0.0

    def fuzzy_parse(self, ocr_text: str) -> tuple[datetime | None, float]:
        """OCR誤認識を考慮した柔軟なパース

        OCRエンジンがよく誤認識する文字を修正してからパースを試みます。
        スペース欠落、スラッシュ欠落などのパターンも補正します。

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
            "O": "0",
            "o": "0",  # O -> 0
            "l": "1",
            "I": "1",  # l,I -> 1
            "S": "5",
            "s": "5",  # S -> 5
            "B": "8",  # B -> 8
            "Z": "2",  # Z -> 2
            "G": "6",  # G -> 6
        }

        corrected = ocr_text
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)

        # 補正の順序が重要: より具体的なパターンを先に処理
        # スラッシュが存在するパターンを先に処理（スラッシュなしのパターンより優先）

        # パターン1: スペース欠落の補正（完全な日付形式の後）
        # 例: "2025/08/2616:04:16" -> "2025/08/26 16:04:16"
        space_missing_pattern = re.compile(r"(\d{4}/\d{2}/\d{2})(\d{2}:\d{2}:\d{2})")
        corrected = space_missing_pattern.sub(r"\1 \2", corrected)

        # パターン2: 最初のスラッシュ欠落の補正（時刻付き、スラッシュが1つ存在）
        # 例: "2025108/26 16:04:16" -> "2025/08/26 16:04:16"
        # または: "2025108/2616:04:16" -> "2025/08/26 16:04:16"
        # スラッシュの前に6桁以上の数字、スラッシュの後に2桁の数字がある場合
        # 注意: スラッシュの前の数字を年(4桁) + 月(2桁)として解釈
        # パターン: 6桁以上の数字 + スラッシュ + 2桁の数字 + スペース/時刻
        first_slash_missing_with_time1 = re.compile(r"^(\d{6,})/(\d{2})\s+(\d{2}:\d{2}:\d{2})")

        def replace_first_slash1(m):
            year_month_str = m.group(1)  # 6桁以上の数字
            day = m.group(2)
            time = m.group(3)
            # 最初の4桁を年、次の2桁を月として解釈（残りは無視）
            if len(year_month_str) >= 6:
                year = year_month_str[:4]
                month = year_month_str[4:6]
                return f"{year}/{month}/{day} {time}"
            return m.group(0)  # マッチしない場合は元の文字列を返す

        corrected = first_slash_missing_with_time1.sub(replace_first_slash1, corrected)

        first_slash_missing_with_time2 = re.compile(r"^(\d{6,})/(\d{2})(\d{2}:\d{2}:\d{2})")

        def replace_first_slash2(m):
            year_month_str = m.group(1)  # 6桁以上の数字
            day = m.group(2)
            time = m.group(3)
            # 最初の4桁を年、次の2桁を月として解釈（残りは無視）
            if len(year_month_str) >= 6:
                year = year_month_str[:4]
                month = year_month_str[4:6]
                return f"{year}/{month}/{day} {time}"
            return m.group(0)  # マッチしない場合は元の文字列を返す

        corrected = first_slash_missing_with_time2.sub(replace_first_slash2, corrected)

        # パターン3: 2番目のスラッシュ欠落の補正（時刻付き、スラッシュが1つ存在）
        # 例: "2025/0826 16:04:16" -> "2025/08/26 16:04:16"
        # または: "2025/082616:04:16" -> "2025/08/26 16:04:16"
        second_slash_missing_with_time1 = re.compile(r"(\d{4}/\d{2})(\d{2})\s+(\d{2}:\d{2}:\d{2})")
        corrected = second_slash_missing_with_time1.sub(r"\1/\2 \3", corrected)
        second_slash_missing_with_time2 = re.compile(r"(\d{4}/\d{2})(\d{2})(\d{2}:\d{2}:\d{2})")
        corrected = second_slash_missing_with_time2.sub(r"\1/\2 \3", corrected)

        # パターン4: 日付部分のスラッシュが全て欠落している場合（時刻付き、スラッシュなし）
        # 例: "20250826 16:04:16" -> "2025/08/26 16:04:16"
        # または: "2025082616:04:16" -> "2025/08/26 16:04:16"
        # スラッシュが存在しない場合のみマッチ（より一般的なパターンなので最後に処理）
        if "/" not in corrected[:10]:  # 最初の10文字にスラッシュがない場合
            all_slashes_missing_with_time1 = re.compile(r"(\d{4})(\d{2})(\d{2})\s+(\d{2}:\d{2}:\d{2})")
            corrected = all_slashes_missing_with_time1.sub(r"\1/\2/\3 \4", corrected)
            all_slashes_missing_with_time2 = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2}:\d{2}:\d{2})")
            corrected = all_slashes_missing_with_time2.sub(r"\1/\2/\3 \4", corrected)

        # パターン5: 日付と時刻の間のスペースが複数ある場合の正規化
        # 例: "2025/08/26  16:04:16" -> "2025/08/26 16:04:16"
        corrected = re.sub(r"(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})", r"\1 \2", corrected)

        # まず補正後のテキストでパースを試みる
        result, confidence = self.parse(corrected)
        if result is not None:
            return result, confidence

        # 補正前のテキストでも試す（元のテキストが正しい場合）
        result, confidence = self.parse(ocr_text)
        if result is not None:
            return result, confidence

        # 最後に補正後のテキストで再試行（信頼度を下げる）
        return None, 0.0
