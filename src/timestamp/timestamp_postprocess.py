"""柔軟なタイムスタンプ正規化モジュール

OCR出力を柔軟に正規化し、候補生成とLevenshtein距離による補正を提供。
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from Levenshtein import distance as levenshtein_distance

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logger.warning(
        "python-Levenshteinがインストールされていません。pip install python-Levenshtein でインストールできます。"
    )


# 柔軟な正規表現パターン
FLEXIBLE_TIMESTAMP_PATTERN = re.compile(
    r"([12]\d{3})[\/\-\._]?(0?\d)[\/\-\._]?(0?\d)[\sT]?([0-2]?\d)[:\.]?([0-5]?\d)[:\.]?([0-5]?\d)"
)


def normalize_text(text: str) -> str:
    """OCRテキストを正規化（全角→半角、文字修正）

    Args:
        text: OCR出力テキスト

    Returns:
        正規化されたテキスト
    """
    if not text:
        return ""

    # 文字変換テーブル
    translation_table = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "D": "0",
            "S": "5",
            "s": "5",
            "I": "1",
            "l": "1",
            "|": "1",
            "B": "8",
            "b": "6",
            "A": "4",
            "Z": "2",
            "z": "2",
            "q": "9",
            "G": "6",
            "T": "7",
            "Q": "0",
            "／": "/",
            "：": ":",
            "－": "-",
        }
    )

    normalized = text.translate(translation_table)
    normalized = re.sub(r"[\t\r\n]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def extract_digits(text: str) -> str:
    """テキストから数字のみを抽出

    Args:
        text: 入力テキスト

    Returns:
        数字のみの文字列
    """
    return re.sub(r"[^0-9]", "", text)


def generate_timestamp_candidates(
    digits: str, reference_timestamp: Optional[datetime] = None
) -> List[Tuple[str, datetime]]:
    """数字列からタイムスタンプ候補を生成

    Args:
        digits: 数字のみの文字列（12-15桁）
        reference_timestamp: 参照タイムスタンプ（妥当性チェック用）

    Returns:
        [(正規化文字列, datetimeオブジェクト), ...] のリスト
    """
    candidates: List[Tuple[str, datetime]] = []

    if len(digits) < 12:
        return candidates

    # 14桁または15桁の場合を処理
    if len(digits) >= 14:
        # 最初の8桁を日付として解釈
        year_str = digits[0:4]
        month_str = digits[4:6]
        day_str = digits[6:8]

        # 残りを時刻として解釈
        if len(digits) == 15:
            # 15桁: 202510826160526 -> 2025/10/08 16:05:26
            hour_str = digits[9:11]
            minute_str = digits[11:13]
            second_str = digits[13:15]
        else:
            # 14桁: 20251082616052 -> 2025/10/08 16:05:26
            hour_str = digits[8:10]
            minute_str = digits[10:12]
            second_str = digits[12:14]

        try:
            year = int(year_str)
            month = int(month_str)
            day = int(day_str)
            hour = int(hour_str)
            minute = int(minute_str)
            second = int(second_str)

            # 妥当性チェック
            if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    dt = datetime(year, month, day, hour, minute, second)

                    # 参照タイムスタンプとの整合性チェック
                    if reference_timestamp:
                        time_diff = abs((dt - reference_timestamp).total_seconds())
                        if time_diff > 3 * 365 * 24 * 3600:  # 3年以上の差は無視
                            return candidates

                    normalized = f"{year:04d}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
                    candidates.append((normalized, dt))
        except (ValueError, OverflowError):
            pass

    # 12桁の場合（秒なし）
    if len(digits) == 12:
        year_str = digits[0:4]
        month_str = digits[4:6]
        day_str = digits[6:8]
        hour_str = digits[8:10]
        minute_str = digits[10:12]

        try:
            year = int(year_str)
            month = int(month_str)
            day = int(day_str)
            hour = int(hour_str)
            minute = int(minute_str)

            if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    dt = datetime(year, month, day, hour, minute, 0)

                    if reference_timestamp:
                        time_diff = abs((dt - reference_timestamp).total_seconds())
                        if time_diff > 3 * 365 * 24 * 3600:
                            return candidates

                    normalized = (
                        f"{year:04d}/{month:02d}/{month:02d} {hour:02d}:{minute:02d}:00"
                    )
                    candidates.append((normalized, dt))
        except (ValueError, OverflowError):
            pass

    return candidates


def score_candidate(
    candidate_text: str, original_text: str, confidence: float, is_valid: bool
) -> float:
    """候補のスコアを計算

    スコア = OCR_confidence * 0.6 + date_validity(0/1) * 0.4 - normed_edit_distance * 0.2

    Args:
        candidate_text: 候補タイムスタンプ文字列
        original_text: 元のOCRテキスト
        confidence: OCR信頼度
        is_valid: 日付が妥当か

    Returns:
        スコア（高いほど良い）
    """
    score = confidence * 0.6

    if is_valid:
        score += 0.4

    if LEVENSHTEIN_AVAILABLE:
        # 正規化編集距離を計算
        max_len = max(len(candidate_text), len(original_text))
        if max_len > 0:
            edit_dist = levenshtein_distance(candidate_text, original_text)
            normed_edit = edit_dist / max_len
            score -= normed_edit * 0.2

    return score


def parse_flexible_timestamp(
    ocr_text: str,
    confidence: float = 0.5,
    reference_timestamp: Optional[datetime] = None,
) -> Optional[str]:
    """柔軟なタイムスタンプ抽出

    Args:
        ocr_text: OCR出力テキスト
        confidence: OCR信頼度
        reference_timestamp: 参照タイムスタンプ（妥当性チェック用）

    Returns:
        正規化されたタイムスタンプ文字列（YYYY/MM/DD HH:MM:SS形式）、失敗時None
    """
    if not ocr_text:
        return None

    # 正規化
    normalized = normalize_text(ocr_text)

    # 柔軟な正規表現でマッチ
    match = FLEXIBLE_TIMESTAMP_PATTERN.search(normalized)
    if match:
        year, month, day, hour, minute, second = match.groups()

        try:
            year_i = int(year)
            month_i = int(month)
            day_i = int(day)
            hour_i = int(hour) if hour else 0
            minute_i = int(minute) if minute else 0
            second_i = int(second) if second else 0

            # 1桁時のゼロ埋め
            if hour_i < 10 and len(hour) == 1:
                hour_i = int(f"0{hour}")

            # 妥当性チェック
            if 2000 <= year_i <= 2100 and 1 <= month_i <= 12 and 1 <= day_i <= 31:
                if 0 <= hour_i <= 23 and 0 <= minute_i <= 59 and 0 <= second_i <= 59:
                    dt = datetime(year_i, month_i, day_i, hour_i, minute_i, second_i)

                    # 参照タイムスタンプとの整合性チェック
                    if reference_timestamp:
                        time_diff = abs((dt - reference_timestamp).total_seconds())
                        if time_diff > 3 * 365 * 24 * 3600:
                            return None

                    return f"{year_i:04d}/{month_i:02d}/{day_i:02d} {hour_i:02d}:{minute_i:02d}:{second_i:02d}"
        except (ValueError, OverflowError):
            pass

    # 正規表現マッチが失敗した場合、数字列から候補生成
    digits = extract_digits(normalized)
    if len(digits) >= 12:
        candidates = generate_timestamp_candidates(digits, reference_timestamp)

        if candidates:
            # スコアでソート
            scored_candidates = []
            for cand_text, cand_dt in candidates:
                score = score_candidate(cand_text, normalized, confidence, True)
                scored_candidates.append((score, cand_text, cand_dt))

            scored_candidates.sort(reverse=True)

            # 最高スコアの候補を返す
            if scored_candidates:
                return scored_candidates[0][1]

    return None
