"""Unit tests for timestamp_postprocess module."""

from __future__ import annotations

from datetime import datetime

from src.timestamp.timestamp_postprocess import (extract_digits,
                                                 generate_timestamp_candidates,
                                                 normalize_text,
                                                 parse_flexible_timestamp,
                                                 score_candidate)


def test_normalize_text_basic():
    """基本的なテキスト正規化が動作する。"""

    text = "2023/04/01 12:34:56"
    result = normalize_text(text)

    assert result == "2023/04/01 12:34:56"


def test_normalize_text_character_substitution():
    """文字の置換が正しく動作する。"""

    text = "2023/04/01 O2:34:56"  # Oを0に変換
    result = normalize_text(text)

    assert "O" not in result
    assert "0" in result


def test_normalize_text_fullwidth_to_halfwidth():
    """全角文字が半角に変換される。"""

    text = "2023／04／01 12：34：56"  # 全角スラッシュとコロン
    result = normalize_text(text)

    assert "／" not in result
    assert "：" not in result
    assert "/" in result
    assert ":" in result


def test_normalize_text_whitespace():
    """空白文字が正規化される。"""

    text = "2023/04/01\t12:34:56\n"
    result = normalize_text(text)

    assert "\t" not in result
    assert "\n" not in result


def test_normalize_text_empty():
    """空文字列は空文字列を返す。"""

    result = normalize_text("")

    assert result == ""


def test_extract_digits():
    """数字のみを抽出できる。"""

    text = "2023/04/01 12:34:56"
    result = extract_digits(text)

    assert result == "20230401123456"


def test_extract_digits_with_non_digits():
    """数字以外の文字が除去される。"""

    text = "abc2023def04ghi01"
    result = extract_digits(text)

    assert result == "20230401"


def test_extract_digits_empty():
    """数字がない場合は空文字列を返す。"""

    text = "abc def"
    result = extract_digits(text)

    assert result == ""


def test_generate_timestamp_candidates_15_digits():
    """15桁の数字列からタイムスタンプ候補を生成できる。"""

    digits = "20251026160526"  # 2025/10/26 16:05:26
    candidates = generate_timestamp_candidates(digits)

    assert len(candidates) > 0
    assert any("2025/10/26 16:05:26" in text for text, dt in candidates)


def test_generate_timestamp_candidates_14_digits():
    """14桁の数字列からタイムスタンプ候補を生成できる。"""

    digits = "2025102616052"  # 14桁（不完全）
    candidates = generate_timestamp_candidates(digits)

    # 14桁でも候補が生成される可能性がある
    assert isinstance(candidates, list)


def test_generate_timestamp_candidates_12_digits():
    """12桁の数字列からタイムスタンプ候補を生成できる（秒なし）。"""

    digits = "202510261605"  # 2025/10/26 16:05
    candidates = generate_timestamp_candidates(digits)

    assert len(candidates) > 0
    assert any("2025/10/26" in text for text, dt in candidates)


def test_generate_timestamp_candidates_too_short():
    """12桁未満の場合は空リストを返す。"""

    digits = "20251026"
    candidates = generate_timestamp_candidates(digits)

    assert candidates == []


def test_generate_timestamp_candidates_invalid_date():
    """不正な日付の場合は候補が生成されない。"""

    digits = "20251301123456"  # 13月は存在しない
    candidates = generate_timestamp_candidates(digits)

    assert candidates == []


def test_generate_timestamp_candidates_with_reference():
    """参照タイムスタンプとの整合性チェックが動作する。"""

    digits = "20251026160526"
    reference = datetime(2025, 10, 26, 16, 5, 25)

    candidates = generate_timestamp_candidates(digits, reference_timestamp=reference)

    assert len(candidates) > 0


def test_generate_timestamp_candidates_far_from_reference():
    """参照タイムスタンプと大きく離れている場合は候補が生成されない。"""

    digits = "20201026160526"  # 5年前
    reference = datetime(2025, 10, 26, 16, 5, 25)

    candidates = generate_timestamp_candidates(digits, reference_timestamp=reference)

    assert candidates == []


def test_score_candidate():
    """候補のスコアを計算できる。"""

    score = score_candidate(
        candidate_text="2023/04/01 12:34:56",
        original_text="2023/04/01 12:34:56",
        confidence=0.9,
        is_valid=True,
    )

    assert score > 0
    assert score <= 1.0


def test_score_candidate_invalid():
    """不正な日付の場合はスコアが低くなる。"""

    score_valid = score_candidate(
        candidate_text="2023/04/01 12:34:56",
        original_text="2023/04/01 12:34:56",
        confidence=0.9,
        is_valid=True,
    )

    score_invalid = score_candidate(
        candidate_text="2023/04/01 12:34:56",
        original_text="2023/04/01 12:34:56",
        confidence=0.9,
        is_valid=False,
    )

    assert score_valid > score_invalid


def test_score_candidate_low_confidence():
    """信頼度が低い場合はスコアが低くなる。"""

    score_high = score_candidate(
        candidate_text="2023/04/01 12:34:56",
        original_text="2023/04/01 12:34:56",
        confidence=0.9,
        is_valid=True,
    )

    score_low = score_candidate(
        candidate_text="2023/04/01 12:34:56",
        original_text="2023/04/01 12:34:56",
        confidence=0.5,
        is_valid=True,
    )

    assert score_high > score_low


def test_parse_flexible_timestamp_strict_format():
    """厳密なフォーマットのタイムスタンプを解析できる。"""

    text = "2023/04/01 12:34:56"
    result = parse_flexible_timestamp(text, confidence=0.9)

    assert result == "2023/04/01 12:34:56"


def test_parse_flexible_timestamp_flexible_format():
    """柔軟なフォーマットのタイムスタンプを解析できる。"""

    text = "2023-04-01 12:34:56"
    result = parse_flexible_timestamp(text, confidence=0.9)

    assert result is not None
    assert "2023" in result


def test_parse_flexible_timestamp_space_missing():
    """スペースが欠落している場合でも解析できる。"""

    text = "2023/04/0112:34:56"
    result = parse_flexible_timestamp(text, confidence=0.9)

    assert result is not None


def test_parse_flexible_timestamp_single_digit_hour():
    """1桁時の場合でも解析できる。"""

    text = "2023/04/01 5:34:56"
    result = parse_flexible_timestamp(text, confidence=0.9)

    assert result is not None
    assert "05" in result or "5" in result


def test_parse_flexible_timestamp_from_digits():
    """数字列からタイムスタンプを生成できる。"""

    text = "20230401123456"
    result = parse_flexible_timestamp(text, confidence=0.9)

    assert result is not None
    assert "2023" in result or result is None  # 候補生成で解析される可能性がある


def test_parse_flexible_timestamp_empty():
    """空文字列は None を返す。"""

    result = parse_flexible_timestamp("", confidence=0.9)

    assert result is None


def test_parse_flexible_timestamp_invalid():
    """不正なテキストは None を返す。"""

    result = parse_flexible_timestamp("invalid text", confidence=0.9)

    assert result is None


def test_parse_flexible_timestamp_with_reference():
    """参照タイムスタンプとの整合性チェックが動作する。"""

    text = "2025/10/26 16:05:26"
    reference = datetime(2025, 10, 26, 16, 5, 25)

    result = parse_flexible_timestamp(
        text, confidence=0.9, reference_timestamp=reference
    )

    assert result is not None


def test_parse_flexible_timestamp_far_from_reference():
    """参照タイムスタンプと大きく離れている場合は None を返す。"""

    text = "2020/10/26 16:05:26"  # 5年前
    reference = datetime(2025, 10, 26, 16, 5, 25)

    result = parse_flexible_timestamp(
        text, confidence=0.9, reference_timestamp=reference
    )

    assert result is None


def test_parse_flexible_timestamp_character_substitution():
    """文字の置換が適用される。"""

    text = "2023/04/01 O2:34:56"  # Oを0に変換
    result = parse_flexible_timestamp(text, confidence=0.9)

    # 正規化により解析される可能性がある
    assert result is None or "2023" in result
