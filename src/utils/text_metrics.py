"""テキスト評価指標（CER, WER, Precision/Recall/F1）を計算するユーティリティ"""

import re
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """テキストを正規化（空白の統一、前後の空白削除）

    Args:
        text: 入力テキスト

    Returns:
        正規化されたテキスト
    """
    if not text:
        return ""
    # 連続する空白を1つに統一
    text = re.sub(r"\s+", " ", text)
    # 前後の空白を削除
    text = text.strip()
    return text


def tokenize(text: str) -> List[str]:
    """テキストをトークン（単語）に分割

    Args:
        text: 入力テキスト

    Returns:
        トークンのリスト
    """
    if not text:
        return []
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def calculate_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """文字エラー率（Character Error Rate, CER）を計算

    Args:
        reference: 参照テキスト（ゴールド）
        hypothesis: 予測テキスト（OCR出力）

    Returns:
        CER指標の辞書（cer, substitutions, insertions, deletions, total_chars）
    """
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if not ref_chars:
        if not hyp_chars:
            return {
                "cer": 0.0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "total_chars": 0,
            }
        return {
            "cer": 1.0,
            "substitutions": 0,
            "insertions": len(hyp_chars),
            "deletions": 0,
            "total_chars": len(ref_chars),
        }

    # 動的計画法で編集距離を計算（Levenshtein距離）
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初期化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # DPテーブルを埋める
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    # 編集操作を逆算
    substitutions = 0
    insertions = 0
    deletions = 0
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == hyp_chars[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1

    total_errors = substitutions + insertions + deletions
    cer = total_errors / len(ref_chars) if ref_chars else 1.0

    return {
        "cer": cer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "total_chars": len(ref_chars),
        "total_errors": total_errors,
    }


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """単語エラー率（Word Error Rate, WER）を計算

    Args:
        reference: 参照テキスト（ゴールド）
        hypothesis: 予測テキスト（OCR出力）

    Returns:
        WER指標の辞書（wer, substitutions, insertions, deletions, total_words）
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens:
        if not hyp_tokens:
            return {
                "wer": 0.0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "total_words": 0,
            }
        return {
            "wer": 1.0,
            "substitutions": 0,
            "insertions": len(hyp_tokens),
            "deletions": 0,
            "total_words": len(ref_tokens),
        }

    # 動的計画法で編集距離を計算
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初期化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # DPテーブルを埋める
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    # 編集操作を逆算
    substitutions = 0
    insertions = 0
    deletions = 0
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1

    total_errors = substitutions + insertions + deletions
    wer = total_errors / len(ref_tokens) if ref_tokens else 1.0

    return {
        "wer": wer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "total_words": len(ref_tokens),
        "total_errors": total_errors,
    }


def calculate_token_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """トークン単位のPrecision/Recall/F1を計算

    Args:
        reference: 参照テキスト（ゴールド）
        hypothesis: 予測テキスト（OCR出力）

    Returns:
        トークン指標の辞書（precision, recall, f1, true_positives, false_positives, false_negatives）
    """
    ref_tokens = set(tokenize(reference))
    hyp_tokens = set(tokenize(hypothesis))

    # トークンレベルでのマッチング
    true_positives = len(ref_tokens & hyp_tokens)
    false_positives = len(hyp_tokens - ref_tokens)
    false_negatives = len(ref_tokens - hyp_tokens)

    # Precision = TP / (TP + FP)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall = TP / (TP + FN)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_ref_tokens": len(ref_tokens),
        "total_hyp_tokens": len(hyp_tokens),
    }
