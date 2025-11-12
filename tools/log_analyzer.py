#!/usr/bin/env python
"""ログファイルの分析ツール

WARNING/ERRORレベルのログを抽出し、頻出エラーパターンを特定します。
"""

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import re
import sys

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import contextlib

from tqdm import tqdm

from src.config import ConfigManager
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_log_file(log_path: Path) -> dict:
    """ログファイルを解析

    Args:
        log_path: ログファイルのパス

    Returns:
        解析結果の辞書
    """
    if not log_path.exists():
        logger.error(f"ログファイルが見つかりません: {log_path}")
        return {}

    errors = []
    warnings = []
    error_patterns = defaultdict(int)
    warning_patterns = defaultdict(int)

    # ログレベルのパターン（複数の形式に対応）
    # 形式1: "2025-11-06 20:15:28,052 - module - ERROR - message"
    # 形式2: "ERROR - message"
    level_pattern1 = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - [^-]+ - (ERROR|WARNING|INFO|DEBUG) - (.+)$"
    )
    level_pattern2 = re.compile(r"^(ERROR|WARNING|INFO|DEBUG)\s+")

    # ファイルの行数を取得（プログレスバー用）
    total_lines = sum(1 for _ in log_path.open("r", encoding="utf-8"))

    with log_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="ログ解析中"), 1):
            line = line.strip()
            if not line:
                continue

            # ログレベルの検出
            match1 = level_pattern1.match(line)
            if match1:
                level = match1.group(1)
                message = match1.group(2).strip()
            else:
                match2 = level_pattern2.match(line)
                if match2:
                    level = match2.group(1)
                    message = line[match2.end() :].strip()
                else:
                    continue

            if level == "ERROR":
                errors.append((line_num, message))
                # エラーパターンの抽出（最初の50文字）
                pattern = message[:50] if len(message) > 50 else message
                error_patterns[pattern] += 1
            elif level == "WARNING":
                warnings.append((line_num, message))
                # 警告パターンの抽出
                pattern = message[:50] if len(message) > 50 else message
                warning_patterns[pattern] += 1

    analysis = {
        "total_errors": len(errors),
        "total_warnings": len(warnings),
        "errors": errors[:100],  # 最初の100件
        "warnings": warnings[:100],  # 最初の100件
        "error_patterns": dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
        "warning_patterns": dict(sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
    }

    return analysis


def analyze_performance_bottlenecks(log_path: Path) -> dict:
    """処理時間のボトルネックを分析

    Args:
        log_path: ログファイルのパス

    Returns:
        ボトルネック分析結果
    """
    if not log_path.exists():
        return {}

    # 処理時間のパターン
    time_pattern = re.compile(r"(\d+\.\d+)秒|(\d+)秒")
    processing_times = []

    # ファイルの行数を取得（プログレスバー用）
    total_lines = sum(1 for _ in log_path.open("r", encoding="utf-8"))

    with log_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="ボトルネック分析中"):
            # "処理時間"や"実行時間"などのキーワードを含む行を探す
            if "処理時間" in line or "実行時間" in line or "elapsed" in line.lower():
                matches = time_pattern.findall(line)
                for match in matches:
                    time_str = match[0] if match[0] else match[1]
                    with contextlib.suppress(ValueError):
                        processing_times.append(float(time_str))

    if not processing_times:
        return {}

    import numpy as np

    bottlenecks = {
        "count": len(processing_times),
        "mean": np.mean(processing_times),
        "median": np.median(processing_times),
        "max": np.max(processing_times),
        "min": np.min(processing_times),
        "std": np.std(processing_times),
    }

    return bottlenecks


def print_log_analysis_report(analysis: dict, bottlenecks: dict):
    """ログ分析レポートを出力

    Args:
        analysis: ログ解析結果
        bottlenecks: ボトルネック分析結果
    """
    logger.info("=" * 80)
    logger.info("ログファイル分析レポート")
    logger.info("=" * 80)

    logger.info("\nエラーログ:")
    logger.info(f"  総数: {analysis.get('total_errors', 0)}")

    if analysis.get("error_patterns"):
        logger.info("\n  頻出エラーパターン（上位10件）:")
        for i, (pattern, count) in enumerate(list(analysis.get("error_patterns", {}).items())[:10], 1):
            logger.info(f"    {i}. [{count}回] {pattern}")

    if analysis.get("errors"):
        logger.info("\n  エラーログ例（最初の5件）:")
        for line_num, message in analysis.get("errors", [])[:5]:
            logger.info(f"    行{line_num}: {message[:100]}")

    logger.info("\n警告ログ:")
    logger.info(f"  総数: {analysis.get('total_warnings', 0)}")

    if analysis.get("warning_patterns"):
        logger.info("\n  頻出警告パターン（上位10件）:")
        for i, (pattern, count) in enumerate(list(analysis.get("warning_patterns", {}).items())[:10], 1):
            logger.info(f"    {i}. [{count}回] {pattern}")

    if analysis.get("warnings"):
        logger.info("\n  警告ログ例（最初の5件）:")
        for line_num, message in analysis.get("warnings", [])[:5]:
            logger.info(f"    行{line_num}: {message[:100]}")

    if bottlenecks:
        logger.info("\n処理時間のボトルネック分析:")
        logger.info(f"  測定回数: {bottlenecks.get('count', 0)}")
        logger.info(f"  平均: {bottlenecks.get('mean', 0):.2f}秒")
        logger.info(f"  中央値: {bottlenecks.get('median', 0):.2f}秒")
        logger.info(f"  最大: {bottlenecks.get('max', 0):.2f}秒")
        logger.info(f"  最小: {bottlenecks.get('min', 0):.2f}秒")
        logger.info(f"  標準偏差: {bottlenecks.get('std', 0):.2f}秒")

    logger.info("=" * 80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="ログファイルの分析")
    parser.add_argument("--log", type=str, help="ログファイルのパス")
    parser.add_argument("--output-dir", type=str, help="出力ディレクトリ（ログが未指定の場合）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # ログパスの決定
    if args.log:
        log_path = Path(args.log)
    elif args.output_dir:
        log_path = Path(args.output_dir) / "system.log"
    else:
        config = ConfigManager(args.config)
        output_dir = Path(config.get("output.directory", "output"))
        log_path = output_dir / "system.log"

    # ログ解析
    analysis = parse_log_file(log_path)

    if not analysis:
        logger.error("ログ解析に失敗しました")
        return 1

    # ボトルネック分析
    bottlenecks = analyze_performance_bottlenecks(log_path)

    # レポート出力
    print_log_analysis_report(analysis, bottlenecks)

    # 結果をJSONで保存
    import json

    output_dir = log_path.parent
    report_path = output_dir / "log_analysis_report.json"

    report = {
        "analysis": analysis,
        "bottlenecks": bottlenecks,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n分析結果を保存しました: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
