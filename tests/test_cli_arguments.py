"""Test cases for CLI arguments."""

from __future__ import annotations

import sys
from unittest.mock import patch

from src.cli.arguments import parse_arguments


def test_parse_arguments_default():
    """デフォルト引数のパース"""
    test_args = ["script_name"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.config == "config.yaml"
        assert args.debug is False
        assert args.evaluate is False
        assert args.fine_tune is False
        assert args.start_time is None
        assert args.end_time is None
        assert args.timestamps_only is False


def test_parse_arguments_config():
    """設定ファイルパスの指定"""
    test_args = ["script_name", "--config", "custom_config.yaml"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.config == "custom_config.yaml"


def test_parse_arguments_debug():
    """デバッグモードの指定"""
    test_args = ["script_name", "--debug"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.debug is True


def test_parse_arguments_evaluate():
    """評価モードの指定"""
    test_args = ["script_name", "--evaluate"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.evaluate is True


def test_parse_arguments_fine_tune():
    """ファインチューニングモードの指定"""
    test_args = ["script_name", "--fine-tune"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.fine_tune is True


def test_parse_arguments_start_time():
    """開始時刻の指定"""
    test_args = ["script_name", "--start-time", "10:30"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.start_time == "10:30"


def test_parse_arguments_end_time():
    """終了時刻の指定"""
    test_args = ["script_name", "--end-time", "18:00"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.end_time == "18:00"


def test_parse_arguments_timestamps_only():
    """タイムスタンプのみモードの指定"""
    test_args = ["script_name", "--timestamps-only"]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.timestamps_only is True


def test_parse_arguments_all_options():
    """全てのオプションを指定"""
    test_args = [
        "script_name",
        "--config",
        "test_config.yaml",
        "--debug",
        "--evaluate",
        "--fine-tune",
        "--start-time",
        "09:00",
        "--end-time",
        "17:00",
        "--timestamps-only",
    ]

    with patch.object(sys, "argv", test_args):
        args = parse_arguments()

        assert args.config == "test_config.yaml"
        assert args.debug is True
        assert args.evaluate is True
        assert args.fine_tune is True
        assert args.start_time == "09:00"
        assert args.end_time == "17:00"
        assert args.timestamps_only is True
