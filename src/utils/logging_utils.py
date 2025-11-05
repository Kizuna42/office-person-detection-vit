"""Logging utilities for the office person detection system."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(debug_mode: bool = False, output_dir: str = 'output') -> None:
    """ロギングを設定する
    
    Args:
        debug_mode: デバッグモードの場合True
        output_dir: 出力ディレクトリ
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 既存のハンドラをクリア
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソール出力
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # ファイル出力
    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    file_handler = logging.FileHandler(log_dir / 'system.log', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # ロガーに設定
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

