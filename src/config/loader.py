"""設定ファイルの読み込み専用モジュール。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config_file(path: str) -> dict[str, Any]:
    """YAML/JSON設定を辞書として読み込む。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    suffix = config_path.suffix.lower()
    with config_path.open(encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise ValueError("YAML設定は辞書形式である必要があります")
            return data
        if suffix == ".json":
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON設定は辞書形式である必要があります")
            return data
        raise ValueError(f"サポートされない設定形式です: {suffix}")
