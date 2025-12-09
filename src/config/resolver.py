"""CLI/環境変数の上書きを行う簡易リゾルバ。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """辞書を浅いマージで統合する。"""
    merged = dict(base)
    for k, v in overrides.items():
        merged[k] = v
    return merged


def apply_env_overrides(config: Mapping[str, Any], prefix: str = "YOLO3_") -> dict[str, Any]:
    """環境変数による単純上書き。"""
    updated = dict(config)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        key = env_key[len(prefix) :].lower()
        updated[key] = env_val
    return updated
