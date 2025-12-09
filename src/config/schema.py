"""設定スキーマの薄い定義。

依存を増やさずに最小の型安全を提供するため、dataclass ベースで保持する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class AppConfig:
    """全体設定のラッパー。実際の検証は ConfigManager 互換層に委譲する。"""

    raw: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self.raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
