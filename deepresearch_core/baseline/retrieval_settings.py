"""Retrieval infrastructure settings persisted in config.json."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalSettings:
    chroma_persist_enabled: bool = True
    kg_aware_retrieval_enabled: bool = False
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            **deepcopy(self.extra),
            "chroma_persist_enabled": self.chroma_persist_enabled,
            "kg_aware_retrieval_enabled": self.kg_aware_retrieval_enabled,
        }


def default_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()


def coerce_retrieval_settings(value: Any) -> RetrievalSettings:
    if value is None:
        return default_retrieval_settings()
    if not isinstance(value, dict):
        raise ValueError("retrieval settings payload must be an object")
    enabled = value.get("chroma_persist_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("chroma_persist_enabled must be a boolean")
    kg_enabled = value.get("kg_aware_retrieval_enabled", False)
    if not isinstance(kg_enabled, bool):
        raise ValueError("kg_aware_retrieval_enabled must be a boolean")
    extra = {
        str(key): deepcopy(item)
        for key, item in value.items()
        if key not in {"chroma_persist_enabled", "kg_aware_retrieval_enabled"}
    }
    return RetrievalSettings(
        chroma_persist_enabled=enabled,
        kg_aware_retrieval_enabled=kg_enabled,
        extra=extra,
    )
