"""User-facing Knowledge Graph settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeGraphConfig:
    """Persisted Knowledge Graph settings exposed through the Settings API."""

    post_run_rebuild_enabled: bool = True
    wiki_auto_refresh_after_kg_rebuild: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "post_run_rebuild_enabled": self.post_run_rebuild_enabled,
            "wiki_auto_refresh_after_kg_rebuild": self.wiki_auto_refresh_after_kg_rebuild,
        }


def default_knowledge_graph_config() -> KnowledgeGraphConfig:
    """Return the product default: keep the graph fresh after usable runs."""

    return KnowledgeGraphConfig()


def coerce_knowledge_graph_config(value: Any) -> KnowledgeGraphConfig:
    """Validate and normalize a Knowledge Graph settings payload."""

    if value is None:
        return default_knowledge_graph_config()
    if not isinstance(value, dict):
        raise ValueError("knowledge graph settings must be an object")
    enabled = value.get("post_run_rebuild_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("post_run_rebuild_enabled must be a boolean")
    wiki_auto = value.get("wiki_auto_refresh_after_kg_rebuild", False)
    if not isinstance(wiki_auto, bool):
        raise ValueError("wiki_auto_refresh_after_kg_rebuild must be a boolean")
    return KnowledgeGraphConfig(
        post_run_rebuild_enabled=enabled,
        wiki_auto_refresh_after_kg_rebuild=wiki_auto,
    )
