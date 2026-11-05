"""DTOs for Knowledge Graph Wiki Map snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]

WIKI_LAYOUT_MODE = "wiki_map"


@dataclass(frozen=True)
class WikiTopicDraft:
    """A normalized topic before it is persisted for one wiki snapshot."""

    topic_key: str
    title: str
    summary: str = ""
    depth: int = 0
    parent_key: str = ""
    session_ids: tuple[str, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    attrs: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class WikiTreeEdgeDraft:
    """A parent-child topic edge before persistence."""

    src_key: str
    dst_key: str
    edge_type: str = "parent_child"
    attrs: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class WikiCrossLinkDraft:
    """A conservative evidence-backed cross-session topic link."""

    src_key: str
    dst_key: str
    evidence_type: str
    evidence_refs: tuple[JsonDict, ...] = ()
    attrs: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class WikiSnapshotDraft:
    """Complete staged Wiki Map payload."""

    workspace_id: str
    graph_version_id: str
    topics: tuple[WikiTopicDraft, ...] = ()
    tree_edges: tuple[WikiTreeEdgeDraft, ...] = ()
    cross_links: tuple[WikiCrossLinkDraft, ...] = ()
    stats: JsonDict = field(default_factory=dict)
