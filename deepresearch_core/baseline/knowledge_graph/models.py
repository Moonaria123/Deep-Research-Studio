"""DTOs for the research knowledge graph API and builder."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


JsonDict = dict[str, Any]


class NodeDraft(BaseModel):
    """A deterministic projected node before it is stored for one graph version."""

    node_type: Literal["session", "source", "report", "note", "memory", "concept"]
    node_key: str
    title: str
    summary: str = ""
    attrs: JsonDict = Field(default_factory=dict)
    confidence: float = 1.0
    visible: bool = True


class EdgeDraft(BaseModel):
    """A deterministic projected edge keyed by type and endpoint node keys."""

    edge_type: str
    src_key: str
    dst_key: str
    weight: float = 1.0
    confidence: float = 1.0
    visible: bool = True
    attrs: JsonDict = Field(default_factory=dict)

    @property
    def owner_key(self) -> str:
        """Return the stable owner key used by provenance rows before storage."""

        return f"edge:{self.edge_type}:{self.src_key}->{self.dst_key}"


class ProvenanceDraft(BaseModel):
    """A provenance record attached to a projected node or edge."""

    owner_kind: Literal["node", "edge"]
    owner_key: str
    source_kind: str
    source_id: str
    span: JsonDict = Field(default_factory=dict)
    evidence: JsonDict = Field(default_factory=dict)


class ProjectedGraph(BaseModel):
    """Complete deterministic graph projection before version storage."""

    workspace_id: str = "default"
    base_snapshot: JsonDict = Field(default_factory=dict)
    nodes: list[NodeDraft] = Field(default_factory=list)
    edges: list[EdgeDraft] = Field(default_factory=list)
    provenance: list[ProvenanceDraft] = Field(default_factory=list)
    stats: JsonDict = Field(default_factory=dict)
