"""Bounded KG candidate discovery that hydrates canonical evidence artifacts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

from ..evidence_trust import evidence_identity
from ..retrieval import EvidenceItem
from .wiki_schema import safe_wiki_concept_key

JsonDict = dict[str, Any]

_SUPPORTED_MODES = frozenset({"multiple", "deep", "council"})
_CONCEPT_LIMIT = 8
_EDGE_LIMIT = 24
_CANDIDATE_LIMIT = 12
_DOCUMENT_CHUNK_LIMIT = 64
_TEXT_LIMIT = 16 * 1024
_CONCEPT_EDGE_TYPES = frozenset(
    {
        "concept_mentioned_in_note",
        "concept_supported_by_source",
        "concept_summarized_by_report",
        "concept_remembered_in_memory",
    }
)
_NODE_ARTIFACT_TYPES = {
    "source": frozenset({"web_document", "library_document"}),
    "note": frozenset({"library_note"}),
    "report": frozenset({"library_report"}),
    "memory": frozenset({"memory"}),
}


def _checkpoint_bool(value: Any, *, default: bool = False) -> bool:
    """Accept only actual booleans from a checkpoint payload."""

    return value if isinstance(value, bool) else default


def _checkpoint_nonnegative_int(value: Any) -> int:
    """Coerce compatible counters without letting malformed data break resume."""

    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _checkpoint_optional_string(value: Any) -> str | None:
    """Restore identifiers only from non-empty strings."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _checkpoint_string_list(value: Any) -> list[str]:
    """Restore a deterministic, de-duplicated list from a compatible payload."""

    if not isinstance(value, (list, tuple)):
        return []
    restored: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized and normalized not in restored:
            restored.append(normalized)
    return restored


@dataclass
class KGRetrievalDiagnostics:
    """Run-visible facts for KG discovery, separate from evidence provenance."""

    requested: bool
    effective: bool = False
    attempted: bool = False
    applied: bool = False
    snapshot_version_id: str | None = None
    wiki_snapshot_id: str | None = None
    candidate_count: int = 0
    hydrated_count: int = 0
    adopted_evidence_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    _discovered_evidence_ids: set[str] = field(
        default_factory=set,
        repr=False,
        compare=False,
    )

    def add_reason(self, reason: str) -> None:
        """Append one stable reason code without duplication."""

        value = str(reason or "").strip()
        if value and value not in self.reason_codes:
            self.reason_codes.append(value)

    def register_discovered(self, evidence: Iterable[EvidenceItem]) -> None:
        """Remember canonical evidence identities without claiming adoption."""

        self._discovered_evidence_ids.update(
            evidence_identity(item)
            for item in evidence
        )

    def mark_adopted(self, evidence: Iterable[EvidenceItem]) -> None:
        """Set ``applied`` only after the final adopted evidence snapshot exists."""

        adopted = {
            evidence_identity(item)
            for item in evidence
        }
        self.adopted_evidence_ids = sorted(
            self._discovered_evidence_ids.intersection(adopted)
        )
        self.applied = bool(self.adopted_evidence_ids)

    def merge(self, other: "KGRetrievalDiagnostics") -> None:
        """Aggregate one planned-query diagnostic into the run-level view."""

        self.requested = self.requested or other.requested
        self.effective = self.effective or other.effective
        self.attempted = self.attempted or other.attempted
        self.candidate_count += max(0, int(other.candidate_count))
        self.hydrated_count += max(0, int(other.hydrated_count))
        self._discovered_evidence_ids.update(other._discovered_evidence_ids)
        if self.snapshot_version_id is None:
            self.snapshot_version_id = other.snapshot_version_id
        elif (
            other.snapshot_version_id is not None
            and other.snapshot_version_id != self.snapshot_version_id
        ):
            self.add_reason("snapshot_version_changed")
        if self.wiki_snapshot_id is None:
            self.wiki_snapshot_id = other.wiki_snapshot_id
        for reason in other.reason_codes:
            self.add_reason(reason)
        # Query-level hydration never proves final adoption.
        self.applied = False
        self.adopted_evidence_ids = []

    def to_dict(self) -> JsonDict:
        """Return the backward-compatible public run-result representation."""

        return {
            "requested": bool(self.requested),
            "effective": bool(self.effective),
            "attempted": bool(self.attempted),
            "applied": bool(self.applied),
            "snapshot_version_id": self.snapshot_version_id,
            "wiki_snapshot_id": self.wiki_snapshot_id,
            "candidate_count": max(0, int(self.candidate_count)),
            "hydrated_count": max(0, int(self.hydrated_count)),
            "adopted_evidence_ids": list(self.adopted_evidence_ids),
            "reason_codes": list(self.reason_codes),
        }

    def to_checkpoint_dict(self) -> JsonDict:
        """Return secret-free lineage state, including non-public discovery identities."""

        return {
            **self.to_dict(),
            "discovered_evidence_ids": sorted(self._discovered_evidence_ids),
        }

    @classmethod
    def from_checkpoint_dict(
        cls,
        value: Any,
        *,
        requested: bool,
    ) -> "KGRetrievalDiagnostics":
        """Restore bounded diagnostic state from a validated checkpoint object."""

        data = value if isinstance(value, dict) else {}
        diagnostics = cls(
            requested=_checkpoint_bool(
                data.get("requested"),
                default=requested,
            ),
            effective=_checkpoint_bool(data.get("effective")),
            attempted=_checkpoint_bool(data.get("attempted")),
            applied=False,
            snapshot_version_id=_checkpoint_optional_string(
                data.get("snapshot_version_id")
            ),
            wiki_snapshot_id=_checkpoint_optional_string(
                data.get("wiki_snapshot_id")
            ),
            candidate_count=_checkpoint_nonnegative_int(
                data.get("candidate_count")
            ),
            hydrated_count=_checkpoint_nonnegative_int(
                data.get("hydrated_count")
            ),
            reason_codes=_checkpoint_string_list(data.get("reason_codes")),
        )
        diagnostics._discovered_evidence_ids.update(
            _checkpoint_string_list(data.get("discovered_evidence_ids"))
        )
        return diagnostics


@dataclass(frozen=True)
class KGRetrievalResult:
    """One bounded KG expansion and its secret-free diagnostics."""

    evidence: tuple[EvidenceItem, ...]
    diagnostics: KGRetrievalDiagnostics


class KnowledgeGraphEvidenceRetriever:
    """Discover canonical evidence through one frozen KG snapshot.

    The graph contributes candidate identities only. Hydrated evidence retains
    the canonical ``local``, ``memory``, or ``external`` provenance family.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._conn = store._conn
        self._lock = store._lock

    def retrieve(
        self,
        query: str,
        *,
        workspace_id: str,
        requested: bool,
        mode: str,
        snapshot_version_id: str,
        snapshot_signature: str,
    ) -> KGRetrievalResult:
        """Return at most twelve canonical artifacts through a one-hop expansion."""

        diagnostics = KGRetrievalDiagnostics(requested=bool(requested))
        if not requested:
            return KGRetrievalResult((), diagnostics)
        normalized_mode = str(getattr(mode, "value", mode) or "").strip().lower()
        if normalized_mode not in _SUPPORTED_MODES:
            diagnostics.add_reason("mode_not_supported")
            return KGRetrievalResult((), diagnostics)

        version_id = str(snapshot_version_id or "").strip()
        signature = str(snapshot_signature or "").strip()
        if not version_id or not signature:
            diagnostics.add_reason("snapshot_unavailable")
            return KGRetrievalResult((), diagnostics)

        with self._lock:
            version = self._conn.execute(
                "SELECT * FROM kg_graph_versions WHERE id=? LIMIT 1",
                (version_id,),
            ).fetchone()
            if version is None:
                diagnostics.add_reason("snapshot_unavailable")
                return KGRetrievalResult((), diagnostics)
            if str(version["workspace_id"] or "default") != str(
                workspace_id or "default"
            ):
                diagnostics.add_reason("workspace_mismatch")
                return KGRetrievalResult((), diagnostics)
            diagnostics.snapshot_version_id = str(version["id"])
            if str(version["status"] or "") != "active":
                diagnostics.add_reason("graph_not_ready")
                return KGRetrievalResult((), diagnostics)
            base_snapshot = _json_object(version["base_snapshot_json"])
            stored_signature = str(
                base_snapshot.get("canonical_evidence_signature") or ""
            ).strip()
            if not stored_signature:
                diagnostics.add_reason("snapshot_signature_unavailable")
                return KGRetrievalResult((), diagnostics)
            if stored_signature != signature:
                diagnostics.add_reason("snapshot_signature_mismatch")
                return KGRetrievalResult((), diagnostics)
            wiki = self._conn.execute(
                "SELECT id FROM kg_wiki_snapshots "
                "WHERE workspace_id=? AND graph_version_id=? "
                "AND is_active=1 AND stale=0 AND status IN ('succeeded','partial') "
                "ORDER BY activated_at DESC,id ASC LIMIT 1",
                (str(workspace_id or "default"), version_id),
            ).fetchone()
            if wiki is not None:
                diagnostics.wiki_snapshot_id = str(wiki["id"])
            candidates, concept_count = self._query_candidates_locked(
                query,
                version_id=version_id,
                wiki_snapshot_id=diagnostics.wiki_snapshot_id,
            )

        diagnostics.effective = True
        diagnostics.attempted = True
        diagnostics.candidate_count = len(candidates)
        if concept_count == 0:
            diagnostics.add_reason("no_concept_match")
            return KGRetrievalResult((), diagnostics)
        if not candidates:
            diagnostics.add_reason("no_candidates")
            return KGRetrievalResult((), diagnostics)

        hydrated: list[EvidenceItem] = []
        seen: set[str] = set()
        retrieved_at = time.time()
        for rank, candidate in enumerate(candidates):
            evidence = self._hydrate_candidate(
                candidate,
                query=query,
                score=1.0 / float(rank + 1),
                retrieved_at=retrieved_at,
            )
            if evidence is None:
                continue
            identity = evidence_identity(evidence)
            if identity in seen:
                continue
            seen.add(identity)
            hydrated.append(evidence)
        diagnostics.hydrated_count = len(hydrated)
        diagnostics.register_discovered(hydrated)
        if not hydrated:
            diagnostics.add_reason("no_hydrated_candidates")
        return KGRetrievalResult(tuple(hydrated), diagnostics)

    def _query_candidates_locked(
        self,
        query: str,
        *,
        version_id: str,
        wiki_snapshot_id: str | None,
    ) -> tuple[list[JsonDict], int]:
        concept_rows = self._conn.execute(
            "SELECT id,node_key,title,summary FROM kg_nodes "
            "WHERE version_id=? AND node_type='concept' AND visible=1 "
            "ORDER BY node_key ASC,id ASC",
            (version_id,),
        ).fetchall()
        concept_matches: list[tuple[float, str, str, JsonDict]] = [
            (
                _concept_match_score(query, row),
                "graph",
                str(row["node_key"] or ""),
                dict(row),
            )
            for row in concept_rows
        ]
        if wiki_snapshot_id:
            topic_rows = self._conn.execute(
                "SELECT id,topic_key,title,attrs_json FROM kg_wiki_topics "
                "WHERE snapshot_id=? ORDER BY topic_key ASC,id ASC",
                (wiki_snapshot_id,),
            ).fetchall()
            for topic in topic_rows:
                attrs = _json_object(topic["attrs_json"])
                concept_key = safe_wiki_concept_key(
                    attrs.get("concept_key")
                    or topic["topic_key"]
                    or topic["title"]
                )
                if not concept_key or concept_key == "topic":
                    continue
                concept_matches.append(
                    (
                        _concept_match_score(
                            query,
                            {
                                "node_key": f"concept:{concept_key}",
                                "title": topic["title"],
                            },
                        ),
                        "wiki",
                        concept_key,
                        {
                            "id": str(topic["id"]),
                            "node_key": f"concept:{concept_key}",
                            "title": str(topic["title"] or concept_key),
                            "evidence_refs": (
                                attrs.get("evidence_refs")
                                if isinstance(attrs.get("evidence_refs"), list)
                                else []
                            ),
                        },
                    )
                )
        ranked_concepts = sorted(
            concept_matches,
            key=lambda item: (
                -item[0],
                item[2],
                item[1],
                str(item[3].get("id") or ""),
            ),
        )
        ranked_concepts = [
            item
            for item in ranked_concepts
            if item[0] > 0.0
        ][:_CONCEPT_LIMIT]
        if not ranked_concepts:
            return [], 0

        graph_rank = {
            str(row["id"]): index
            for index, (_, kind, _, row) in enumerate(ranked_concepts)
            if kind == "graph"
        }
        traversed: list[tuple[int, str, str, str, JsonDict]] = []
        if graph_rank:
            concept_ids = list(graph_rank)
            concept_placeholders = ",".join("?" for _ in concept_ids)
            edge_placeholders = ",".join("?" for _ in _CONCEPT_EDGE_TYPES)
            edge_rows = self._conn.execute(
                "SELECT e.id AS edge_id,e.edge_type,e.src_node_id,e.dst_node_id,"
                "n.node_type,n.node_key,n.title,n.attrs_json "
                "FROM kg_edges e JOIN kg_nodes n "
                "ON n.id=e.dst_node_id AND n.version_id=e.version_id "
                f"WHERE e.version_id=? AND e.src_node_id IN ({concept_placeholders}) "
                f"AND e.edge_type IN ({edge_placeholders}) "
                "AND e.visible=1 AND n.visible=1",
                (
                    version_id,
                    *concept_ids,
                    *sorted(_CONCEPT_EDGE_TYPES),
                ),
            ).fetchall()
            traversed.extend(
                (
                    graph_rank.get(
                        str(row["src_node_id"] or ""),
                        _CONCEPT_LIMIT,
                    ),
                    str(row["edge_type"] or ""),
                    str(row["node_key"] or ""),
                    str(row["edge_id"] or ""),
                    dict(row),
                )
                for row in edge_rows
            )

        for rank, (_, kind, _, concept) in enumerate(ranked_concepts):
            if kind != "wiki":
                continue
            refs = sorted(
                (
                    dict(ref)
                    for ref in concept.get("evidence_refs", [])
                    if isinstance(ref, dict)
                ),
                key=lambda ref: (
                    str(ref.get("id") or ref.get("node_key") or ""),
                    str(ref.get("type") or ""),
                ),
            )[:_EDGE_LIMIT]
            for ref in refs:
                artifact_key = str(
                    ref.get("id")
                    or ref.get("node_key")
                    or ""
                ).strip()
                if not artifact_key:
                    continue
                node = self._conn.execute(
                    "SELECT id AS dst_node_id,node_type,node_key,title,attrs_json "
                    "FROM kg_nodes WHERE version_id=? AND visible=1 "
                    "AND (node_key=? OR id=?) "
                    "ORDER BY CASE WHEN node_key=? THEN 0 ELSE 1 END,id ASC LIMIT 1",
                    (version_id, artifact_key, artifact_key, artifact_key),
                ).fetchone()
                if node is None:
                    continue
                row = dict(node)
                attrs = _json_object(row.get("attrs_json"))
                if not _wiki_ref_matches_node(
                    str(ref.get("type") or ""),
                    node_type=str(row.get("node_type") or ""),
                    artifact_type=str(attrs.get("artifact_type") or ""),
                ):
                    continue
                edge_type = _edge_type_for_node(
                    str(row.get("node_type") or "")
                )
                if not edge_type:
                    continue
                row.update(
                    {
                        "edge_id": f"wiki:{concept['id']}:{artifact_key}",
                        "edge_type": edge_type,
                        "src_node_id": str(concept["id"]),
                    }
                )
                traversed.append(
                    (
                        rank,
                        edge_type,
                        str(row.get("node_key") or ""),
                        str(row.get("edge_id") or ""),
                        row,
                    )
                )

        ordered_edges = [
            item[4]
            for item in sorted(
                traversed,
                key=lambda item: item[:4],
            )[:_EDGE_LIMIT]
        ]

        candidates: list[JsonDict] = []
        seen: set[tuple[str, str]] = set()
        for row in ordered_edges:
            node_type = str(row.get("node_type") or "")
            attrs = _json_object(row.get("attrs_json"))
            artifact_type = str(attrs.get("artifact_type") or "")
            artifact_id = str(
                attrs.get("artifact_id")
                or attrs.get("document_id")
                or attrs.get("memory_id")
                or attrs.get("note_id")
                or attrs.get("report_id")
                or ""
            ).strip()
            if (
                artifact_type not in _NODE_ARTIFACT_TYPES.get(node_type, ())
                or not artifact_id
            ):
                continue
            identity = (artifact_type, artifact_id)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                {
                    "node_type": node_type,
                    "artifact_type": artifact_type,
                    "artifact_id": artifact_id,
                    "attrs": attrs,
                }
            )
            if len(candidates) >= _CANDIDATE_LIMIT:
                break
        return candidates, len(ranked_concepts)

    def _hydrate_candidate(
        self,
        candidate: JsonDict,
        *,
        query: str,
        score: float,
        retrieved_at: float,
    ) -> EvidenceItem | None:
        artifact_type = str(candidate.get("artifact_type") or "")
        artifact_id = str(candidate.get("artifact_id") or "")
        if artifact_type in {"web_document", "library_document"}:
            return self._hydrate_document(
                artifact_id,
                expected_type=artifact_type,
                query=query,
                score=score,
                retrieved_at=retrieved_at,
            )
        if artifact_type == "library_report":
            report = self._store.get_report(artifact_id)
            if not report:
                return None
            text = _bounded_text(report.get("body_md"))
            if not text:
                return None
            return EvidenceItem(
                chunk_id=0,
                document_id="",
                url="",
                title=str(report.get("title") or "Library report"),
                text=text,
                score=score,
                channels=["knowledge_graph"],
                origin="local",
                artifact_type="library_report",
                artifact_id=artifact_id,
                provenance_refs=[f"local:library_report:{artifact_id}"],
                source_status="active",
                retrieved_at=retrieved_at,
            )
        if artifact_type in {"library_note", "memory"}:
            memory = self._store.get_memory(artifact_id)
            if not memory:
                return None
            namespace = str(memory.get("namespace") or "")
            if artifact_type == "library_note" and namespace != "note":
                return None
            if artifact_type == "memory" and namespace not in {
                "session",
                "workspace",
                "user",
            }:
                return None
            text = _bounded_text(memory.get("content"))
            if not text:
                return None
            metadata = (
                memory.get("metadata")
                if isinstance(memory.get("metadata"), dict)
                else {}
            )
            origin = "local" if artifact_type == "library_note" else "memory"
            return EvidenceItem(
                chunk_id=0,
                document_id="",
                url="",
                title=str(
                    metadata.get("title")
                    or ("Library note" if origin == "local" else f"Memory ({namespace})")
                ),
                text=text,
                score=score,
                channels=["knowledge_graph"],
                origin=origin,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                provenance_refs=[f"{origin}:{artifact_type}:{artifact_id}"],
                source_status=(
                    "archived"
                    if memory.get("archived_at") is not None
                    else "active"
                ),
                retrieved_at=retrieved_at,
            )
        return None

    def _hydrate_document(
        self,
        document_id: str,
        *,
        expected_type: str,
        query: str,
        score: float,
        retrieved_at: float,
    ) -> EvidenceItem | None:
        document = self._store.get_document(document_id)
        if not document:
            return None
        metadata = _json_object(document.get("metadata"))
        url = str(document.get("url") or "")
        canonical_type = (
            "library_document"
            if str(metadata.get("artifact_type") or "") == "library_document"
            or url.startswith("library://local/")
            else "web_document"
        )
        if canonical_type != expected_type:
            return None
        chunk = self._best_document_chunk(document_id, query)
        if chunk is None:
            return None
        if canonical_type == "web_document" and not _is_http_url(url):
            return None
        origin = "local" if canonical_type == "library_document" else "external"
        provenance_ref = (
            f"local:library_document:{document_id}"
            if origin == "local"
            else str(metadata.get("provenance_root") or url or document_id)
        )
        return EvidenceItem(
            chunk_id=int(chunk["id"]),
            document_id=document_id,
            url="" if origin == "local" else url,
            title=str(document.get("title") or ""),
            text=str(chunk["text"] or ""),
            score=score,
            channels=["knowledge_graph"],
            origin=origin,
            artifact_type=canonical_type,
            artifact_id=document_id,
            provenance_refs=[provenance_ref],
            source_status="active",
            published_at=_timestamp(metadata.get("published_at")),
            updated_at=_timestamp(metadata.get("updated_at")),
            effective_at=_timestamp(metadata.get("effective_at")),
            retrieved_at=(
                _timestamp(metadata.get("retrieved_at"))
                or _timestamp(document.get("created_at"))
                or retrieved_at
            ),
            search_provider=str(metadata.get("search_provider") or ""),
            fetch_provider=str(metadata.get("fetch_provider") or ""),
        )

    def _best_document_chunk(self, document_id: str, query: str) -> JsonDict | None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,chunk_index,text FROM chunks "
                "WHERE document_id=? ORDER BY chunk_index ASC,id ASC LIMIT ?",
                (document_id, _DOCUMENT_CHUNK_LIMIT),
            ).fetchall()
        if not rows:
            return None
        return dict(
            max(
                rows,
                key=lambda row: (
                    _text_match_score(query, str(row["text"] or "")),
                    -int(row["chunk_index"] or 0),
                    -int(row["id"]),
                ),
            )
        )


def _concept_match_score(query: str, row: Any) -> float:
    query_key = safe_wiki_concept_key(query)
    raw_node_key = str(row["node_key"] or "")
    concept_key = (
        raw_node_key.split(":", 1)[1]
        if raw_node_key.startswith("concept:")
        else safe_wiki_concept_key(row["title"])
    )
    if concept_key == query_key:
        return 4.0
    if concept_key and concept_key in query_key:
        return 3.0 + min(0.99, len(concept_key) / max(1, len(query_key)))
    if query_key and query_key in concept_key:
        return 2.5 + min(0.49, len(query_key) / max(1, len(concept_key)))
    query_terms = {term for term in query_key.split("-") if term}
    concept_terms = {term for term in concept_key.split("-") if term}
    overlap = query_terms.intersection(concept_terms)
    if not overlap:
        return 0.0
    return len(overlap) / max(1, len(concept_terms))


def _text_match_score(query: str, text: str) -> float:
    query_terms = {
        term
        for term in safe_wiki_concept_key(query).split("-")
        if len(term) > 1
    }
    text_terms = set(safe_wiki_concept_key(text).split("-"))
    return float(len(query_terms.intersection(text_terms)))


def _edge_type_for_node(node_type: str) -> str:
    return {
        "source": "concept_supported_by_source",
        "note": "concept_mentioned_in_note",
        "report": "concept_summarized_by_report",
        "memory": "concept_remembered_in_memory",
    }.get(str(node_type or ""), "")


def _wiki_ref_matches_node(
    ref_type: str,
    *,
    node_type: str,
    artifact_type: str,
) -> bool:
    normalized = str(ref_type or "").strip().lower()
    if not normalized:
        return True
    allowed = {
        str(node_type or "").strip().lower(),
        str(artifact_type or "").strip().lower(),
    }
    if node_type == "source":
        allowed.update({"source", "library_document", "web_document"})
    return normalized in allowed


def _json_object(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _timestamp(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_text(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= _TEXT_LIMIT:
        return text
    half = (_TEXT_LIMIT - 1) // 2
    return f"{text[:half]}\n{text[-half:]}"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(str(value or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
