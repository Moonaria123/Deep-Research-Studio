"""Deterministic, bounded concept projection over canonical KG artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .models import EdgeDraft, JsonDict, NodeDraft, ProvenanceDraft
from .wiki_schema import safe_wiki_concept_key


MAX_CONCEPTS_PER_ARTIFACT = 8
MAX_CONCEPTS_PER_WORKSPACE = 64
MAX_P2_EDGES_PER_WORKSPACE = 256
_MAX_TITLE = 90
_MAX_SUMMARY = 280
_MAX_CONTENT_SCAN = 64 * 1024
_ARTIFACT_NODE_TYPES = {"source", "note", "report", "memory"}
_STRUCTURED_KEYS = ("concepts", "topics", "keywords", "tags")
_EDGE_TYPES = {
    "source": "concept_supported_by_source",
    "note": "concept_mentioned_in_note",
    "report": "concept_summarized_by_report",
    "memory": "concept_remembered_in_memory",
}
_WIKI_CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.5}
_SECRET_RE = re.compile(
    r"("
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{6,}|"
    r"(?<![A-Za-z0-9])ghp_[A-Za-z0-9_]{6,}|"
    r"AKIA[0-9A-Z]{8,}|"
    r"(?<![A-Za-z0-9])xox[abp]-[A-Za-z0-9-]{6,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r")",
    re.IGNORECASE,
)
_GENERIC_TITLE_RE = re.compile(
    r"^(?:untitled|source|report|note|memory)(?:\s+"
    r"(?:[A-Za-z0-9_-]{6,}|document|source|report|note|memory))?$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ConceptArtifact:
    """One canonical artifact eligible for deterministic concept projection."""

    node_key: str
    node_type: Literal["source", "note", "report", "memory"]
    title: str = ""
    summary: str = ""
    metadata: JsonDict = field(default_factory=dict)
    content: str = ""


@dataclass(frozen=True)
class CanonicalEvidenceInput:
    """Secret-free identity material used to compute KG freshness."""

    stable_key: str
    content_hash: str
    source_status: str = "active"
    deleted: bool = False
    provenance_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalEvidenceSignature:
    """Digest and cardinality for one canonical evidence surface."""

    digest: str
    input_count: int

    def to_dict(self) -> JsonDict:
        return {
            "canonical_evidence_signature": self.digest,
            "input_count": self.input_count,
        }


@dataclass(frozen=True)
class ConceptProjection:
    """Bounded concept nodes, P2 edges, provenance and audit statistics."""

    nodes: tuple[NodeDraft, ...] = ()
    edges: tuple[EdgeDraft, ...] = ()
    provenance: tuple[ProvenanceDraft, ...] = ()
    stats: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    concept_key: str
    title: str
    summary: str
    artifact_key: str
    confidence: float
    priority: int
    source_kind: str
    source_id: str
    method: str
    evidence: JsonDict


def content_sha256(value: str) -> str:
    """Return a stable digest without retaining source content."""

    secret_free = _SECRET_RE.sub("[redacted]", str(value or ""))
    return hashlib.sha256(secret_free.encode("utf-8")).hexdigest()


def compute_canonical_evidence_signature(
    inputs: Iterable[CanonicalEvidenceInput],
) -> CanonicalEvidenceSignature:
    """Compute the canonical KG freshness signature from selected artifacts."""

    grouped: dict[str, JsonDict] = {}
    for item in inputs:
        stable_key = _safe_identifier(item.stable_key, limit=240)
        if not stable_key:
            continue
        current = grouped.setdefault(
            stable_key,
            {
                "stable_key": stable_key,
                "content_hashes": set(),
                "source_statuses": set(),
                "deleted": False,
                "provenance_refs": set(),
            },
        )
        content_hash = _safe_identifier(item.content_hash, limit=128)
        if content_hash:
            current["content_hashes"].add(content_hash)
        status = _safe_identifier(item.source_status or "active", limit=32)
        current["source_statuses"].add(status or "active")
        current["deleted"] = bool(current["deleted"] or item.deleted)
        current["provenance_refs"].update(
            ref
            for ref in (
                _safe_identifier(raw, limit=240)
                for raw in item.provenance_refs
            )
            if ref
        )
    payload = [
        {
            "stable_key": stable_key,
            "content_hashes": sorted(value["content_hashes"]),
            "source_statuses": sorted(value["source_statuses"]),
            "deleted": bool(value["deleted"]),
            "provenance_refs": sorted(value["provenance_refs"]),
        }
        for stable_key, value in sorted(grouped.items())
    ]
    encoded = json.dumps(
        {
            "schema": "canonical-evidence-signature/v1",
            "inputs": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return CanonicalEvidenceSignature(
        digest=hashlib.sha256(encoded).hexdigest(),
        input_count=len(payload),
    )


def extract_concepts(
    *,
    artifacts: Iterable[ConceptArtifact],
    wiki_topics: Iterable[JsonDict],
    wiki_snapshot_id: str,
) -> ConceptProjection:
    """Project bounded, evidence-backed concept nodes without remote calls."""

    artifact_by_key = {
        artifact.node_key: artifact
        for artifact in artifacts
        if artifact.node_key and artifact.node_type in _ARTIFACT_NODE_TYPES
    }
    candidates = [
        *_wiki_candidates(
            artifact_by_key,
            wiki_topics=wiki_topics,
            wiki_snapshot_id=wiki_snapshot_id,
        ),
        *_heuristic_candidates(artifact_by_key),
    ]
    candidates = _dedupe_candidates(candidates)
    candidates_before_artifact_limit = len(candidates)
    candidates = _cap_candidates_per_artifact(candidates)
    candidates_after_artifact_limit = len(candidates)

    grouped_by_concept: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped_by_concept[candidate.concept_key].append(candidate)
    concepts_before_workspace_limit = len(grouped_by_concept)
    retained_concepts = {
        key
        for key, _ in sorted(
            grouped_by_concept.items(),
            key=lambda item: (
                min(candidate.priority for candidate in item[1]),
                -max(candidate.confidence for candidate in item[1]),
                item[0],
            ),
        )[:MAX_CONCEPTS_PER_WORKSPACE]
    }
    candidates = [
        candidate
        for candidate in candidates
        if candidate.concept_key in retained_concepts
    ]

    grouped_edges: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped_edges[(candidate.concept_key, candidate.artifact_key)].append(candidate)
    edges_before_workspace_limit = len(grouped_edges)
    retained_edge_groups = sorted(
        grouped_edges.items(),
        key=lambda item: (
            item[0][0],
            _EDGE_TYPES[artifact_by_key[item[0][1]].node_type],
            item[0][1],
        ),
    )[:MAX_P2_EDGES_PER_WORKSPACE]

    retained_candidates: list[_Candidate] = []
    edges: list[EdgeDraft] = []
    provenance: dict[tuple[str, str, str, str], ProvenanceDraft] = {}
    artifact_keys_by_concept: dict[str, set[str]] = defaultdict(set)
    for (concept_key, artifact_key), edge_candidates in retained_edge_groups:
        artifact = artifact_by_key[artifact_key]
        edge_type = _EDGE_TYPES[artifact.node_type]
        edge = EdgeDraft(
            edge_type=edge_type,
            src_key=f"concept:{concept_key}",
            dst_key=artifact_key,
            weight=max(candidate.confidence for candidate in edge_candidates),
            confidence=max(candidate.confidence for candidate in edge_candidates),
            attrs={
                "concept_key": concept_key,
                "artifact_key": artifact_key,
                "source_status": _safe_identifier(
                    str(artifact.metadata.get("source_status") or "active"),
                    limit=32,
                )
                or "active",
                "source_count": len(
                    {
                        (candidate.source_kind, candidate.source_id)
                        for candidate in edge_candidates
                    }
                ),
            },
        )
        edges.append(edge)
        artifact_keys_by_concept[concept_key].add(artifact_key)
        retained_candidates.extend(edge_candidates)
        for candidate in edge_candidates:
            _add_provenance(
                provenance,
                ProvenanceDraft(
                    owner_kind="edge",
                    owner_key=edge.owner_key,
                    source_kind=candidate.source_kind,
                    source_id=candidate.source_id,
                    evidence={
                        **candidate.evidence,
                        "method": candidate.method,
                        "concept_key": concept_key,
                        "artifact_key": artifact_key,
                    },
                ),
            )

    retained_by_concept: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in retained_candidates:
        retained_by_concept[candidate.concept_key].append(candidate)
    nodes: list[NodeDraft] = []
    wiki_source_ids: set[str] = set()
    for concept_key, concept_candidates in sorted(retained_by_concept.items()):
        ranked = sorted(concept_candidates, key=_candidate_sort_key)
        title = next((candidate.title for candidate in ranked if candidate.title), concept_key)
        summary = next((candidate.summary for candidate in ranked if candidate.summary), "")
        artifact_keys = sorted(artifact_keys_by_concept[concept_key])
        node = NodeDraft(
            node_type="concept",
            node_key=f"concept:{concept_key}",
            title=_bounded(title, _MAX_TITLE),
            summary=_bounded(summary, _MAX_SUMMARY),
            attrs={
                "concept_key": concept_key,
                "canonical_name": _bounded(title, _MAX_TITLE),
                "source_count": len(artifact_keys),
                "artifact_keys": artifact_keys[:16],
                "extraction_methods": sorted(
                    {candidate.method for candidate in concept_candidates}
                ),
            },
            confidence=max(candidate.confidence for candidate in concept_candidates),
        )
        nodes.append(node)
        for candidate in concept_candidates:
            if candidate.source_kind == "wiki_topic":
                wiki_source_ids.add(candidate.source_id)
            _add_provenance(
                provenance,
                ProvenanceDraft(
                    owner_kind="node",
                    owner_key=node.node_key,
                    source_kind=candidate.source_kind,
                    source_id=candidate.source_id,
                    evidence={
                        **candidate.evidence,
                        "method": candidate.method,
                        "concept_key": concept_key,
                        "artifact_key": candidate.artifact_key,
                    },
                ),
            )

    return ConceptProjection(
        nodes=tuple(nodes),
        edges=tuple(edges),
        provenance=tuple(
            provenance[key]
            for key in sorted(provenance)
        ),
        stats={
            "concept_nodes": len(nodes),
            "p2_edges": len(edges),
            "concept_provenance": len(provenance),
            "concept_candidates": len(candidates),
            "concept_candidates_before_artifact_limit": candidates_before_artifact_limit,
            "concept_candidates_truncated_by_artifact_limit": max(
                0,
                candidates_before_artifact_limit - candidates_after_artifact_limit,
            ),
            "concepts_before_workspace_limit": concepts_before_workspace_limit,
            "concepts_truncated_by_workspace_limit": max(
                0,
                concepts_before_workspace_limit - len(nodes),
            ),
            "p2_edges_before_workspace_limit": edges_before_workspace_limit,
            "p2_edges_truncated_by_workspace_limit": max(
                0,
                edges_before_workspace_limit - len(edges),
            ),
            "concept_limit": MAX_CONCEPTS_PER_WORKSPACE,
            "p2_edge_limit": MAX_P2_EDGES_PER_WORKSPACE,
            "concepts_per_artifact_limit": MAX_CONCEPTS_PER_ARTIFACT,
            "wiki_seed_snapshot_id": _safe_identifier(wiki_snapshot_id, limit=120),
            "wiki_seed_count": len(wiki_source_ids),
        },
    )


def _wiki_candidates(
    artifact_by_key: dict[str, ConceptArtifact],
    *,
    wiki_topics: Iterable[JsonDict],
    wiki_snapshot_id: str,
) -> list[_Candidate]:
    out: list[_Candidate] = []
    sorted_topics = sorted(
        (dict(topic) for topic in wiki_topics if isinstance(topic, dict)),
        key=lambda topic: str(
            topic.get("id")
            or topic.get("key")
            or topic.get("topic_key")
            or ""
        ),
    )
    for topic in sorted_topics:
        attrs = topic.get("attrs")
        attrs = attrs if isinstance(attrs, dict) else {}
        raw_key = str(attrs.get("concept_key") or topic.get("concept_key") or "")
        normalized_key_source = _clean_term(raw_key)
        title = _clean_term(str(topic.get("title") or raw_key))
        if not normalized_key_source and not title:
            continue
        concept_key = safe_wiki_concept_key(normalized_key_source or title)
        if not concept_key or concept_key == "topic":
            continue
        confidence_label = str(
            attrs.get("confidence")
            or topic.get("confidence")
            or "medium"
        ).lower()
        confidence = _WIKI_CONFIDENCE.get(confidence_label, 0.7)
        priority = {"high": 0, "medium": 1, "low": 2}.get(confidence_label, 1)
        summary = _safe_summary(str(topic.get("summary") or ""))
        topic_id = _safe_identifier(
            str(topic.get("key") or topic.get("topic_key") or topic.get("id") or ""),
            limit=120,
        )
        if not topic_id:
            continue
        refs = attrs.get("evidence_refs")
        refs = refs if isinstance(refs, list) else []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            artifact_key = str(ref.get("id") or ref.get("node_key") or "").strip()
            artifact = artifact_by_key.get(artifact_key)
            if artifact is None:
                continue
            ref_type = str(ref.get("type") or "").strip().lower()
            if ref_type and ref_type not in {
                artifact.node_type,
                "library_document" if artifact.node_type == "source" else artifact.node_type,
            }:
                continue
            out.append(
                _Candidate(
                    concept_key=concept_key,
                    title=title or concept_key,
                    summary=summary,
                    artifact_key=artifact_key,
                    confidence=confidence,
                    priority=priority,
                    source_kind="wiki_topic",
                    source_id=topic_id,
                    method="wiki_evidence_ref",
                    evidence={
                        "wiki_snapshot_id": _safe_identifier(
                            wiki_snapshot_id,
                            limit=120,
                        ),
                        "confidence": confidence_label,
                        "excerpt": _safe_summary(str(ref.get("excerpt") or "")),
                    },
                )
            )
    return out


def _heuristic_candidates(
    artifact_by_key: dict[str, ConceptArtifact],
) -> list[_Candidate]:
    out: list[_Candidate] = []
    for artifact_key, artifact in sorted(artifact_by_key.items()):
        terms: list[tuple[int, float, str, str]] = []
        for term in _structured_terms(artifact.metadata):
            terms.append((3, 0.65, term, "structured_metadata"))
        parsed_content = _safe_json(artifact.content)
        if isinstance(parsed_content, (dict, list)):
            for term in _structured_terms(parsed_content):
                terms.append((3, 0.65, term, "structured_content"))
        for heading in _MARKDOWN_HEADING_RE.findall(
            str(artifact.content or "")[:_MAX_CONTENT_SCAN]
        ):
            terms.append((4, 0.6, heading, "markdown_heading"))
        if not _is_generic_title(artifact.title):
            terms.append((5, 0.55, artifact.title, "title_heuristic"))

        seen: set[str] = set()
        for priority, confidence, raw_term, method in sorted(
            terms,
            key=lambda item: (
                item[0],
                safe_wiki_concept_key(item[2]),
                item[3],
            ),
        ):
            title = _clean_term(raw_term)
            if not title:
                continue
            concept_key = safe_wiki_concept_key(title)
            if not concept_key or concept_key == "topic" or concept_key in seen:
                continue
            seen.add(concept_key)
            out.append(
                _Candidate(
                    concept_key=concept_key,
                    title=title,
                    summary=_safe_summary(artifact.summary),
                    artifact_key=artifact_key,
                    confidence=confidence,
                    priority=priority,
                    source_kind="artifact_heuristic",
                    source_id=_safe_identifier(artifact_key, limit=240),
                    method=method,
                    evidence={
                        "source_status": _safe_identifier(
                            str(artifact.metadata.get("source_status") or "active"),
                            limit=32,
                        )
                        or "active",
                    },
                )
            )
    return out


def _structured_terms(value: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key in _STRUCTURED_KEYS:
            if key in value:
                values.append(value.get(key))
    elif isinstance(value, list):
        values.append(value)
    terms: list[str] = []
    for raw in values:
        items = raw if isinstance(raw, list) else [raw]
        for item in items[:32]:
            if isinstance(item, dict):
                for key in ("title", "name", "topic", "concept", "key"):
                    if isinstance(item.get(key), str):
                        terms.append(str(item[key]))
                        break
            elif isinstance(item, str):
                terms.extend(re.split(r"[,;\n|]+", item))
    return terms[:64]


def _safe_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text or len(text) > _MAX_CONTENT_SCAN or text[:1] not in {"{", "["}:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _dedupe_candidates(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    deduped: dict[tuple[str, str, str, str], _Candidate] = {}
    for candidate in candidates:
        key = (
            candidate.concept_key,
            candidate.artifact_key,
            candidate.source_kind,
            candidate.source_id,
        )
        current = deduped.get(key)
        if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
            deduped[key] = candidate
    return sorted(deduped.values(), key=_candidate_sort_key)


def _cap_candidates_per_artifact(
    candidates: Iterable[_Candidate],
) -> list[_Candidate]:
    grouped: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.artifact_key].append(candidate)
    out: list[_Candidate] = []
    for artifact_key in sorted(grouped):
        unique_concepts: set[str] = set()
        for candidate in sorted(grouped[artifact_key], key=_candidate_sort_key):
            if (
                candidate.concept_key not in unique_concepts
                and len(unique_concepts) >= MAX_CONCEPTS_PER_ARTIFACT
            ):
                continue
            unique_concepts.add(candidate.concept_key)
            out.append(candidate)
    return out


def _candidate_sort_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        candidate.priority,
        -candidate.confidence,
        candidate.concept_key,
        candidate.artifact_key,
        candidate.source_kind,
        candidate.source_id,
        candidate.title,
    )


def _add_provenance(
    target: dict[tuple[str, str, str, str], ProvenanceDraft],
    item: ProvenanceDraft,
) -> None:
    key = (
        item.owner_kind,
        item.owner_key,
        item.source_kind,
        item.source_id,
    )
    current = target.get(key)
    if current is None:
        target[key] = item
        return
    target[key] = ProvenanceDraft(
        owner_kind=item.owner_kind,
        owner_key=item.owner_key,
        source_kind=item.source_kind,
        source_id=item.source_id,
        span=_merge_json(current.span, item.span),
        evidence=_merge_json(current.evidence, item.evidence),
    )


def _merge_json(left: JsonDict, right: JsonDict) -> JsonDict:
    merged: JsonDict = dict(left)
    for key in sorted(set(left) | set(right)):
        if key not in left:
            merged[key] = right[key]
            continue
        if key not in right or left[key] == right[key]:
            merged[key] = left[key]
            continue
        values = []
        for value in (left[key], right[key]):
            values.extend(value if isinstance(value, list) else [value])
        unique = {
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str): value
            for value in values
        }
        merged[key] = [unique[item] for item in sorted(unique)]
    return merged


def _is_generic_title(value: str) -> bool:
    title = " ".join(str(value or "").split())
    if not title:
        return True
    return bool(_GENERIC_TITLE_RE.fullmatch(title))


def _clean_term(value: str) -> str:
    text = " ".join(str(value or "").split()).strip("`#*_~[](){}<>:;,. ")
    if (
        not text
        or _SECRET_RE.search(text)
        or text.lower().startswith(("http://", "https://"))
        or len(text) > _MAX_TITLE * 2
    ):
        return ""
    return _bounded(text, _MAX_TITLE)


def _safe_summary(value: str) -> str:
    text = " ".join(str(value or "").split())
    text = _SECRET_RE.sub("[redacted]", text)
    return _bounded(text, _MAX_SUMMARY)


def _safe_identifier(value: str, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text or _SECRET_RE.search(text):
        return ""
    return re.sub(r"[^A-Za-z0-9:._/-]+", "", text)[:limit]


def _bounded(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
