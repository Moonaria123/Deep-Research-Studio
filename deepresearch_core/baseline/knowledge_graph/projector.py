"""Deterministic P0/P1 projection from existing research data into KG drafts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from .concept_extractor import (
    CanonicalEvidenceInput,
    ConceptArtifact,
    compute_canonical_evidence_signature,
    content_sha256,
    extract_concepts,
)
from .models import EdgeDraft, JsonDict, NodeDraft, ProjectedGraph, ProvenanceDraft
from .wiki_store import KnowledgeGraphWikiStore


def _safe_json(raw: Any) -> Any:
    if raw in (None, ""):
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _row(row: Any) -> JsonDict:
    return dict(row)


def _short(text: str, *, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _merge_json(left: JsonDict, right: JsonDict) -> JsonDict:
    """Deterministically retain evidence contributed by duplicate owners."""

    merged: JsonDict = dict(left)
    for key in sorted(set(left) | set(right)):
        if key not in left:
            merged[key] = right[key]
            continue
        if key not in right or left[key] == right[key]:
            merged[key] = left[key]
            continue
        values: list[Any] = []
        for value in (left[key], right[key]):
            values.extend(value if isinstance(value, list) else [value])
        unique = {
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str): value
            for value in values
        }
        merged[key] = [unique[item] for item in sorted(unique)]
    return merged


_SECRET_LIKE_RE = re.compile(
    r"("
    r"sk-[A-Za-z0-9_-]{8,}|"
    r"ghp_[A-Za-z0-9_]{8,}|"
    r"AKIA[0-9A-Z]{8,}|"
    r"xox[bp]-[A-Za-z0-9-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r")",
    re.IGNORECASE,
)


def _redact_secret_like(text: str) -> str:
    return _SECRET_LIKE_RE.sub("[redacted]", str(text or ""))


def _memory_preview(memory: JsonDict, *, summary_limit: int = 120, excerpt_limit: int = 320) -> JsonDict:
    """Build redacted, bounded previews from memory/note content for KG nodes."""
    content = _redact_secret_like(str(memory.get("content") or ""))
    cleaned = " ".join(content.split())
    if not cleaned:
        return {"summary": "", "content_excerpt": ""}
    return {
        "summary": _short(cleaned, limit=summary_limit),
        "content_excerpt": _short(cleaned, limit=excerpt_limit),
    }


def _assistant_summary(messages: list[JsonDict], *, limit: int = 220) -> JsonDict:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "assistant":
            continue
        content = _redact_secret_like(str(message.get("content") or ""))
        summary = _short(content, limit=limit)
        if not summary:
            continue
        return {
            "assistant_summary": summary,
            "assistant_message_id": str(message.get("id") or ""),
            "assistant_created_at": str(message.get("created_at") or ""),
        }
    return {}


def _host(url: str) -> str:
    parsed = urlparse(str(url or ""))
    return parsed.netloc or str(url or "")


def _url_key(url: str) -> str:
    digest = hashlib.sha1(str(url or "").encode("utf-8")).hexdigest()[:16]
    return f"source:url:{digest}"


def _source_key(document_id: str | None, url: str | None = None) -> str:
    doc = str(document_id or "").strip()
    if doc:
        return f"source:{doc}"
    return _url_key(str(url or ""))


def _result_citations(runs: list[JsonDict]) -> list[JsonDict]:
    """Read rich T46 citations from the existing run-result JSON contract."""

    citations: list[JsonDict] = []
    for run in runs:
        result = _safe_json(run.get("result"))
        answer = result.get("answer") if isinstance(result, dict) else None
        raw = answer.get("citations") if isinstance(answer, dict) else None
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            citations.append({**item, "run_id": str(run.get("id") or "")})
    return citations


def _citation_artifact_type(citation: JsonDict) -> str:
    artifact_type = str(citation.get("artifact_type") or "").strip()
    return (
        artifact_type
        if artifact_type
        in {"web_document", "library_note", "library_report", "library_document", "memory"}
        else "web_document"
    )


def _citation_evidence_key(citation: JsonDict) -> str:
    artifact_type = _citation_artifact_type(citation)
    artifact_id = str(citation.get("artifact_id") or "").strip()
    origin = str(citation.get("origin") or ("external" if artifact_type == "web_document" else "local"))
    if artifact_type == "memory":
        origin = "memory"
    if artifact_id:
        return f"{origin}:{artifact_type}:{artifact_id}"
    refs = citation.get("provenance_refs")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    document_id = str(citation.get("document_id") or "").strip()
    return f"external:web_document:{document_id or citation.get('url') or citation.get('id') or 'unknown'}"


class KnowledgeGraphProjector:
    """Build an evidence-first graph from persisted local research artifacts."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._conn = store._conn

    def project(self, *, workspace_id: str = "default") -> ProjectedGraph:
        workspace_id = str(workspace_id or "default").strip() or "default"
        sessions = self._sessions(workspace_id)
        runs = self._runs()
        all_documents = self._documents()
        all_reports = self._reports()
        all_citations = self._citations()
        all_memories = self._memories()
        all_provenance_rows = self._library_provenance()
        snapshots = self._snapshots()
        deleted_document_ids = {
            str(metadata.get("deleted_document_id") or "")
            for row in all_provenance_rows
            if str(row.get("artifact_type") or "") == "_document_delete_tombstone"
            for metadata in [_safe_json(row.get("metadata"))]
            if isinstance(metadata, dict) and metadata.get("deleted_document_id")
        }

        session_ids = {row["id"] for row in sessions}
        run_to_session = {
            row["id"]: row.get("session_id")
            for row in runs
            if row.get("session_id") in session_ids
        }
        workspace_run_ids = set(run_to_session)
        result_citations = [
            {**citation, "_citation_source": "run_result"}
            for citation in _result_citations(runs)
            if str(citation.get("run_id") or "") in workspace_run_ids
        ]
        legacy_citations = [
            {**row, "_citation_source": "legacy_table"}
            for row in all_citations
            if str(row.get("run_id") or "") in workspace_run_ids
        ]
        citations = [*result_citations, *legacy_citations]
        cited_note_ids = {
            str(row.get("artifact_id") or "")
            for row in citations
            if _citation_artifact_type(row) == "library_note" and row.get("artifact_id")
        }
        cited_memory_ids = {
            str(row.get("artifact_id") or "")
            for row in citations
            if _citation_artifact_type(row) == "memory" and row.get("artifact_id")
        }
        memory_by_id = {str(row.get("id") or ""): row for row in all_memories}

        def memory_belongs_to_workspace(memory: JsonDict | None) -> bool:
            if not memory:
                return False
            namespace = str(memory.get("namespace") or "")
            scope_key = str(memory.get("scope_key") or "")
            if namespace == "session":
                return scope_key in session_ids
            if namespace == "workspace":
                return scope_key == workspace_id
            return False

        provenance_rows: list[JsonDict] = []
        note_ids: set[str] = set(cited_note_ids)
        provenance_memory_ids: set[str] = set()
        for row in all_provenance_rows:
            item = dict(row)
            artifact_type = str(item.get("artifact_type") or "")
            artifact_id = str(item.get("artifact_id") or "")
            session_id = str(item.get("session_id") or "")
            run_id = str(item.get("run_id") or "")
            if not session_id and run_id:
                session_id = str(run_to_session.get(run_id) or "")
                item["session_id"] = session_id
            belongs = session_id in session_ids or run_id in workspace_run_ids
            if artifact_type == "memory" and memory_belongs_to_workspace(memory_by_id.get(artifact_id)):
                belongs = True
            if not belongs:
                continue
            provenance_rows.append(item)
            if artifact_type == "note" and artifact_id:
                note_ids.add(artifact_id)
            if artifact_type == "memory" and artifact_id:
                provenance_memory_ids.add(artifact_id)

        memories = [
            row
            for row in all_memories
            if str(row.get("id") or "") in note_ids
            or str(row.get("id") or "") in provenance_memory_ids
            or str(row.get("id") or "") in cited_memory_ids
            or memory_belongs_to_workspace(row)
        ]
        reports: list[JsonDict] = []
        for report in all_reports:
            item = dict(report)
            session_id = str(item.get("session_id") or "")
            run_id = str(item.get("run_id") or "")
            if not session_id and run_id:
                session_id = str(run_to_session.get(run_id) or "")
                item["session_id"] = session_id
            if session_id in session_ids:
                reports.append(item)

        allowed_document_ids: set[str] = set()
        for row in citations:
            artifact_type = _citation_artifact_type(row)
            document_id = str(row.get("document_id") or "")
            artifact_id = str(row.get("artifact_id") or "")
            if document_id and artifact_type in {"web_document", "library_document"}:
                allowed_document_ids.add(document_id)
            if artifact_type in {"web_document", "library_document"} and artifact_id:
                allowed_document_ids.add(artifact_id)
        for row in provenance_rows:
            if str(row.get("artifact_type") or "") not in {"source", "library_document"}:
                continue
            document_id = str(row.get("document_id") or "")
            artifact_id = str(row.get("artifact_id") or "")
            if document_id:
                allowed_document_ids.add(document_id)
            if artifact_id:
                allowed_document_ids.add(artifact_id)
        document_by_id = {
            row["id"]: row
            for row in all_documents
            if str(row.get("id") or "") in allowed_document_ids
        }
        nodes: dict[str, NodeDraft] = {}
        edges: dict[tuple[str, str, str], EdgeDraft] = {}
        provenance: dict[tuple[str, str, str, str], ProvenanceDraft] = {}

        def add_provenance(item: ProvenanceDraft) -> None:
            key = (
                item.owner_kind,
                item.owner_key,
                item.source_kind,
                item.source_id,
            )
            current = provenance.get(key)
            if current is None:
                provenance[key] = item
                return
            provenance[key] = ProvenanceDraft(
                owner_kind=item.owner_kind,
                owner_key=item.owner_key,
                source_kind=item.source_kind,
                source_id=item.source_id,
                span=_merge_json(current.span, item.span),
                evidence=_merge_json(current.evidence, item.evidence),
            )

        def add_node(node: NodeDraft, source_kind: str, source_id: str, evidence: JsonDict | None = None) -> None:
            if node.node_key not in nodes:
                nodes[node.node_key] = node
            add_provenance(
                ProvenanceDraft(
                    owner_kind="node",
                    owner_key=node.node_key,
                    source_kind=source_kind,
                    source_id=source_id,
                    evidence=evidence or {},
                )
            )

        def add_edge(edge: EdgeDraft, source_kind: str, source_id: str, evidence: JsonDict | None = None) -> None:
            if edge.src_key not in nodes or edge.dst_key not in nodes:
                return
            key = (edge.edge_type, edge.src_key, edge.dst_key)
            current = edges.get(key)
            if current is None:
                edges[key] = edge
            else:
                edges[key] = EdgeDraft(
                    edge_type=current.edge_type,
                    src_key=current.src_key,
                    dst_key=current.dst_key,
                    weight=max(current.weight, edge.weight),
                    confidence=max(current.confidence, edge.confidence),
                    visible=current.visible or edge.visible,
                    attrs=current.attrs,
                )
            add_provenance(
                ProvenanceDraft(
                    owner_kind="edge",
                    owner_key=edge.owner_key,
                    source_kind=source_kind,
                    source_id=source_id,
                    evidence=evidence or {},
                )
            )

        def ensure_source(document_id: str | None, url: str | None, *, title: str = "", confidence: float = 1.0) -> str:
            doc = document_by_id.get(str(document_id or ""))
            metadata = _safe_json((doc or {}).get("metadata"))
            metadata = metadata if isinstance(metadata, dict) else {}
            local_document = str(metadata.get("artifact_type") or "") == "library_document"
            node_key = _source_key(document_id, url)
            source_url = "" if local_document else str((doc or {}).get("url") or url or "")
            source_title = str(
                (doc or {}).get("title")
                or title
                or metadata.get("filename")
                or _host(source_url)
                or node_key
            )
            origin = "local" if local_document else "external"
            artifact_type = "library_document" if local_document else "web_document"
            artifact_id = str(document_id or "")
            add_node(
                NodeDraft(
                    node_type="source",
                    node_key=node_key,
                    title=_short(source_title, limit=100),
                    summary=_host(source_url),
                    attrs={
                        "document_id": str(document_id or ""),
                        "url": source_url,
                        "jump_uri": f"/library/sources/{document_id}" if document_id else source_url,
                        "origin": origin,
                        "artifact_type": artifact_type,
                        "artifact_id": artifact_id,
                        "source_status": "active",
                        "evidence_key": f"{origin}:{artifact_type}:{artifact_id or source_url}",
                        "filename": str(metadata.get("filename") or ""),
                    },
                    confidence=confidence,
                ),
                "document" if document_id else "url",
                str(document_id or url or node_key),
                {"reason": "source node"},
            )
            return node_key

        assistant_summary_by_session = {
            str(session["id"]): _assistant_summary(self._store.list_messages(str(session["id"])))
            for session in sessions
        }

        for session in sessions:
            session_attrs = {
                "session_id": session["id"],
                "workspace": session.get("workspace") or "default",
                "jump_uri": f"/session/{session['id']}",
            }
            session_attrs.update(assistant_summary_by_session.get(str(session["id"]), {}))
            add_node(
                NodeDraft(
                    node_type="session",
                    node_key=f"session:{session['id']}",
                    title=_short(str(session.get("title") or "Untitled session"), limit=100),
                    summary=str(session_attrs.get("assistant_summary") or ""),
                    attrs=session_attrs,
                ),
                "session",
                str(session["id"]),
            )

        for memory in memories:
            namespace = str(memory.get("namespace") or "")
            memory_id = str(memory.get("id") or "")
            content_length = int(memory.get("content_length") or len(str(memory.get("content") or "")))
            preview = _memory_preview(memory)
            if namespace == "note" or memory_id in note_ids:
                add_node(
                    NodeDraft(
                        node_type="note",
                        node_key=f"note:{memory_id}",
                        title=f"Note {memory_id}",
                        summary=str(preview.get("summary") or ""),
                        attrs={
                            "note_id": memory_id,
                            "namespace": namespace,
                            "scope_key": memory.get("scope_key") or "",
                            "content_length": content_length,
                            "content_excerpt": str(preview.get("content_excerpt") or ""),
                            "jump_uri": f"/library/notes/{memory_id}",
                            "origin": "local",
                            "artifact_type": "library_note",
                            "artifact_id": memory_id,
                            "source_status": "archived" if memory.get("archived_at") is not None else "active",
                            "evidence_key": f"local:library_note:{memory_id}",
                        },
                    ),
                    "memory",
                    memory_id,
                )
            elif namespace in {"workspace", "user", "session"}:
                add_node(
                    NodeDraft(
                        node_type="memory",
                        node_key=f"memory:{memory_id}",
                        title=f"Memory {memory_id}",
                        summary=str(preview.get("summary") or ""),
                        attrs={
                            "memory_id": memory_id,
                            "namespace": namespace,
                            "scope_key": memory.get("scope_key") or "",
                            "content_length": content_length,
                            "content_excerpt": str(preview.get("content_excerpt") or ""),
                            "jump_uri": f"/memory/{memory_id}",
                            "origin": "memory",
                            "artifact_type": "memory",
                            "artifact_id": memory_id,
                            "source_status": "archived" if memory.get("archived_at") is not None else "active",
                            "evidence_key": f"memory:memory:{memory_id}",
                        },
                    ),
                    "memory",
                    memory_id,
                )

        for report in reports:
            report_id = str(report.get("id") or "")
            if not report_id:
                continue
            session_id = str(report.get("session_id") or "")
            run_id = str(report.get("run_id") or "")
            if not session_id and run_id:
                session_id = str(run_to_session.get(run_id) or "")
            if session_id and session_id not in session_ids:
                continue
            add_node(
                NodeDraft(
                    node_type="report",
                    node_key=f"report:{report_id}",
                    title=_short(str(report.get("title") or "Report"), limit=100),
                    attrs={
                        "report_id": report_id,
                        "mode": report.get("mode") or "",
                        "session_id": session_id,
                        "run_id": run_id,
                        "jump_uri": f"/library/reports/{report_id}",
                        "origin": "local",
                        "artifact_type": "library_report",
                        "artifact_id": report_id,
                        "source_status": "active",
                        "evidence_key": f"local:library_report:{report_id}",
                    },
                ),
                "report",
                report_id,
            )
            if session_id:
                add_edge(
                    EdgeDraft(
                        edge_type="session_has_report",
                        src_key=f"session:{session_id}",
                        dst_key=f"report:{report_id}",
                    ),
                    "report",
                    report_id,
                    {"field": "reports.session_id" if report.get("session_id") else "reports.run_id"},
                )

        def citation_target(citation: JsonDict) -> tuple[str, str, float] | None:
            artifact_type = _citation_artifact_type(citation)
            if artifact_type in {"web_document", "library_document"}:
                document_id = str(citation.get("artifact_id") or citation.get("document_id") or "")
                url = str(citation.get("url") or "")
                if not document_id and not url:
                    return None
                if document_id in deleted_document_ids:
                    return None
                if artifact_type == "library_document" and document_id not in document_by_id:
                    return None
                confidence = 1.0 if document_id else 0.65
                return (
                    ensure_source(
                        document_id,
                        url,
                        title=str(citation.get("title") or ""),
                        confidence=confidence,
                    ),
                    "source",
                    confidence,
                )
            artifact_id = str(citation.get("artifact_id") or "").strip()
            if not artifact_id:
                return None
            prefix = {
                "library_note": "note",
                "library_report": "report",
                "memory": "memory",
            }[artifact_type]
            target_key = f"{prefix}:{artifact_id}"
            if target_key not in nodes:
                return None
            return target_key, prefix, 1.0

        citations_by_run: dict[str, list[JsonDict]] = defaultdict(list)
        for citation in citations:
            run_id = str(citation.get("run_id") or "")
            if run_id:
                citations_by_run[run_id].append(citation)
            session_id = str(run_to_session.get(run_id) or "")
            target = citation_target(citation)
            if not session_id or target is None:
                continue
            target_key, target_type, confidence = target
            citation_id = f"{run_id}:{citation.get('id') or _citation_evidence_key(citation)}"
            add_edge(
                EdgeDraft(
                    edge_type=f"session_used_{target_type}",
                    src_key=f"session:{session_id}",
                    dst_key=target_key,
                    confidence=confidence,
                    attrs={
                        "run_id": run_id,
                        "evidence_key": _citation_evidence_key(citation),
                    },
                ),
                "citation_result" if citation.get("_citation_source") == "run_result" else "citation",
                citation_id,
                {
                    "field": "runs.result.answer.citations"
                    if citation.get("_citation_source") == "run_result"
                    else "citations.run_id"
                },
            )

        for report in reports:
            report_id = str(report.get("id") or "")
            run_id = str(report.get("run_id") or "")
            if not report_id or not run_id:
                continue
            for citation in citations_by_run.get(run_id, []):
                target = citation_target(citation)
                if target is None:
                    continue
                target_key, target_type, confidence = target
                citation_id = f"{run_id}:{citation.get('id') or _citation_evidence_key(citation)}"
                add_edge(
                    EdgeDraft(
                        edge_type=f"report_cites_{target_type}",
                        src_key=f"report:{report_id}",
                        dst_key=target_key,
                        confidence=confidence,
                        attrs={
                            "run_id": run_id,
                            "evidence_key": _citation_evidence_key(citation),
                        },
                    ),
                    "citation_result" if citation.get("_citation_source") == "run_result" else "citation",
                    citation_id,
                    {
                        "field": "runs.result.answer.citations"
                        if citation.get("_citation_source") == "run_result"
                        else "reports.run_id"
                    },
                )

        for row in provenance_rows:
            artifact_type = str(row.get("artifact_type") or "")
            artifact_id = str(row.get("artifact_id") or "")
            session_id = str(row.get("session_id") or "")
            run_id = str(row.get("run_id") or "")
            if not session_id and run_id:
                session_id = str(run_to_session.get(run_id) or "")
            document_id = str(row.get("document_id") or "")
            url = str(row.get("url") or "")

            if artifact_type in {"source", "library_document"}:
                if session_id in session_ids:
                    source_key = ensure_source(document_id or artifact_id, url, confidence=1.0 if document_id else 0.65)
                    add_edge(
                        EdgeDraft(
                            edge_type="session_used_source",
                            src_key=f"session:{session_id}",
                            dst_key=source_key,
                            confidence=1.0 if document_id else 0.65,
                            attrs={"run_id": run_id},
                        ),
                        "library_provenance",
                        str(row.get("id") or ""),
                        {"field": "library_provenance.session_id/run_id"},
                    )

            if artifact_type == "note" and artifact_id:
                note_key = f"note:{artifact_id}"
                if session_id in session_ids:
                    add_edge(
                        EdgeDraft(
                            edge_type="session_has_note",
                            src_key=f"session:{session_id}",
                            dst_key=note_key,
                            attrs={"run_id": run_id},
                        ),
                        "library_provenance",
                        str(row.get("id") or ""),
                    )
                if note_key in nodes and (document_id or url):
                    source_key = ensure_source(document_id, url, confidence=1.0 if document_id else 0.65)
                    add_edge(
                        EdgeDraft(
                            edge_type="note_cites_source",
                            src_key=note_key,
                            dst_key=source_key,
                            confidence=1.0 if document_id else 0.65,
                        ),
                        "library_provenance",
                        str(row.get("id") or ""),
                    )

            if artifact_type == "memory" and artifact_id:
                memory_key = f"memory:{artifact_id}"
                if session_id in session_ids:
                    add_edge(
                        EdgeDraft(
                            edge_type="session_wrote_memory",
                            src_key=f"session:{session_id}",
                            dst_key=memory_key,
                            attrs={"run_id": run_id},
                        ),
                        "library_provenance",
                        str(row.get("id") or ""),
                    )
                if memory_key in nodes and (document_id or url):
                    source_key = ensure_source(document_id, url, confidence=1.0 if document_id else 0.65)
                    add_edge(
                        EdgeDraft(
                            edge_type="memory_derived_from_source",
                            src_key=memory_key,
                            dst_key=source_key,
                            confidence=1.0 if document_id else 0.65,
                        ),
                        "library_provenance",
                        str(row.get("id") or ""),
                    )

        for memory in memories:
            if memory.get("namespace") != "session":
                continue
            session_id = str(memory.get("scope_key") or "")
            memory_id = str(memory.get("id") or "")
            if session_id in session_ids and f"memory:{memory_id}" in nodes:
                add_edge(
                    EdgeDraft(
                        edge_type="session_wrote_memory",
                        src_key=f"session:{session_id}",
                        dst_key=f"memory:{memory_id}",
                    ),
                    "memory",
                    memory_id,
                    {"field": "memories.namespace/scope_key"},
                )

        for snapshot in snapshots:
            document_id = str(snapshot.get("document_id") or "")
            url = str(snapshot.get("url") or "")
            source_key = _source_key(document_id, url)
            if source_key in nodes:
                add_provenance(
                    ProvenanceDraft(
                        owner_kind="node",
                        owner_key=source_key,
                        source_kind="snapshot",
                        source_id=str(snapshot.get("id") or ""),
                        evidence={
                            "run_id": snapshot.get("run_id") or "",
                            "status": snapshot.get("status") or "",
                        },
                    )
                )

        report_by_id = {
            str(row.get("id") or ""): row
            for row in reports
            if row.get("id")
        }
        concept_artifacts: list[ConceptArtifact] = []
        for node in sorted(nodes.values(), key=lambda item: item.node_key):
            if node.node_type not in {"source", "note", "report", "memory"}:
                continue
            metadata: JsonDict = {}
            content = ""
            if node.node_type == "source":
                document_id = str(node.attrs.get("document_id") or "")
                document = document_by_id.get(document_id) or {}
                raw_metadata = document.get("metadata")
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                content = str(document.get("text") or "")
            elif node.node_type == "report":
                report_id = str(node.attrs.get("report_id") or "")
                report = report_by_id.get(report_id) or {}
                raw_metadata = report.get("metadata")
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                content = str(report.get("body_md") or "")
            else:
                artifact_id = str(
                    (
                        node.attrs.get("note_id")
                        if node.node_type == "note"
                        else node.attrs.get("memory_id")
                    )
                    or ""
                )
                memory = memory_by_id.get(artifact_id) or {}
                raw_metadata = memory.get("metadata")
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                content = str(memory.get("content") or "")
            metadata["source_status"] = str(
                node.attrs.get("source_status") or "active"
            )
            concept_artifacts.append(
                ConceptArtifact(
                    node_key=node.node_key,
                    node_type=node.node_type,
                    title=node.title,
                    summary=node.summary,
                    metadata=metadata,
                    content=content,
                )
            )

        wiki_store = KnowledgeGraphWikiStore(self._store)
        active_wiki = wiki_store.get_active_snapshot(workspace_id)
        wiki_snapshot_id = str((active_wiki or {}).get("id") or "")
        wiki_topics = (
            wiki_store.get_snapshot_topics(wiki_snapshot_id)
            if wiki_snapshot_id
            else []
        )
        concept_projection = extract_concepts(
            artifacts=concept_artifacts,
            wiki_topics=wiki_topics,
            wiki_snapshot_id=wiki_snapshot_id,
        )
        for node in concept_projection.nodes:
            nodes.setdefault(node.node_key, node)
        for edge in concept_projection.edges:
            if edge.src_key not in nodes or edge.dst_key not in nodes:
                continue
            key = (edge.edge_type, edge.src_key, edge.dst_key)
            current = edges.get(key)
            if current is None:
                edges[key] = edge
            else:
                edges[key] = EdgeDraft(
                    edge_type=current.edge_type,
                    src_key=current.src_key,
                    dst_key=current.dst_key,
                    weight=max(current.weight, edge.weight),
                    confidence=max(current.confidence, edge.confidence),
                    visible=current.visible or edge.visible,
                    attrs=current.attrs,
                )
        for item in concept_projection.provenance:
            add_provenance(item)

        node_provenance_refs: dict[str, set[str]] = defaultdict(set)
        for item in provenance.values():
            if item.owner_kind != "node":
                continue
            node_provenance_refs[item.owner_key].add(
                f"{item.source_kind}:{item.source_id}"
            )
        signature_inputs: list[CanonicalEvidenceInput] = []
        for artifact in concept_artifacts:
            node = nodes.get(artifact.node_key)
            if node is None:
                continue
            signature_material = json.dumps(
                {
                    "title": artifact.title,
                    "summary": artifact.summary,
                    "content": artifact.content,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            signature_inputs.append(
                CanonicalEvidenceInput(
                    stable_key=str(
                        node.attrs.get("evidence_key") or artifact.node_key
                    ),
                    content_hash=content_sha256(signature_material),
                    source_status=str(
                        node.attrs.get("source_status") or "active"
                    ),
                    provenance_refs=tuple(
                        sorted(node_provenance_refs.get(artifact.node_key, set()))
                    ),
                )
            )
        for row in provenance_rows:
            if str(row.get("artifact_type") or "") != "_document_delete_tombstone":
                continue
            metadata = _safe_json(row.get("metadata"))
            metadata = metadata if isinstance(metadata, dict) else {}
            deleted_document_id = str(metadata.get("deleted_document_id") or "")
            if not deleted_document_id:
                continue
            signature_inputs.append(
                CanonicalEvidenceInput(
                    stable_key=f"tombstone:document:{deleted_document_id}",
                    content_hash=content_sha256(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                            default=str,
                        )
                    ),
                    source_status="deleted",
                    deleted=True,
                    provenance_refs=(
                        f"library_provenance:{row.get('id') or ''}",
                    ),
                )
            )
        evidence_signature = compute_canonical_evidence_signature(
            signature_inputs
        )
        base_snapshot = {
            "schema": "canonical-evidence-signature/v1",
            "workspace_id": workspace_id,
            **evidence_signature.to_dict(),
            "wiki_seed_snapshot_id": concept_projection.stats.get(
                "wiki_seed_snapshot_id",
                "",
            ),
            "wiki_seed_count": int(
                concept_projection.stats.get("wiki_seed_count") or 0
            ),
        }
        stats = {
            "sessions": len(sessions),
            "documents": len(document_by_id),
            "reports": len(reports),
            "memories": len(memories),
            "nodes": len(nodes),
            "edges": len(edges),
            "provenance": len(provenance),
            **concept_projection.stats,
            "canonical_evidence_input_count": evidence_signature.input_count,
        }
        return ProjectedGraph(
            workspace_id=workspace_id,
            base_snapshot=base_snapshot,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            provenance=[
                provenance[key]
                for key in sorted(provenance)
            ],
            stats=stats,
        )

    def _sessions(self, workspace_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE workspace=? ORDER BY created_at ASC, id ASC",
            (workspace_id,),
        ).fetchall()
        return [_row(row) for row in rows]

    def _runs(self) -> list[JsonDict]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at ASC, id ASC").fetchall()
        return [_row(row) for row in rows]

    def _documents(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT id,url,title,text,hash,content_type,created_at,metadata "
            "FROM documents ORDER BY created_at ASC, id ASC"
        ).fetchall()
        out: list[JsonDict] = []
        for row in rows:
            item = _row(row)
            item["metadata"] = _safe_json(item.get("metadata"))
            out.append(item)
        return out

    def _reports(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT id,title,body_md,mode,parent_report_id,session_id,run_id,created_at,metadata "
            "FROM reports ORDER BY created_at ASC, id ASC"
        ).fetchall()
        out: list[JsonDict] = []
        for row in rows:
            item = _row(row)
            item["metadata"] = _safe_json(item.get("metadata"))
            out.append(item)
        return out

    def _citations(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM citations ORDER BY created_at ASC, id ASC"
        ).fetchall()
        return [_row(row) for row in rows]

    def _memories(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT id,namespace,scope_key,salience,uses,created_at,last_used_at,metadata,archived_at,"
            "length(content) AS content_length, content FROM memories ORDER BY created_at ASC, id ASC"
        ).fetchall()
        out: list[JsonDict] = []
        for row in rows:
            item = _row(row)
            item["metadata"] = _safe_json(item.get("metadata"))
            out.append(item)
        return out

    def _library_provenance(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM library_provenance ORDER BY created_at ASC, id ASC"
        ).fetchall()
        return [_row(row) for row in rows]

    def _snapshots(self) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT id,document_id,url,status,run_id,captured_at,metadata "
            "FROM page_snapshots ORDER BY captured_at ASC, id ASC"
        ).fetchall()
        return [_row(row) for row in rows]
