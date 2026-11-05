import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepresearch_core.answer_evidence import normalize_answer_evidence
from deepresearch_core.answer_guardrails import normalize_answer_guardrails
from deepresearch_core.conflict_resolution import normalize_conflict_resolution
from deepresearch_core.evidence_repair_run import normalize_evidence_repair_run
from deepresearch_core.report_confidence import normalize_report_confidence
from deepresearch_core.report_provenance import normalize_report_provenance
from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.research_plan_execution import normalize_research_plan_execution
from deepresearch_core.source_quality import normalize_source_quality
from deepresearch_core.standalone.routing import normalize_route_boundary
from deepresearch_core.reports import render_report_html
from deepresearch_core.standalone.llm import redact_text
from deepresearch_core.standalone.reporting import format_source_appendix, safe_run_id


HISTORY_FILE_NAME = "research-history.json"
MAX_HISTORY_ITEMS = 50
MAX_MARKDOWN_CHARS = 80000


def list_history(
    history_dir: Path,
    *,
    limit: int = MAX_HISTORY_ITEMS,
    query: str = "",
    confidence: str = "",
) -> list[dict[str, Any]]:
    """Return sanitized local research history records from the file store."""

    confidence_filter = _confidence_filter(confidence)
    search_text = _search_query(query)
    records = _read_history_file(history_dir)
    filtered = []
    for record in records:
        if confidence_filter and record.get("confidence", {}).get("level") != confidence_filter:
            continue
        if search_text and not _history_matches_query(record, search_text):
            continue
        filtered.append(record)
    return filtered[: max(0, int(limit or 0))]


def read_history_item(history_dir: Path, history_id: str) -> dict[str, Any]:
    """Read one sanitized local research history record by id."""

    normalized_id = normalize_history_id(history_id)
    for record in _read_history_file(history_dir):
        if record.get("id") == normalized_id:
            return record
    raise FileNotFoundError(f"History item not found: {normalized_id}")


def save_history_item(history_dir: Path, item: dict[str, Any], *, limit: int = MAX_HISTORY_ITEMS) -> dict[str, Any]:
    """Sanitize and persist one local research history record."""

    record = sanitize_history_item(item)
    records = [record]
    for existing in _read_history_file(history_dir):
        if existing.get("id") == record["id"]:
            continue
        records.append(existing)
        if len(records) >= limit:
            break
    _write_history_file(history_dir, records)
    return record


def clear_history(history_dir: Path) -> None:
    """Clear the local research history file if it exists."""

    path = history_file_path(history_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def render_history_html(record: dict[str, Any]) -> str:
    """Render one sanitized local history record as safe standalone HTML."""

    safe = sanitize_history_item(record)
    markdown = str(safe.get("reportMarkdown") or "").strip()
    if not markdown:
        title = str(safe.get("title") or "Untitled research").strip()
        summary = str(safe.get("summary") or "").strip()
        markdown = f"# {title}\n\n{summary}".strip()
    sources = safe.get("sources") if isinstance(safe.get("sources"), list) else []
    if sources and "## 参考来源" not in markdown:
        markdown = f"{markdown}\n\n{format_source_appendix(sources)}"
    return render_report_html({"id": safe.get("id") or "history", "text": markdown})


def sanitize_history_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return one safe history record from untrusted browser input."""

    if not isinstance(item, dict):
        item = {}
    sources = sanitize_report_sources(item.get("sources"), limit=12)
    confidence = normalize_report_confidence(item.get("confidence"), sources=sources) or {}
    report_markdown = _safe_markdown(item.get("reportMarkdown"))
    provenance = normalize_report_provenance(item.get("provenance"), markdown=report_markdown, sources=sources)
    answer_evidence = normalize_answer_evidence(
        item.get("answer_evidence"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
    )
    source_quality = normalize_source_quality(
        item.get("source_quality"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
        answer_evidence=answer_evidence,
    )
    answer_guardrails = normalize_answer_guardrails(
        item.get("answer_guardrails"),
        provenance=provenance,
        source_quality=source_quality,
        answer_evidence=answer_evidence,
    )
    evidence_repair_run = normalize_evidence_repair_run(item.get("evidence_repair_run"), sources=sources)
    conflict_resolution = normalize_conflict_resolution(item.get("conflict_resolution"))
    research_plan_execution = normalize_research_plan_execution(item.get("research_plan_execution"), sources=sources)
    route_boundary = normalize_route_boundary(item.get("route_boundary"))
    return {
        "id": _safe_history_id(item.get("id")),
        "title": _compact_text(item.get("title"), limit=72) or "Untitled research",
        "createdAt": _safe_timestamp(item.get("createdAt")),
        "summary": _compact_text(item.get("summary"), limit=180),
        "reportUrl": _safe_local_report_url(item.get("reportUrl")),
        "reportMarkdown": report_markdown,
        "sources": sources,
        "source_count": len(sources),
        "confidence": confidence,
        "evidence_repair_run": evidence_repair_run,
        "conflict_resolution": conflict_resolution,
        "research_plan_execution": research_plan_execution,
        "answer_evidence": answer_evidence,
        "source_quality": source_quality,
        "answer_guardrails": answer_guardrails,
        "route_boundary": route_boundary,
        "provenance": provenance,
    }


def history_file_path(history_dir: Path) -> Path:
    """Return the canonical JSON file path for local research history."""

    return history_dir / HISTORY_FILE_NAME


def normalize_history_id(history_id: str) -> str:
    """Normalize and validate a local history id for external reads."""

    value = str(history_id or "").strip()
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError("History id must be a simple local history id.")
    safe = safe_run_id(value)
    if safe != value:
        raise ValueError("History id contains unsupported characters.")
    return value


def _read_history_file(history_dir: Path) -> list[dict[str, Any]]:
    """Read and sanitize the history file, tolerating missing or invalid JSON."""

    path = history_file_path(history_dir)
    if not path.exists() or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    records = []
    seen: set[str] = set()
    for item in data:
        record = sanitize_history_item(item if isinstance(item, dict) else {})
        if record["id"] in seen:
            continue
        seen.add(record["id"])
        records.append(record)
        if len(records) >= MAX_HISTORY_ITEMS:
            break
    return records


def _write_history_file(history_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write sanitized records to disk with a compact JSON format."""

    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_file_path(history_dir)
    payload = [sanitize_history_item(record) for record in records[:MAX_HISTORY_ITEMS]]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_history_id(value: Any) -> str:
    """Return a stable non-path identifier for one history item."""

    text = _compact_text(value, limit=80)
    if not text:
        text = f"history-{datetime.now(timezone.utc).timestamp():.0f}"
    safe = safe_run_id(text)
    return safe or "history"


def _safe_timestamp(value: Any) -> str:
    """Return a compact timestamp string without accepting arbitrary blobs."""

    text = _compact_text(value, limit=64)
    if text:
        return text
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_text(value: Any, *, limit: int) -> str:
    """Redact sensitive text and collapse whitespace."""

    text = " ".join(_scrub_credential_labels(redact_text(str(value or ""))).split()).strip()
    return text[:limit]


def _safe_markdown(value: Any) -> str:
    """Return a redacted bounded Markdown snapshot for local history."""

    return _scrub_credential_labels(redact_text(str(value or ""))).strip()[:MAX_MARKDOWN_CHARS]


def _safe_local_report_url(value: Any) -> str:
    """Return a safe same-origin report URL for local history links."""

    text = redact_text(str(value or "")).strip()
    if not text or not text.startswith("/") or text.startswith("//") or "\\" in text:
        return ""
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in text):
        return ""
    if not (text.startswith("/reports/") or text.startswith("/api/reports/")):
        return ""
    return text[:512]


def _scrub_credential_labels(value: str) -> str:
    """Remove credential assignment shapes after value redaction."""

    text = str(value)
    text = re.sub(
        r"(?i)\b(?:api[_ -]?key|token|access[_ -]?token|secret)\s*[:=]\s*\[redacted\]",
        "[redacted credential]",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+\[redacted\]", "Bearer [redacted credential]", text)
    return text


def _search_query(query: str) -> str:
    """Return a normalized local history search query."""

    return _normalize_search_text(query)


def _confidence_filter(value: str) -> str:
    """Return a normalized confidence filter or raise on unsupported values."""

    normalized = str(value or "").strip().casefold()
    if not normalized:
        return ""
    if normalized not in {"high", "medium", "low"}:
        raise ValueError("confidence must be one of: high, medium, low")
    return normalized


def _history_matches_query(record: dict[str, Any], query: str) -> bool:
    """Return whether a history record contains a query in public fields."""

    sources = record.get("sources") if isinstance(record.get("sources"), list) else []
    source_text = " ".join(
        " ".join(str(source.get(field) or "") for field in ("title", "url", "site"))
        for source in sources
        if isinstance(source, dict)
    )
    confidence = record.get("confidence") if isinstance(record.get("confidence"), dict) else {}
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    confidence_text = " ".join(
        [
            str(confidence.get("level") or ""),
            str(confidence.get("label") or ""),
            str(confidence.get("note") or ""),
            " ".join(str(signal or "") for signal in confidence.get("signals") or []),
        ]
    )
    provenance_text = " ".join(
        str(claim.get("text") or "")
        for claim in provenance.get("claims", [])
        if isinstance(claim, dict)
    )
    haystack = _normalize_search_text(
        " ".join(
            [
                str(record.get("id") or ""),
                str(record.get("title") or ""),
                str(record.get("summary") or ""),
                str(record.get("reportMarkdown") or ""),
                source_text,
                confidence_text,
                provenance_text,
            ]
        )
    )
    return query in haystack


def _normalize_search_text(value: Any) -> str:
    """Collapse case, whitespace, and punctuation for local history search."""

    text = re.sub(r"[\W_]+", " ", str(value or "").casefold(), flags=re.UNICODE)
    return " ".join(text.split())
