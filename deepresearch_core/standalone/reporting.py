import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

from deepresearch_core.answer_guardrails import normalize_answer_guardrails
from deepresearch_core.answer_evidence import normalize_answer_evidence
from deepresearch_core.conflict_resolution import build_conflict_resolution, normalize_conflict_resolution
from deepresearch_core.evidence_repair_run import build_evidence_repair_run, normalize_evidence_repair_run
from deepresearch_core.report_confidence import assess_quality_confidence, normalize_report_confidence
from deepresearch_core.report_provenance import normalize_report_provenance
from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.research_plan_execution import normalize_research_plan_execution
from deepresearch_core.search_recovery import build_search_recovery, normalize_search_recovery
from deepresearch_core.source_quality import normalize_source_quality
from deepresearch_core.standalone.routing import normalize_route_boundary

from .llm import redact_text


def safe_run_id(run_id: str) -> str:
    """Return a filesystem-safe run id."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id or "local").strip("-")
    return cleaned or "local"


def save_markdown_report(
    markdown: str,
    *,
    run_id: str,
    reports_dir: Path,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, object]:
    """Save markdown to the reports directory and return public metadata."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_id = safe_run_id(run_id)
    name = f"deepresearch-{safe_id}.md"
    path = reports_dir / name
    path.write_text(markdown, encoding="utf-8")
    result: Dict[str, object] = {"name": name, "size": len(markdown.encode("utf-8")), "url": f"/reports/{name}"}
    if metadata:
        metadata_name = f"deepresearch-{safe_id}.json"
        sidecar = reports_dir / metadata_name
        public_metadata = safe_report_metadata(metadata)
        sidecar.write_text(json.dumps(public_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["metadata_name"] = metadata_name
    return result


def build_report_metadata(
    *,
    sources: Iterable[dict[str, Any]],
    validation: Dict[str, Any],
    fact_checking: Dict[str, Any],
    route: Dict[str, Any],
    markdown: Any = "",
    provider_verification: Any = None,
    query: Any = "",
    evidence_repair_request: Any = None,
    answer_evidence: Any = None,
    source_quality: Any = None,
    answer_guardrails: Any = None,
    conflict_resolution: Any = None,
) -> Dict[str, Any]:
    """Build public generation-time report metadata."""

    raw_sources = list(sources)
    safe_sources = sanitize_report_sources(raw_sources)
    confidence = assess_quality_confidence(validation, source_count=len(safe_sources))
    provenance = normalize_report_provenance(
        None,
        markdown=markdown,
        sources=raw_sources,
        provider_verification=provider_verification,
        query=query,
    )
    metadata = {
        "sources": safe_sources,
        "confidence": confidence,
        "search_recovery": build_search_recovery(raw_sources),
        "provenance": provenance,
        "evidence_repair_run": build_evidence_repair_run(evidence_repair_request, safe_sources, query=query),
        "conflict_resolution": conflict_resolution
        if conflict_resolution is not None
        else build_conflict_resolution(evidence_repair_request, raw_sources, query=query),
        "query": query,
        "quality": {
            "validation_summary": str(validation.get("summary") or ""),
            "fact_check_verdict": str(fact_checking.get("verdict") or ""),
        },
        "route": route,
        "route_boundary": normalize_route_boundary(None, routing=route),
    }
    if answer_evidence is not None:
        metadata["answer_evidence"] = answer_evidence
    if source_quality is not None:
        metadata["source_quality"] = source_quality
    if answer_guardrails is not None:
        metadata["answer_guardrails"] = answer_guardrails
    return safe_report_metadata(metadata)


def safe_report_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return sidecar metadata without report bodies, local paths, or secrets."""

    sources = sanitize_report_sources(metadata.get("sources") or [])
    confidence = normalize_report_confidence(metadata.get("confidence"), sources=sources) or assess_quality_confidence(
        {}, source_count=len(sources)
    )
    provenance_input = metadata.get("provenance")
    provenance_has_source_relevance = isinstance(provenance_input, dict) and isinstance(
        provenance_input.get("source_relevance"),
        dict,
    )
    provenance = normalize_report_provenance(
        provenance_input,
        sources=sources,
        query="" if provenance_has_source_relevance else metadata.get("query"),
    )
    search_recovery = normalize_search_recovery(metadata.get("search_recovery"), sources=sources)
    evidence_repair_run = normalize_evidence_repair_run(
        metadata.get("evidence_repair_run"),
        sources=sources,
        query=metadata.get("query"),
    )
    conflict_resolution = normalize_conflict_resolution(metadata.get("conflict_resolution"))
    answer_evidence = normalize_answer_evidence(
        metadata.get("answer_evidence"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
        query=metadata.get("query"),
    )
    source_quality = normalize_source_quality(
        metadata.get("source_quality"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
        answer_evidence=answer_evidence,
    )
    answer_guardrails = normalize_answer_guardrails(
        metadata.get("answer_guardrails"),
        provenance=provenance,
        source_quality=source_quality,
        answer_evidence=answer_evidence,
    )
    research_plan_execution = normalize_research_plan_execution(metadata.get("research_plan_execution"), sources=sources)
    quality = metadata.get("quality") if isinstance(metadata.get("quality"), dict) else {}
    route = metadata.get("route") if isinstance(metadata.get("route"), dict) else {}
    route_boundary = normalize_route_boundary(metadata.get("route_boundary"), routing=route)
    return {
        "schema_version": 1,
        "source_count": len(sources),
        "sources": sources,
        "confidence": confidence,
        "search_recovery": search_recovery,
        "evidence_repair_run": evidence_repair_run,
        "conflict_resolution": conflict_resolution,
        "answer_evidence": answer_evidence,
        "source_quality": source_quality,
        "answer_guardrails": answer_guardrails,
        "research_plan_execution": research_plan_execution,
        "provenance": provenance,
        "route_boundary": route_boundary,
        "quality": {
            "validation_summary": _metadata_text(quality.get("validation_summary") or ""),
            "fact_check_verdict": _metadata_text(quality.get("fact_check_verdict") or ""),
        },
        "route": {
            "level": _metadata_text(route.get("level") or ""),
            "label": _metadata_text(route.get("label") or ""),
        },
    }


def merge_report(
    title: str,
    sections: list[str],
    *,
    source_count: int,
    route_label: str,
    sources: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Merge generated sections into a final markdown report."""

    body = "\n\n".join(section.strip() for section in sections if section and section.strip())
    source_appendix = format_source_appendix(sources or [])
    return (
        f"# {title}\n\n"
        f"> 报告类型：{route_label}；来源数量：{source_count}\n\n"
        f"{body}\n\n"
        f"{source_appendix}\n\n"
        "---\n"
        "- 本报告由 DeepResearch local runtime 生成。\n"
    )


def format_source_appendix(sources: Iterable[dict[str, Any]], *, limit: int = 12) -> str:
    """Return a Markdown source appendix with safe HTTP(S) links."""

    rows = []
    seen = set()
    for item in sources:
        if len(rows) >= limit:
            break
        url = _safe_source_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        title = _markdown_text(str(item.get("title") or url), fallback=url)
        site = _markdown_text(str(item.get("source") or urlparse(url).netloc), fallback=urlparse(url).netloc)
        description = _markdown_text(str(item.get("description") or ""), fallback="")
        row = f"{len(rows) + 1}. [{title}]({url})"
        if site:
            row += f" - {site}"
        if description:
            row += f"\n   摘要：{description}"
        rows.append(row)
    if not rows:
        return "## 参考来源\n\n- 本次运行未记录可展示来源。"
    return "## 参考来源\n\n" + "\n".join(rows)


def _safe_source_url(value: str) -> str:
    """Return a displayable HTTP(S) URL or an empty string."""

    url = str(value or "").strip()
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url.replace(")", "%29").replace(" ", "%20")


def _markdown_text(value: str, *, fallback: str) -> str:
    """Normalize one short Markdown text fragment."""

    text = _redact_local_paths(redact_text(value or fallback))
    text = html.unescape(text)
    text = re.sub(r"(?is)<(?:script|style)\b[^>]*>.*?</(?:script|style)>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"<[^<>\n]{1,160}>", " ", text)
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    return text.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|")


def _redact_local_paths(value: str) -> str:
    """Redact common local filesystem path fragments from source metadata."""

    text = str(value)
    text = re.sub(r"(?i)file:///[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b(?:/home|/users|/var|/tmp|/mnt)/[^\s,;)]+", "[local path]", text)
    return text


def _metadata_text(value: Any, *, max_length: int = 240) -> str:
    """Return compact metadata text with secrets and local paths removed."""

    text = _redact_local_paths(redact_text(str(value or "")))
    return " ".join(text.split()).strip()[:max_length]
