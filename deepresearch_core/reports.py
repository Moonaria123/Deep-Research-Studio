import io
import html
import json
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from deepresearch_core.answer_evidence import normalize_answer_evidence
from deepresearch_core.answer_guardrails import normalize_answer_guardrails
from deepresearch_core.conflict_resolution import normalize_conflict_resolution
from deepresearch_core.evidence_repair_run import normalize_evidence_repair_run
from deepresearch_core.report_confidence import assess_report_confidence, normalize_report_confidence
from deepresearch_core.report_provenance import build_claim_provenance, normalize_report_provenance
from deepresearch_core.report_sources import extract_report_sources, sanitize_report_sources
from deepresearch_core.research_plan_execution import normalize_research_plan_execution
from deepresearch_core.source_quality import normalize_source_quality
from deepresearch_core.standalone.llm import redact_text
from deepresearch_core.standalone.reporting import safe_run_id
from deepresearch_core.standalone.routing import normalize_route_boundary


REPORT_PREFIX = "deepresearch-"
REPORT_SUFFIX = ".md"
_INLINE_LINK_RE = re.compile(r"\[([^\]\n]{1,200})\]\(([^\s)]+)\)", re.IGNORECASE)
_CODE_SPAN_RE = re.compile(r"`([^`\n]{1,400})`")
_LOCAL_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:file:///)?[A-Za-z]:[\\/][^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])(?:file://)?/(?:Users|home|tmp|var|private|mnt|workspace|root|etc|opt)/[^\r\n,;)]*",
    re.IGNORECASE,
)
_XML_INVALID_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def list_reports(
    reports_dir: Path,
    *,
    limit: int = 50,
    query: str = "",
    confidence: str = "",
) -> list[dict[str, Any]]:
    """Return local DeepResearch Markdown report metadata."""

    confidence_filter = _confidence_filter(confidence)
    if not reports_dir.exists():
        return []
    search_text = _search_query(query)
    reports = []
    for path in reports_dir.glob(f"{REPORT_PREFIX}*{REPORT_SUFFIX}"):
        if not path.is_file():
            continue
        report_id = _id_from_name(path.name)
        if not report_id:
            continue
        text = _report_text(path)
        metadata = _report_sidecar_metadata(path)
        sources = _report_sources(text, metadata)
        confidence = _report_confidence(text, sources, metadata)
        provenance = _report_provenance(text, sources, metadata)
        answer_evidence = _report_answer_evidence(sources, confidence, provenance, metadata)
        source_quality = _report_source_quality(sources, confidence, provenance, answer_evidence, metadata)
        answer_guardrails = _report_answer_guardrails(provenance, source_quality, answer_evidence, metadata)
        evidence_repair_run = normalize_evidence_repair_run(metadata.get("evidence_repair_run"), sources=sources)
        conflict_resolution = normalize_conflict_resolution(metadata.get("conflict_resolution"))
        research_plan_execution = normalize_research_plan_execution(metadata.get("research_plan_execution"), sources=sources)
        route_boundary = normalize_route_boundary(metadata.get("route_boundary"), routing=metadata.get("route"))
        if confidence_filter and confidence.get("level") != confidence_filter:
            continue
        if search_text and not _report_matches_query(report_id, path.name, text, sources, confidence, search_text):
            continue
        stat = path.stat()
        reports.append(
            {
                "id": report_id,
                "name": path.name,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "uri": resource_uri(report_id),
                "preview": _preview_text(text),
                "source_count": len(sources),
                "confidence": confidence,
                "evidence_repair_run": evidence_repair_run,
                "conflict_resolution": conflict_resolution,
                "research_plan_execution": research_plan_execution,
                "answer_evidence": answer_evidence,
                "source_quality": source_quality,
                "answer_guardrails": answer_guardrails,
                "route_boundary": route_boundary,
            }
        )
    reports.sort(key=lambda item: (item["mtime"], item["name"]), reverse=True)
    return reports[:limit]


def read_report(reports_dir: Path, report_id: str) -> dict[str, Any]:
    """Read one local report by id with path traversal protection."""

    path = report_path(reports_dir, report_id)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Report not found: {report_id}")
    text = _report_text(path)
    metadata = _report_sidecar_metadata(path)
    sources = _report_sources(text, metadata)
    confidence = _report_confidence(text, sources, metadata)
    provenance = _report_provenance(text, sources, metadata)
    answer_evidence = _report_answer_evidence(sources, confidence, provenance, metadata)
    source_quality = _report_source_quality(sources, confidence, provenance, answer_evidence, metadata)
    answer_guardrails = _report_answer_guardrails(provenance, source_quality, answer_evidence, metadata)
    evidence_repair_run = normalize_evidence_repair_run(metadata.get("evidence_repair_run"), sources=sources)
    conflict_resolution = normalize_conflict_resolution(metadata.get("conflict_resolution"))
    research_plan_execution = normalize_research_plan_execution(metadata.get("research_plan_execution"), sources=sources)
    route_boundary = normalize_route_boundary(metadata.get("route_boundary"), routing=metadata.get("route"))
    stat = path.stat()
    normalized_id = normalize_report_id(report_id)
    return {
        "id": normalized_id,
        "name": path.name,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "uri": resource_uri(normalized_id),
        "text": text,
        "preview": _preview_text(text),
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


def render_report_html(report: dict[str, Any]) -> str:
    """Render one local report as safe standalone HTML."""

    title = _report_title(report)
    body = _markdown_to_html(str(report.get("text") or ""))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            "body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:880px;margin:40px auto;padding:0 20px;line-height:1.7;color:#111827;background:#ffffff}",
            "h1,h2,h3{line-height:1.25;color:#0f172a}pre{padding:12px;overflow:auto;background:#f8fafc;border:1px solid #e5e7eb}code{font-family:Consolas,monospace}a{color:#2563eb}",
            "</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def render_report_pdf(report: dict[str, Any]) -> bytes:
    """Render one local report as a small text PDF."""

    lines = _wrapped_plain_report_lines(report, width=78)
    pages = _paginate(lines, lines_per_page=48)
    objects: list[bytes] = []
    page_ids = [6 + index * 2 for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects.append(_pdf_object(1, "<< /Type /Catalog /Pages 2 0 R >>"))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(_pdf_object(2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"))
    objects.append(
        _pdf_object(3, "<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [4 0 R] >>")
    )
    objects.append(
        _pdf_object(
            4,
            "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> "
            "/FontDescriptor 5 0 R >>",
        )
    )
    objects.append(
        _pdf_object(
            5,
            "<< /Type /FontDescriptor /FontName /STSong-Light /Flags 4 /FontBBox [0 -120 1000 880] "
            "/ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 880 /StemV 80 >>",
        )
    )
    for index, page_lines in enumerate(pages):
        page_id = page_ids[index]
        content_id = content_ids[index]
        stream = _pdf_text_stream(page_lines)
        objects.append(
            _pdf_object(
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>",
            )
        )
        objects.append(_pdf_stream_object(content_id, stream))
    return _build_pdf(objects)


def render_report_docx(report: dict[str, Any]) -> bytes:
    """Render one local report as a minimal DOCX package."""

    lines = _plain_report_lines(report)
    paragraphs = "\n".join(_docx_paragraph(line) for line in (lines or [_report_title(report)]))
    document_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">',
            "<w:body>",
            paragraphs,
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>',
            "</w:body>",
            "</w:document>",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                    '<Default Extension="xml" ContentType="application/xml"/>',
                    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
                    "</Types>",
                ]
            ),
        )
        archive.writestr(
            "_rels/.rels",
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>',
                    "</Relationships>",
                ]
            ),
        )
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def report_export_name(report: dict[str, Any], suffix: str) -> str:
    """Return a safe export filename for one local report."""

    report_id = normalize_report_id(str(report.get("id") or "local"))
    clean_suffix = str(suffix or "").lower().lstrip(".")
    if clean_suffix not in {"html", "pdf", "docx", "md"}:
        raise ValueError("Unsupported report export suffix.")
    return f"{REPORT_PREFIX}{report_id}.{clean_suffix}"


def report_path(reports_dir: Path, report_id: str) -> Path:
    """Return the absolute path for a local report id."""

    normalized_id = normalize_report_id(report_id)
    root = reports_dir.resolve()
    path = (root / f"{REPORT_PREFIX}{normalized_id}{REPORT_SUFFIX}").resolve()
    if root not in path.parents and path != root:
        raise ValueError("Report id escapes the reports directory.")
    return path


def resource_uri(report_id: str) -> str:
    """Return the MCP resource URI for a report id."""

    return f"deepresearch://reports/{normalize_report_id(report_id)}"


def report_id_from_uri(uri: str) -> str:
    """Extract and validate a report id from a DeepResearch report resource URI."""

    prefix = "deepresearch://reports/"
    if not uri.startswith(prefix):
        raise ValueError("Unsupported report resource URI.")
    return normalize_report_id(uri[len(prefix) :])


def normalize_report_id(report_id: str) -> str:
    """Normalize a report id or report file name into a safe report id."""

    value = str(report_id or "").strip()
    if value.startswith(REPORT_PREFIX):
        value = value[len(REPORT_PREFIX) :]
    if value.endswith(REPORT_SUFFIX):
        value = value[: -len(REPORT_SUFFIX)]
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError("Report id must be a simple local report id.")
    safe = safe_run_id(value)
    if safe != value:
        raise ValueError("Report id contains unsupported characters.")
    return value


def _id_from_name(name: str) -> str | None:
    """Return a report id from a report filename when valid."""

    if not name.startswith(REPORT_PREFIX) or not name.endswith(REPORT_SUFFIX):
        return None
    try:
        return normalize_report_id(name)
    except ValueError:
        return None


def _report_text(path: Path) -> str:
    """Read report text with tolerant UTF-8 handling."""

    return path.read_text(encoding="utf-8-sig", errors="replace")


def _report_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Read safe report metadata sidecar when present."""

    sidecar = path.with_suffix(".json")
    if not sidecar.exists() or not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _report_sources(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sidecar sources first, falling back to Markdown appendix parsing."""

    return sanitize_report_sources(metadata.get("sources")) or extract_report_sources(text)


def _report_confidence(text: str, sources: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Return sidecar confidence first, falling back to Markdown parsing."""

    return normalize_report_confidence(metadata.get("confidence"), sources=sources) or assess_report_confidence(text, sources)


def _report_provenance(text: str, sources: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Return sidecar provenance first, falling back to Markdown citation parsing."""

    provenance = normalize_report_provenance(metadata.get("provenance"), sources=sources)
    if provenance.get("claim_count"):
        return provenance
    return build_claim_provenance(text, sources)


def _report_answer_evidence(
    sources: list[dict[str, Any]],
    confidence: dict[str, Any],
    provenance: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return sidecar answer evidence or derive it from public report metadata."""

    return normalize_answer_evidence(
        metadata.get("answer_evidence"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
    )


def _report_source_quality(
    sources: list[dict[str, Any]],
    confidence: dict[str, Any],
    provenance: dict[str, Any],
    answer_evidence: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return sidecar source quality or derive it from public report metadata."""

    return normalize_source_quality(
        metadata.get("source_quality"),
        sources=sources,
        confidence=confidence,
        provenance=provenance,
        answer_evidence=answer_evidence,
    )


def _report_answer_guardrails(
    provenance: dict[str, Any],
    source_quality: dict[str, Any],
    answer_evidence: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return normalized answer-use guardrails for stored reports."""

    return normalize_answer_guardrails(
        metadata.get("answer_guardrails"),
        provenance=provenance,
        source_quality=source_quality,
        answer_evidence=answer_evidence,
    )


def _preview_text(text: str) -> str:
    """Return a compact secret-redacted report preview."""

    lines = [line.strip() for line in redact_text(text).splitlines() if line.strip()]
    preview = " ".join(lines)
    return preview[:240]


def _search_query(query: str) -> str:
    """Return a normalized local report search query."""

    return " ".join(str(query or "").casefold().split())


def _confidence_filter(value: str) -> str:
    """Return a normalized confidence filter or raise on unsupported values."""

    normalized = str(value or "").strip().casefold()
    if not normalized:
        return ""
    if normalized not in {"high", "medium", "low"}:
        raise ValueError("confidence must be one of: high, medium, low")
    return normalized


def _report_matches_query(
    report_id: str,
    name: str,
    text: str,
    sources: list[dict[str, Any]],
    confidence: dict[str, Any],
    query: str,
) -> bool:
    """Return whether a report contains a query in public report fields."""

    source_text = " ".join(
        " ".join(str(source.get(field) or "") for field in ("title", "url", "site")) for source in sources
    )
    confidence_text = " ".join(
        [
            str(confidence.get("level") or ""),
            str(confidence.get("label") or ""),
            str(confidence.get("note") or ""),
            " ".join(str(signal or "") for signal in confidence.get("signals") or []),
        ]
    )
    haystack = " ".join([report_id, name, redact_text(text), source_text, redact_text(confidence_text)]).casefold()
    return query in haystack


def _report_title(report: dict[str, Any]) -> str:
    """Return a human-readable report title."""

    text = str(report.get("text") or "")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or str(report.get("id") or "DeepResearch Report")
    return str(report.get("id") or "DeepResearch Report")


def _markdown_to_html(markdown: str) -> str:
    """Render a small safe Markdown subset for exported reports."""

    parts: list[str] = []
    paragraph: list[str] = []
    code_lines: list[str] = []
    in_code = False
    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                _flush_html_paragraph(parts, paragraph)
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            _flush_html_paragraph(parts, paragraph)
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            _flush_html_paragraph(parts, paragraph)
            level = len(heading.group(1))
            parts.append(f"<h{level}>{_inline_markdown_to_html(heading.group(2))}</h{level}>")
            continue
        list_item = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
        if list_item:
            _flush_html_paragraph(parts, paragraph)
            parts.append(f"<p>{_inline_markdown_to_html(list_item.group(1))}</p>")
            continue
        paragraph.append(stripped)
    if in_code:
        parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    _flush_html_paragraph(parts, paragraph)
    return "\n".join(parts)


def _flush_html_paragraph(parts: list[str], paragraph: list[str]) -> None:
    """Append pending paragraph text to an HTML part list."""

    if not paragraph:
        return
    parts.append(f"<p>{_inline_markdown_to_html(' '.join(paragraph))}</p>")
    paragraph.clear()


def _inline_markdown_to_html(text: str) -> str:
    """Render safe HTTP(S) inline Markdown links and escape everything else."""

    result: list[str] = []
    last_index = 0
    for match in _INLINE_LINK_RE.finditer(text):
        result.append(html.escape(text[last_index : match.start()]))
        href = _safe_html_href(match.group(2))
        if href:
            title = html.escape(match.group(1))
            result.append(f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noreferrer noopener">{title}</a>')
        else:
            result.append(html.escape(match.group(1)))
        last_index = match.end()
    result.append(html.escape(text[last_index:]))
    return "".join(result)


def _safe_html_href(value: str) -> str:
    """Return a safe HTTP(S) href for HTML export links."""

    href = str(value or "").strip()
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in href):
        return ""
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return href


def _plain_report_lines(report: dict[str, Any]) -> list[str]:
    """Return a redacted plain-text representation of report Markdown."""

    text = str(report.get("text") or "")
    lines: list[str] = []
    in_code = False
    in_private_key = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if re.match(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", stripped):
            in_private_key = True
            lines.append("[redacted]")
            continue
        if in_private_key:
            if re.match(r"-----END [A-Z0-9 ]*PRIVATE KEY-----", stripped):
                in_private_key = False
            continue
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            stripped = _plain_markdown_line(stripped)
        line = _safe_export_text(stripped).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines or [_report_title(report)]


def _plain_markdown_line(line: str) -> str:
    """Convert one Markdown-ish line to safe plain text."""

    heading = re.match(r"^(#{1,6})\s+(.*)$", line)
    if heading:
        line = heading.group(2)
    line = re.sub(r"^\s*[-*+]\s+", "- ", line)
    line = re.sub(r"^\s*\d+\.\s+", "- ", line)
    if "|" in line and line.strip("|").strip():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            line = " | ".join(cells)
    line = _INLINE_LINK_RE.sub(_plain_link, line)
    return _CODE_SPAN_RE.sub(r"\1", line)


def _plain_link(match: re.Match[str]) -> str:
    """Return plain text for a Markdown link, keeping only safe URLs."""

    title = match.group(1).replace("\\[", "[").replace("\\]", "]").strip()
    href = match.group(2).strip()
    if _safe_html_href(href):
        return f"{title} ({href})"
    return title


def _wrapped_plain_report_lines(report: dict[str, Any], *, width: int) -> list[str]:
    """Return plain report lines wrapped for PDF pages."""

    wrapped: list[str] = []
    for line in _plain_report_lines(report):
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width, break_long_words=True, replace_whitespace=False) or [""])
    return wrapped


def _paginate(lines: list[str], *, lines_per_page: int) -> list[list[str]]:
    """Split lines into non-empty page chunks."""

    chunks = [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)]
    return chunks or [["DeepResearch Report"]]


def _pdf_text_stream(lines: list[str]) -> bytes:
    """Return one PDF content stream for a page of text."""

    commands = ["BT", "/F1 11 Tf", "50 760 Td", "14 TL"]
    for line in lines:
        commands.append(f"<{_pdf_utf16_hex(line)}> Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("ascii")


def _pdf_utf16_hex(text: str) -> str:
    """Return PDF hex text using UTF-16BE with BOM."""

    return ("\ufeff" + text).encode("utf-16-be", errors="replace").hex().upper()


def _pdf_object(object_id: int, body: str) -> bytes:
    """Return one PDF object with a textual body."""

    return f"{object_id} 0 obj\n{body}\nendobj\n".encode("ascii")


def _pdf_stream_object(object_id: int, stream: bytes) -> bytes:
    """Return one PDF stream object."""

    header = f"{object_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
    return header + stream + b"endstream\nendobj\n"


def _build_pdf(objects: list[bytes]) -> bytes:
    """Build a minimal PDF document from serialized objects."""

    output = bytearray(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def _docx_paragraph(text: str) -> str:
    """Return one DOCX paragraph containing escaped text."""

    return f"<w:p><w:r><w:t>{html.escape(_safe_xml_text(text), quote=False)}</w:t></w:r></w:p>"


def _safe_export_text(value: Any) -> str:
    """Return report-export text without credentials, local paths, or XML-breaking chars."""

    text = redact_text(str(value or ""))
    text = _LOCAL_PATH_TOKEN_RE.sub(_redact_local_path_match, text)
    return _safe_xml_text(text)


def _safe_xml_text(value: Any) -> str:
    """Return XML 1.0-safe text."""

    return _XML_INVALID_CHAR_RE.sub("", str(value or ""))


def _redact_local_path_match(match: re.Match[str]) -> str:
    """Redact one local path while preserving simple trailing prose punctuation."""

    value = match.group(0)
    trailing = ""
    while value.endswith(".") and not re.search(r"\.[A-Za-z0-9]{1,8}$", value):
        trailing = "." + trailing
        value = value[:-1]
    return "[local path]" + trailing
