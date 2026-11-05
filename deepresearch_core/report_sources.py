import re
from typing import Any
from urllib.parse import urlparse

from .standalone.llm import redact_text


JsonDict = dict[str, Any]
_SOURCE_LINE_RE = re.compile(
    r"^\s*\d+\.\s+\[((?:\\.|[^\]\n]){1,200})\]\((https?://[^\s)]+)\)(?:\s+-\s+(.+))?\s*$",
    re.IGNORECASE,
)


def extract_report_sources(markdown: Any, *, limit: int = 12) -> list[JsonDict]:
    """Extract safe HTTP(S) source metadata from a report source appendix."""

    lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((index for index, line in enumerate(lines) if line.strip() == "## 参考来源"), -1)
    if start < 0:
        return []
    sources: list[JsonDict] = []
    seen: set[str] = set()
    for line in lines[start + 1 :]:
        if len(sources) >= limit:
            break
        text = line.strip()
        if re.match(r"^#{1,3}\s+", text):
            break
        source = _source_from_line(text)
        if not source or source["url"] in seen:
            continue
        seen.add(source["url"])
        sources.append(source)
    return sources


def sanitize_report_sources(value: Any, *, limit: int = 12) -> list[JsonDict]:
    """Return safe HTTP(S) source metadata from structured report metadata."""

    if not isinstance(value, list):
        return []
    sources: list[JsonDict] = []
    seen: set[str] = set()
    for item in value:
        if len(sources) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = _safe_source_url(item.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "title": _safe_source_label(item.get("title") or url) or url,
                "url": url,
                "site": _safe_source_label(item.get("site") or item.get("source") or ""),
            }
        )
    return sources


def report_source_rows(report: dict[str, Any]) -> list[JsonDict]:
    """Return report source rows with body-only citation counts."""

    markdown = str(report.get("text") or "")
    sources = sanitize_report_sources(report.get("sources")) or extract_report_sources(markdown)
    rows: list[JsonDict] = []
    for index, source in enumerate(sources, start=1):
        rows.append(
            {
                "index": index,
                "title": source.get("title") or source.get("url") or f"Source {index}",
                "url": source.get("url") or "",
                "site": source.get("site") or "",
                "body_citations": report_source_body_citation_count(markdown, index),
            }
        )
    return rows


def report_source_body_citation_count(markdown: Any, source_index: int) -> int:
    """Count bracket citations for one source before the source appendix."""

    if source_index <= 0:
        return 0
    lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body_lines = []
    for line in lines:
        if line.strip() == "## 参考来源":
            break
        body_lines.append(line)
    count = 0
    for match in re.finditer(r"\[(\d{1,3})\]", "\n".join(body_lines)):
        if int(match.group(1)) == source_index:
            count += 1
    return count


def _source_from_line(text: str) -> JsonDict | None:
    """Return one source item from a Markdown ordered-list line."""

    match = _SOURCE_LINE_RE.match(text)
    if not match:
        return None
    url = _safe_source_url(match.group(2))
    if not url:
        return None
    return {
        "title": _unescape_markdown_source_title(match.group(1)) or url,
        "url": url,
        "site": _safe_source_label(match.group(3) or ""),
    }


def _safe_source_url(value: Any) -> str:
    """Return a source URL only when it is a printable HTTP(S) URL."""

    text = redact_text(str(value or "")).strip()
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in text):
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _unescape_markdown_source_title(value: Any) -> str:
    """Restore simple Markdown escapes in a source title."""

    return _safe_source_label(re.sub(r"\\([\[\]\\])", r"\1", str(value or "")))


def _safe_source_label(value: Any) -> str:
    """Return compact source label text without local paths or secrets."""

    text = redact_text(str(value or ""))
    text = re.sub(r"(?i)file:///[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b(?:/home|/users|/var|/tmp|/mnt)/[^\s,;)]+", "[local path]", text)
    return " ".join(text.split()).strip()
