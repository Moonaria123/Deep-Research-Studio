"""Ingest-time content sanitizer for externally fetched documents (DR-ITER-052).

Web pages fetched for indexing (Brave + Firecrawl path) are untrusted text. Before chunking
and embedding we normalize and defang them so the corpus the Answer Composer later cites can
never carry control-character obfuscation, leaked secrets, executable markup, or prompt-
injection directives aimed at the LLM.

Design follows the legacy ``sanitize_report_sources`` intent (redact + keep only safe content)
but operates on *document body text* rather than source URL metadata. It is conservative: it
strips clearly dangerous spans and normalizes whitespace, but does not rewrite legitimate
prose, so retrieval quality is unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..standalone.llm import redact_text

# Zero-width and bidirectional control characters used to hide injection payloads.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# Other C0/C1 control chars except tab/newline.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Executable / non-content markup blocks.
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_HTML_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
# Chat-template / role markers that only make sense as an injection attempt.
_ROLE_MARKER_RE = re.compile(r"(?i)<\|?\s*(?:im_(?:start|end)|system|assistant|user)\s*\|?>")
# Imperative prompt-injection directives, matched per line.
_INJECTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|the\s+above)"
    r"|forget\s+(?:everything|all\s+previous)"
    r"|you\s+are\s+now\s+(?:a|an|the)\b.*"
    r"|(?:new|updated)\s+(?:system\s+)?(?:instructions?|prompt)\s*:.*"
    r"|act\s+as\s+(?:a|an)\b.*"
    r").*$"
)
_BLANKLINES_RE = re.compile(r"\n{3,}")

_REDACTION = "[redacted-injection]"


@dataclass(frozen=True)
class SanitizeResult:
    """Sanitized text plus counters describing what was removed."""

    text: str
    injections_removed: int = 0
    markup_removed: int = 0
    truncated: bool = False


def sanitize_ingest_text(text: object, *, max_chars: int = 200_000) -> SanitizeResult:
    """Return defanged ingest text plus a small report of what changed.

    Steps (all conservative and deterministic):
      1. normalize newlines; drop invisible/control characters;
      2. remove ``<script>``/``<style>`` blocks and HTML comments;
      3. neutralize chat-role markers and prompt-injection directive lines;
      4. redact secrets via :func:`redact_text`;
      5. collapse excess blank lines and cap length.
    """

    raw = str(text or "")
    if not raw.strip():
        return SanitizeResult(text="")

    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _INVISIBLE_RE.sub("", cleaned)
    cleaned = _CONTROL_RE.sub("", cleaned)

    cleaned, script_hits = _SCRIPT_STYLE_RE.subn("", cleaned)
    cleaned, comment_hits = _HTML_COMMENT_RE.subn("", cleaned)
    cleaned, role_hits = _ROLE_MARKER_RE.subn(" ", cleaned)
    cleaned, injection_hits = _INJECTION_LINE_RE.subn(_REDACTION, cleaned)

    # Redaction of secrets (API keys, local paths) — never index a leaked credential.
    cleaned = redact_text(cleaned)

    cleaned = _BLANKLINES_RE.sub("\n\n", cleaned).strip()

    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = cleaned[:max_chars].rstrip()

    return SanitizeResult(
        text=cleaned,
        injections_removed=role_hits + injection_hits,
        markup_removed=script_hits + comment_hits,
        truncated=truncated,
    )
