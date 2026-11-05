"""Thin Deep Research stage helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .adapters import LLMAdapter
from .errors import BaselineError, ErrorCode
from .retrieval import EvidenceItem


def expand_queries(subtask: str, *, remaining_budget: int) -> list[str]:
    """Return search queries for a subtask, bounded by the remaining budget."""

    return [subtask] if remaining_budget > 0 else []


def extract_notes(
    llm: LLMAdapter,
    subtask: str,
    evidence: Sequence[EvidenceItem],
    *,
    max_tokens: int,
    temperature: float,
    thinking_budget: int | None = None,
) -> str:
    """Extract concise analyst notes for one Deep subtask."""

    sources = "\n".join(
        f"[c{index}] {item.title}\n{item.text[:500]}" for index, item in enumerate(evidence, start=1)
    )
    if not sources:
        return "No retrieved evidence for this subtask."
    prompt = (
        "ANALYST-NOTES\n"
        f"Subtask: {subtask}\n"
        "Extract concise research notes from the sources. Keep citations when present.\n\n"
        f"{sources}"
    )
    try:
        result = llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
    except BaselineError as error:
        if error.code in {ErrorCode.RUN_TIMEOUT, ErrorCode.RUN_CANCELLED}:
            raise
        return "Analyst notes unavailable; use retrieved evidence directly."
    except Exception:  # noqa: BLE001 - notes are best-effort
        return "Analyst notes unavailable; use retrieved evidence directly."
    return result.text.strip() or "Analyst notes unavailable; use retrieved evidence directly."


def render_evidence_matrix(rows: Sequence[dict[str, Any]]) -> str:
    """Render a compact evidence matrix for the final writer prompt."""

    lines = ["Evidence matrix:"]
    for row in rows:
        gaps = ", ".join(row.get("gaps") or []) or "none"
        lines.append(
            f"- {row.get('subtask', '')}: {row.get('notes', '')} "
            f"(sources={row.get('source_count', 0)}, gaps={gaps})"
        )
    return "\n".join(lines)
