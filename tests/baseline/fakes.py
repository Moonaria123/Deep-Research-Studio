"""Deterministic, hermetic fakes for the baseline test suite."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure the repo root is importable when pytest is invoked from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepresearch_core.baseline.adapters import Document, LLMResult, SearchHit  # noqa: E402


class FakeLLM:
    """Deterministic LLMAdapter: echoes source snippets so answers stay grounded."""

    name = "fake-llm"

    def __init__(self, model: str = "fake-1", *, scripted: dict[str, str] | None = None) -> None:
        self.model = model
        self.name = model
        self._scripted = scripted or {}
        self.calls: list[list[dict]] = []

    def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.2,
        max_tokens=1200,
        thinking_budget=None,
        response_format=None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        for needle, reply in self._scripted.items():
            if needle in user:
                return LLMResult(text=reply, model=model or self.model)
        # Echo the snippet bodies (line after each "[cN] ..." header) to guarantee overlap.
        snippets = re.findall(r"\[c\d+\][^\n]*\n([^\n]+)", user)
        if snippets:
            body = " ".join(f"{snippet} [c{i}]" for i, snippet in enumerate(snippets, start=1))
            return LLMResult(text=body, model=model or self.model)
        # Synthesis/judge case: echo any already-cited lines so the result stays grounded.
        cited_lines = [ln.strip() for ln in user.splitlines() if re.search(r"\[c\d+\]", ln)]
        if cited_lines:
            return LLMResult(text=" ".join(cited_lines), model=model or self.model)
        return LLMResult(text="No grounded answer available.", model=model or self.model)

    def embed(self, texts, *, model=None) -> list[list[float]]:
        # Tiny deterministic bag-of-chars embedding (length 8).
        out = []
        for text in texts:
            vec = [0.0] * 8
            for ch in text.lower():
                vec[ord(ch) % 8] += 1.0
            out.append(vec)
        return out


class FakeSearch:
    """Deterministic SearchToolAdapter backed by an in-memory page table."""

    name = "fake-search"

    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001
        hits = []
        for rank, (url, text) in enumerate(self._pages.items(), start=1):
            if rank > count:
                break
            hits.append(SearchHit(title=f"Doc {rank}", url=url, snippet=text[:80], source="fake", rank=rank))
        return hits

    def fetch(self, url: str) -> Document:
        return Document(url=url, title=url.rsplit("/", 1)[-1], text=self._pages.get(url, ""), content_type="web")
