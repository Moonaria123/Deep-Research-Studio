"""Dreaming v2: distill recent local activity into workspace memories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .adapters import LLMAdapter
from .memory import MemoryEngine, MemoryLayer
from .store import Store

JsonDict = dict[str, Any]

_PROFILE_KINDS = {
    "topics": "topic",
    "domains": "domain",
    "preferences": "preference",
    "constraints": "constraint",
}
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{6,}|ghp_[A-Za-z0-9_]{6,}|AKIA[0-9A-Z]{8,}|"
    r"xox[abp]-[A-Za-z0-9-]{6,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


class DreamingError(RuntimeError):
    """Raised when a Dreaming run cannot produce a safe profile."""


@dataclass(frozen=True)
class DreamingConfig:
    """Bounds for a single Dreaming prompt."""

    recent_session_limit: int = 8
    messages_per_session: int = 10
    existing_memory_limit: int = 40
    report_limit: int = 6
    max_input_chars: int = 12000
    max_entry_chars: int = 240


class DreamingService:
    """Distill local history and Library artifacts into reusable memories."""

    def __init__(
        self,
        *,
        store: Store,
        memory: MemoryEngine,
        llm: LLMAdapter,
        config: DreamingConfig | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._llm = llm
        self._config = config or DreamingConfig()

    def run(self, *, dream_run_id: str, workspace_key: str = "default", user_key: str = "default") -> JsonDict:
        """Run one distillation pass and return creation stats."""

        prompt = self._build_prompt(workspace_key=workspace_key, user_key=user_key)
        result = self._llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=900,
        )
        profile = _parse_profile(result.text)
        created = 0
        skipped = 0
        existing = {
            _normalize(row.get("content", ""))
            for row in self._store.list_memories(
                namespace=MemoryLayer.WORKSPACE.value,
                scope_key=workspace_key,
                limit=500,
            )
        }
        seen = set(existing)

        for source_key, kind in _PROFILE_KINDS.items():
            values = profile.get(source_key) or []
            if not isinstance(values, list):
                continue
            for raw in values:
                content = _clean_entry(str(raw or ""), max_chars=self._config.max_entry_chars)
                normalized = _normalize(content)
                if not content or normalized in seen or _SECRET_RE.search(content):
                    skipped += 1
                    continue
                metadata = {
                    "source": "dreaming_v2",
                    "dream_run_id": dream_run_id,
                    "kind": kind,
                    "workspace_key": workspace_key,
                    "content_hash": _content_hash(content),
                }
                self._memory.remember(
                    MemoryLayer.WORKSPACE,
                    workspace_key,
                    content,
                    salience=2.5,
                    metadata=metadata,
                )
                seen.add(normalized)
                created += 1

        if created == 0 and skipped == 0:
            raise DreamingError("dreaming produced no profile entries")
        return {"memories_created": created, "skipped": skipped}

    def _build_prompt(self, *, workspace_key: str, user_key: str) -> str:
        sections = [
            "Distill a local research profile from the user's recent DeepResearch activity.",
            "Return strict JSON only with keys: topics, domains, preferences, constraints.",
            "Each value must be an array of short, durable, non-sensitive memory statements.",
            "Do not include API keys, credentials, private URLs, internal file paths, or implementation details.",
            "",
            "Recent sessions:",
            self._recent_sessions(),
            "",
            "Existing workspace memories:",
            self._memory_lines(MemoryLayer.WORKSPACE.value, workspace_key),
            "",
            "Existing user memories:",
            self._memory_lines(MemoryLayer.USER.value, user_key),
            "",
            "Library notes:",
            self._memory_lines("note", "default"),
            "",
            "Recent reports:",
            self._report_lines(),
        ]
        return _truncate("\n".join(sections), self._config.max_input_chars)

    def _recent_sessions(self) -> str:
        out: list[str] = []
        for session in self._store.list_sessions(limit=self._config.recent_session_limit):
            title = str(session.get("title") or "Untitled")[:120]
            out.append(f"- Session: {title}")
            messages = self._store.list_messages(str(session.get("id") or ""))[-self._config.messages_per_session :]
            for message in messages:
                role = str(message.get("role") or "message")[:20]
                content = _one_line(str(message.get("content") or ""), 500)
                if content:
                    out.append(f"  - {role}: {content}")
        return "\n".join(out) if out else "(none)"

    def _memory_lines(self, namespace: str, scope_key: str) -> str:
        rows = self._store.list_memories(
            namespace=namespace,
            scope_key=scope_key,
            limit=self._config.existing_memory_limit,
        )
        lines = [_one_line(str(row.get("content") or ""), 300) for row in rows]
        lines = [line for line in lines if line]
        return "\n".join(f"- {line}" for line in lines) if lines else "(none)"

    def _report_lines(self) -> str:
        rows = self._store.list_reports(limit=self._config.report_limit)
        lines: list[str] = []
        for row in rows:
            title = _one_line(str(row.get("title") or "Untitled report"), 120)
            preview = _one_line(str(row.get("body_preview") or ""), 300)
            lines.append(f"- {title}: {preview}" if preview else f"- {title}")
        return "\n".join(lines) if lines else "(none)"


def _parse_profile(text: str) -> JsonDict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DreamingError("dreaming returned invalid JSON") from error
    if not isinstance(data, dict):
        raise DreamingError("dreaming returned a non-object profile")
    if not any(isinstance(data.get(key), list) and data.get(key) for key in _PROFILE_KINDS):
        raise DreamingError("dreaming produced no profile entries")
    return data


def _clean_entry(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\t\r\n")
    return cleaned[:max_chars].strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _one_line(text: str, max_chars: int) -> str:
    return _truncate(re.sub(r"\s+", " ", text).strip(), max_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
