"""Memory duplicate preview and user-confirmed cleanup actions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from .adapters import LLMAdapter
from .errors import BaselineError, ErrorCode
from .memory import MemoryEngine, MemoryLayer
from .memory_lifecycle import MemoryLifecycleService
from .retrieval import cosine_similarity
from .store import Store

JsonDict = dict[str, Any]

_DEFAULT_THRESHOLD = 0.82
_MIN_THRESHOLD = 0.5
_MAX_THRESHOLD = 0.99
_MAX_SUMMARY_CHARS = 600
_MANAGED = (MemoryLayer.WORKSPACE.value, MemoryLayer.USER.value)


class MemoryOptimizer:
    """Find duplicate memories and apply explicit user-selected cleanup."""

    def __init__(
        self,
        store: Store,
        *,
        memory: MemoryEngine,
        lifecycle: MemoryLifecycleService,
        index_memory: Callable[..., None],
        memory_llm: Callable[[], LLMAdapter | None],
    ) -> None:
        self._store = store
        self._memory = memory
        self._lifecycle = lifecycle
        self._index_memory = index_memory
        self._memory_llm = memory_llm

    def preview_duplicates(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        threshold: float = _DEFAULT_THRESHOLD,
        limit: int = 200,
    ) -> JsonDict:
        """Return candidate duplicate groups without mutating memories."""

        threshold = _coerce_threshold(threshold)
        rows = [
            row
            for row in self._active_rows(workspace_key=workspace_key, user_key=user_key, limit=limit)
            if not _is_pinned(row)
        ]
        vectors = self._embedding_map(workspace_key=workspace_key, user_key=user_key, limit=limit)
        used: set[str] = set()
        groups: list[JsonDict] = []
        for index, row in enumerate(rows):
            mid = str(row.get("id") or "")
            if not mid or mid in used:
                continue
            matches: list[tuple[JsonDict, float]] = [(row, 1.0)]
            for other in rows[index + 1 :]:
                oid = str(other.get("id") or "")
                if not oid or oid in used:
                    continue
                score = _memory_similarity(row, other, vectors)
                if score >= threshold:
                    matches.append((other, score))
            if len(matches) < 2:
                continue
            ids = [str(item.get("id") or "") for item, _score in matches]
            used.update(ids)
            keep_id = _choose_keep_id([item for item, _score in matches])
            groups.append(
                {
                    "group_id": _group_id(ids),
                    "similarity": round(sum(score for _item, score in matches[1:]) / (len(matches) - 1), 4),
                    "items": [_public_memory(item) for item, _score in matches],
                    "suggested_summary": _deterministic_summary([item for item, _score in matches]),
                    "suggestions": [
                        {"type": "merge", "memory_ids": ids},
                        {"type": "archive", "memory_ids": [mid for mid in ids if mid != keep_id], "keep_id": keep_id},
                        {"type": "downrank", "memory_ids": [mid for mid in ids if mid != keep_id], "factor": 0.5},
                    ],
                }
            )
        return {"groups": groups, "threshold": threshold}

    def apply(self, payload: JsonDict) -> JsonDict:
        """Apply one confirmed optimizer action."""

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "optimizer payload must be an object")
        strategy = str(payload.get("strategy") or "merge").strip().lower()
        memory_ids = [str(item).strip() for item in payload.get("memory_ids") or [] if str(item).strip()]
        if not memory_ids:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory_ids are required")
        if strategy == "merge":
            return self._apply_merge(memory_ids)
        if strategy == "archive":
            self._assert_not_pinned(memory_ids)
            archived = [self._lifecycle.archive_memory(memory_id)["memory_id"] for memory_id in memory_ids]
            return {"strategy": "archive", "archived_ids": archived}
        if strategy == "downrank":
            self._assert_not_pinned(memory_ids)
            factor = _coerce_factor(payload.get("factor", 0.5))
            updated: list[str] = []
            for memory_id in memory_ids:
                row = self._managed_active(memory_id)
                salience = float(row.get("salience") or 1.0) * factor
                if self._store.update_memory_salience(memory_id, salience):
                    updated.append(memory_id)
            return {"strategy": "downrank", "memory_ids": updated, "factor": factor}
        raise BaselineError(ErrorCode.INVALID_REQUEST, "strategy must be merge, archive, or downrank")

    def _apply_merge(self, memory_ids: list[str]) -> JsonDict:
        if len(set(memory_ids)) < 2:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "merge requires at least two memories")
        rows = [self._managed_active(memory_id) for memory_id in dict.fromkeys(memory_ids)]
        self._assert_not_pinned([str(row.get("id") or "") for row in rows])
        llm = self._memory_llm()
        if llm is None:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "memory processing model is not configured",
                details={"role": "memory"},
            )
        summary = self._summarize(rows, llm=llm)
        target_layer = MemoryLayer.USER if any(row.get("namespace") == MemoryLayer.USER.value for row in rows) else MemoryLayer.WORKSPACE
        target_scope = next(
            str(row.get("scope_key") or "default")
            for row in rows
            if row.get("namespace") == target_layer.value
        )
        salience = max(float(row.get("salience") or 1.0) for row in rows)
        source_ids = [str(row.get("id")) for row in rows]
        try:
            merged_id = self._store.merge_memories_atomic(
                namespace=target_layer.value,
                scope_key=target_scope,
                content=summary,
                salience=salience,
                metadata={"source": "memory_optimizer", "merged_from": source_ids},
                source_ids=source_ids,
            )
        except Exception as error:  # noqa: BLE001
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "memory merge could not be applied atomically",
                details={"memory_ids": source_ids},
            ) from error
        return {
            "strategy": "merge",
            "memory_id": merged_id,
            "archived_ids": source_ids,
            "summary": summary,
            "library_index_warning": None,
        }

    def _summarize(self, rows: list[JsonDict], *, llm: LLMAdapter) -> str:
        bullets = "\n".join(f"- {str(row.get('content') or '').strip()}" for row in rows)
        prompt = (
            "Merge these memory entries into one durable, non-sensitive memory. "
            "Return one concise sentence, no bullets, no explanations.\n\n"
            f"{bullets}"
        )
        try:
            result = llm.complete([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=180)
        except Exception as error:  # noqa: BLE001
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, "memory processing model failed") from error
        summary = re.sub(r"\s+", " ", result.text).strip(" -\t\r\n")
        if not summary:
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, "memory processing model returned an empty summary")
        return summary[:_MAX_SUMMARY_CHARS].strip()

    def _active_rows(self, *, workspace_key: str, user_key: str, limit: int) -> list[JsonDict]:
        rows: list[JsonDict] = []
        for namespace, scope_key in (
            (MemoryLayer.WORKSPACE.value, workspace_key),
            (MemoryLayer.USER.value, user_key),
        ):
            rows.extend(
                self._store.list_memories(namespace=namespace, scope_key=scope_key, limit=limit)
            )
        return rows[:limit]

    def _embedding_map(self, *, workspace_key: str, user_key: str, limit: int) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for namespace, scope_key in ((MemoryLayer.WORKSPACE.value, workspace_key), (MemoryLayer.USER.value, user_key)):
            for row in self._store.iter_memories_with_embeddings(namespace=namespace, scope_key=scope_key, limit=limit):
                vec = row.get("embedding")
                if isinstance(vec, list):
                    out[str(row.get("id") or "")] = [float(value) for value in vec]
        return out

    def _managed_active(self, memory_id: str) -> JsonDict:
        row = self._store.get_memory(memory_id)
        if not row:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory not found")
        if row.get("archived_at") is not None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "archived memories cannot be optimized")
        if str(row.get("namespace") or "") not in _MANAGED:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "only workspace/user memories can be optimized")
        return row

    def _assert_not_pinned(self, memory_ids: list[str]) -> None:
        pinned_ids = [
            memory_id
            for memory_id in dict.fromkeys(memory_ids)
            if _is_pinned(self._managed_active(memory_id))
        ]
        if pinned_ids:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "pinned memories cannot be optimized automatically",
                details={"pinned_ids": pinned_ids},
            )


def _memory_similarity(left: JsonDict, right: JsonDict, vectors: dict[str, list[float]]) -> float:
    left_id = str(left.get("id") or "")
    right_id = str(right.get("id") or "")
    vector_score = cosine_similarity(vectors.get(left_id) or [], vectors.get(right_id) or [])
    return max(vector_score, _token_jaccard(str(left.get("content") or ""), str(right.get("content") or "")))


def _token_jaccard(left: str, right: str) -> float:
    a = set(_tokens(left))
    b = set(_tokens(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    union = len(a | b)
    containment = overlap / min(len(a), len(b))
    return max(overlap / union, containment)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _choose_keep_id(rows: list[JsonDict]) -> str:
    winner = max(rows, key=lambda row: (float(row.get("salience") or 0), float(row.get("last_used_at") or 0)))
    return str(winner.get("id") or "")


def _is_pinned(row: JsonDict) -> bool:
    return bool(row.get("pinned"))


def _deterministic_summary(rows: list[JsonDict]) -> str:
    ordered = sorted(rows, key=lambda row: (float(row.get("salience") or 0), float(row.get("last_used_at") or 0)), reverse=True)
    text = " ".join(str(row.get("content") or "").strip() for row in ordered if row.get("content"))
    return re.sub(r"\s+", " ", text).strip()[:_MAX_SUMMARY_CHARS]


def _group_id(memory_ids: list[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(memory_ids)).encode("utf-8")).hexdigest()[:12]
    return f"memgrp_{digest}"


def _public_memory(row: JsonDict) -> JsonDict:
    namespace = str(row.get("namespace") or "")
    salience = float(row.get("salience") or 0.0)
    return {
        **row,
        "layer": namespace,
        "effective_salience": float(row.get("effective_salience") or salience),
    }


def _coerce_threshold(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_THRESHOLD
    return min(_MAX_THRESHOLD, max(_MIN_THRESHOLD, parsed))


def _coerce_factor(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return min(0.95, max(0.05, parsed))
