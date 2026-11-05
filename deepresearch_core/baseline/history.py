"""Conversation-history vector index (intra- and cross-conversation recall).

Embeds chat messages into ``messages.embedding`` on persist and offers semantic
search over them. Like the rest of the memory stack, this degrades gracefully:
without an embedder (or on any embedding error) messages are still stored, just
without a vector, and ``search_history`` returns an empty list.
"""

from __future__ import annotations

from typing import Any

from .store import JsonDict, Store

# Scope keywords for ``search_history``.
SCOPE_INTRA = "intra"  # within a single conversation
SCOPE_CROSS = "cross"  # across all conversations


class ConversationIndex:
    """Persists chat turns with embeddings and searches them semantically."""

    def __init__(self, store: Store, *, embedder: Any | None = None) -> None:
        self._store = store
        self._embedder = embedder

    def index_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: JsonDict | None = None,
        embed: bool = True,
    ) -> str:
        """Persist a message, optionally embedding it when an embedder is configured."""

        return self._store.add_message(
            session_id,
            role,
            content,
            metadata=metadata,
            embedding=self._embed_one(content) if embed else None,
        )

    def search_history(
        self, query: str, *, session_id: str = "", scope: str = SCOPE_CROSS, limit: int = 10
    ) -> list[JsonDict]:
        """Return messages most relevant to ``query``.

        ``scope=intra`` restricts the search to ``session_id`` (which must be
        provided); ``scope=cross`` searches every conversation. Returns ``[]``
        when no embedder is available or nothing matches.
        """

        query_vec = self._embed_one(query)
        if query_vec is None:
            return []
        target_session = session_id if scope == SCOPE_INTRA else ""
        if scope == SCOPE_INTRA and not session_id:
            return []
        return self._store.search_messages_semantic(query_vec, session_id=target_session, limit=limit)

    def _embed_one(self, text: str) -> list[float] | None:
        """Embed a single string, returning ``None`` when unavailable/failed."""

        if not self._embedder or not text:
            return None
        try:
            vectors = self._embedder.embed([text])
        except Exception:  # noqa: BLE001 - embedding is best-effort (graceful degradation)
            return None
        if vectors and isinstance(vectors[0], list):
            return [float(x) for x in vectors[0]]
        return None
