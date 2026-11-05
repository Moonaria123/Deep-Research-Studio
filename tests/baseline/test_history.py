"""Conversation-history vector index: intra/cross search + graceful degradation."""

from __future__ import annotations

from deepresearch_core.baseline.history import SCOPE_CROSS, SCOPE_INTRA, ConversationIndex
from deepresearch_core.baseline.store import Store


class _AxisEmbedder:
    """Deterministic keyword->axis embedder for stable cosine ordering."""

    _AXES = {"apple": [1.0, 0.0, 0.0], "orange": [0.0, 1.0, 0.0], "banana": [0.0, 0.0, 1.0]}

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            for word, axis in self._AXES.items():
                if word in text.lower():
                    vec = list(axis)
                    break
            out.append(vec)
        return out


def test_index_turn_embeds_and_is_searchable_intra():
    store = Store(":memory:")
    index = ConversationIndex(store, embedder=_AxisEmbedder())
    sid = store.create_session(title="t")
    index.index_turn(sid, "user", "tell me about apple")
    index.index_turn(sid, "user", "tell me about banana")
    hits = index.search_history("apple please", session_id=sid, scope=SCOPE_INTRA)
    assert hits[0]["content"] == "tell me about apple"


def test_search_history_cross_spans_sessions():
    store = Store(":memory:")
    index = ConversationIndex(store, embedder=_AxisEmbedder())
    s1 = store.create_session(title="a")
    s2 = store.create_session(title="b")
    index.index_turn(s1, "user", "apple notes")
    index.index_turn(s2, "user", "apple ideas")
    hits = index.search_history("apple", scope=SCOPE_CROSS)
    sessions = {h["session_id"] for h in hits}
    assert sessions == {s1, s2}


def test_search_history_intra_requires_session_id():
    store = Store(":memory:")
    index = ConversationIndex(store, embedder=_AxisEmbedder())
    sid = store.create_session(title="t")
    index.index_turn(sid, "user", "apple")
    assert index.search_history("apple", scope=SCOPE_INTRA) == []


def test_search_history_without_embedder_returns_empty():
    store = Store(":memory:")
    index = ConversationIndex(store)  # no embedder
    sid = store.create_session(title="t")
    index.index_turn(sid, "user", "apple")  # stored, but not embedded
    assert index.search_history("apple", session_id=sid, scope=SCOPE_INTRA) == []


def test_index_turn_without_embedder_still_persists():
    store = Store(":memory:")
    index = ConversationIndex(store)
    sid = store.create_session(title="t")
    mid = index.index_turn(sid, "user", "hello world")
    assert mid
    assert store.list_messages(sid)[0]["content"] == "hello world"


def test_index_turn_can_skip_optional_embedding_and_still_persist():
    class _RecordingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts):  # noqa: ANN001, ANN201
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    store = Store(":memory:")
    embedder = _RecordingEmbedder()
    index = ConversationIndex(store, embedder=embedder)
    sid = store.create_session(title="raw")

    index.index_turn(sid, "user", "deadline-safe message", embed=False)

    assert embedder.calls == 0
    assert store.list_messages(sid)[0]["content"] == "deadline-safe message"
    assert store.search_messages_semantic([1.0, 0.0], session_id=sid) == []
    store.close()
