"""Sessions persist messages on run completion; latest_run_id surfaces for History."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


def _orchestrator(fake_llm, fake_search) -> tuple[Orchestrator, Store]:
    store = Store(":memory:")
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=fake_search,
        memory=MemoryEngine(store),
        enforce_grounding_gate=True,
    )
    return orch, store


def test_run_completion_persists_user_and_assistant_messages(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))

    messages = store.list_messages(sid)
    roles = [m["role"] for m in messages]
    assert roles[0] == "user"
    assert "assistant" in roles
    assert messages[0]["content"] == "quantum computing qubits"
    store.close()


def test_post_terminal_message_persistence_does_not_call_optional_embedder(fake_llm, fake_search):
    from deepresearch_core.baseline.history import ConversationIndex

    class _RecordingEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts):  # noqa: ANN001, ANN201
            self.calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    store = Store(":memory:")
    embedder = _RecordingEmbedder()
    history = ConversationIndex(store, embedder=embedder)
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=fake_search,
        memory=None,
        enforce_grounding_gate=True,
        history=history,
    )
    sid = store.create_session(title="deadline-safe")

    orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))

    assert embedder.calls == []
    assert [item["role"] for item in store.list_messages(sid)] == ["user", "assistant"]
    store.close()


def test_followup_injects_prior_turn_context(fake_llm, fake_search):
    """Second turn in a session sees the first turn; first turn sees no history."""

    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))
    first_prompt = fake_llm.calls[-1][-1]["content"]
    assert "Conversation so far" not in first_prompt

    orch.run(RunRequest(query="how is error correction done", mode=Mode.FAST, session_id=sid))
    second_prompt = fake_llm.calls[-1][-1]["content"]
    assert "Conversation so far" in second_prompt
    assert "quantum computing qubits" in second_prompt
    store.close()


def test_run_without_session_has_no_context(fake_llm, fake_search):
    """A run with no session_id behaves like a single turn (no injected history)."""

    orch, store = _orchestrator(fake_llm, fake_search)
    orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST))
    assert "Conversation so far" not in fake_llm.calls[-1][-1]["content"]
    store.close()


class _AxisEmbedder:
    """Deterministic keyword->axis embedder for stable cosine ordering (DR-ITER-053)."""

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


def test_followup_injects_semantic_related_turns(fake_llm):
    """DR-ITER-053: an older relevant turn (outside the recent window) is recalled."""

    from deepresearch_core.baseline.history import ConversationIndex
    from tests.baseline.fakes import FakeSearch

    store = Store(":memory:")
    index = ConversationIndex(store, embedder=_AxisEmbedder())
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch(
            {
                "https://example.com/quantum-apple": (
                    "Quantum computing methods can model apple orchard pruning schedules. "
                    "This fixture supports the composite query while the test isolates semantic history recall."
                )
            }
        ),
        memory=MemoryEngine(store),
        enforce_grounding_gate=True,
        history=index,
    )
    sid = store.create_session(title="t")
    # Old relevant turn, then >6 unrelated turns push it out of the recent window.
    index.index_turn(sid, "user", "apple cultivation in cold climates")
    for i in range(4):
        index.index_turn(sid, "user", f"banana question {i}")
        index.index_turn(sid, "assistant", f"banana answer {i}")

    orch.run(
        RunRequest(query="quantum computing apple orchard pruning", mode=Mode.FAST, session_id=sid)
    )
    prompt = fake_llm.calls[-1][-1]["content"]
    assert "Related earlier turns" in prompt
    assert "apple cultivation in cold climates" in prompt
    store.close()


def test_followup_without_embedder_has_no_semantic_block(fake_llm, fake_search):
    """No embedder => linear-only context (no semantic block), matching DR-ITER-026A."""

    orch, store = _orchestrator(fake_llm, fake_search)  # default ConversationIndex, no embedder
    sid = store.create_session(title="t")
    orch.run(RunRequest(query="quantum computing apple basics", mode=Mode.FAST, session_id=sid))
    orch.run(RunRequest(query="quantum computing apple advanced", mode=Mode.FAST, session_id=sid))
    prompt = fake_llm.calls[-1][-1]["content"]
    assert "Related earlier turns" not in prompt
    store.close()


def test_semantic_history_skips_embed_when_no_message_exists_outside_recent_window(fake_llm, fake_search):
    class _HistorySpy:
        calls = 0

        def search_history(self, *_args, **_kwargs):
            self.calls += 1
            return []

    history = _HistorySpy()
    store = Store(":memory:")
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=fake_search,
        memory=MemoryEngine(store),
        enforce_grounding_gate=True,
        history=history,
    )
    sid = store.create_session(title="new")
    request = RunRequest(query="first live question", mode=Mode.FAST, session_id=sid)
    cfg = orch._resolve_context_budget(request)  # noqa: SLF001

    assert orch._semantic_history_block(request, [], cfg, budget=1000) == ""  # noqa: SLF001
    store.add_message(sid, "user", "recent question")
    messages = store.list_messages(sid)
    assert orch._semantic_history_block(request, messages, cfg, budget=1000) == ""  # noqa: SLF001
    assert history.calls == 0
    store.close()


def test_format_history_keeps_newest_within_budget():
    from deepresearch_core.baseline.orchestrator import _format_history

    messages = [
        {"role": "user", "content": "A" * 50},
        {"role": "assistant", "content": "B" * 50},
        {"role": "user", "content": "C" * 50},
    ]
    out = _format_history(messages, turns=6, max_chars=70, msg_chars=1000)
    assert "C" * 50 in out  # newest turn survives
    assert "B" * 50 not in out  # older turns drop once the char budget is spent


def test_get_latest_run_id_returns_most_recent():
    store = Store(":memory:")
    sid = store.create_session(title="t")
    assert store.get_latest_run_id(sid) is None
    store.create_run(session_id=sid, mode="fast", query="a", budget={})
    second = store.create_run(session_id=sid, mode="fast", query="b", budget={})
    assert store.get_latest_run_id(sid) == second
    store.close()


def test_service_list_sessions_includes_latest_run_id():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    sid = service.store.create_session(title="t")
    rid = service.store.create_run(session_id=sid, mode="fast", query="q", budget={})
    target = next(s for s in service.list_sessions() if s["id"] == sid)
    assert target["latest_run_id"] == rid


def test_get_latest_run_for_session_returns_id_and_mode():
    store = Store(":memory:")
    sid = store.create_session(title="t")
    assert store.get_latest_run_for_session(sid) is None
    store.create_run(session_id=sid, mode="fast", query="a", budget={})
    second = store.create_run(session_id=sid, mode="deep", query="b", budget={})
    latest = store.get_latest_run_for_session(sid)
    assert latest == (second, "deep")
    store.close()


def test_service_list_sessions_includes_latest_run_mode():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    sid = service.store.create_session(title="t")
    service.store.create_run(session_id=sid, mode="fast", query="a", budget={})
    service.store.create_run(session_id=sid, mode="council", query="b", budget={})
    target = next(s for s in service.list_sessions() if s["id"] == sid)
    assert target["latest_run_mode"] == "council"


def test_service_list_sessions_without_runs_has_null_mode():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    sid = service.store.create_session(title="empty")
    target = next(s for s in service.list_sessions() if s["id"] == sid)
    assert target["latest_run_id"] is None
    assert target["latest_run_mode"] is None


def test_service_list_messages_roundtrip():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    sid = service.store.create_session(title="t")
    service.store.add_message(sid, "user", "hello")
    service.store.add_message(sid, "assistant", "world")
    contents = [m["content"] for m in service.list_messages(sid)["messages"]]
    assert contents == ["hello", "world"]


# --- HTTP --------------------------------------------------------------------

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def test_messages_endpoint():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    sid = service.store.create_session(title="t")
    service.store.add_message(sid, "user", "endpoint q")
    client = TestClient(create_app(service))
    res = client.get(f"/api/v1/sessions/{sid}/messages")
    assert res.status_code == 200
    assert [m["content"] for m in res.json()["messages"]] == ["endpoint q"]


def test_messages_endpoint_unknown_session_404():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    client = TestClient(create_app(service))
    res = client.get("/api/v1/sessions/does-not-exist/messages")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "SESSION_NOT_FOUND"
