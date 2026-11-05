"""T54 deadline-safe semantic enrichment contracts."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.deadline import current_provider_call_budget
from deepresearch_core.baseline.history import SCOPE_INTRA, ConversationIndex
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.semantic_enrichment import (
    SemanticEnricher,
    SemanticEnrichmentRequest,
    ThreadSemanticBackend,
    validate_embedding_vectors,
)
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.llm import ProviderRequestError
from tests.baseline.fakes import FakeSearch


class _RecordingEmbedder:
    name = "recording-embedder"

    def __init__(self, response=None) -> None:  # noqa: ANN001
        self.calls: list[list[str]] = []
        self.response = response

    def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
        values = [str(text) for text in texts]
        self.calls.append(values)
        if self.response is not None:
            return self.response
        return [[float(index + 1), 1.0] for index in range(len(values))]


def _raw_embedding(store: Store, table: str, row_id: str):
    row = store._conn.execute(f"SELECT embedding FROM {table} WHERE id=?", (row_id,)).fetchone()
    assert row is not None
    return json.loads(row["embedding"]) if row["embedding"] else None


def _seed_missing_rows(store: Store, session_id: str) -> tuple[str, str, str]:
    user_id = store.add_message(session_id, "user", "alpha request")
    assistant_id = store.add_message(session_id, "assistant", "alpha response")
    memory_id = store.add_memory(
        namespace="session",
        scope_key=session_id,
        content="alpha session memory",
    )
    return user_id, assistant_id, memory_id


def test_store_lists_only_missing_continuity_rows_in_stable_pages() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="active")
    user_id, assistant_id, memory_id = _seed_missing_rows(store, session_id)
    store.add_message(session_id, "system", "not eligible")
    store.add_message(session_id, "user", "already embedded", embedding=[1.0, 0.0])
    store.add_memory(namespace="workspace", scope_key=session_id, content="wrong namespace")
    with store._lock:
        store._conn.execute(
            "UPDATE messages SET created_at=100 WHERE id IN (?,?)",
            (user_id, assistant_id),
        )
        store._conn.execute("UPDATE memories SET created_at=100 WHERE id=?", (memory_id,))
        store._conn.commit()

    first = store.list_missing_embedding_rows(session_id, limit=2)
    second = store.list_missing_embedding_rows(
        session_id,
        after_created_at=first[-1]["created_at"],
        after_id=first[-1]["id"],
        limit=2,
    )

    assert len(first) == 2
    assert len(second) == 1
    assert [row["id"] for row in first + second] == sorted([user_id, assistant_id, memory_id])
    assert {row["kind"] for row in first + second} == {"message", "memory"}
    store.close()


def test_store_includes_ttl_archived_session_memory_for_enrichment() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="active")
    memory_id = store.add_memory(namespace="session", scope_key=session_id, content="cold memory")
    assert store.archive_memory(memory_id)

    rows = store.list_missing_embedding_rows(session_id, limit=10)

    assert [row["id"] for row in rows] == [memory_id]
    store.close()


def test_store_excludes_rows_after_session_archive() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="active")
    _seed_missing_rows(store, session_id)
    assert store.archive_session(session_id)

    assert store.is_session_active(session_id) is False
    assert store.list_missing_embedding_rows(session_id, limit=10) == []
    store.close()


def test_store_content_aware_cas_is_idempotent_and_rejects_changed_content() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="cas")
    message_id = store.add_message(session_id, "user", "original")

    assert store.compare_and_set_embedding("message", message_id, "original", [1.0, 2.0]) is True
    assert store.compare_and_set_embedding("message", message_id, "original", [9.0, 9.0]) is False
    assert _raw_embedding(store, "messages", message_id) == [1.0, 2.0]

    changed_id = store.add_message(session_id, "assistant", "before")
    with store._lock:
        store._conn.execute("UPDATE messages SET content='after' WHERE id=?", (changed_id,))
        store._conn.commit()
    assert store.compare_and_set_embedding("message", changed_id, "before", [3.0, 4.0]) is False
    assert _raw_embedding(store, "messages", changed_id) is None
    store.close()


def test_store_lists_only_active_sessions_with_missing_embeddings() -> None:
    store = Store(":memory:")
    active = store.create_session(title="active")
    archived = store.create_session(title="archived")
    complete = store.create_session(title="complete")
    store.add_message(active, "user", "missing")
    store.add_message(archived, "user", "missing but archived")
    store.add_message(complete, "user", "done", embedding=[1.0])
    assert store.archive_session(archived)

    rows = store.list_sessions_missing_embeddings(limit=10)

    assert rows == [{"session_id": active, "messages": 1, "memories": 0}]
    store.close()


def test_validate_embedding_vectors_restores_explicit_index_order() -> None:
    raw = [
        {"index": 1, "embedding": [0.0, 2.0]},
        {"index": 0, "embedding": [1.0, 0.0]},
    ]

    assert validate_embedding_vectors(raw, 2) == [[1.0, 0.0], [0.0, 2.0]]


@pytest.mark.parametrize(
    "raw,expected_count",
    [
        ([[1.0, 0.0]], 2),
        ([{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}], 2),
        ([{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}], 2),
        ([[math.nan, 1.0]], 1),
        ([[math.inf, 1.0]], 1),
        ([[0.0, 0.0]], 1),
        ([[1.0], [1.0, 2.0]], 2),
    ],
)
def test_validate_embedding_vectors_rejects_unsafe_batch(raw, expected_count) -> None:
    with pytest.raises(ValueError):
        validate_embedding_vectors(raw, expected_count)


def test_enricher_writes_all_missing_rows_once() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="enrich")
    row_ids = _seed_missing_rows(store, session_id)
    embedder = _RecordingEmbedder()
    enricher = SemanticEnricher(store=store, embedder=embedder)
    request = SemanticEnrichmentRequest(
        session_id=session_id,
        run_id="run_1",
        trigger="post_run",
        attempt=0,
    )

    first = enricher.run(request)
    second = enricher.run(request)

    assert first.status == "completed"
    assert first.written_count == 3
    assert second.status == "empty"
    assert len(embedder.calls) == 1
    assert all(
        _raw_embedding(store, "messages" if row_id.startswith("msg_") else "memories", row_id)
        for row_id in row_ids
    )
    store.close()


def test_enriched_message_and_session_memory_support_synonym_vector_recall() -> None:
    class _SynonymEmbedder:
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            vectors = []
            for text in texts:
                normalized = str(text).lower()
                if "feline" in normalized or "cat" in normalized:
                    vectors.append([1.0, 0.0])
                elif "automobile" in normalized or "car" in normalized:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([1.0, 1.0])
            return vectors

    store = Store(":memory:")
    session_id = store.create_session(title="synonym-recall")
    message_text = "Feline behavior notes"
    memory_text = "Automobile maintenance notes"
    store.add_message(session_id, "user", message_text)
    store.add_memory(
        namespace="session",
        scope_key=session_id,
        content=memory_text,
    )
    embedder = _SynonymEmbedder()

    outcome = SemanticEnricher(store=store, embedder=embedder).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="post_run")
    )
    history_hits = ConversationIndex(store, embedder=embedder).search_history(
        "cat",
        session_id=session_id,
        scope=SCOPE_INTRA,
    )
    memory_hits = MemoryEngine(store, embedder=embedder).recall(
        session_key=session_id,
        query="car",
    )

    assert outcome.status == "completed"
    assert history_hits[0]["content"] == message_text
    assert memory_hits[0]["content"] == memory_text
    assert "cat" not in message_text.lower()
    assert "car" not in memory_text.lower()
    store.close()


@pytest.mark.parametrize(
    "response",
    [
        [[1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[math.nan, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0], [1.0, 0.0], [0.0, 1.0]],
    ],
)
def test_enricher_invalid_provider_batch_performs_zero_writes(response) -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="invalid")
    row_ids = _seed_missing_rows(store, session_id)
    outcome = SemanticEnricher(store=store, embedder=_RecordingEmbedder(response)).run(
        SemanticEnrichmentRequest(session_id=session_id, run_id=None, trigger="backlog", attempt=0)
    )

    assert outcome.status == "failed"
    assert outcome.reason_codes == ("invalid_embedding_response",)
    assert outcome.written_count == 0
    assert all(
        _raw_embedding(store, "messages" if row_id.startswith("msg_") else "memories", row_id) is None
        for row_id in row_ids
    )
    store.close()


@pytest.mark.parametrize(
    "status_code,retryable,expected_reason,expected_attempts",
    [
        (401, False, "credential_rejected", 1),
        (402, False, "provider_action_required", 1),
        (429, True, "rate_limited", 2),
        (503, True, "service_unavailable", 2),
    ],
)
def test_enricher_classifies_provider_failures_and_retries_only_transient_errors(
    status_code,
    retryable,
    expected_reason,
    expected_attempts,
) -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="provider-error")
    _seed_missing_rows(store, session_id)

    class _FailingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            self.calls += 1
            raise ProviderRequestError(
                "synthetic provider failure",
                status_code=status_code,
                retryable=retryable,
            )

    embedder = _FailingEmbedder()
    outcome = SemanticEnricher(
        store=store,
        embedder=embedder,
        sleeper=lambda _: None,
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "failed"
    assert outcome.reason_codes == (expected_reason,)
    assert outcome.provider_attempts == expected_attempts
    assert embedder.calls == expected_attempts
    assert len(store.list_missing_embedding_rows(session_id, limit=10)) == 3
    store.close()


def test_enricher_installs_independent_ten_second_transport_budget() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="maintenance-budget")
    _seed_missing_rows(store, session_id)
    observed: list[float] = []

    class _BudgetAwareEmbedder(_RecordingEmbedder):
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            budget = current_provider_call_budget()
            assert budget is not None
            observed.append(budget.call_timeout_s(99.0))
            return super().embed(texts, model=model)

    outcome = SemanticEnricher(store=store, embedder=_BudgetAwareEmbedder()).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="post_run")
    )

    assert outcome.status == "completed"
    assert len(observed) == 1
    assert 0.0 < observed[0] <= 10.0
    store.close()


def test_enricher_reports_total_maintenance_deadline_after_provider_timeout() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="maintenance-timeout")
    _seed_missing_rows(store, session_id)
    now = [0.0]

    class _DeadlineEmbedder:
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            now[0] = 31.0
            raise TimeoutError("synthetic timeout")

    outcome = SemanticEnricher(
        store=store,
        embedder=_DeadlineEmbedder(),
        clock=lambda: now[0],
        sleeper=lambda _: None,
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "timeout"
    assert outcome.reason_codes == ("maintenance_deadline_exceeded",)
    assert outcome.provider_attempts == 1
    assert len(store.list_missing_embedding_rows(session_id, limit=10)) == 3
    store.close()


def test_enricher_does_not_mutate_terminal_run_or_append_events() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="terminal")
    _seed_missing_rows(store, session_id)
    run_id = store.create_run(session_id=session_id, mode="fast", query="q")
    store.update_run(run_id, status="completed", result={"answer": {"text": "done"}})
    store.append_event(run_id, "workflow_end", {"status": "completed"})
    before_run = store.get_run(run_id)
    before_events = store.list_events(run_id)

    SemanticEnricher(store=store, embedder=_RecordingEmbedder()).run(
        SemanticEnrichmentRequest(
            session_id=session_id,
            run_id=run_id,
            trigger="post_run",
            attempt=0,
        )
    )

    assert store.get_run(run_id) == before_run
    assert store.list_events(run_id) == before_events
    store.close()


def test_enricher_without_embedder_skips_without_reading_content() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="no embed role")
    _seed_missing_rows(store, session_id)

    outcome = SemanticEnricher(store=store, embedder=None).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="backlog")
    )

    assert outcome.status == "skipped"
    assert outcome.scanned_count == 0
    assert outcome.reason_codes == ("skipped_no_embedder",)
    store.close()


def test_enricher_defers_before_provider_call_when_research_is_busy() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="busy")
    _seed_missing_rows(store, session_id)
    embedder = _RecordingEmbedder()

    outcome = SemanticEnricher(
        store=store,
        embedder=embedder,
        research_busy=lambda: True,
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "deferred"
    assert outcome.reason_codes == ("deferred_research_busy",)
    assert embedder.calls == []
    store.close()


def test_enricher_normalizes_research_queue_check_failure() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="busy-check-failure")
    _seed_missing_rows(store, session_id)
    embedder = _RecordingEmbedder()

    outcome = SemanticEnricher(
        store=store,
        embedder=embedder,
        research_busy=lambda: (_ for _ in ()).throw(ConnectionError("synthetic")),
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "failed"
    assert outcome.reason_codes == ("research_queue_check_failed",)
    assert embedder.calls == []
    store.close()


def test_enricher_rechecks_research_queue_before_each_batch() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="batch-priority")
    for index in range(33):
        store.add_message(session_id, "user", f"row-{index}")

    busy = False

    class _BusyAfterFirstBatch(_RecordingEmbedder):
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            nonlocal busy
            result = super().embed(texts, model=model)
            busy = True
            return result

    embedder = _BusyAfterFirstBatch()
    outcome = SemanticEnricher(
        store=store,
        embedder=embedder,
        research_busy=lambda: busy,
    ).run(SemanticEnrichmentRequest(session_id=session_id, run_id="run-priority"))

    assert outcome.status == "deferred"
    assert outcome.reason_codes == ("deferred_research_busy",)
    assert outcome.written_count == 32
    assert len(embedder.calls) == 1
    assert len(store.list_missing_embedding_rows(session_id, limit=64)) == 1
    store.close()


def test_enricher_rechecks_session_after_provider_call_before_cas() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="archive race")
    row_ids = _seed_missing_rows(store, session_id)

    class _ArchiveDuringEmbed(_RecordingEmbedder):
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            response = super().embed(texts, model=model)
            assert store.archive_session(session_id)
            return response

    outcome = SemanticEnricher(store=store, embedder=_ArchiveDuringEmbed()).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="post_run")
    )

    assert outcome.status == "skipped"
    assert outcome.reason_codes == ("skipped_session_inactive",)
    assert all(
        _raw_embedding(store, "messages" if row_id.startswith("msg_") else "memories", row_id) is None
        for row_id in row_ids
    )
    store.close()


def test_enricher_rechecks_session_after_resolver_before_provider_call() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="archive before provider")
    _seed_missing_rows(store, session_id)
    embedder = _RecordingEmbedder()

    def resolve_after_archive():
        assert store.archive_session(session_id)
        return embedder

    outcome = SemanticEnricher(
        store=store,
        embedder=None,
        embedder_resolver=resolve_after_archive,
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "skipped"
    assert outcome.reason_codes == ("skipped_session_inactive",)
    assert embedder.calls == []
    store.close()


def test_enricher_does_not_resurrect_session_deleted_during_provider_call() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="delete race")
    _seed_missing_rows(store, session_id)

    class _DeleteDuringEmbed(_RecordingEmbedder):
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            response = super().embed(texts, model=model)
            assert store.delete_session(session_id)
            return response

    outcome = SemanticEnricher(store=store, embedder=_DeleteDuringEmbed()).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="post_run")
    )

    assert outcome.status == "skipped"
    assert outcome.reason_codes == ("skipped_session_inactive",)
    assert store.get_session(session_id) is None
    assert store.list_missing_embedding_rows(session_id, limit=10) == []
    store.close()


def test_provider_call_does_not_hold_sqlite_lock() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="network-lock")
    other_session_id = store.create_session(title="concurrent-writer")
    _seed_missing_rows(store, session_id)
    provider_started = threading.Event()
    release_provider = threading.Event()

    class _BlockingEmbedder(_RecordingEmbedder):
        def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            provider_started.set()
            assert release_provider.wait(timeout=5)
            return super().embed(texts, model=model)

    outcome_holder = []
    worker = threading.Thread(
        target=lambda: outcome_holder.append(
            SemanticEnricher(store=store, embedder=_BlockingEmbedder()).run(
                SemanticEnrichmentRequest(session_id=session_id, trigger="post_run")
            )
        )
    )
    worker.start()
    assert provider_started.wait(timeout=5)

    started_at = time.monotonic()
    store.add_message(
        other_session_id,
        "user",
        "concurrent write",
        embedding=[1.0, 0.0],
    )
    write_elapsed = time.monotonic() - started_at
    release_provider.set()
    worker.join(timeout=5)

    assert write_elapsed < 1.0
    assert outcome_holder[0].status == "completed"
    store.close()


def test_enricher_uses_bounded_batches_and_truncates_oversized_items() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="batch bounds")
    oversized = "head-" + ("x" * (20 * 1024)) + "-tail"
    store.add_message(session_id, "user", oversized)
    for index in range(32):
        store.add_message(session_id, "assistant", f"row {index}")
    embedder = _RecordingEmbedder()

    outcome = SemanticEnricher(store=store, embedder=embedder).run(
        SemanticEnrichmentRequest(session_id=session_id, trigger="backlog")
    )

    assert outcome.status == "completed"
    assert outcome.written_count == 33
    assert [len(call) for call in embedder.calls] == [32, 1]
    assert all(len(text.encode("utf-8")) <= 16 * 1024 for call in embedder.calls for text in call)
    assert "content_truncated_for_embedding" in outcome.reason_codes
    store.close()


def test_enricher_captures_current_embedder_at_each_batch_boundary() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="batch-reconfigure")
    for index in range(33):
        store.add_message(session_id, "user", f"row {index}")
    first = _RecordingEmbedder()
    second = _RecordingEmbedder()
    resolvers = iter((first, second))

    outcome = SemanticEnricher(
        store=store,
        embedder=None,
        embedder_resolver=lambda: next(resolvers),
    ).run(SemanticEnrichmentRequest(session_id=session_id, trigger="post_run"))

    assert outcome.status == "completed"
    assert [len(call) for call in first.calls] == [32]
    assert [len(call) for call in second.calls] == [1]
    store.close()


def test_orchestrator_commits_raw_rows_before_post_continuity_callback(fake_llm) -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="barrier")
    embedder = _RecordingEmbedder()
    callback_rows: list[dict[str, list[str]]] = []

    def after_persisted(request, outcome, run_id, persisted_rows) -> None:  # noqa: ANN001
        assert request.session_id == session_id
        assert outcome.status in {"completed", "completed_degraded"}
        assert run_id.startswith("run_")
        message_ids = persisted_rows["message_ids"]
        memory_ids = persisted_rows["session_memory_ids"]
        assert len(message_ids) == 2
        assert len(memory_ids) == 1
        assert all(
            store._conn.execute("SELECT 1 FROM messages WHERE id=?", (row_id,)).fetchone()
            for row_id in message_ids
        )
        assert all(
            store._conn.execute("SELECT 1 FROM memories WHERE id=?", (row_id,)).fetchone()
            for row_id in memory_ids
        )
        callback_rows.append(persisted_rows)

    orchestrator = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store, embedder=fake_llm),
        indexer=Indexer(store, embedder=fake_llm),
        search=FakeSearch(
            {"https://example.test/alpha": "Alpha evidence directly supports the alpha request."}
        ),
        memory=MemoryEngine(store, embedder=embedder),
        enforce_grounding_gate=True,
        post_continuity_persisted=after_persisted,
    )

    orchestrator.run(
        RunRequest(query="Alpha evidence request", mode=Mode.FAST, session_id=session_id)
    )

    assert len(callback_rows) == 1
    assert embedder.calls == []
    store.close()


def test_thread_backend_coalesces_same_session_and_stops_cleanly() -> None:
    started = threading.Event()
    release = threading.Event()
    handled: list[str] = []

    def handle(request: SemanticEnrichmentRequest) -> None:
        handled.append(request.session_id)
        started.set()
        assert release.wait(timeout=5)

    backend = ThreadSemanticBackend(handle, max_pending=128)
    request = SemanticEnrichmentRequest(
        session_id="sess_1",
        run_id="run_1",
        trigger="post_run",
        attempt=0,
    )
    assert backend.enqueue(request) is True
    assert started.wait(timeout=5)
    assert backend.enqueue(request) is True
    release.set()

    backend.shutdown(timeout_s=5)
    assert handled == ["sess_1", "sess_1"]
    assert backend.enqueue(request) is False


def test_thread_backend_reruns_when_new_rows_arrive_after_final_scan() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="lost-wakeup")
    first_id = store.add_message(session_id, "user", "first")
    embedder = _RecordingEmbedder()
    first_scan_finished = threading.Event()
    release_first = threading.Event()
    outcomes = []

    def handle(request: SemanticEnrichmentRequest) -> None:
        outcomes.append(SemanticEnricher(store=store, embedder=embedder).run(request))
        if len(outcomes) == 1:
            first_scan_finished.set()
            assert release_first.wait(timeout=5)

    backend = ThreadSemanticBackend(handle)
    request = SemanticEnrichmentRequest(session_id=session_id, run_id="run-1")
    assert backend.enqueue(request) is True
    assert first_scan_finished.wait(timeout=5)
    second_id = store.add_message(session_id, "assistant", "second")
    assert backend.enqueue(
        SemanticEnrichmentRequest(session_id=session_id, run_id="run-2")
    ) is True
    release_first.set()
    backend.shutdown(timeout_s=5)

    assert [outcome.status for outcome in outcomes] == ["completed", "completed"]
    assert _raw_embedding(store, "messages", first_id) is not None
    assert _raw_embedding(store, "messages", second_id) is not None
    assert len(embedder.calls) == 2
    store.close()


def test_thread_backend_reports_handler_failure_and_continues() -> None:
    failures = []
    handled = []
    completed = threading.Event()

    def handle(request: SemanticEnrichmentRequest) -> None:
        handled.append(request.session_id)
        if request.session_id == "broken":
            raise RuntimeError("synthetic")
        completed.set()

    backend = ThreadSemanticBackend(handle, failure_sink=failures.append)
    assert backend.enqueue(SemanticEnrichmentRequest(session_id="broken"))
    assert backend.enqueue(SemanticEnrichmentRequest(session_id="healthy"))
    assert completed.wait(timeout=5)
    backend.shutdown(timeout_s=5)

    assert handled == ["broken", "healthy"]
    assert len(failures) == 1
    assert failures[0].status == "failed"
    assert failures[0].reason_codes == ("failed_internal",)


def test_thread_backend_full_queue_still_exits_after_shutdown_drain() -> None:
    started = threading.Event()
    release = threading.Event()

    def handle(request: SemanticEnrichmentRequest) -> None:
        if request.session_id == "session-a":
            started.set()
            assert release.wait(timeout=5)

    backend = ThreadSemanticBackend(handle, max_pending=1)
    assert backend.enqueue(SemanticEnrichmentRequest(session_id="session-a"))
    assert started.wait(timeout=5)
    assert backend.enqueue(SemanticEnrichmentRequest(session_id="session-b"))

    backend.shutdown(timeout_s=0.01)
    assert backend.enqueue(SemanticEnrichmentRequest(session_id="session-c")) is False
    release.set()
    backend._thread.join(timeout=5)

    assert backend._thread.is_alive() is False
    assert backend.pending_count == 0


def test_backlog_inventory_is_dry_run_and_secret_free(tmp_path: Path) -> None:
    db_path = tmp_path / "continuity.db"
    store = Store(db_path)
    session_id = store.create_session(title="inventory")
    _seed_missing_rows(store, session_id)
    store.close()
    before_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_files = sorted(path.name for path in tmp_path.iterdir())
    script = Path(__file__).resolve().parents[2] / "scripts" / "backfill_semantic_continuity.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--db-path", str(db_path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["sessions"] == 1
    assert payload["messages"] == 2
    assert payload["memories"] == 1
    assert payload["estimated_batches"] == 1
    assert "alpha request" not in completed.stdout
    assert "embedding" not in completed.stdout.lower()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_hash
    after_files = sorted(path.name for path in tmp_path.iterdir())
    assert set(after_files) - set(before_files) <= {
        f"{db_path.name}-shm",
        f"{db_path.name}-wal",
    }
    reopened = Store(db_path)
    assert len(reopened.list_missing_embedding_rows(session_id, limit=10)) == 3
    reopened.close()


def test_read_only_store_requires_existing_file_and_rejects_writes(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        Store(missing_path, read_only=True)
    assert missing_path.exists() is False

    db_path = tmp_path / "existing.db"
    writable = Store(db_path)
    session_id = writable.create_session(title="read-only")
    writable.add_message(session_id, "user", "missing vector")
    writable.close()

    read_only = Store(db_path, read_only=True)
    assert read_only.list_sessions_missing_embeddings(limit=10)[0]["session_id"] == session_id
    with pytest.raises(sqlite3.OperationalError):
        read_only._conn.execute(
            "UPDATE sessions SET title='mutated' WHERE id=?",
            (session_id,),
        )
    read_only.close()


def test_backlog_inventory_estimates_byte_bounded_batches(tmp_path: Path) -> None:
    db_path = tmp_path / "byte-bounded.db"
    store = Store(db_path)
    session_id = store.create_session(title="byte-bounded")
    for index in range(5):
        store.add_message(session_id, "user", f"{index}-" + ("x" * (20 * 1024)))
    store.close()
    script = Path(__file__).resolve().parents[2] / "scripts" / "backfill_semantic_continuity.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--db-path", str(db_path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["sessions"] == 1
    assert payload["messages"] == 5
    assert payload["estimated_batches"] == 2
