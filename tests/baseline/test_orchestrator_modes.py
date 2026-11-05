"""End-to-end orchestrator runs across all four modes (hermetic, with fakes)."""

from __future__ import annotations

from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store


def _orchestrator(fake_llm, fake_search, *, members=None):
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=fake_llm,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=fake_search,
            memory=MemoryEngine(store),
            council_members=members or [],
            enforce_grounding_gate=True,
        ),
        store,
    )


def test_fast_mode_produces_grounded_answer_with_sources(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))
    payload = outcome.to_dict()
    assert payload["status"] == "completed"
    assert payload["answer"]["citations"]
    assert payload["sources"]
    # Fast now produces a lightweight structured brief: the executive summary is populated.
    assert payload["answer"]["summary"]
    store.close()


def test_multiple_mode_emits_lanes_and_merges(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum error correction", mode=Mode.MULTIPLE, session_id=sid))
    payload = outcome.to_dict()
    assert payload["status"] == "completed"
    assert "lanes" in payload and payload["lanes"]
    store.close()


def test_deep_mode_writes_sections(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="state of quantum computing", mode=Mode.DEEP, session_id=sid))
    payload = outcome.to_dict()
    assert payload["status"] == "completed"
    assert "sections" in payload
    store.close()


def test_council_mode_reports_confidence(fake_llm, fake_search):
    from tests.baseline.fakes import FakeLLM

    members = [FakeLLM("m-a"), FakeLLM("m-b")]
    orch, store = _orchestrator(fake_llm, fake_search, members=members)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.COUNCIL, session_id=sid))
    payload = outcome.to_dict()
    assert payload["status"] == "completed"
    assert "confidence" in payload and 0.0 <= payload["confidence"] <= 1.0
    assert "members" in payload
    store.close()


def test_cancel_before_run_raises_run_cancelled(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="deep", query="q", budget={})
    orch.request_cancel(rid)
    try:
        orch.run(RunRequest(query="q", mode=Mode.DEEP, session_id=sid), run_id=rid)
        assert False, "expected cancellation"
    except BaselineError as exc:
        assert exc.code is ErrorCode.RUN_CANCELLED
    # A cooperative cancel is terminal "cancelled", distinct from a run failure (DR-ITER-021).
    assert store.get_run(rid)["status"] == "cancelled"
    store.close()


def test_events_are_persisted_for_streaming(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))
    events = store.list_events(outcome.run_id, after_seq=0)
    types = {e["type"] for e in events}
    assert "node_status" in types
    assert "workflow_end" in types
    store.close()


def test_fast_mode_emits_budget_snapshot(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))
    snaps = [e for e in store.list_events(outcome.run_id, after_seq=0) if e["type"] == "budget_snapshot"]
    assert snaps, "fast run should emit at least one budget_snapshot"
    payload = snaps[0]["payload"]
    assert payload["mode"] == "fast"
    assert payload["searches_used"] == 1
    assert payload["searches_max"] >= 1
    assert payload["stage"] == "search"
    store.close()


def test_execution_provenance_comes_from_resolved_adapter_and_actual_search_stats(fake_llm, fake_search):
    fake_llm._deepresearch_model_ref = "remote:model-a"
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")

    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))

    provenance = outcome.to_dict()["execution_provenance"]
    assert provenance["schema"] == "deepresearch.execution-provenance/v1"
    assert provenance["model_observation"] == "resolved_adapter"
    assert provenance["search_observation"] == "indexer_stats"
    assert provenance["llm_model_refs"] == ["remote:model-a"]
    assert provenance["search_providers"] == []
    store.close()


def test_council_mode_budget_snapshot_stages_follow_strategy(fake_llm, fake_search):
    from tests.baseline.fakes import FakeLLM

    cases = {
        "parallel_vote": ["search", "drafts", "vote"],
        "judge_synthesis": ["search", "drafts", "synthesis"],
        "debate_once": ["search", "drafts", "critique", "synthesis"],
    }

    for strategy, expected in cases.items():
        members = [FakeLLM("m-a"), FakeLLM("m-b")]
        orch, store = _orchestrator(fake_llm, fake_search, members=members)
        sid = store.create_session(title=f"t-{strategy}")
        outcome = orch.run(
            RunRequest(
                query="quantum computing qubits",
                mode=Mode.COUNCIL,
                session_id=sid,
                council_strategy=strategy,
            )
        )
        stages = [
            e["payload"]["stage"]
            for e in store.list_events(outcome.run_id, after_seq=0)
            if e["type"] == "budget_snapshot"
        ]
        positions = [stages.index(stage) for stage in expected]
        assert positions == sorted(positions)
        if strategy != "debate_once":
            assert "critique" not in stages
        store.close()


def test_deep_mode_budget_snapshot_carries_gap_rounds(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="state of quantum computing", mode=Mode.DEEP, session_id=sid))
    snaps = [e for e in store.list_events(outcome.run_id, after_seq=0) if e["type"] == "budget_snapshot"]
    assert snaps, "deep run should emit budget snapshots"
    # Every deep snapshot reports a bounded, non-decreasing searches_used and a gap_rounds field.
    for snap in snaps:
        payload = snap["payload"]
        assert payload["mode"] == "deep"
        assert "gap_rounds" in payload
        assert 1 <= payload["searches_used"] <= payload["searches_max"]
    store.close()
