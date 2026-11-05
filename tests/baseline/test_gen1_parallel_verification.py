"""Report delivery is independent of advisory verification."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from deepresearch_core.baseline import orchestrator as runtime
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM


def test_report_is_delivered_while_verification_is_still_running(monkeypatch, fake_llm, fake_search):
    verification_entered, release_verification, delivered = Event(), Event(), Event()

    def slow_verifier(*args, **kwargs):
        verification_entered.set()
        assert release_verification.wait(5)
        return [], []

    monkeypatch.setattr(runtime, "analyze_evidence_pack_conflicts", slow_verifier)
    store = Store(":memory:")
    sid = store.create_session(title="Parallel delivery")
    rid = store.create_run(session_id=sid, mode="fast", query="quantum computing", budget={})
    orch = Orchestrator(store=store, llm=fake_llm, search=fake_search)

    def on_event(event):
        if event["type"] == "workflow_end":
            assert verification_entered.is_set()
            delivered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(orch.run, RunRequest(query="quantum computing", mode=Mode.FAST, session_id=sid), run_id=rid, sink=on_event)
        try:
            assert delivered.wait(5)
            run = store.get_run(rid)
            assert run["status"] == "completed"
            assert run["result"]["answer"]["text"]
            assert run["result"]["verification"]["status"] == "running"
            assert not future.done()
            original_answer = run["result"]["answer"]["text"]
        finally:
            release_verification.set()
        future.result(timeout=5)
    final = store.get_run(rid)
    assert final["result"]["verification"]["status"] == "completed"
    assert final["result"]["answer"]["text"] == original_answer
    assert [e["type"] for e in store.list_events(rid, after_seq=0)][-1] == "workflow_end"
    store.close()


@pytest.mark.parametrize("mode", list(Mode))
def test_low_confidence_never_retracts_a_generated_report(mode, fake_search):
    llm = FakeLLM(scripted={"": "The unsupported machine has exactly 9999 quantum processors. [c99]"})
    store = Store(":memory:")
    sid = store.create_session(title="Advisory verification")
    orch = Orchestrator(store=store, llm=llm, search=fake_search, council_members=[llm, llm])
    outcome = orch.run(RunRequest(query="quantum computing", mode=mode, session_id=sid))
    run = store.get_run(outcome.run_id)
    assert run["status"] in {"completed", "completed_degraded"}
    assert run["result"]["answer"]["text"]
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    assert run["result"]["confidence_warning"]["reasons"]
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    assert not any(e["type"] == "answer_reset" for e in store.list_events(outcome.run_id, after_seq=0))
    store.close()


def test_verifier_failure_retains_report_and_exposes_incomplete_check(monkeypatch, fake_llm, fake_search):
    def fail_verifier(*args, **kwargs):
        raise RuntimeError("private provider error must not reach metadata")

    monkeypatch.setattr(runtime, "analyze_evidence_pack_conflicts", fail_verifier)
    store = Store(":memory:")
    sid = store.create_session(title="Verification outage")
    outcome = Orchestrator(store=store, llm=fake_llm, search=fake_search).run(
        RunRequest(query="quantum computing", mode=Mode.FAST, session_id=sid),
    )
    result = store.get_run(outcome.run_id)["result"]
    assert result["answer"]["text"]
    assert result["verification"]["status"] == "failed"
    assert result["confidence_warning"]["reasons"] == ["verification_incomplete"]
    assert "private provider" not in str(result)
    store.close()


def test_regeneration_reuses_evidence_and_bypasses_a_warm_search_cache():
    from deepresearch_core.baseline.adapters import ResilientSearchAdapter
    from deepresearch_core.baseline.retrieval import Retriever
    from tests.baseline.test_t49_live_path_budget import _CountingCacheClient

    store = Store(":memory:")
    client = _CountingCacheClient()
    llm = FakeLLM()
    search = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    orch = Orchestrator(store=store, llm=llm, search=search)
    sid = store.create_session(title="Reissue")
    request = RunRequest(query="Alpha evidence", mode=Mode.FAST, session_id=sid)
    parent = orch.run(request)
    orch.run(request)
    assert client.search_calls == 1
    prior_text = "Alpha evidence records a previous documented capability."
    doc = store.upsert_document(url="https://example.test/prior", title="Alpha evidence", text=prior_text)
    chunk = store.add_chunk(document_id=doc, chunk_index=0, text=prior_text)
    child = store.create_run(session_id=sid, mode="fast", query=request.query, budget={}, metadata={
        "regeneration_context": {
            "parent_run_id": parent.run_id, "fresh_search_required": True,
            "seed_evidence": [{"document_id": doc, "chunk_id": chunk, "url": "https://example.test/prior",
                               "title": "Alpha evidence", "snippet": prior_text, "origin": "external",
                               "artifact_id": doc, "artifact_type": "web_document", "source_status": "active",
                               "grounding_eligible": True, "provenance_refs": ["https://example.test/prior"],
                               "score": 0.8, "published_at": 100.0, "retrieved_at": 200.0}],
        },
    })
    outcome = orch.run(request, run_id=child)
    assert client.search_calls == 2
    assert any(prior_text in str(message) for message in llm.calls[-1])
    prior = next(c for c in outcome.answer.citations if c.url == "https://example.test/prior")
    assert prior.published_at == 100.0
    assert store.get_document(doc)["text"] == prior_text
    assert store.get_run(parent.run_id)["result"]["answer"]["text"] == parent.answer.text
    store.close()
