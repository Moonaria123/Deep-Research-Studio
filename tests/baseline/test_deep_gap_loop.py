"""Deep Research bounded gap loop: detect gaps -> re-search -> extra sections, capped."""

from __future__ import annotations

from deepresearch_core.baseline.budget import Budget, Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store

from tests.baseline.fakes import FakeLLM, FakeSearch


def _orchestrator(llm, search):
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=llm,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=search,
            memory=MemoryEngine(store),
            enforce_grounding_gate=False,
        ),
        store,
    )


def _search() -> FakeSearch:
    return FakeSearch(
        {
            "https://example.com/quantum": (
                "Quantum computing uses qubits to represent superpositions of states. "
                "Error correction is essential for scalable quantum computers."
            ),
            "https://example.com/qubits": (
                "A qubit can be in a superposition of zero and one. Surface codes help correct errors."
            ),
        }
    )


def test_deep_loop_adds_a_gap_section_when_gaps_detected():
    llm = FakeLLM(scripted={"GAP-ANALYSIS": "Quantum error rates benchmark"})
    orch, store = _orchestrator(llm, _search())
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid))
    payload = outcome.to_dict()
    titles = [s["title"] for s in payload["sections"]]
    assert "Quantum error rates benchmark" in titles
    assert payload["gap_rounds"] >= 1
    assert payload["research_state"]["evidence_matrix"]
    assert payload["evidence_matrix"] == payload["research_state"]["evidence_matrix"]
    assert any(row["subtask"] == "Quantum error rates benchmark" for row in payload["evidence_matrix"])
    store.close()


def test_deep_loop_breaks_when_no_gaps():
    # Default FakeLLM returns its no-grounded sentinel for the gap prompt -> treated as no gaps.
    llm = FakeLLM()
    orch, store = _orchestrator(llm, _search())
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid))
    assert outcome.to_dict()["gap_rounds"] == 0
    store.close()


def test_deep_loop_is_budget_capped():
    llm = FakeLLM(scripted={"GAP-ANALYSIS": "Gap one\nGap two\nGap three\nGap four"})
    orch, store = _orchestrator(llm, _search())
    sid = store.create_session(title="t")
    # max_searches=2 caps total search rounds (initial sections + gap rounds).
    budget = Budget(max_searches=2, max_think_tokens=256, max_output_tokens=800, concurrency=1, timeout_s=60, top_k=6)
    outcome = orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid, budget=budget)
    )
    payload = outcome.to_dict()
    assert len(payload["sections"]) <= 2
    assert payload["answer"]["text"]
    assert payload["research_state"]["searches_used"] <= 2
    assert payload["research_state"]["evidence_matrix"]
    store.close()


def test_deep_loop_dedups_repeated_gap_lines():
    # The LLM repeats the same gap; it must be researched at most once.
    llm = FakeLLM(scripted={"GAP-ANALYSIS": "Same gap line\nSame gap line\nSame gap line"})
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://example.com/quantum-gap": (
                    "Quantum computing uses qubits. The same gap line identifies a documented research gap."
                )
            }
        ),
    )
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid))
    payload = outcome.to_dict()
    titles = [s["title"] for s in payload["sections"]]
    assert titles.count("Same gap line") == 1
    assert payload["gap_rounds"] == 1
    store.close()


def test_deep_loop_preserves_numeric_titles():
    # A gap title containing digits must not be mangled by list-marker cleanup.
    llm = FakeLLM(scripted={"GAP-ANALYSIS": "Market size in 2025"})
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://example.com/quantum-market": (
                    "Quantum computing uses qubits. The market size in 2025 is documented here."
                )
            }
        ),
    )
    sid = store.create_session(title="t")
    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid))
    titles = [s["title"] for s in outcome.to_dict()["sections"]]
    assert "Market size in 2025" in titles
    store.close()


def test_deep_loop_respects_cancellation():
    llm = FakeLLM(scripted={"GAP-ANALYSIS": "Quantum error rates benchmark"})
    orch, store = _orchestrator(llm, _search())
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="deep", query="q", budget={})
    orch.request_cancel(rid)
    try:
        orch.run(RunRequest(query="quantum computing qubits", mode=Mode.DEEP, session_id=sid), run_id=rid)
        raise AssertionError("expected cancellation")
    except BaselineError as exc:
        assert exc.code is ErrorCode.RUN_CANCELLED
    store.close()


def test_deep_explicit_six_sentence_shape_caps_representative_sections() -> None:
    outline = "\n".join(
        (
            "Overview of SQLite FTS5 and Its Use Cases",
            "SQLite FTS5 MATCH Query Syntax and Search Semantics",
            "SQLite FTS5 Tokenization, Phrase Queries, and Boolean Prefix Search",
            "SQLite FTS5 Ranking Results with BM25 and Custom Scoring",
            "SQLite FTS5 Practical Limitations and Examples",
        )
    )
    final = " ".join(
        (
            "SQLite FTS5 is a virtual table module [c1].",
            "It provides full-text search [c1].",
            "MATCH accepts phrase queries [c1].",
            "MATCH also supports prefix and boolean queries [c1].",
            "BM25 ranks matching rows [c1].",
            "Better matches receive lower BM25 values [c1].",
        )
    )
    llm = FakeLLM(
        scripted={
            "Create a concise research outline": outline,
            "Evidence matrix:": final,
            "GAP-ANALYSIS": "This call should be skipped while planned sections fill the budget",
        }
    )
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://www.sqlite.org/fts5.html": (
                    "SQLite FTS5 is a virtual table module that provides full-text search. "
                    "MATCH accepts phrase, prefix, NEAR, boolean, and column-filter queries. "
                    "The BM25 function ranks rows, with better matches receiving lower values."
                )
            }
        ),
    )
    sid = store.create_session(title="t")

    outcome = orch.run(
        RunRequest(
            query=(
                "Write a concise report explaining SQLite FTS5, its MATCH query syntax, and its "
                "ranking capabilities using authoritative SQLite documentation. Use exactly 6 "
                "concise sentences with citations."
            ),
            mode=Mode.DEEP,
            session_id=sid,
        )
    )
    state = outcome.to_dict()["research_state"]

    assert state["searches_max"] == 3
    assert state["searches_used"] == 3
    assert state["plan"] == [
        "Overview of SQLite FTS5 and Its Use Cases",
        "SQLite FTS5 MATCH Query Syntax and Search Semantics",
        "SQLite FTS5 Ranking Results with BM25 and Custom Scoring",
    ]
    assert not any(
        "GAP-ANALYSIS" in str(message.get("content") or "")
        for call in llm.calls
        for message in call
    )
    store.close()


def test_deep_explicit_twelve_sentence_shape_can_plan_all_six_rounds() -> None:
    expected = [
        "SQLite FTS5 overview",
        "SQLite FTS5 table setup",
        "SQLite FTS5 MATCH syntax",
        "SQLite FTS5 tokenizers",
        "SQLite FTS5 BM25 ranking",
        "SQLite FTS5 operational limits",
    ]
    outline = "\n".join(expected)
    llm = FakeLLM(scripted={"Create a concise research outline": outline})
    orch, store = _orchestrator(llm, _search())
    request = RunRequest(
        query="Explain SQLite FTS5 in exactly 12 concise sentences with citations.",
        mode=Mode.DEEP,
    )
    budget = Budget(
        max_searches=6,
        max_think_tokens=1024,
        max_output_tokens=3600,
        concurrency=3,
        timeout_s=180,
        top_k=10,
    )

    sections = orch._plan(  # noqa: SLF001 - regression coverage for the planner boundary
        request.query,
        budget,
        orch._build_context(request),  # noqa: SLF001
        section_count=6,
    )

    assert sections == expected
    store.close()
