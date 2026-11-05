"""DR-ITER-073: cooperative run deadline and partial-timeout semantics."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.adapters import Document
from deepresearch_core.baseline.budget import Budget, Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import (
    Orchestrator,
    _deep_section_work_reserve_s,
    _RunDeadline,
)
from deepresearch_core.baseline.budget import retrieval_budget_for
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store

from tests.baseline.fakes import FakeLLM, FakeSearch


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TimeoutAfterFetchSearch(FakeSearch):
    def __init__(self, pages: dict[str, str], clock: ManualClock) -> None:
        super().__init__(pages)
        self._clock = clock

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        hits = super().search(query, count=count, options=options)
        self._clock.now = 10.0
        return hits

    def fetch(self, url: str) -> Document:
        document = super().fetch(url)
        self._clock.now = 10.0
        return document


class TimeoutBeforeEvidenceSearch(FakeSearch):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__({})
        self._clock = clock

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        self._clock.now = 10.0
        return []


class TimeoutDuringTemporalAssuranceSearch(FakeSearch):
    def __init__(self, pages: dict[str, str], clock: ManualClock) -> None:
        super().__init__(pages)
        self._clock = clock

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        if "latest current status effective date" in query:
            self._clock.now = 10.0
        return super().search(query, count=count, options=options)


class TimeoutOnSecondDeepSearch(FakeSearch):
    def __init__(self, pages: dict[str, str], clock: ManualClock) -> None:
        super().__init__(pages)
        self._clock = clock
        self.calls = 0

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        self.calls += 1
        hits = super().search(query, count=count, options=options)
        if self.calls == 2:
            self._clock.now = 60.0
        return hits


def _orchestrator(search, clock: ManualClock, llm: FakeLLM | None = None) -> tuple[Orchestrator, Store]:
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=llm or FakeLLM(),
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=search,
            memory=MemoryEngine(store),
            enforce_grounding_gate=False,
            clock=clock,
        ),
        store,
    )


def test_run_deadline_expires_with_injected_clock() -> None:
    clock = ManualClock()
    deadline = _RunDeadline(timeout_s=5, clock=clock)

    assert not deadline.expired()
    clock.now = 5
    assert deadline.expired()


def test_deep_section_work_reserve_uses_only_default_budget_surplus() -> None:
    retrieval = retrieval_budget_for(Mode.DEEP)

    assert _deep_section_work_reserve_s(60, retrieval) == 0.0
    assert _deep_section_work_reserve_s(180, retrieval) == 15.0


def test_deep_retrieval_cannot_consume_reserved_writer_time() -> None:
    clock = ManualClock()
    search = TimeoutAfterFetchSearch(
        {
            "https://example.com/quantum": (
                "Quantum computing uses qubits. Error correction is essential for scale."
            )
        },
        clock,
    )
    llm = FakeLLM(scripted={"Create a concise research outline": "quantum computing"})
    orch, store = _orchestrator(search, clock, llm)
    sid = store.create_session(title="t")
    budget = Budget(max_searches=2, max_think_tokens=256, max_output_tokens=800, concurrency=1, timeout_s=50, top_k=6)

    with pytest.raises(BaselineError) as exc:
        orch.run(RunRequest(query="quantum computing", mode=Mode.DEEP, session_id=sid, budget=budget))

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    run_id = store.get_latest_run_id(sid) or ""
    run = store.get_run(run_id)
    assert run["status"] == "error"
    assert run["metadata"]["deadline_summary"]["writer_started_at"] is None
    store.close()


def test_deep_timeout_before_evidence_is_run_timeout_error() -> None:
    clock = ManualClock()
    orch, store = _orchestrator(TimeoutBeforeEvidenceSearch(clock), clock)
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="deep", query="quantum computing")
    budget = Budget(max_searches=1, max_think_tokens=256, max_output_tokens=800, concurrency=1, timeout_s=50, top_k=6)

    with pytest.raises(BaselineError) as exc:
        orch.run(RunRequest(query="quantum computing", mode=Mode.DEEP, session_id=sid, budget=budget), run_id=rid)

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    run = store.get_run(rid)
    assert run["status"] == "error"
    assert run["error"]["error"]["code"] == "RUN_TIMEOUT"
    store.close()


def test_deep_budget_exhausted_completes_without_timeout() -> None:
    clock = ManualClock()
    search = FakeSearch({"https://example.com/quantum": "Quantum computing evidence with qubits."})
    orch, store = _orchestrator(search, clock)
    sid = store.create_session(title="t")
    budget = Budget(max_searches=1, max_think_tokens=256, max_output_tokens=800, concurrency=1, timeout_s=60, top_k=6)

    outcome = orch.run(RunRequest(query="quantum computing", mode=Mode.DEEP, session_id=sid, budget=budget))
    payload = outcome.to_dict()

    assert outcome.status == "completed"
    assert payload["budget_exhausted"] is True
    store.close()


def test_deep_can_commit_an_already_grounded_section_as_partial_timeout() -> None:
    clock = ManualClock()
    search = TimeoutOnSecondDeepSearch(
        {
            "https://example.com/quantum": (
                "First and second quantum sections are directly supported by this grounded evidence."
            )
        },
        clock,
    )
    llm = FakeLLM(
        scripted={"Create a concise research outline": "first quantum\nsecond quantum"}
    )
    orch, store = _orchestrator(search, clock, llm)
    sid = store.create_session(title="t")
    budget = Budget(
        max_searches=2,
        max_think_tokens=256,
        max_output_tokens=800,
        concurrency=1,
        timeout_s=60,
        top_k=6,
    )

    outcome = orch.run(
        RunRequest(query="quantum", mode=Mode.DEEP, session_id=sid, budget=budget)
    )

    assert outcome.status == "partial_timeout"
    assert outcome.answer.text
    assert outcome.answer.citations
    assert outcome.to_dict()["partial_stage"] == "deep_section"
    assert outcome.to_dict()["deadline_summary"]["deadline_met"] is False
    assert store.get_run(outcome.run_id)["status"] == "partial_timeout"
    store.close()


@pytest.mark.parametrize(
    ("mode", "timeout_s"),
    [(Mode.MULTIPLE, 25), (Mode.DEEP, 50)],
)
def test_temporal_assurance_timeout_does_not_start_an_unbudgeted_writer(
    mode: Mode,
    timeout_s: int,
) -> None:
    clock = ManualClock()
    search = TimeoutDuringTemporalAssuranceSearch(
        {"https://example.com/current": "Current product status is supported by dated evidence."},
        clock,
    )
    orch, store = _orchestrator(search, clock)
    sid = store.create_session(title="t")
    budget = Budget(
        max_searches=1,
        max_think_tokens=256,
        max_output_tokens=800,
        concurrency=1,
        timeout_s=timeout_s,
        top_k=6,
    )

    if mode is Mode.DEEP:
        outcome = orch.run(
            RunRequest(query="latest product status", mode=mode, session_id=sid, budget=budget)
        )
        assert outcome.status == "partial_timeout"
        assert outcome.to_dict()["partial_stage"] == "deep_section"
        assert outcome.to_dict()["deadline_summary"]["writer_started_at"] is None
    else:
        with pytest.raises(BaselineError) as exc:
            orch.run(
                RunRequest(query="latest product status", mode=mode, session_id=sid, budget=budget)
            )
        assert exc.value.code is ErrorCode.RUN_TIMEOUT
        run_id = store.get_latest_run_id(sid) or ""
        assert store.get_run(run_id)["metadata"]["deadline_summary"]["writer_started_at"] is None
    store.close()
