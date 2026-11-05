"""DR-ITER-060 T9b: Library retrieval scope and non-citable context."""

from __future__ import annotations

import time

from deepresearch_core.baseline.adapters import Document
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.deadline import ProviderCallBudget, use_provider_call_budget
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.library_retrieval import LibraryRetriever, render_library_context
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store


def _index_library_artifacts(store: Store, *, embedder=None) -> None:
    indexer = Indexer(store, embedder=embedder)
    library = LibraryIndexer(store, indexer=indexer)
    library.index_note(note_id="note_1", content="private AlphaMemo note detail")
    library.index_report(
        report_id="rep_1",
        title="Alpha report",
        body_md="# Alpha report\nprivate AlphaMemo report detail",
        mode="deep",
    )
    library.index_memory(
        memory_id="mem_1",
        namespace="workspace",
        scope_key="default",
        content="private AlphaMemo memory detail",
    )


def _index_web_source(store: Store, *, embedder=None) -> None:
    Indexer(store, embedder=embedder).index_document(
        Document(
            url="https://example.com/source",
            title="Public source",
            text="public AlphaMemo web detail",
            content_type="web",
        )
    )


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):  # noqa: ANN001, ANN201
        values = [str(text) for text in texts]
        self.calls.append(values)
        return [[1.0] for _ in values]


def test_retriever_scope_off_excludes_local_library_artifacts():
    store = Store(":memory:")
    _index_library_artifacts(store)
    _index_web_source(store)

    results = Retriever(store).retrieve("AlphaMemo", top_k=5)

    assert results
    assert all(not item.url.startswith("local://library/") for item in results)
    assert all(item.title != "Library note" for item in results)
    store.close()


def test_library_retrieval_skips_embedding_when_only_external_documents_exist():
    store = Store(":memory:")
    _index_web_source(store)
    embedder = _CountingEmbedder()

    hits = LibraryRetriever(store, embedder=embedder).retrieve(
        "AlphaMemo", scope="all", top_k=5
    )

    assert hits == []
    assert embedder.calls == []
    store.close()


def test_library_evidence_uses_lexical_hit_without_blocking_on_optional_query_embedding():
    store = Store(":memory:")
    embedder = _CountingEmbedder()
    _index_library_artifacts(store, embedder=embedder)
    embedder.calls.clear()
    budget = ProviderCallBudget(
        request_deadline=time.monotonic() + 10.0,
        phase_deadline=time.monotonic() + 8.0,
    )

    with use_provider_call_budget(budget):
        hits = LibraryRetriever(store, embedder=embedder).retrieve_evidence(
            "AlphaMemo",
            top_k=5,
        )

    assert hits
    assert embedder.calls == []
    assert all(item.channels == ["lexical"] for item in hits)
    store.close()


def test_library_retriever_scopes_notes_reports_and_all_excludes_memory(fake_llm):
    store = Store(":memory:")
    _index_library_artifacts(store, embedder=fake_llm)
    _index_web_source(store, embedder=fake_llm)
    retriever = LibraryRetriever(store, embedder=fake_llm)

    note_hits = retriever.retrieve("AlphaMemo", scope="notes", top_k=5)
    report_hits = retriever.retrieve("AlphaMemo", scope="reports", top_k=5)
    all_hits = retriever.retrieve("AlphaMemo", scope="all", top_k=5)

    assert {hit.artifact_type for hit in note_hits} == {"note"}
    assert {hit.artifact_type for hit in report_hits} == {"report"}
    assert {hit.artifact_type for hit in all_hits} == {"note", "report"}
    assert all(not hit.title.startswith("Public") for hit in all_hits)
    assert all(hit.artifact_type != "memory" for hit in all_hits)
    store.close()


def test_library_context_renderer_is_bounded_and_non_citable(fake_llm):
    store = Store(":memory:")
    _index_library_artifacts(store, embedder=fake_llm)
    hits = LibraryRetriever(store, embedder=fake_llm).retrieve("AlphaMemo", scope="all", top_k=5)

    context = render_library_context(hits, max_chars=180)

    assert context.startswith("Library context")
    assert "do not cite as web sources" in context
    assert "[note]" in context or "[report]" in context
    assert len(context) <= 180
    assert "local://library" not in context
    store.close()


def test_orchestrator_keeps_scoped_background_and_also_retrieves_citable_library_evidence(
    fake_llm, fake_search
):
    store = Store(":memory:")
    _index_library_artifacts(store, embedder=fake_llm)
    library_retriever = LibraryRetriever(store, embedder=fake_llm)
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store, embedder=fake_llm),
        indexer=Indexer(store, embedder=fake_llm),
        search=fake_search,
        library_retriever=library_retriever,
        enforce_grounding_gate=False,
    )
    sid = store.create_session(title="t")

    outcome = orch.run(
        RunRequest(query="AlphaMemo quantum", mode=Mode.FAST, session_id=sid, library_scope="notes")
    )

    prompts = [message["content"] for call in fake_llm.calls for message in call if message.get("role") == "user"]
    compose_prompt = next(prompt for prompt in prompts if "Sources:" in prompt)
    assert "Library context" in compose_prompt
    assert "private AlphaMemo note detail" in compose_prompt
    sources_block = compose_prompt.split("Sources:", 1)[1]
    assert "local://library" not in sources_block
    internal = [citation for citation in outcome.answer.citations if citation.origin == "local"]
    assert {citation.artifact_type for citation in internal} == {"library_note", "library_report"}
    assert all(citation.url == "" for citation in internal)
    store.close()


def test_council_receives_library_context_but_never_as_citation(fake_llm, fake_search):
    from tests.baseline.fakes import FakeLLM

    store = Store(":memory:")
    _index_library_artifacts(store, embedder=fake_llm)
    members = [FakeLLM("m-a"), FakeLLM("m-b")]
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store, embedder=fake_llm),
        indexer=Indexer(store, embedder=fake_llm),
        search=fake_search,
        council_members=members,
        library_retriever=LibraryRetriever(store, embedder=fake_llm),
    )
    sid = store.create_session(title="t")

    outcome = orch.run(
        RunRequest(query="AlphaMemo quantum", mode=Mode.COUNCIL, session_id=sid, library_scope="reports")
    )

    member_prompts = [
        message["content"]
        for member in members
        for call in member.calls
        for message in call
        if message.get("role") == "user"
    ]
    assert any("Library context" in prompt for prompt in member_prompts)
    assert any("private AlphaMemo report detail" in prompt for prompt in member_prompts)
    assert all(not citation.url.startswith("local://library/") for citation in outcome.answer.citations)
    store.close()


def test_library_retriever_failure_does_not_fail_run(fake_llm, fake_search):
    class ExplodingLibraryRetriever:
        def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    store = Store(":memory:")
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=fake_search,
        library_retriever=ExplodingLibraryRetriever(),
    )
    sid = store.create_session(title="t")

    outcome = orch.run(RunRequest(query="quantum computing", mode=Mode.FAST, session_id=sid, library_scope="all"))

    assert outcome.status == "completed"
    store.close()
