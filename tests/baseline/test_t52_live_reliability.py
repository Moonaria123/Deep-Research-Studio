"""T52 contracts for reliable OpenAI-compatible output and final writer diagnostics."""

from __future__ import annotations

import json
import time

import pytest

from deepresearch_core.baseline import adapters as adapters_module
from deepresearch_core.baseline.adapters import (
    LLMResult,
    LLMStreamChunk,
    OpenAICompatLLMAdapter,
    classify_llm_terminal,
)
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.citations import build_citations, enforce_grounding, ground_answer
from deepresearch_core.baseline.composer import AnswerComposer
from deepresearch_core.baseline.deadline import (
    ProviderCallBudget,
    current_provider_call_budget,
    use_provider_call_budget,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.evidence_eligibility import output_shape_valid, parse_research_intent
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.llm_call_params import resolve_llm_call_options
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.model_router import ModelRouter
from deepresearch_core.baseline.orchestrator import (
    Orchestrator,
    _fuse_evidence_rankings,
    _safe_writer_diagnostics,
)
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.llm import LLMClient
from scripts.verify_t52_live_reliability import _answer_is_relevant
from tests.baseline.fakes import FakeLLM, FakeSearch


def test_deep_complementary_candidate_and_fetched_excerpts_remain_bindable() -> None:
    provenance_root = "https://www.sqlite.org/fts5.html"
    overview = EvidenceItem(
        chunk_id=-1,
        document_id=provenance_root,
        url=provenance_root,
        title="SQLite FTS5 Extension",
        text=(
            "SQLite FTS5 is a virtual table module for full-text search. "
            "FTS5 tables are created with CREATE VIRTUAL TABLE."
        ),
        provenance_refs=[provenance_root],
    )
    match = EvidenceItem(
        chunk_id=21,
        document_id="doc-fetched-fts5",
        url=provenance_root,
        title="SQLite FTS5 Extension",
        text=(
            "FTS5 query strings are supplied to the MATCH operator. "
            "MATCH supports phrases, prefix terms, NEAR queries, and boolean combinations."
        ),
        provenance_refs=[provenance_root],
    )
    ranking = EvidenceItem(
        chunk_id=42,
        document_id="doc-indexed-fts5",
        url=provenance_root,
        title="SQLite FTS5 Extension",
        text=(
            "ORDER BY rank sorts FTS5 results from most relevant to least relevant. "
            "The bm25 function returns smaller values for better matches."
        ),
        provenance_refs=[provenance_root],
    )

    evidence = _fuse_evidence_rankings(
        [[overview], [match], [ranking]],
        top_k=5,
        preserve_ranking_diversity=True,
        merge_complementary_text=True,
    )
    citations = build_citations(evidence)
    answer = ground_answer(
        " ".join(
            (
                "SQLite FTS5 is a virtual table module for full-text search.[c1]",
                "FTS5 tables are created with CREATE VIRTUAL TABLE.[c1]",
                "FTS5 query strings are supplied to the MATCH operator.[c1]",
                "MATCH supports phrases, prefix terms, NEAR queries, and boolean combinations.[c1]",
                "ORDER BY rank sorts FTS5 results from most relevant to least relevant.[c1]",
                "The bm25 function returns smaller values for better matches.[c1]",
            )
        ),
        evidence,
        citations=citations,
    )

    assert len(evidence) == 1
    assert "complementary_excerpt" in evidence[0].channels
    assert "MATCH supports phrases" in citations[0].snippet
    assert "smaller values for better matches" in citations[0].snippet
    assert enforce_grounding(answer, min_coverage=0.85).coverage == 1.0


def test_duplicate_same_source_excerpt_keeps_normal_citation_window() -> None:
    provenance_root = "https://www.sqlite.org/fts5.html"
    text = "SQLite FTS5 full-text search evidence. " * 20
    first = EvidenceItem(
        chunk_id=-1,
        document_id=provenance_root,
        url=provenance_root,
        title="SQLite FTS5 Extension",
        text=text,
        provenance_refs=[provenance_root],
    )
    duplicate = EvidenceItem(
        chunk_id=3,
        document_id="doc-fetched-fts5",
        url=provenance_root,
        title="SQLite FTS5 Extension",
        text=text,
        provenance_refs=[provenance_root],
    )

    evidence = _fuse_evidence_rankings(
        [[first], [duplicate]],
        top_k=5,
        preserve_ranking_diversity=True,
        merge_complementary_text=True,
    )
    citations = build_citations(evidence)

    assert len(evidence) == 1
    assert "complementary_excerpt" not in evidence[0].channels
    assert len(citations[0].snippet) == 240


def test_exact_sentence_shape_ignores_question_marks_inside_inline_code() -> None:
    intent = parse_research_intent(
        "Explain SQLite FTS5 in exactly 6 concise sentences with citations."
    )
    answer = " ".join(
        (
            "FTS5 tokenizes each query into terms [c1].",
            "Its index records token locations [c1].",
            "Columns may be declared UNINDEXED [c1].",
            "A query may use `WHERE fts MATCH ? ORDER BY bm25(fts)` [c1].",
            "MATCH supports phrase and prefix syntax [c1].",
            "The bm25 function ranks matching rows [c1].",
        )
    )

    assert output_shape_valid(intent, answer)


def test_live_relevance_accepts_bm25_as_the_requested_ranking_capability() -> None:
    answer = (
        "SQLite FTS5 uses MATCH for full-text queries [c1]. "
        "ORDER BY bm25(fts) returns results using the documented score [c1]."
    )

    assert _answer_is_relevant(answer, ("fts5", "match", ("rank", "bm25")))


def test_poe_chat_uses_custom_parameter_container_and_prompt_validation() -> None:
    options = resolve_llm_call_options(
        provider_type="openai",
        provider_id="poe",
        base_url="https://api.poe.com/v1",
        thinking_budget=4096,
        max_visible_tokens=1600,
        response_format={"type": "json_schema", "json_schema": {"name": "answer"}},
    )

    assert options.provider_family == "poe"
    assert options.endpoint_type == "chat_completions"
    assert options.structured_output_strategy == "prompt_validated"
    assert options.request_fields == {
        "max_tokens": 5696,
        "extra_body": {"reasoning_effort": "medium"},
    }
    assert "reasoning_effort" not in options.request_fields
    assert "response_format" not in options.request_fields
    assert options.to_dict()["applied"]["reasoning_effort"] == "extra_body"
    assert options.to_dict()["ignored"]["response_format"] == "unsupported_by_poe_chat"
    assert options.to_dict()["shared_output_cap"] is True
    assert options.to_dict()["visible_output_target"] == 1600
    assert options.to_dict()["reasoning_reserve_tokens"] == 4096
    assert options.to_dict()["provider_output_cap"] == 5696


def test_poe_shared_cap_reserves_reasoning_without_consuming_visible_target() -> None:
    options = resolve_llm_call_options(
        provider_type="openai",
        provider_id="poe",
        base_url="https://api.poe.com/v1",
        model="gpt-5.4",
        thinking_budget=4096,
        max_visible_tokens=600,
        response_format=None,
    )

    assert options.request_fields == {
        "max_tokens": 4696,
        "extra_body": {"reasoning_effort": "medium"},
    }
    assert options.to_dict()["visible_output_target"] == 600
    assert options.to_dict()["reasoning_reserve_tokens"] == 4096
    assert options.to_dict()["provider_output_cap"] == 4696
    assert "shared_output_cap_includes_reasoning_reserve" in options.reason_codes


def test_openai_chat_keeps_native_reasoning_and_structured_output() -> None:
    response_format = {"type": "json_object"}
    options = resolve_llm_call_options(
        provider_type="openai",
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5",
        thinking_budget=1024,
        max_visible_tokens=900,
        response_format=response_format,
    )

    assert options.provider_family == "openai"
    assert options.structured_output_strategy == "native"
    assert options.request_fields == {
        "max_completion_tokens": 1924,
        "reasoning_effort": "low",
        "response_format": response_format,
    }
    assert options.to_dict()["reasoning_reserve_tokens"] == 1024
    assert options.to_dict()["provider_output_cap"] == 1924
    assert options.to_dict()["shared_output_cap"] is True


def test_openai_does_not_apply_reasoning_to_an_unconfirmed_model() -> None:
    options = resolve_llm_call_options(
        provider_type="openai",
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        thinking_budget=4096,
        max_visible_tokens=700,
        response_format=None,
    )

    assert options.request_fields == {"max_tokens": 700}
    assert options.ignored["reasoning_effort"] == "unsupported_or_unconfirmed"
    assert options.to_dict()["shared_output_cap"] is False


def test_sync_chat_preserves_finish_usage_and_applies_resolved_payload() -> None:
    captured: list[dict] = []

    def transport(request):  # noqa: ANN001, ANN202
        captured.append(json.loads((request.data or b"{}").decode("utf-8")))
        return {
            "model": "Claude-Sonnet",
            "choices": [{"message": {"content": "Grounded. [c1]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }

    client = LLMClient(
        base_url="https://api.poe.com/v1",
        api_key="test-only",
        model="Claude-Sonnet",
        transport=transport,
    )
    result = client.chat_result(
        [{"role": "user", "content": "answer"}],
        request_fields={
            "max_tokens": 1200,
            "extra_body": {"reasoning_effort": "high"},
        },
    )

    assert result.text == "Grounded. [c1]"
    assert result.model == "Claude-Sonnet"
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert captured[0]["extra_body"] == {"reasoning_effort": "high"}
    assert "reasoning_effort" not in captured[0]


def test_poe_adapter_captured_request_and_diagnostics_match_actual_payload() -> None:
    captured: list[dict] = []

    def transport(request):  # noqa: ANN001, ANN202
        captured.append(json.loads((request.data or b"{}").decode("utf-8")))
        return {
            "model": "Claude-Sonnet",
            "choices": [{"message": {"content": "Grounded. [c1]"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }

    adapter = OpenAICompatLLMAdapter(
        LLMClient(
            base_url="https://api.poe.com/v1",
            api_key="test-only",
            model="Claude-Sonnet",
            transport=transport,
        ),
        provider_type="openai",
        provider_id="poe",
    )
    result = adapter.complete(
        [{"role": "user", "content": "answer"}],
        thinking_budget=4096,
        response_format={"type": "json_object"},
    )

    assert captured[0]["extra_body"] == {"reasoning_effort": "medium"}
    assert captured[0]["max_tokens"] == 5296
    assert "reasoning_effort" not in captured[0]
    assert "response_format" not in captured[0]
    assert result.finish_reason == "stop"
    assert result.usage == {"completion_tokens": 3}
    assert result.diagnostics["call_options"]["provider_family"] == "poe"
    assert result.diagnostics["call_options"]["ignored"] == {
        "response_format": "unsupported_by_poe_chat"
    }


def test_sync_adapter_classifies_a_missing_choice_as_malformed_terminal() -> None:
    adapter = OpenAICompatLLMAdapter(
        LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="test-only",
            model="gpt-5",
            transport=lambda _request: {"model": "gpt-5"},
        ),
        provider_id="openai",
    )

    result = adapter.complete([{"role": "user", "content": "answer"}])

    assert result.text == ""
    assert result.diagnostics["classification"] == "malformed_terminal"


def test_poe_stream_preserves_finish_usage_and_custom_parameter_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    b'data: {"model":"Claude-Sonnet","choices":[{"delta":{"content":"Grounded [c1]"},"finish_reason":null}]}\n',
                    b'data: {"model":"Claude-Sonnet","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":3}}\n',
                    b"data: [DONE]\n",
                ]
            )

    def urlopen(request, *, timeout):  # noqa: ANN001, ANN202, ARG001
        captured.append(json.loads((request.data or b"{}").decode("utf-8")))
        return _Response()

    monkeypatch.setattr(adapters_module.urllib.request, "urlopen", urlopen)
    adapter = OpenAICompatLLMAdapter(
        LLMClient(
            base_url="https://api.poe.com/v1",
            api_key="test-only",
            model="Claude-Sonnet",
        ),
        provider_id="poe",
    )
    chunks = list(
        adapter.stream(
            [{"role": "user", "content": "answer"}],
            thinking_budget=8192,
            response_format={"type": "json_object"},
        )
    )

    assert captured[0]["extra_body"] == {"reasoning_effort": "high"}
    assert captured[0]["max_tokens"] == 9392
    assert "reasoning_effort" not in captured[0]
    assert "response_format" not in captured[0]
    assert "".join(chunk.delta for chunk in chunks) == "Grounded [c1]"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == {"completion_tokens": 3}
    assert chunks[-1].diagnostics["classification"] == "completed"


@pytest.mark.parametrize(
    ("finish_reason", "usage", "expected"),
    [
        (None, {"completion_tokens": 100}, "usage_only"),
        ("length", {"completion_tokens": 100}, "length_without_visible_output"),
        ("content_filter", {}, "content_filtered"),
        ("stop", {}, "provider_stop_empty"),
        (None, {}, "empty_stream"),
    ],
)
def test_terminal_classification_is_stable(
    finish_reason: str | None,
    usage: dict,
    expected: str,
) -> None:
    assert classify_llm_terminal(
        text="",
        finish_reason=finish_reason,
        usage=usage,
    ) == expected


class _EmptyStreamBackend:
    name = "primary-poe"

    def __init__(self) -> None:
        self.complete_calls = 0
        self.budgets: list[tuple[str, int, int]] = []

    def stream(self, _messages, **_kwargs):  # noqa: ANN001, ANN202
        budget = current_provider_call_budget()
        if budget is not None:
            self.budgets.append(("stream", budget.attempt, budget.max_attempts))
        yield LLMStreamChunk(
            done=True,
            model="poe-model",
            provider=self.name,
            finish_reason="length",
            usage={"completion_tokens": 256},
            diagnostics={"classification": "length_without_visible_output"},
        )

    def complete(self, _messages, **_kwargs):  # noqa: ANN001, ANN202
        self.complete_calls += 1
        budget = current_provider_call_budget()
        if budget is not None:
            self.budgets.append(("sync", budget.attempt, budget.max_attempts))
        return LLMResult(
            text="Grounded answer. [c1]",
            model="poe-model",
            finish_reason="stop",
            usage={"completion_tokens": 5},
            diagnostics={"classification": "completed"},
        )

    def embed(self, _texts, **_kwargs):  # noqa: ANN001, ANN202
        return []


def test_router_preserves_pre_delta_terminal_and_owns_single_sync_fallback() -> None:
    backend = _EmptyStreamBackend()
    router = ModelRouter([backend])
    observed = list(router.stream([{"role": "user", "content": "answer"}]))

    assert backend.complete_calls == 1
    assert observed[0].finish_reason == "length"
    assert observed[0].usage == {"completion_tokens": 256}
    assert observed[0].diagnostics["classification"] == "length_without_visible_output"
    assert "Grounded answer. [c1]" == "".join(chunk.delta for chunk in observed)
    assert observed[-1].diagnostics["fallback_scope"] == "same_provider_sync"


class _MidStreamFailureBackend(_EmptyStreamBackend):
    def stream(self, _messages, **_kwargs):  # noqa: ANN001, ANN202
        yield LLMStreamChunk(delta="unverified draft", provider=self.name, model="poe-model")
        raise RuntimeError("provider disconnected")


def test_mid_stream_failure_is_diagnostic_error_and_never_falls_back() -> None:
    backend = _MidStreamFailureBackend()
    composer = AnswerComposer(ModelRouter([backend]))

    with pytest.raises(BaselineError) as exc:
        composer.compose_stream("answer", [], on_delta=lambda _delta: None)

    assert exc.value.code is ErrorCode.UPSTREAM_ERROR
    diagnostics = exc.value.details["writer_diagnostics"]
    assert diagnostics["classification"] == "provider_error"
    assert diagnostics["visible_chars"] == len("unverified draft")
    assert diagnostics["attempts"][0]["status"] == "failed"
    assert backend.complete_calls == 0
    assert composer.last_stream_meta.classification == "provider_error"


class _AlwaysEmptyBackend(_EmptyStreamBackend):
    def complete(self, _messages, **_kwargs):  # noqa: ANN001, ANN202
        self.complete_calls += 1
        budget = current_provider_call_budget()
        if budget is not None:
            self.budgets.append(("sync", budget.attempt, budget.max_attempts))
        return LLMResult(
            text="",
            model="poe-model",
            finish_reason="stop",
            usage={},
            diagnostics={"classification": "provider_stop_empty"},
        )


def test_composer_does_not_repeat_router_owned_fallback() -> None:
    backend = _AlwaysEmptyBackend()
    composer = AnswerComposer(ModelRouter([backend]))

    with pytest.raises(Exception):
        composer.compose_stream("answer", [], on_delta=lambda _delta: None)

    assert backend.complete_calls == 1
    assert composer.last_stream_meta.classification == "provider_stop_empty"
    assert composer.last_stream_meta.attempts


class _HealthySecondaryBackend(_EmptyStreamBackend):
    name = "secondary-backend"

    def stream(self, _messages, **_kwargs):  # noqa: ANN001, ANN202
        budget = current_provider_call_budget()
        if budget is not None:
            self.budgets.append(("stream", budget.attempt, budget.max_attempts))
        yield LLMStreamChunk(
            delta="Secondary grounded answer. [c1]",
            model="secondary-model",
            provider=self.name,
        )
        yield LLMStreamChunk(
            done=True,
            model="secondary-model",
            provider=self.name,
            finish_reason="stop",
            usage={"completion_tokens": 6},
            diagnostics={"classification": "completed"},
        )


def test_cross_backend_success_is_explicit_and_cannot_masquerade_as_primary_success() -> None:
    primary = _AlwaysEmptyBackend()
    secondary = _HealthySecondaryBackend()
    with use_provider_call_budget(ProviderCallBudget(max_attempts=3)):
        chunks = list(
            ModelRouter([primary, secondary]).stream([{"role": "user", "content": "answer"}])
        )

    assert "Secondary grounded answer. [c1]" == "".join(chunk.delta for chunk in chunks)
    terminal = chunks[-1]
    assert terminal.provider == "secondary-backend"
    assert terminal.diagnostics["fallback_scope"] == "cross_backend"
    attempts = terminal.diagnostics["attempts"]
    assert [(item["provider"], item["status"]) for item in attempts] == [
        ("primary-poe", "empty"),
        ("primary-poe", "empty"),
        ("secondary-backend", "completed"),
    ]
    assert primary.budgets == [("stream", 0, 1), ("sync", 1, 2)]
    assert secondary.budgets == [("stream", 2, 3)]


@pytest.mark.parametrize("mode", [Mode.MULTIPLE, Mode.DEEP])
def test_multiple_and_deep_finalize_only_citations_bound_to_adopted_evidence(mode: Mode) -> None:
    store = Store(":memory:")
    session_id = store.create_session(title=f"t52-{mode.value}")
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch(
            {
                "https://example.com/a": (
                    "Quantum computing uses qubits for information processing, and quantum error "
                    "correction protects encoded information."
                ),
                "https://example.com/b": (
                    "Quantum error correction detects errors without directly measuring the logical "
                    "quantum state."
                ),
            }
        ),
        memory=MemoryEngine(store),
        enforce_grounding_gate=True,
    )

    payload = orchestrator.run(
        RunRequest(query="quantum error correction", mode=mode, session_id=session_id)
    ).to_dict()

    assert payload["status"] == "completed"
    assert payload["answer"]["coverage"] == 1.0
    assert payload["answer"]["citations"]
    adopted_document_ids = {item["document_id"] for item in payload["sources"]}
    cited_document_ids = {item["document_id"] for item in payload["answer"]["citations"]}
    assert cited_document_ids <= adopted_document_ids
    assert all(item["grounding_eligible"] for item in payload["answer"]["citations"])
    assert payload["writer_diagnostics"]["visible_chars"] > 0
    store.close()


def test_error_run_persists_secret_free_writer_diagnostics() -> None:
    backend = _AlwaysEmptyBackend()
    store = Store(":memory:")
    session_id = store.create_session(title="t52")
    run_id = store.create_run(session_id=session_id, mode="fast", query="Alpha")
    orchestrator = Orchestrator(
        store=store,
        llm=ModelRouter([backend]),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(
            RunRequest(
                query="Alpha",
                mode=Mode.FAST,
                session_id=session_id,
                requested_at=time.time(),
            ),
            run_id=run_id,
        )

    assert exc.value.code is ErrorCode.UPSTREAM_ERROR
    persisted = store.get_run(run_id)
    assert persisted is not None
    diagnostics = persisted["metadata"]["writer_diagnostics"]
    assert diagnostics["classification"] == "provider_stop_empty"
    assert diagnostics["visible_chars"] == 0
    assert diagnostics["fallback_scope"] == "same_provider_sync"
    serialized = json.dumps(diagnostics)
    assert "test-only" not in serialized
    assert "Sources:" not in serialized
    store.close()


def test_writer_diagnostics_redact_labels_and_drop_string_usage_payloads() -> None:
    diagnostics = _safe_writer_diagnostics(
        {
            "classification": "provider_error",
            "provider": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456",
            "usage": {"completion_tokens": 3, "raw": "prompt or reasoning text"},
            "call_options": {
                "visible_output_target": 600,
                "reasoning_reserve_tokens": 4096,
                "provider_output_cap": 4696,
            },
            "attempts": [
                {
                    "attempt": 0,
                    "provider": "token=secret-value",
                    "status": "failed",
                    "usage": {"total_tokens": 4, "raw": "response body"},
                }
            ],
        }
    )

    serialized = json.dumps(diagnostics)
    assert "abcdefghijklmnopqrstuvwxyz123456" not in serialized
    assert "secret-value" not in serialized
    assert "prompt or reasoning text" not in serialized
    assert "response body" not in serialized
    assert diagnostics["usage"] == {"completion_tokens": 3}
    assert diagnostics["call_options"]["reasoning_reserve_tokens"] == 4096
