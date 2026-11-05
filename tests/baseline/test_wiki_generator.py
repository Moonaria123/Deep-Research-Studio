"""Wiki Map v2 generator contracts."""

from __future__ import annotations

import json

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.knowledge_graph.builder import KnowledgeGraphBuilder
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM
from tests.baseline.test_knowledge_graph import _seed_graph_fixture


def _wiki_reply(
    evidence_id: str,
    title: str = "Risk taxonomy",
    *,
    concept_key: str = "risk-taxonomy",
    confidence: str = "high",
) -> str:
    return json.dumps(
        {
            "topics": [
                {
                    "title": title,
                    "summary": "Groups model and operational risks into a reviewable structure.",
                    "concept_key": concept_key,
                    "confidence": confidence,
                    "evidence_refs": [
                        {
                            "type": "message",
                            "id": evidence_id,
                            "excerpt": "The answer grouped model and operational risks.",
                        }
                    ],
                    "source_message_ids": [evidence_id],
                    "children": [],
                }
            ]
        }
    )


def _service_with_active_graph() -> tuple[BaselineService, dict[str, str], str]:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    ids = _seed_graph_fixture(service.store)
    message_id = service.store.add_message(
        ids["session_id"],
        "assistant",
        "The answer grouped model and operational risks into a taxonomy.",
    )
    KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    return service, ids, message_id


def _service_with_message_window() -> tuple[BaselineService, dict[str, str], str, str, list[str]]:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    ids = _seed_graph_fixture(service.store)
    user_id = service.store.add_message(ids["session_id"], "user", "Please compare safety evaluation approaches.")
    assistant_ids = [
        service.store.add_message(
            ids["session_id"],
            "assistant",
            f"Assistant answer {index} grouped safety evaluation evidence.",
        )
        for index in range(1, 7)
    ]
    KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    return service, ids, user_id, assistant_ids[0], assistant_ids


class FailingLLM:
    name = "failing-llm"

    def __init__(self, code: ErrorCode = ErrorCode.UPSTREAM_ERROR) -> None:
        self.code = code
        self.calls: list[list[dict]] = []

    def complete(self, messages, **_kwargs):
        self.calls.append(list(messages))
        raise BaselineError(
            self.code,
            "provider call failed",
            details={"request_id": "req-hidden", "api_key": "redacted-token"},
        )

    def embed(self, texts, *, model=None) -> list[list[float]]:
        return [[0.0] for _ in texts]


class SelectiveFailingLLM(FakeLLM):
    """Fail only the Wiki prompt containing one Session evidence id."""

    def __init__(self, *, fail_on: str, scripted: dict[str, str]) -> None:
        super().__init__(scripted=scripted)
        self.fail_on = fail_on

    def complete(self, messages, **kwargs):
        user = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
        if self.fail_on in user:
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, "provider call failed")
        return super().complete(messages, **kwargs)


def _service_with_two_active_sessions() -> tuple[BaselineService, dict[str, str], dict[str, str], str, str]:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    first = _seed_graph_fixture(service.store)
    second = _seed_graph_fixture(service.store)
    first_message = service.store.add_message(
        first["session_id"],
        "assistant",
        "The first answer grouped model and operational risks into a taxonomy.",
    )
    second_message = service.store.add_message(
        second["session_id"],
        "assistant",
        "The second answer grouped governance and deployment risks into a taxonomy.",
    )
    KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    return service, first, second, first_message, second_message


def test_wiki_generator_persists_v2_attrs_and_keeps_topic_key_unique() -> None:
    service, ids, message_id = _service_with_active_graph()
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(message_id)}))

    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    wiki_map = service.get_knowledge_graph_wiki_map(workspace_id="default")

    assert result["stats"]["source"] == "wiki_refresh_v2"
    assert result["stats"]["topic_count"] == 1
    assert result["stats"]["top_level_topic_count"] == 1
    assert result["stats"]["max_depth"] == 1
    assert result["stats"]["topics_with_evidence"] == 1
    assert result["stats"]["evidence_coverage"] == 1.0
    topic = next(item for item in wiki_map["topics"] if item["attrs"].get("kind") == "topic")
    session_root = next(item for item in wiki_map["topics"] if item["attrs"].get("kind") == "session_root")
    assert topic["key"].startswith(f"wiki:topic:{ids['session_id']}:")
    assert topic["attrs"]["concept_key"] == "risk-taxonomy"
    assert topic["attrs"]["schema_version"] == "2"
    assert topic["attrs"]["confidence"] == "high"
    assert topic["attrs"]["evidence_refs"][0]["id"] == message_id
    assert topic["source_message_ids"] == [message_id]
    assert session_root["summary"]
    assert session_root["attrs"]["generation_state"] == "fresh"
    service.store.close()


def test_wiki_generator_carries_failed_session_without_orphan_or_coverage_loss() -> None:
    service, first, second, first_message, second_message = _service_with_two_active_sessions()
    service.configure_runtime_for_tests(
        memory_llm=FakeLLM(
            scripted={
                first_message: _wiki_reply(first_message, title="Initial model risks", concept_key="model-risks"),
                second_message: _wiki_reply(second_message, title="Initial governance risks", concept_key="governance-risks"),
            }
        )
    )
    initial = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    service.configure_runtime_for_tests(
        memory_llm=SelectiveFailingLLM(
            fail_on=second_message,
            scripted={
                first_message: _wiki_reply(first_message, title="Updated model risks", concept_key="model-risks"),
            },
        )
    )
    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    wiki_map = service.get_knowledge_graph_wiki_map(workspace_id="default")

    assert result["status"] == "partial"
    assert result["stats"]["fresh_sessions"] == 1
    assert result["stats"]["carried_sessions"] == 1
    assert result["stats"]["uncovered_sessions"] == 0
    assert result["stats"]["candidate_coverage"] == 1.0
    assert wiki_map["active_snapshot"]["id"] != initial["snapshot_id"]

    first_topics = [topic for topic in wiki_map["topics"] if topic["attrs"].get("session_id") == first["session_id"]]
    second_topics = [topic for topic in wiki_map["topics"] if topic["attrs"].get("session_id") == second["session_id"]]
    assert any(topic["attrs"].get("kind") == "topic" for topic in first_topics)
    assert any(topic["attrs"].get("kind") == "topic" for topic in second_topics)
    assert all(topic["attrs"]["generation_state"] == "fresh" for topic in first_topics)
    assert all(topic["attrs"]["generation_state"] == "carried" for topic in second_topics)
    assert all(topic["attrs"]["source_snapshot_id"] == initial["snapshot_id"] for topic in second_topics)
    service.store.close()


def test_wiki_generator_prompt_evidence_matches_visible_assistant_window() -> None:
    service, _, user_id, oldest_assistant_id, assistant_ids = _service_with_message_window()
    fake = FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(oldest_assistant_id)})
    service.configure_runtime_for_tests(memory_llm=fake)

    service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    prompt = fake.calls[0][0]["content"]

    assert oldest_assistant_id in prompt
    assert f"- {oldest_assistant_id} [message]:" in prompt
    assert f"- {user_id} [user]:" not in prompt
    assert "user-message [user]:" in prompt
    assert all(message_id in prompt for message_id in assistant_ids)
    service.store.close()


def test_wiki_generator_drops_illegal_evidence_ref_when_topic_still_grounded() -> None:
    service, _, message_id = _service_with_active_graph()
    reply = json.loads(_wiki_reply(message_id))
    reply["topics"][0]["evidence_refs"].append(
        {"type": "message", "id": "unknown-message-id", "excerpt": "This id is not in the catalog."}
    )
    reply["topics"][0]["source_message_ids"] = [message_id, "unknown-message-id"]
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": json.dumps(reply)}))

    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    wiki_map = service.get_knowledge_graph_wiki_map(workspace_id="default")

    topic = next(item for item in wiki_map["topics"] if item["attrs"].get("kind") == "topic")
    assert result["stats"]["evidence_ref_dropped"] == 1
    assert [ref["id"] for ref in topic["attrs"]["evidence_refs"]] == [message_id]
    assert topic["source_message_ids"] == [message_id]
    service.store.close()


def test_wiki_generator_normalizes_id_only_evidence_refs_before_validation() -> None:
    service, _, message_id = _service_with_active_graph()
    reply = {
        "title": "extra root title should be ignored",
        "topics": [
            {
                "title": "Risk taxonomy",
                "summary": "Groups model and operational risks into a reviewable structure.",
                "evidence_refs": [{"id": message_id}],
                "confidence": "not-a-confidence",
            }
        ],
    }
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": json.dumps(reply)}))

    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    wiki_map = service.get_knowledge_graph_wiki_map(workspace_id="default")

    topic = next(item for item in wiki_map["topics"] if item["attrs"].get("kind") == "topic")
    assert result["status"] == "succeeded"
    assert topic["title"] == "Risk taxonomy"
    assert topic["summary"] == "Groups model and operational risks into a reviewable structure."
    assert topic["attrs"]["confidence"] == "medium"
    assert topic["attrs"]["concept_key"] == "risk-taxonomy"
    assert topic["attrs"]["evidence_refs"] == [
        {
            "type": "message",
            "id": message_id,
            "excerpt": "The answer grouped model and operational risks into a taxonomy.",
        }
    ]
    assert topic["source_message_ids"] == [message_id]
    service.store.close()


def test_wiki_generator_llm_call_failure_has_secret_free_attempt_diagnostics() -> None:
    service, _, _message_id = _service_with_active_graph()
    service.configure_runtime_for_tests(memory_llm=FailingLLM())

    with pytest.raises(Exception) as exc:
        service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    diagnostics = getattr(exc.value, "diagnostics", {})
    error = diagnostics["errors"][0]
    serialized = json.dumps(error)
    assert error["error_code"] == "WIKI_LLM_FAILED"
    assert error["attempts"] >= 1
    assert error["validation_error"] == "UPSTREAM_ERROR"
    assert error["structured_strategy"] == "llm_call"
    assert error["failure_class"] == "llm_call"
    assert "req-hidden" not in serialized
    assert "redacted-token" not in serialized
    service.store.close()


def test_wiki_generator_repair_prompt_includes_catalog_and_error_code() -> None:
    service, _, message_id = _service_with_active_graph()
    invalid_reply = json.dumps({"topics": [{"title": "Ungrounded topic", "evidence_refs": [{"id": "unknown-message"}]}]})
    valid_reply = _wiki_reply(message_id, title="Recovered risk taxonomy")
    fake = FakeLLM(
        scripted={
            "Knowledge Graph Wiki Map": invalid_reply,
            "Repair the Wiki Map JSON": valid_reply,
        }
    )
    service.configure_runtime_for_tests(memory_llm=fake)

    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    assert result["status"] == "succeeded"
    assert len(fake.calls) == 2
    repair_prompt = fake.calls[1][0]["content"]
    assert "Error class: EMPTY_CONTENT" in repair_prompt
    assert message_id in repair_prompt
    assert "type, id, excerpt" in repair_prompt
    service.store.close()


def test_wiki_generator_repairs_topic_title_that_repeats_session_title() -> None:
    service, _, message_id = _service_with_active_graph()
    fake = FakeLLM(
        scripted={
            "Knowledge Graph Wiki Map": _wiki_reply(message_id, title="AI safety"),
            "Repair the Wiki Map JSON": _wiki_reply(message_id, title="Safety risk taxonomy"),
        }
    )
    service.configure_runtime_for_tests(memory_llm=fake)

    result = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    assert result["status"] == "succeeded"
    assert len(fake.calls) == 2
    repair_prompt = fake.calls[1][0]["content"]
    assert "Error class: LOW_QUALITY_TOPIC_TITLE" in repair_prompt
    assert "AI safety" in repair_prompt
    service.store.close()


def test_wiki_generator_failure_diagnostics_include_repair_drop_stats() -> None:
    service, _, _message_id = _service_with_active_graph()
    invalid_reply = json.dumps({"topics": [{"title": "Ungrounded topic", "evidence_refs": [{"id": "unknown-message"}]}]})
    service.configure_runtime_for_tests(
        memory_llm=FakeLLM(
            scripted={
                "Knowledge Graph Wiki Map": invalid_reply,
                "Repair the Wiki Map JSON": invalid_reply,
            }
        )
    )

    with pytest.raises(Exception) as exc:
        service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    diagnostics = getattr(exc.value, "diagnostics", {})
    error = diagnostics["errors"][0]
    assert error["error_code"] == "EMPTY_CONTENT"
    assert error["evidence_ref_dropped"] >= 1
    assert error["topic_dropped"] >= 1
    assert "raw" not in json.dumps(error).lower()
    service.store.close()


def test_wiki_generator_rejects_unlisted_evidence_ref_without_overwriting_active() -> None:
    service, _, message_id = _service_with_active_graph()
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(message_id)}))
    first = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    bad_reply = json.loads(_wiki_reply(message_id))
    bad_reply["topics"][0]["evidence_refs"][0]["id"] = "fake_evidence"
    bad_reply["topics"][0]["source_message_ids"] = ["fake_evidence"]
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": json.dumps(bad_reply)}))

    try:
        service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    except Exception:
        pass

    current = service.get_knowledge_graph_wiki_map(workspace_id="default")
    assert current["active_snapshot"]["id"] == first["snapshot_id"]
    service.store.close()
