"""T25 Wiki Map API and generation contracts."""

from __future__ import annotations

import json
import re
import time

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.knowledge_graph.builder import KnowledgeGraphBuilder
from deepresearch_core.baseline.knowledge_graph import wiki_worker
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM
from tests.baseline.test_knowledge_graph import _seed_graph_fixture


def _wiki_reply(evidence_id: str) -> str:
    return json.dumps(
        {
            "topics": [
                {
                    "title": "Safety evaluation",
                    "summary": "Compare risks, evaluation methods, and mitigation evidence.",
                    "concept_key": "safety-evaluation",
                    "confidence": "high",
                    "evidence_refs": [{"type": "message", "id": evidence_id, "excerpt": "Safety evaluation evidence."}],
                    "source_message_ids": [evidence_id],
                    "children": [
                        {
                            "title": "Risk taxonomy",
                            "summary": "Group operational and model behavior risks.",
                            "concept_key": "risk-taxonomy",
                            "confidence": "high",
                            "evidence_refs": [{"type": "message", "id": evidence_id, "excerpt": "Risk taxonomy evidence."}],
                            "source_message_ids": [evidence_id],
                            "children": [],
                        }
                    ],
                }
            ]
        }
    )


def test_wiki_map_get_is_read_only_and_refresh_creates_active_snapshot() -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    ids = _seed_graph_fixture(service.store)
    assistant_id = service.store.add_message(
        ids["session_id"],
        "assistant",
        "The session concluded that AI safety evaluation needs risk taxonomy and evidence.",
    )
    KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    fake = FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(assistant_id)})
    service.configure_runtime_for_tests(memory_llm=fake)
    client = TestClient(create_app(service))

    empty = client.get("/api/v1/knowledge-graph/wiki-map").json()
    assert empty["layout_mode"] == "wiki_map"
    assert empty["active_snapshot"] is None
    assert fake.calls == []

    created = client.post("/api/v1/knowledge-graph/wiki-map/refresh", json={"workspace_id": "default"}).json()
    job_id = created["id"]
    for _ in range(50):
        job = client.get(f"/api/v1/knowledge-graph/wiki-map/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "partial", "failed"}:
            break
        time.sleep(0.03)
    else:
        raise AssertionError("Wiki Map refresh did not finish")

    assert job["status"] == "succeeded"
    wiki_map = client.get("/api/v1/knowledge-graph/wiki-map").json()
    assert wiki_map["active_snapshot"]["id"] == job["snapshot_id"]
    assert wiki_map["stats"]["topics"] >= 3
    titles = {topic["title"] for topic in wiki_map["topics"]}
    assert "Workspace Wiki" in titles
    assert "Safety evaluation" in titles
    assert all(re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{6,}", json.dumps(topic)) is None for topic in wiki_map["topics"])
    source_topic = next(topic for topic in wiki_map["topics"] if topic["title"] == "Safety evaluation")
    assert assistant_id in source_topic["source_message_ids"]
    service.store.close()


def test_wiki_refresh_all_failed_does_not_overwrite_previous_snapshot() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    ids = _seed_graph_fixture(service.store)
    assistant_id = service.store.add_message(ids["session_id"], "assistant", "Safety answer.")
    KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(assistant_id)}))

    first = service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": "not json"}))

    with pytest.raises(Exception):
        service.refresh_knowledge_graph_wiki_now({"workspace_id": "default"})

    current = service.get_knowledge_graph_wiki_map(workspace_id="default")
    assert current["active_snapshot"]["id"] == first["snapshot_id"]
    service.store.close()


def test_wiki_rq_worker_preserves_provider_not_configured(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    _seed_graph_fixture(service.store)
    graph = KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    job = service._wiki_store.create_refresh_job(workspace_id="default", graph_version_id=graph["version_id"], backend="rq")  # noqa: SLF001
    monkeypatch.setattr(wiki_worker, "_get_service", lambda: service)

    wiki_worker.refresh_wiki_job({"workspace_id": "default"}, job["id"])

    failed = service.get_knowledge_graph_wiki_refresh_job(job["id"])
    assert failed["status"] == "failed"
    assert failed["error_code"] == "PROVIDER_NOT_CONFIGURED"
    assert failed["attempted_sessions"] == 0
    service.store.close()


def test_wiki_settings_roundtrip_and_auto_refresh_hook(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    ids = _seed_graph_fixture(service.store)
    assistant_id = service.store.add_message(ids["session_id"], "assistant", "Safety answer.")
    service.configure_runtime_for_tests(memory_llm=FakeLLM(scripted={"Knowledge Graph Wiki Map": _wiki_reply(assistant_id)}))

    assert service.get_knowledge_graph_config() == {
        "post_run_rebuild_enabled": True,
        "wiki_auto_refresh_after_kg_rebuild": False,
    }
    saved = service.update_knowledge_graph_config(
        {
            "post_run_rebuild_enabled": False,
            "wiki_auto_refresh_after_kg_rebuild": True,
        }
    )
    assert saved["wiki_auto_refresh_after_kg_rebuild"] is True

    calls: list[tuple[dict, str]] = []

    class RecordingBackend:
        name = "thread"

        def enqueue(self, payload, job_id):  # noqa: ANN001, ANN201
            calls.append((dict(payload), str(job_id)))
            return str(job_id)

    service._wiki_job_backend = RecordingBackend()  # noqa: SLF001 - verify auto-refresh enqueue boundary
    service.knowledge_graph_rebuild_now({"workspace_id": "default"})

    assert calls
    assert calls[0][0] == {"workspace_id": "default"}
    reloaded = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    assert reloaded.get_knowledge_graph_config()["wiki_auto_refresh_after_kg_rebuild"] is True
    service.store.close()
    reloaded.store.close()
