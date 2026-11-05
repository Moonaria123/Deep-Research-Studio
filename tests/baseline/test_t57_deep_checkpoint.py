"""T57 Deep checkpoint persistence, state-machine, and codec contracts."""

from __future__ import annotations

import json
import sqlite3

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.budget import Budget, Mode, RunRequest
from deepresearch_core.baseline.deep_checkpoint import (
    CheckpointAlreadyContinued,
    CheckpointCodecError,
    DeepCheckpointV1,
    decode_deep_checkpoint,
    encode_deep_checkpoint,
)
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import DeepRunPaused, Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM, FakeSearch


def _checkpoint_payload(*, run_id: str = "run_parent", stage: str = "round_complete") -> dict:
    return {
        "schema_version": 1,
        "stage": stage,
        "lineage_id": "lineage_1",
        "parent_run_id": run_id,
        "request": {
            "query": "Compare two public standards",
            "mode": "deep",
            "session_id": "sess_1",
            "requested_at": 1_700_000_000.0,
            "deadline_at": 1_700_000_180.0,
        },
        "research_intent": {"research_focus": "public standards"},
        "plan": {"sections": [{"id": "s1", "title": "Scope"}]},
        "pending_tasks": [{"id": "s2", "title": "Findings"}],
        "completed_sections": [{"id": "s1", "title": "Scope", "body": "Grounded draft."}],
        "evidence_snapshot": [
            {
                "id": "ev_1",
                "origin": "local",
                "artifact_type": "library_document",
                "artifact_id": "doc_1",
                "text": "A bounded evidence excerpt.",
            }
        ],
        "retrieval_state": {
            "searches_used": 1,
            "pages_fetched": 0,
            "gap_rounds": 0,
        },
        "temporal_coverage": {"intent": "none", "status": "not_requested"},
        "conflicts": [],
        "evidence_eligibility": [],
        "grounding": {"coverage": 1.0},
        "remediation": {"attempted": False, "attempt_index": 0},
        "phase_audit": [{"stage": "primary:local", "status": "completed"}],
        "budget_usage": {
            "searches_used": 1,
            "fetches_used": 0,
            "provider_attempts": 2,
            "recovery_searches_used": 0,
        },
    }


def _running_deep_run(store: Store) -> tuple[str, str]:
    session_id = store.create_session(title="Checkpoint test")
    run_id = store.create_run(
        session_id=session_id,
        mode="deep",
        query="Research this",
        budget={"timeout_s": 180},
        metadata={"enqueue_confirmed": True},
    )
    assert store.claim_queued_run(run_id) is True
    return session_id, run_id


def _deep_orchestrator(store: Store) -> Orchestrator:
    search = FakeSearch(
        {
            "https://example.com/checkpoint": (
                "Checkpoint-safe Deep Research resumes completed planning and "
                "uses grounded evidence for the remaining report sections."
            )
        }
    )
    return Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        memory=MemoryEngine(store),
        enforce_grounding_gate=False,
    )


def test_checkpoint_codec_round_trips_canonical_payload():
    value = DeepCheckpointV1.from_dict(_checkpoint_payload())

    encoded = encode_deep_checkpoint(value)
    decoded = decode_deep_checkpoint(encoded.payload_json, encoded.payload_sha256)

    assert decoded == value
    assert encoded.size_bytes <= 4 * 1024 * 1024
    assert encoded.payload_json == encode_deep_checkpoint(decoded).payload_json
    assert encoded.payload_sha256 == encode_deep_checkpoint(decoded).payload_sha256


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"unexpected": "field"}),
        lambda payload: payload["request"].update({"api_key": "synthetic-secret"}),
        lambda payload: payload["phase_audit"].append({"prompt": "hidden prompt"}),
        lambda payload: payload.update({"stage": "mid_provider"}),
    ],
)
def test_checkpoint_codec_rejects_unknown_secret_or_unsafe_payload(mutator):
    payload = _checkpoint_payload()
    mutator(payload)

    with pytest.raises(CheckpointCodecError):
        DeepCheckpointV1.from_dict(payload)


def test_checkpoint_codec_hash_mismatch_fails_closed():
    encoded = encode_deep_checkpoint(DeepCheckpointV1.from_dict(_checkpoint_payload()))

    with pytest.raises(CheckpointCodecError):
        decode_deep_checkpoint(encoded.payload_json, "0" * 64)


def test_checkpoint_table_migration_is_idempotent_and_old_rows_survive(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, title TEXT, workspace TEXT DEFAULT 'default',
            created_at REAL, updated_at REAL, metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, session_id TEXT, mode TEXT, query TEXT, status TEXT,
            budget TEXT DEFAULT '{}', result TEXT, error TEXT,
            created_at REAL, updated_at REAL, metadata TEXT DEFAULT '{}'
        );
        INSERT INTO sessions(id,title,created_at,updated_at) VALUES ('sess_old','Old',1,1);
        INSERT INTO runs(id,session_id,mode,query,status,created_at,updated_at)
        VALUES ('run_old','sess_old','deep','Old query','completed',1,1);
        """
    )
    connection.commit()
    connection.close()

    first = Store(db_path)
    columns = {
        row["name"]
        for row in first._conn.execute("PRAGMA table_info(deep_checkpoints)").fetchall()  # noqa: SLF001
    }
    first.close()
    second = Store(db_path)
    old_run = second.get_run("run_old")
    second.close()

    assert {
        "run_id",
        "lineage_id",
        "revision",
        "schema_version",
        "stage",
        "payload_json",
        "payload_sha256",
        "disposition",
        "claimed_child_run_id",
        "claim_idempotency_key",
        "created_at",
        "updated_at",
    }.issubset(columns)
    assert old_run is not None
    assert old_run["status"] == "completed"


def test_user_pause_is_persisted_and_safe_boundary_commit_is_atomic():
    store = Store(":memory:")
    _, run_id = _running_deep_run(store)

    pause = store.request_run_pause(run_id)
    assert pause["status"] == "pausing"
    assert [item["type"] for item in store.list_events(run_id)] == ["pause_requested"]

    encoded = encode_deep_checkpoint(
        DeepCheckpointV1.from_dict(_checkpoint_payload(run_id=run_id))
    )
    saved = store.save_deep_checkpoint_and_pause(
        run_id=run_id,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=encoded.payload_json,
        payload_sha256=encoded.payload_sha256,
    )

    run = store.get_run(run_id)
    events = store.list_events(run_id)
    checkpoint = store.get_deep_checkpoint(run_id, include_payload=True)
    store.close()

    assert saved["revision"] == 1
    assert run is not None and run["status"] == "paused"
    assert run["result"] is None
    assert run["error"] is None
    assert [item["type"] for item in events] == [
        "pause_requested",
        "checkpoint_saved",
        "workflow_end",
    ]
    assert events[-1]["payload"]["status"] == "paused"
    assert checkpoint is not None
    assert checkpoint["integrity"] == "valid"


def test_cancel_wins_before_checkpoint_pause_commit():
    store = Store(":memory:")
    _, run_id = _running_deep_run(store)
    store.request_run_pause(run_id)
    assert store.transition_run(
        run_id,
        expected_statuses={"pausing"},
        status="cancelling",
    )
    encoded = encode_deep_checkpoint(
        DeepCheckpointV1.from_dict(_checkpoint_payload(run_id=run_id))
    )

    assert (
        store.save_deep_checkpoint_and_pause(
            run_id=run_id,
            lineage_id="lineage_1",
            stage="round_complete",
            schema_version=1,
            payload_json=encoded.payload_json,
            payload_sha256=encoded.payload_sha256,
        )
        is None
    )
    assert store.get_deep_checkpoint(run_id) is None
    assert store.get_run(run_id)["status"] == "cancelling"
    store.close()


def test_checkpoint_claim_is_single_child_and_same_idempotency_is_replayed():
    store = Store(":memory:")
    session_id, run_id = _running_deep_run(store)
    store.request_run_pause(run_id)
    encoded = encode_deep_checkpoint(
        DeepCheckpointV1.from_dict(_checkpoint_payload(run_id=run_id))
    )
    checkpoint = store.save_deep_checkpoint_and_pause(
        run_id=run_id,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=encoded.payload_json,
        payload_sha256=encoded.payload_sha256,
    )

    first = store.claim_deep_checkpoint_and_create_child(
        run_id=run_id,
        expected_revision=checkpoint["revision"],
        idempotency_key="resume-once",
        mode="deep",
        query="Research this",
        budget={"timeout_s": 180},
        metadata={"checkpoint_parent_run_id": run_id},
    )
    repeated = store.claim_deep_checkpoint_and_create_child(
        run_id=run_id,
        expected_revision=checkpoint["revision"],
        idempotency_key="resume-once",
        mode="deep",
        query="Research this",
        budget={"timeout_s": 180},
        metadata={"checkpoint_parent_run_id": run_id},
    )

    assert first["session_id"] == session_id
    assert repeated["run_id"] == first["run_id"]
    with pytest.raises(CheckpointAlreadyContinued):
        store.claim_deep_checkpoint_and_create_child(
            run_id=run_id,
            expected_revision=checkpoint["revision"],
            idempotency_key="different-request",
            mode="deep",
            query="Research this",
            budget={"timeout_s": 180},
            metadata={"checkpoint_parent_run_id": run_id},
        )
    store.close()


def test_session_delete_removes_checkpoint_but_not_other_sessions():
    store = Store(":memory:")
    first_session, first_run = _running_deep_run(store)
    _, second_run = _running_deep_run(store)
    for run_id in (first_run, second_run):
        encoded = encode_deep_checkpoint(
            DeepCheckpointV1.from_dict(_checkpoint_payload(run_id=run_id))
        )
        store.save_deep_checkpoint(
            run_id=run_id,
            lineage_id="lineage_1",
            stage="round_complete",
            schema_version=1,
            payload_json=encoded.payload_json,
            payload_sha256=encoded.payload_sha256,
        )

    assert store.delete_session(first_session) is True
    assert store.get_deep_checkpoint(first_run) is None
    assert store.get_deep_checkpoint(second_run) is not None
    store.close()


def test_deep_checkpoint_setting_defaults_off_persists_and_freezes(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"deep":{"future_option":"preserve"},"unrelated":{"enabled":true}}',
        encoding="utf-8",
    )
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        config_path=config_path,
    )

    assert service.get_deep_checkpoint_settings() == {
        "auto_checkpoint_enabled": False
    }
    assert service.update_deep_checkpoint_settings(
        {"auto_checkpoint_enabled": True}
    ) == {"auto_checkpoint_enabled": True}

    class CaptureBackend:
        request = None

        def enqueue(self, request, run_id):  # noqa: ANN001
            self.request = request

        def health(self):
            return {"job_backend": "capture"}

    backend = CaptureBackend()
    monkeypatch.setattr(service, "_llm_ready", True)
    monkeypatch.setattr(service, "_job_backend", backend)
    created = service.create_run(
        {"query": "Checkpoint this Deep run", "mode": "deep"},
        background=True,
    )

    run = service.store.get_run(created["run_id"])
    assert backend.request.auto_checkpoint_enabled is True
    assert run["metadata"]["auto_checkpoint_enabled"] is True
    reloaded = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        config_path=config_path,
    )
    assert reloaded.get_deep_checkpoint_settings()["auto_checkpoint_enabled"] is True
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["deep"]["future_option"] == "preserve"
    assert persisted["unrelated"] == {"enabled": True}
    service.shutdown()
    reloaded.shutdown()


def test_pause_checkpoint_continue_and_abandon_http_contract(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    monkeypatch.setattr(service, "_llm_ready", True)

    class CaptureBackend:
        requests = []

        def enqueue(self, request, run_id):  # noqa: ANN001
            self.requests.append((request, run_id))

        def health(self):
            return {"job_backend": "capture"}

    backend = CaptureBackend()
    monkeypatch.setattr(service, "_job_backend", backend)
    session_id, run_id = _running_deep_run(service.store)
    client = TestClient(create_app(service))

    pause = client.post(f"/api/v1/runs/{run_id}/pause")
    assert pause.status_code == 200
    assert pause.json()["status"] == "pausing"

    checkpoint_payload = _checkpoint_payload(run_id=run_id)
    checkpoint_payload["evidence_snapshot"] = []
    checkpoint_payload["request"] = {
        "query": "Research this",
        "mode": "deep",
        "session_id": session_id,
        "requested_at": 1_700_000_000.0,
        "deadline_at": 1_700_000_180.0,
        "budget": {},
        "resolved_search_usage": {},
    }
    encoded = encode_deep_checkpoint(DeepCheckpointV1.from_dict(checkpoint_payload))
    checkpoint = service.store.save_deep_checkpoint_and_pause(
        run_id=run_id,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=encoded.payload_json,
        payload_sha256=encoded.payload_sha256,
    )

    fetched = client.get(f"/api/v1/runs/{run_id}/checkpoint")
    assert fetched.status_code == 200
    assert fetched.json()["integrity"] == "valid"
    assert fetched.json()["resume_readiness"] == "ready"
    assert "payload" not in fetched.json()

    continued = client.post(
        f"/api/v1/runs/{run_id}/continue",
        json={
            "expected_checkpoint_revision": checkpoint["revision"],
            "idempotency_key": "http-resume-once",
            "action": "continue",
            "incremental_budget": {"timeout_s": 180},
        },
    )
    assert continued.status_code == 200
    assert continued.json()["parent_run_id"] == run_id
    assert len(backend.requests) == 1
    queued_request, child_run_id = backend.requests[0]
    assert queued_request.checkpoint_parent_run_id == run_id
    assert queued_request.checkpoint_revision == checkpoint["revision"]
    assert queued_request.deadline_at > queued_request.requested_at
    assert service.store.get_run(child_run_id)["metadata"]["lineage_id"] == "lineage_1"

    replay = client.post(
        f"/api/v1/runs/{run_id}/continue",
        json={
            "expected_checkpoint_revision": checkpoint["revision"],
            "idempotency_key": "http-resume-once",
            "action": "continue",
        },
    )
    assert replay.status_code == 200
    assert replay.json()["run_id"] == child_run_id
    assert replay.json()["idempotent_replay"] is True
    assert len(backend.requests) == 1

    second_session, second_run = _running_deep_run(service.store)
    service.store.request_run_pause(second_run)
    second_payload = _checkpoint_payload(run_id=second_run)
    second_payload["evidence_snapshot"] = []
    second_payload["request"]["session_id"] = second_session
    second_encoded = encode_deep_checkpoint(
        DeepCheckpointV1.from_dict(second_payload)
    )
    second_checkpoint = service.store.save_deep_checkpoint_and_pause(
        run_id=second_run,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=second_encoded.payload_json,
        payload_sha256=second_encoded.payload_sha256,
    )
    abandoned = client.post(
        f"/api/v1/runs/{second_run}/checkpoint/abandon",
        json={"expected_checkpoint_revision": second_checkpoint["revision"]},
    )
    assert abandoned.status_code == 200
    assert abandoned.json()["disposition"] == "abandoned"
    blocked = client.post(
        f"/api/v1/runs/{second_run}/continue",
        json={
            "expected_checkpoint_revision": second_checkpoint["revision"],
            "idempotency_key": "cannot-resume",
            "action": "continue",
        },
    )
    assert blocked.status_code == 409
    service.shutdown()


def test_non_deep_run_pause_is_rejected_without_state_change():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service.store.create_session(title="Fast")
    run_id = service.store.create_run(
        session_id=session_id,
        mode="fast",
        query="Fast query",
        metadata={"enqueue_confirmed": True},
    )
    assert service.store.claim_queued_run(run_id)
    client = TestClient(create_app(service))

    response = client.post(f"/api/v1/runs/{run_id}/pause")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RUN_NOT_PAUSABLE"
    assert service.store.get_run(run_id)["status"] == "running"
    service.shutdown()


def test_deep_plan_boundary_pauses_parent_without_result_side_effects(monkeypatch):
    store = Store(":memory:")
    orchestrator = _deep_orchestrator(store)
    session_id = store.create_session(title="Pause at plan")
    request = RunRequest(
        query="checkpoint continuation",
        mode=Mode.DEEP,
        session_id=session_id,
    )
    run_id = store.create_run(
        session_id=session_id,
        mode="deep",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
        metadata={"enqueue_confirmed": True},
    )

    def pause_after_plan(query, budget, ctx, *, section_count=None):  # noqa: ANN001, ARG001
        assert store.request_run_pause(run_id)["accepted"] is True
        return ["Checkpoint continuation"]

    monkeypatch.setattr(orchestrator, "_plan", pause_after_plan)

    with pytest.raises(DeepRunPaused):
        orchestrator.run(request, run_id=run_id)

    run = store.get_run(run_id)
    checkpoint = store.get_deep_checkpoint(run_id, include_payload=True)
    event_types = [item["type"] for item in store.list_events(run_id)]
    citation_count = store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM citations WHERE run_id=?",
        (run_id,),
    ).fetchone()[0]
    message_count = store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM messages WHERE session_id=?",
        (session_id,),
    ).fetchone()[0]

    assert run is not None and run["status"] == "paused"
    assert run["result"] is None and run["error"] is None
    assert checkpoint is not None and checkpoint["stage"] == "plan_ready"
    assert checkpoint["payload"]["pending_tasks"] == [
        {"title": "Checkpoint continuation"}
    ]
    assert event_types[-3:] == [
        "pause_requested",
        "checkpoint_saved",
        "workflow_end",
    ]
    assert citation_count == 0
    assert message_count == 0
    store.close()


def test_user_pause_checkpoint_write_failure_rolls_back_and_run_continues(monkeypatch):
    store = Store(":memory:")
    orchestrator = _deep_orchestrator(store)
    session_id = store.create_session(title="Pause write failure")
    request = RunRequest(
        query="checkpoint continuation",
        mode=Mode.DEEP,
        session_id=session_id,
    )
    run_id = store.create_run(
        session_id=session_id,
        mode="deep",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
        metadata={"enqueue_confirmed": True},
    )

    def pause_after_plan(query, budget, ctx, *, section_count=None):  # noqa: ANN001, ARG001
        assert store.request_run_pause(run_id)["accepted"] is True
        return ["Checkpoint continuation"]

    def fail_checkpoint_write(**kwargs):  # noqa: ANN003
        raise OSError("synthetic checkpoint write failure")

    monkeypatch.setattr(orchestrator, "_plan", pause_after_plan)
    monkeypatch.setattr(store, "save_deep_checkpoint_and_pause", fail_checkpoint_write)

    outcome = orchestrator.run(request, run_id=run_id)

    run = store.get_run(run_id)
    events = store.list_events(run_id)
    assert outcome.status in {"completed", "completed_degraded"}
    assert run is not None and run["status"] == outcome.status
    assert store.get_deep_checkpoint(run_id) is None
    assert [item["type"] for item in events].count("pause_requested") == 1
    failures = [item for item in events if item["type"] == "checkpoint_save_failed"]
    assert len(failures) == 1
    assert failures[0]["payload"]["reason"] == "checkpoint_write_failed"
    store.close()


def test_child_run_restores_plan_checkpoint_without_replanning(monkeypatch):
    store = Store(":memory:")
    orchestrator = _deep_orchestrator(store)
    session_id = store.create_session(title="Continue child")
    parent_request = RunRequest(
        query="checkpoint continuation",
        mode=Mode.DEEP,
        session_id=session_id,
    )
    parent_run_id = store.create_run(
        session_id=session_id,
        mode="deep",
        query=parent_request.query,
        budget=parent_request.resolved_budget().to_dict(),
        metadata={"enqueue_confirmed": True},
    )

    def pause_after_plan(query, budget, ctx, *, section_count=None):  # noqa: ANN001, ARG001
        store.request_run_pause(parent_run_id)
        return ["Checkpoint continuation"]

    monkeypatch.setattr(orchestrator, "_plan", pause_after_plan)
    with pytest.raises(DeepRunPaused):
        orchestrator.run(parent_request, run_id=parent_run_id)
    checkpoint = store.get_deep_checkpoint(parent_run_id)
    assert checkpoint is not None
    claimed = store.claim_deep_checkpoint_and_create_child(
        run_id=parent_run_id,
        expected_revision=checkpoint["revision"],
        idempotency_key="execute-child-once",
        mode="deep",
        query=parent_request.query,
        budget=parent_request.resolved_budget().to_dict(),
        metadata={
            "enqueue_confirmed": True,
            "checkpoint_parent_run_id": parent_run_id,
            "checkpoint_revision": checkpoint["revision"],
            "lineage_id": checkpoint["lineage_id"],
        },
    )
    child_run_id = claimed["run_id"]

    def unexpected_replan(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("continued child must not rerun the completed planner")

    monkeypatch.setattr(orchestrator, "_plan", unexpected_replan)
    child_request = RunRequest(
        query=parent_request.query,
        mode=Mode.DEEP,
        session_id=session_id,
        checkpoint_parent_run_id=parent_run_id,
        checkpoint_revision=checkpoint["revision"],
        lineage_id=checkpoint["lineage_id"],
        checkpoint_action="continue",
    )

    outcome = orchestrator.run(child_request, run_id=child_run_id)

    assert outcome.status in {"completed", "completed_degraded"}
    assert outcome.to_dict()["research_state"]["searches_used"] >= 1
    assert store.get_run(parent_run_id)["status"] == "paused"
    assert store.get_run(child_run_id)["status"] == outcome.status
    assert store.get_run(child_run_id)["result"]["answer"]["text"]
    store.close()


def test_revise_uses_incremental_search_budget_without_resetting_lineage_usage(monkeypatch):
    store = Store(":memory:")
    orchestrator = _deep_orchestrator(store)
    session_id, parent_run_id = _running_deep_run(store)
    store.request_run_pause(parent_run_id)
    payload = _checkpoint_payload(run_id=parent_run_id)
    payload["request"]["session_id"] = session_id
    payload["plan"] = {
        "sections": [
            {"id": "completed", "title": "Completed section"},
            {"id": "old-pending", "title": "Old pending section"},
        ]
    }
    payload["pending_tasks"] = [{"title": "Old pending section"}]
    payload["completed_sections"] = []
    payload["evidence_snapshot"] = [
        {
            "chunk_id": 1,
            "document_id": "doc_checkpoint_external",
            "url": "https://example.com/checkpoint",
            "title": "Checkpoint continuation evidence",
            "text": (
                "Research continuation preserves completed work while revised evidence "
                "and implications use an explicit incremental search budget."
            ),
            "score": 0.99,
            "channels": ["web"],
            "origin": "external",
            "artifact_type": "web_document",
            "artifact_id": "doc_checkpoint_external",
            "grounding_eligible": True,
            "provenance_refs": ["https://example.com/checkpoint"],
            "source_status": "active",
        }
    ]
    payload["retrieval_state"] = {
        "research_state": {
            "plan": ["Completed section", "Old pending section"],
            "evidence_matrix": [
                {
                    "subtask": "Completed section",
                    "queries": ["completed evidence"],
                    "notes": "Already researched.",
                    "gaps": [],
                    "source_count": 1,
                    "status": "researched",
                }
            ],
            "searches_used": 1,
            "searches_max": 3,
            "pages_fetched": 1,
            "think_tokens_used": 0,
            "gap_rounds": 0,
            "pending": ["Old pending section"],
        },
        "collected_rankings": [],
        "retrieval_outcomes": [],
        "gap_tasks": [],
    }
    payload["budget_usage"] = {
        "ordinary_searches_used": 1,
        "recovery_searches_used": 0,
        "providers_attempted": ["fake-search"],
    }
    encoded = encode_deep_checkpoint(DeepCheckpointV1.from_dict(payload))
    checkpoint = store.save_deep_checkpoint_and_pause(
        run_id=parent_run_id,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=encoded.payload_json,
        payload_sha256=encoded.payload_sha256,
    )
    claimed = store.claim_deep_checkpoint_and_create_child(
        run_id=parent_run_id,
        expected_revision=checkpoint["revision"],
        idempotency_key="revise-with-increment",
        mode="deep",
        query="Research this",
        budget={"max_searches": 2, "timeout_s": 180},
        metadata={
            "enqueue_confirmed": True,
            "checkpoint_parent_run_id": parent_run_id,
            "checkpoint_revision": checkpoint["revision"],
            "lineage_id": "lineage_1",
        },
    )

    def unexpected_replan(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("revised child must restore and edit only the pending plan")

    monkeypatch.setattr(orchestrator, "_plan", unexpected_replan)
    child_request = RunRequest(
        query="Research this",
        mode=Mode.DEEP,
        session_id=session_id,
        checkpoint_parent_run_id=parent_run_id,
        checkpoint_revision=checkpoint["revision"],
        lineage_id="lineage_1",
        checkpoint_action="revise",
        checkpoint_remaining_plan={
            "sections": ["Revised evidence", "Revised implications"]
        },
        checkpoint_incremental_budget={"max_searches": 2},
        overrides={"max_searches": 2, "timeout_s": 180},
    )

    outcome = orchestrator.run(child_request, run_id=claimed["run_id"])
    research_state = outcome.to_dict()["research_state"]

    assert research_state["plan"] == [
        "Completed section",
        "Revised evidence",
        "Revised implications",
    ]
    assert research_state["searches_used"] == 3
    assert research_state["searches_max"] == 3
    assert store.get_run(parent_run_id)["status"] == "paused"
    store.close()


def test_auto_checkpoint_off_and_non_deep_modes_never_persist_checkpoints():
    store = Store(":memory:")
    search = FakeSearch(
        {
            "https://example.com/checkpoint": (
                "Checkpoint-safe research uses grounded evidence and deterministic "
                "execution boundaries across supported research modes."
            )
        }
    )
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        memory=MemoryEngine(store),
        council_members=[FakeLLM("m-a"), FakeLLM("m-b")],
        enforce_grounding_gate=False,
    )

    deep_session = store.create_session(title="Deep auto off")
    deep = orchestrator.run(
        RunRequest(
            query="checkpoint continuation",
            mode=Mode.DEEP,
            session_id=deep_session,
        )
    )
    assert store.get_deep_checkpoint(deep.run_id) is None
    assert all(
        event["type"] != "checkpoint_saved"
        for event in store.list_events(deep.run_id)
    )

    for mode in (Mode.FAST, Mode.MULTIPLE, Mode.COUNCIL):
        session_id = store.create_session(title=f"{mode.value} no checkpoint")
        outcome = orchestrator.run(
            RunRequest(
                query="checkpoint continuation",
                mode=mode,
                session_id=session_id,
                auto_checkpoint_enabled=True,
            )
        )
        assert store.get_deep_checkpoint(outcome.run_id) is None
        assert all(
            event["type"] != "checkpoint_saved"
            for event in store.list_events(outcome.run_id)
        )
    store.close()


def test_auto_checkpoint_records_all_three_safe_boundaries():
    store = Store(":memory:")
    orchestrator = _deep_orchestrator(store)
    session_id = store.create_session(title="Automatic checkpoints")

    outcome = orchestrator.run(
        RunRequest(
            query="checkpoint continuation",
            mode=Mode.DEEP,
            session_id=session_id,
            auto_checkpoint_enabled=True,
        )
    )

    events = [
        item["payload"]
        for item in store.list_events(outcome.run_id)
        if item["type"] == "checkpoint_saved"
    ]
    checkpoint = store.get_deep_checkpoint(outcome.run_id, include_payload=True)

    assert {"plan_ready", "round_complete", "final_evidence_ready"}.issubset(
        {str(item["stage"]) for item in events}
    )
    assert all(item["automatic"] is True for item in events)
    assert checkpoint is not None
    assert checkpoint["stage"] == "final_evidence_ready"
    assert checkpoint["payload"]["evidence_snapshot"]
    assert store.get_run(outcome.run_id)["status"] in {
        "completed",
        "completed_degraded",
    }
    store.close()


def test_resume_readiness_degrades_for_source_tombstone_but_allows_archived_memory():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id, run_id = _running_deep_run(service.store)
    memory_id = service.store.add_memory(
        namespace="session",
        scope_key=session_id,
        content="Archived memories remain valid internal evidence.",
    )
    assert service.store.archive_memory(memory_id) is True
    service.store.request_run_pause(run_id)
    payload = _checkpoint_payload(run_id=run_id)
    payload["request"]["session_id"] = session_id
    payload["evidence_snapshot"] = [
        {
            "id": "ev_deleted_document",
            "origin": "local",
            "artifact_type": "library_document",
            "artifact_id": "doc_missing",
            "text": "Historical evidence remains frozen in the checkpoint.",
        },
        {
            "id": "ev_archived_memory",
            "origin": "memory",
            "artifact_type": "memory",
            "artifact_id": memory_id,
            "text": "Archived memories remain citable.",
        },
    ]
    encoded = encode_deep_checkpoint(DeepCheckpointV1.from_dict(payload))
    service.store.save_deep_checkpoint_and_pause(
        run_id=run_id,
        lineage_id="lineage_1",
        stage="round_complete",
        schema_version=1,
        payload_json=encoded.payload_json,
        payload_sha256=encoded.payload_sha256,
    )

    view = service.get_run_checkpoint(run_id)

    assert view["integrity"] == "valid"
    assert view["resume_readiness"] == "degraded"
    assert "source_tombstone" in view["reason_codes"]
    assert "memory_missing" not in view["reason_codes"]
    service.shutdown()


def test_final_writer_fence_atomically_rejects_late_pause():
    store = Store(":memory:")
    _, run_id = _running_deep_run(store)

    assert store.fence_deep_final_writer(run_id) is True
    pause = store.request_run_pause(run_id)

    assert pause["accepted"] is False
    assert pause["reason"] == "final_writer_started"
    assert store.get_run(run_id)["status"] == "running"
    store.close()
