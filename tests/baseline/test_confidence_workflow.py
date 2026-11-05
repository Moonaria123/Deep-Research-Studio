"""Advisory confidence persistence and report regeneration contracts."""

from __future__ import annotations

import time

import pytest

from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.api import create_app
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def _delivered_run(store: Store) -> tuple[str, str]:
    session_id = store.create_session(title="Confidence")
    run_id = store.create_run(
        session_id=session_id,
        mode="fast",
        query="Evidence-backed question",
        metadata={"deadline_at": time.time() + 30.0},
    )
    store.update_run(
        run_id,
        status="completed",
        result={
            "answer": {
                "text": "Delivered immediately. [c1]",
                "citations": [{"id": "c1"}],
            },
            "verification": {"status": "running", "started_at": time.time()},
        },
    )
    store.add_citation(
        run_id=run_id,
        document_id="doc-1",
        url="https://example.com/source",
        title="Source",
        snippet="Existing source evidence.",
        score=0.9,
    )
    return session_id, run_id


def test_late_verification_preserves_delivered_report_and_terminal_events() -> None:
    store = Store(":memory:")
    _session_id, run_id = _delivered_run(store)

    updated = store.update_run_verification(
        run_id,
        verification={
            "status": "completed",
            "completed_at": time.time(),
            "reason_codes": ["unsupported_claim"],
        },
        confidence_label="uncertain",
        confidence_warning={"reasons": ["unsupported_claim"], "acknowledged": False},
        conflicts=[],
    )

    run = store.get_run(run_id)
    assert updated is True
    assert run["status"] == "completed"
    assert run["result"]["answer"]["text"] == "Delivered immediately. [c1]"
    assert run["result"]["answer"]["citations"] == [{"id": "c1"}]
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    assert run["result"]["answer"]["conflicts"] == []
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    assert store.list_citations(run_id)[0]["document_id"] == "doc-1"
    assert store.list_events(run_id) == []
    store.close()


def test_confidence_acknowledgement_is_persistent_and_idempotent() -> None:
    store = Store(":memory:")
    _session_id, run_id = _delivered_run(store)
    assert store.update_run_verification(
        run_id,
        verification={"status": "failed", "reason_codes": ["verification_interrupted"]},
        confidence_label="uncertain",
        confidence_warning={"reasons": ["verification_interrupted"], "acknowledged": False},
    )

    first = store.acknowledge_run_confidence(run_id)
    second = store.acknowledge_run_confidence(run_id)

    assert first == second
    assert first and first["acknowledged"] is True
    run = store.get_run(run_id)
    assert run["result"]["confidence_warning"]["acknowledged_at"] == first["acknowledged_at"]
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    store.close()


def test_legacy_uncertain_report_can_be_acknowledged_without_fake_verification() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="Legacy confidence")
    run_id = store.create_run(
        session_id=session_id,
        mode="fast",
        query="Legacy report",
    )
    store.update_run(
        run_id,
        status="completed",
        result={
            "answer": {
                "text": "Historical report body. [c1]",
                "citations": [{"id": "c1"}],
                "confidence_label": "uncertain",
            }
        },
    )

    acknowledged = store.acknowledge_run_confidence(run_id)
    run = store.get_run(run_id)

    assert acknowledged and acknowledged["acknowledged"] is True
    assert "verification" not in run["result"]
    assert run["result"]["answer"]["text"] == "Historical report body. [c1]"
    assert run["result"]["answer"]["citations"] == [{"id": "c1"}]
    assert run["result"]["confidence_warning"]["reasons"] == ["legacy_low_confidence"]
    assert run["result"]["confidence_warning"]["acknowledged"] is True
    store.close()


def test_service_expires_abandoned_verification_as_low_confidence(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    _session_id, run_id = _delivered_run(service.store)
    run_before = service.store.get_run(run_id)
    monkeypatch.setattr(
        "deepresearch_core.baseline.service.time.time",
        lambda: float(run_before["updated_at"]) + 61.0,
    )

    run = service.get_run(run_id)

    assert run["status"] == "completed"
    assert run["result"]["verification"]["status"] == "failed"
    assert run["result"]["verification"]["reason_codes"] == ["verification_interrupted"]
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    service.shutdown()


def test_regenerate_uses_same_session_and_server_owned_evidence_snapshot(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM())
    session_id, parent_id = _delivered_run(service.store)
    parent = service.store.get_run(parent_id)
    metadata = dict(parent["metadata"])
    metadata["run_request_snapshot"] = {
        "query": "Evidence-backed question",
        "mode": "fast",
        "session_id": session_id,
        "llm_model": "provider:model-a",
        "search_options": {"freshness": "pm"},
        "budget": {"timeout_s": 29},
    }
    service.store.update_run(parent_id, metadata_patch=metadata)
    service.store.add_citation(
        run_id=parent_id,
        document_id="doc-deleted",
        url="https://example.com/deleted",
        title="Deleted source",
        snippet="Must not be reused.",
        score=1.0,
        source_status="deleted",
    )
    service.store.add_citation(
        run_id=parent_id,
        document_id="",
        url=None,
        title="Missing Memory artifact",
        snippet="Must not be resurrected.",
        score=0.95,
        artifact_type="memory",
        artifact_id="",
    )

    class CaptureBackend:
        request = None
        run_id = ""

        def enqueue(self, request, run_id):  # noqa: ANN001
            self.request = request
            self.run_id = run_id

        def health(self):
            return {"job_backend": "capture"}

    backend = CaptureBackend()
    monkeypatch.setattr(service, "_job_backend", backend)

    created = service.regenerate_run(parent_id)
    child = service.store.get_run(created["run_id"])
    context = child["metadata"]["regeneration_context"]

    assert created["session_id"] == session_id
    assert created["parent_run_id"] == parent_id
    assert child["mode"] == "fast"
    assert backend.request.llm_model == "provider:model-a"
    assert context["parent_run_id"] == parent_id
    assert context["fresh_search_required"] is True
    assert context["fresh_search_available"] is False
    assert context["diagnostic"] == "external_search_unavailable"
    assert context["seed_evidence"][0]["chunk_id"] == 0
    assert context["seed_evidence"][0]["snippet"] == "Existing source evidence."
    assert len(context["seed_evidence"]) == 1
    assert "regeneration_context" not in child["metadata"]["run_request_snapshot"]
    assert service.store.get_run(parent_id)["result"]["answer"]["text"] == "Delivered immediately. [c1]"
    service.shutdown()


def test_acknowledge_rejects_report_without_completed_warning() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    _session_id, run_id = _delivered_run(service.store)

    with pytest.raises(BaselineError) as exc:
        service.acknowledge_run_confidence(run_id)

    assert exc.value.code is ErrorCode.INVALID_REQUEST
    service.shutdown()


def test_confidence_http_routes_delegate_to_service(monkeypatch) -> None:
    testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM())
    session_id, parent_id = _delivered_run(service.store)
    assert service.store.update_run_verification(
        parent_id,
        verification={"status": "completed"},
        confidence_label="uncertain",
        confidence_warning={"reasons": ["evidence_insufficient"], "acknowledged": False},
    )

    class CaptureBackend:
        def enqueue(self, request, run_id):  # noqa: ANN001
            self.request = request
            self.run_id = run_id

        def health(self):
            return {"job_backend": "capture"}

    monkeypatch.setattr(service, "_job_backend", CaptureBackend())
    with testclient.TestClient(create_app(service)) as client:
        acknowledged = client.post(
            f"/api/v1/runs/{parent_id}/confidence/acknowledge"
        )
        regenerated = client.post(f"/api/v1/runs/{parent_id}/regenerate")

    assert acknowledged.status_code == 200
    assert acknowledged.json()["acknowledged"] is True
    assert regenerated.status_code == 200
    assert regenerated.json()["session_id"] == session_id
    assert regenerated.json()["parent_run_id"] == parent_id
