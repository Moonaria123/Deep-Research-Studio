"""DR-ITER-060 T8b: reports, snapshots, and post-run Library artifacts."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.adapters import Document, SearchHit
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.snapshots import SnapshotWriter
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM, FakeSearch


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def test_report_crud_and_children_roundtrip():
    store = Store(":memory:")

    parent_id = store.add_report(
        title="Deep report",
        body_md="# Deep report\nBody",
        mode="deep",
        session_id="s1",
        run_id="r1",
        metadata={"source": "post_run"},
    )
    child_id = store.add_report(
        title="Member A",
        body_md="member draft",
        mode="council_member",
        parent_report_id=parent_id,
        session_id="s1",
        run_id="r1",
    )

    assert store.get_report(parent_id)["body_md"].startswith("# Deep")
    report_ids = {row["id"] for row in store.list_reports()}
    assert {parent_id, child_id}.issubset(report_ids)
    assert "body_md" not in store.list_reports()[0]
    assert store.list_child_reports(parent_id)[0]["id"] == child_id


def test_snapshot_writer_stays_inside_root_and_store_roundtrips(tmp_path):
    store = Store(":memory:")
    writer = SnapshotWriter(tmp_path)

    snapshot = writer.write_text_snapshot(
        url="https://example.com/a",
        title="Example",
        text="Hello<script>alert(1)</script>\x00 world",
    )
    snapshot_id = store.add_page_snapshot(
        document_id="doc_1",
        url=snapshot.url,
        content_hash=snapshot.content_hash,
        storage_path=snapshot.storage_path,
        mime_type=snapshot.mime_type,
        size_bytes=snapshot.size_bytes,
        status="ok",
        run_id="r1",
    )

    assert "<script>" not in writer.read_text(snapshot.storage_path)
    assert "alert" not in writer.read_text(snapshot.storage_path)
    assert writer.resolve(snapshot.storage_path).relative_to(tmp_path.resolve())
    assert store.get_page_snapshot(snapshot_id)["status"] == "ok"

    with pytest.raises(BaselineError) as exc:
        writer.read_text("../escape.txt")
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def _client_with_service(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    return TestClient(create_app(service)), service


def test_library_reports_api_roundtrip_and_indexes_report(tmp_path):
    client, service = _client_with_service(tmp_path)

    res = client.post(
        "/api/v1/library/reports",
        json={
            "title": "Report",
            "body_md": "# Report\nUseful answer",
            "mode": "deep",
            "session_id": "s1",
            "run_id": "r1",
        },
    )

    assert res.status_code == 200
    report_id = res.json()["report_id"]
    listed = client.get("/api/v1/library/reports").json()["reports"]
    assert listed[0]["id"] == report_id
    assert "body_md" not in listed[0]
    full = client.get(f"/api/v1/library/reports/{report_id}").json()["report"]
    assert full["body_md"].startswith("# Report")
    assert service.store.search_chunks_lexical("Useful", limit=5)


def test_library_notes_api_exposes_linked_run(tmp_path):
    client, _service = _client_with_service(tmp_path)

    created = client.post(
        "/api/v1/library/notes",
        json={"content": "Run-linked note", "session_id": "s1", "run_id": "r1"},
    ).json()

    listed = client.get("/api/v1/library/notes").json()["notes"]

    assert created["note_id"]
    assert listed[0]["session_id"] == "s1"
    assert listed[0]["run_id"] == "r1"


def test_library_report_upload_api_accepts_markdown_and_indexes_report(tmp_path):
    client, service = _client_with_service(tmp_path)

    res = client.post(
        "/api/v1/library/reports/upload",
        data={"title": "Uploaded report"},
        files={"file": ("report.md", b"# Uploaded\n\nAlpha upload body", "text/markdown")},
    )

    assert res.status_code == 200
    report_id = res.json()["report_id"]
    report = client.get(f"/api/v1/library/reports/{report_id}").json()["report"]
    assert report["title"] == "Uploaded report"
    assert report["mode"] == "upload"
    assert "Alpha upload body" in report["body_md"]
    assert service.store.search_chunks_lexical("Alpha", limit=5)


def test_library_source_sessions_and_snapshots_api(tmp_path):
    client, service = _client_with_service(tmp_path)
    doc_id = service.store.upsert_document(
        url="https://e.com/a",
        title="Doc A",
        text="snap text",
        content_type="web",
    )
    service.store.add_library_provenance(
        artifact_type="source",
        artifact_id=doc_id,
        document_id=doc_id,
        session_id="s1",
        run_id="r1",
    )
    snap_id = service.capture_document_snapshot(document_id=doc_id, run_id="r1")["snapshot_id"]

    sessions = client.get(f"/api/v1/library/sources/{doc_id}/sessions").json()
    assert sessions["artifacts"][0]["session_id"] == "s1"
    snapshots = client.get(f"/api/v1/library/sources/{doc_id}/snapshots").json()
    assert snapshots["snapshots"][0]["id"] == snap_id
    preview = client.get(f"/api/v1/library/snapshots/{snap_id}").json()
    assert preview["snapshot"]["text"] == "snap text"
    assert "storage_path" not in preview["snapshot"]


def test_deep_run_persists_report_and_snapshot_without_default_post_run_note(fake_llm, fake_search, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.configure_runtime_for_tests(llm=fake_llm, search=fake_search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(
        RunRequest(query="state of quantum computing", mode=Mode.DEEP, session_id=sid)
    )

    reports = service.store.list_reports()
    notes = service.list_notes()["notes"]
    assert any(r["run_id"] == outcome.run_id and r["mode"] == "deep" for r in reports)
    assert not any(note.get("metadata", {}).get("source") == "post_run" for note in notes)
    assert service.store.list_page_snapshots_for_document(outcome.answer.citations[0].document_id)


def test_deep_run_writes_post_run_note_when_mode_enabled(fake_llm, fake_search, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.configure_runtime_for_tests(llm=fake_llm, search=fake_search)
    service.update_post_run_notes_config({"enabled_modes": ["deep"]})
    sid = service.store.create_session(title="t")

    service.orchestrator.run(RunRequest(query="state of quantum computing", mode=Mode.DEEP, session_id=sid))

    notes = service.list_notes()["notes"]
    assert any(note.get("metadata", {}).get("source") == "post_run" for note in notes)


def test_council_run_persists_parent_and_member_reports(fake_llm, fake_search, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.configure_runtime_for_tests(llm=fake_llm, search=fake_search, council_members=[FakeLLM("m-a"), FakeLLM("m-b")])
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(
        RunRequest(query="quantum computing qubits", mode=Mode.COUNCIL, session_id=sid)
    )

    reports = service.store.list_reports()
    parent = next(row for row in reports if row["run_id"] == outcome.run_id and row["mode"] == "council")
    children = service.store.list_child_reports(parent["id"])
    assert {child["metadata"]["member"] for child in children} == {"m-a", "m-b"}


def test_post_run_artifact_failure_does_not_rewrite_completed_run(fake_llm, fake_search):
    store = Store(":memory:")
    sid = store.create_session(title="t")

    def explode(_request, _outcome):
        raise RuntimeError("boom")

    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        search=fake_search,
        post_run_artifacts=explode,
    )

    query = "quantum computing qubits"
    rid = store.create_run(session_id=sid, mode="deep", query=query, budget={})
    orch.run(RunRequest(query=query, mode=Mode.DEEP, session_id=sid), run_id=rid)

    assert store.get_run(rid)["status"] == "completed"
    events = store.list_events(rid)
    assert any(event["type"] == "library_artifact_warning" for event in events)


def test_snapshot_failure_isolation_emits_library_warning(fake_llm, fake_search, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.configure_runtime_for_tests(llm=fake_llm, search=fake_search)
    sid = service.store.create_session(title="t")

    class ExplodingSnapshotWriter:
        def write_text_snapshot(self, **_kwargs):  # noqa: ANN003
            raise RuntimeError("snapshot boom")

    service._snapshot_writer = ExplodingSnapshotWriter()  # noqa: SLF001

    outcome = service.orchestrator.run(
        RunRequest(query="state of quantum computing", mode=Mode.DEEP, session_id=sid)
    )

    assert service.store.get_run(outcome.run_id)["status"] == "completed"
    events = service.store.list_events(outcome.run_id)
    assert any(
        event["type"] == "library_artifact_warning"
        and event["payload"].get("stage") == "citation_snapshot"
        for event in events
    )


class _ChangingFetchSearch:
    name = "changing-fetch"

    def __init__(self, *, fail_refetch: bool = False) -> None:
        self.fetches = 0
        self.fail_refetch = fail_refetch

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001
        return [SearchHit(title="Quantum", url="https://example.com/live", snippet="old quantum baseline", rank=1)]

    def fetch(self, url: str):
        self.fetches += 1
        if self.fetches > 1 and self.fail_refetch:
            raise RuntimeError("live fetch failed")
        text = "old quantum baseline" if self.fetches == 1 else "live quantum update"
        return Document(url=url, title="Quantum", text=text, content_type="web")


def test_citation_snapshot_reuses_adopted_text_without_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch()
    service.configure_runtime_for_tests(llm=fake_llm, search=search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum", mode=Mode.DEEP, session_id=sid))

    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert search.fetches == 0
    assert preview["text"] == "old quantum baseline"


def test_council_citation_snapshot_reuses_adopted_text_without_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch()
    service.configure_runtime_for_tests(
        llm=fake_llm,
        search=search,
        council_members=[FakeLLM("m-a"), FakeLLM("m-b")],
    )
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum council", mode=Mode.COUNCIL, session_id=sid))

    assert search.fetches == 0
    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert preview["text"] == "old quantum baseline"


def test_fast_citation_snapshot_reuses_adopted_text_without_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch()
    service.configure_runtime_for_tests(llm=fake_llm, search=search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum fast", mode=Mode.FAST, session_id=sid))

    assert search.fetches == 0
    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert preview["text"] == "old quantum baseline"
    events = service.store.list_events(outcome.run_id)
    assert not any(
        event["type"] == "library_artifact_warning"
        and event["payload"].get("stage") == "citation_refetch"
        for event in events
    )


def test_multiple_citation_snapshot_reuses_adopted_text_without_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch()
    service.configure_runtime_for_tests(llm=fake_llm, search=search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum multiple", mode=Mode.MULTIPLE, session_id=sid))

    assert search.fetches == 0
    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert preview["text"] == "old quantum baseline"


def test_deep_snapshot_does_not_call_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch(fail_refetch=True)
    service.configure_runtime_for_tests(llm=fake_llm, search=search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum", mode=Mode.DEEP, session_id=sid))

    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert preview["text"] == "old quantum baseline"
    events = service.store.list_events(outcome.run_id)
    assert search.fetches == 0
    assert not any(
        event["type"] == "library_artifact_warning"
        and event["payload"].get("stage") == "citation_refetch"
        for event in events
    )


def test_fast_snapshot_does_not_call_post_terminal_refetch(fake_llm, tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    search = _ChangingFetchSearch(fail_refetch=True)
    service.configure_runtime_for_tests(llm=fake_llm, search=search)
    sid = service.store.create_session(title="t")

    outcome = service.orchestrator.run(RunRequest(query="quantum fast", mode=Mode.FAST, session_id=sid))

    doc_id = outcome.answer.citations[0].document_id
    snapshots = service.store.list_page_snapshots_for_document(doc_id)
    preview = service.get_library_snapshot(snapshots[0]["id"])["snapshot"]
    assert preview["text"] == "old quantum baseline"
    events = service.store.list_events(outcome.run_id)
    assert search.fetches == 0
    assert not any(
        event["type"] == "library_artifact_warning"
        and event["payload"].get("stage") == "citation_refetch"
        for event in events
    )
