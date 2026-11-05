"""T56 local document import, evidence, and constrained deletion contracts."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import threading
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import Workbook
from pypdf import PdfWriter

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline import local_document_import as local_import
from deepresearch_core.baseline.adapters import Document
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.knowledge_graph.builder import KnowledgeGraphBuilder
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import DocumentWriteRejected
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM, FakeSearch


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../CON.pdf", "_CON.pdf"),
        ("folder\\LPT1.xlsx", "_LPT1.xlsx"),
        ("report:quarter?one*.docx", "report_quarter_one_.docx"),
        ("\u202e hidden.pdf ", "hidden.pdf"),
    ],
)
def test_import_filename_is_safe_for_private_cross_platform_temp_files(raw, expected):
    assert local_import.sanitize_import_filename(raw) == expected


def _client_with_service(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        snapshot_root=tmp_path / "snapshots",
    )
    return TestClient(create_app(service)), service


def _docx_bytes(text: str) -> bytes:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p><w:sectPr/></w:body>
</w:document>"""
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
    return payload.getvalue()


def _xlsx_bytes(
    *,
    visible: str,
    hidden: str = "",
    hidden_row: str = "",
    hidden_column: str = "",
    formula_without_cache: bool = False,
) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Visible"
    sheet["A1"] = visible
    if hidden_row:
        sheet["A2"] = hidden_row
        sheet.row_dimensions[2].hidden = True
    if hidden_column:
        sheet["B1"] = hidden_column
        sheet.column_dimensions["B"].hidden = True
    if formula_without_cache:
        sheet["C1"] = "=1+1"
    if hidden:
        hidden_sheet = workbook.create_sheet("Hidden")
        hidden_sheet.sheet_state = "hidden"
        hidden_sheet["A1"] = hidden
    payload = BytesIO()
    workbook.save(payload)
    workbook.close()
    return payload.getvalue()


def _rewrite_zip(
    content: bytes,
    *,
    replacements: dict[str, bytes] | None = None,
    additions: dict[str, bytes] | None = None,
) -> bytes:
    replacements = replacements or {}
    additions = additions or {}
    target = BytesIO()
    with zipfile.ZipFile(BytesIO(content)) as source, zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as destination:
        for item in source.infolist():
            destination.writestr(item.filename, replacements.get(item.filename, source.read(item.filename)))
        for name, payload in additions.items():
            destination.writestr(name, payload)
    return target.getvalue()


def _encrypted_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("test-password")
    payload = BytesIO()
    writer.write(payload)
    return payload.getvalue()


def _pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


@pytest.mark.parametrize(
    ("filename", "mime_type", "content", "needle"),
    [
        ("evidence.pdf", "application/pdf", _pdf_bytes("T56 PDF authority marker"), "authority"),
        (
            "evidence.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _docx_bytes("T56 DOCX evidence marker"),
            "DOCX",
        ),
        (
            "evidence.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            _xlsx_bytes(visible="T56 XLSX evidence marker", hidden="do not index hidden secret"),
            "XLSX",
        ),
    ],
)
def test_import_is_local_lexical_only_and_idempotent(
    tmp_path,
    monkeypatch,
    filename,
    mime_type,
    content,
    needle,
):
    client, service = _client_with_service(tmp_path)
    embed_calls = 0

    def fail_if_embedded(_pieces):
        nonlocal embed_calls
        embed_calls += 1
        raise AssertionError("local import must not request embeddings")

    monkeypatch.setattr(service._indexer, "_embed", fail_if_embedded)

    first = client.post(
        "/api/v1/library/documents/import",
        data={"title": "First title"},
        files={"file": (filename, content, mime_type)},
    )
    duplicate = client.post(
        "/api/v1/library/documents/import",
        data={"title": "Replacement title"},
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert first.status_code == 200, first.text
    assert duplicate.status_code == 200, duplicate.text
    created = first.json()
    existing = duplicate.json()
    assert created["status"] == "created"
    assert existing["status"] == "existing"
    assert existing["document"]["id"] == created["document"]["id"]
    assert existing["document"]["title"] == "First title"
    assert created["document"]["origin"] == "local"
    assert created["document"]["artifact_type"] == "library_document"
    assert created["document"]["content_sha256"]
    assert created["document"]["chunk_count"] > 0
    assert embed_calls == 0

    listed = client.get("/api/v1/library/documents").json()["documents"]
    assert [item["id"] for item in listed].count(created["document"]["id"]) == 1
    assert listed[0]["filename"] == filename
    assert listed[0]["deletable"] is True

    local = service._library_retriever.retrieve_evidence(needle, top_k=5)
    assert any(
        item.document_id == created["document"]["id"]
        and item.origin == "local"
        and item.artifact_type == "library_document"
        and item.artifact_id == created["document"]["id"]
        and item.url == ""
        for item in local
    )
    generic = Retriever(service.store).retrieve(needle, top_k=5)
    assert all(item.document_id != created["document"]["id"] for item in generic)
    if filename.endswith(".xlsx"):
        assert not service.store.search_chunks_lexical("secret", limit=5)


def test_import_rejects_size_type_and_textless_documents(tmp_path):
    client, _service = _client_with_service(tmp_path)

    too_large = client.post(
        "/api/v1/library/documents/import",
        headers={"Content-Length": "1"},
        files={"file": ("large.pdf", b"%PDF-" + b"x" * (10 * 1024 * 1024), "application/pdf")},
    )
    mismatch = client.post(
        "/api/v1/library/documents/import",
        files={"file": ("fake.pdf", _docx_bytes("not a PDF"), "application/pdf")},
    )
    textless = client.post(
        "/api/v1/library/documents/import",
        files={"file": ("empty.xlsx", _xlsx_bytes(visible=""), "application/octet-stream")},
    )

    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "DOCUMENT_TOO_LARGE"
    assert mismatch.status_code == 415
    assert mismatch.json()["error"]["code"] == "DOCUMENT_TYPE_UNSUPPORTED"
    assert textless.status_code == 422
    assert textless.json()["error"]["code"] == "DOCUMENT_NO_TEXT"


@pytest.mark.parametrize(
    ("filename", "content", "expected_code"),
    [
        (
            "unsafe.docx",
            _rewrite_zip(_docx_bytes("safe"), additions={"../escape.xml": b"unsafe"}),
            "DOCUMENT_LIMIT_EXCEEDED",
        ),
        (
            "macro.docx",
            _rewrite_zip(_docx_bytes("safe"), additions={"word/vbaProject.bin": b"macro"}),
            "DOCUMENT_TYPE_UNSUPPORTED",
        ),
        (
            "ratio.docx",
            _rewrite_zip(_docx_bytes("safe"), additions={"word/bomb.bin": b"0" * 20_000}),
            "DOCUMENT_LIMIT_EXCEEDED",
        ),
        (
            "encrypted.pdf",
            _encrypted_pdf_bytes(),
            "DOCUMENT_ENCRYPTED",
        ),
    ],
)
def test_import_rejects_unsafe_packages_and_encrypted_pdf(
    tmp_path,
    filename,
    content,
    expected_code,
):
    client, _service = _client_with_service(tmp_path)

    response = client.post(
        "/api/v1/library/documents/import",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code in {415, 422}
    assert response.json()["error"]["code"] == expected_code


def test_import_rejects_duplicate_ooxml_entries(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = _rewrite_zip(
            _docx_bytes("original"),
            additions={
                "word/document.xml": (
                    b"<?xml version=\"1.0\"?><w:document "
                    b"xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"/>"
                )
            },
        )
    client, _service = _client_with_service(tmp_path)

    response = client.post(
        "/api/v1/library/documents/import",
        files={"file": ("duplicate.docx", duplicate, "application/octet-stream")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DOCUMENT_LIMIT_EXCEEDED"


def test_xlsx_indexes_only_visible_cached_values_and_reports_missing_formula_cache(tmp_path):
    client, service = _client_with_service(tmp_path)
    response = client.post(
        "/api/v1/library/documents/import",
        files={
            "file": (
                "visibility.xlsx",
                _xlsx_bytes(
                    visible="visible T56 spreadsheet evidence",
                    hidden="HIDDENSHEETMARKER",
                    hidden_row="HIDDENROWMARKER",
                    hidden_column="HIDDENCOLUMNMARKER",
                    formula_without_cache=True,
                ),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200, response.text
    assert "formula_cache_missing" in response.json()["warnings"]
    assert service.store.search_chunks_lexical("visible", limit=5)
    for secret in ("HIDDENSHEETMARKER", "HIDDENROWMARKER", "HIDDENCOLUMNMARKER"):
        assert not service.store.search_chunks_lexical(secret, limit=5)


def test_xlsx_extraction_enforces_character_limit_incrementally(tmp_path, monkeypatch):
    workbook = Workbook()
    worksheet = workbook.active
    for row in range(1, 101):
        worksheet.cell(row=row, column=1, value=f"row-{row}-" + ("x" * 40))
    source = tmp_path / "bounded.xlsx"
    workbook.save(source)
    workbook.close()
    monkeypatch.setattr(local_import, "MAX_EXTRACTED_CHARS", 256)

    text, warnings = local_import._parse_xlsx(source)

    assert len(text) <= 256
    assert "text_truncated" in warnings


def test_parser_timeout_cleans_its_private_temporary_directory(tmp_path, monkeypatch):
    real_temporary_directory = tempfile.TemporaryDirectory

    def isolated_temporary_directory(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return real_temporary_directory(*args, dir=tmp_path, **kwargs)

    def time_out(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or args[0], timeout=0.1)

    monkeypatch.setattr(local_import.tempfile, "TemporaryDirectory", isolated_temporary_directory)
    monkeypatch.setattr(local_import.subprocess, "run", time_out)

    with pytest.raises(BaselineError) as exc:
        local_import.parse_local_document(
            _docx_bytes("timeout marker"),
            filename="timeout.docx",
            mime_type="application/octet-stream",
            timeout_s=0.1,
        )

    assert exc.value.code is ErrorCode.DOCUMENT_PARSE_FAILED
    assert exc.value.details["reason"] == "parser_timeout"
    assert list(tmp_path.iterdir()) == []


def test_import_parsing_runs_off_the_api_event_loop(tmp_path, monkeypatch):
    import httpx2 as httpx

    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        snapshot_root=tmp_path / "snapshots",
    )
    parser_started = threading.Event()
    release_parser = threading.Event()

    def blocking_import(_payload):
        parser_started.set()
        assert release_parser.wait(timeout=2)
        return {
            "status": "created",
            "document": {
                "id": "doc-event-loop",
                "title": "Event loop",
                "filename": "event-loop.docx",
                "content_type": "docx",
                "origin": "local",
                "artifact_type": "library_document",
                "content_sha256": "a" * 64,
                "chunk_count": 1,
            },
            "warnings": [],
        }

    monkeypatch.setattr(service, "import_local_document", blocking_import)

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            import_task = asyncio.create_task(
                client.post(
                    "/api/v1/library/documents/import",
                    files={
                        "file": (
                            "event-loop.docx",
                            _docx_bytes("event loop marker"),
                            "application/octet-stream",
                        )
                    },
                )
            )
            assert await asyncio.to_thread(parser_started.wait, 1)
            health = await asyncio.wait_for(client.get("/api/v1/health"), timeout=0.5)
            assert health.status_code == 200
            release_parser.set()
            imported = await asyncio.wait_for(import_task, timeout=1)
            assert imported.status_code == 200

    try:
        asyncio.run(exercise())
    finally:
        release_parser.set()
        service.store.close()


@pytest.mark.parametrize("mode", [Mode.FAST, Mode.MULTIPLE, Mode.DEEP, Mode.COUNCIL])
def test_all_research_modes_cite_imported_documents_as_local_evidence(tmp_path, mode):
    client, service = _client_with_service(tmp_path)
    imported = client.post(
        "/api/v1/library/documents/import",
        files={
            "file": (
                "mode-evidence.docx",
                _docx_bytes(
                    "T56ModeMarker local document evidence confirms the project launch target is July 2026."
                ),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    ).json()["document"]
    llm = FakeLLM(f"writer-{mode.value}")
    orchestrator = Orchestrator(
        store=service.store,
        llm=llm,
        retriever=Retriever(service.store, embedder=llm),
        indexer=Indexer(service.store, embedder=llm),
        search=FakeSearch({}),
        memory=MemoryEngine(service.store, embedder=llm),
        library_retriever=service._library_retriever,
        council_members=[FakeLLM("member-a"), FakeLLM("member-b")],
        enforce_grounding_gate=True,
    )

    outcome = orchestrator.run(
        RunRequest(
            query="T56ModeMarker project launch target July 2026",
            mode=mode,
            library_scope="off",
            memory_policy="off",
        )
    )

    local_citations = [
        citation
        for citation in outcome.answer.citations
        if citation.artifact_type == "library_document"
    ]
    assert outcome.status in {"completed", "completed_degraded"}
    assert local_citations
    assert all(citation.origin == "local" for citation in local_citations)
    assert all(citation.artifact_id == imported["id"] for citation in local_citations)
    assert all(citation.url == "" for citation in local_citations)
    assert all(
        source["artifact_type"] != "web_document"
        for source in outcome.to_dict()["sources"]
    )


def test_delete_cascades_current_state_but_preserves_history_and_shared_snapshot(tmp_path):
    client, service = _client_with_service(tmp_path)
    imported = client.post(
        "/api/v1/library/documents/import",
        files={
            "file": (
                "delete-me.docx",
                _docx_bytes("T56 deletion evidence marker"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    ).json()["document"]
    document_id = imported["id"]
    service.store.add_citation(
        run_id="historical-run",
        document_id=document_id,
        url=None,
        title=imported["title"],
        snippet="T56 deletion evidence marker",
        score=1.0,
        origin="local",
        artifact_type="library_document",
        artifact_id=document_id,
        provenance_refs=[f"local:library_document:{document_id}"],
    )

    shared = service._snapshot_writer.write_text_snapshot(
        url="https://example.test/shared",
        title="Shared",
        text="shared snapshot",
    )
    external_id, _ = service.store.upsert_document_with_chunks(
        url="https://example.test/shared",
        title="Shared external",
        text="shared external text",
        chunks=[("shared external text", None)],
    )
    service.store.add_page_snapshot(
        document_id=document_id,
        url="",
        content_hash=shared.content_hash,
        storage_path=shared.storage_path,
    )
    service.store.add_page_snapshot(
        document_id=external_id,
        url="https://example.test/shared",
        content_hash=shared.content_hash,
        storage_path=shared.storage_path,
    )

    invalidations = 0

    class _VectorBackend:
        name = "test"

        def invalidate(self):
            nonlocal invalidations
            invalidations += 1

        def health(self):
            return {"vector_backend": self.name}

        def search(self, _query_embedding, *, limit):
            del limit
            return []

    service._vector_backend = _VectorBackend()
    response = client.delete(f"/api/v1/library/documents/{document_id}")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "deleted"
    assert service.store.get_document(document_id) is None
    assert not service.store.search_chunks_lexical("deletion", limit=5)
    assert service.store.list_page_snapshots_for_document(document_id) == []
    assert service.store.list_library_provenance(
        artifact_type="library_document",
        artifact_id=document_id,
    ) == []
    citations = service.store.list_citations("historical-run")
    assert citations[0]["artifact_id"] == document_id
    assert shared.storage_path
    assert service._snapshot_writer.resolve(shared.storage_path).is_file()
    assert invalidations == 1

    repeated = client.delete(f"/api/v1/library/documents/{document_id}")
    assert repeated.status_code == 404
    assert repeated.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_delete_cleanup_warning_is_recovered_by_next_startup_sweep(tmp_path, monkeypatch):
    db_path = tmp_path / "recovery.sqlite3"
    snapshot_root = tmp_path / "snapshots"
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=db_path,
        snapshot_root=snapshot_root,
    )
    monkeypatch.setattr(service, "_run_knowledge_graph_refresh_safe", lambda **_kwargs: None)
    snapshot = service._snapshot_writer.write_text_snapshot(
        url="https://example.test/orphan",
        title="Orphan recovery",
        text="snapshot cleanup recovery marker",
    )
    document_id, _ = service.store.upsert_document_with_chunks(
        url="https://example.test/orphan",
        title="Orphan recovery",
        text="snapshot cleanup recovery marker",
        chunks=[("snapshot cleanup recovery marker", None)],
    )
    service.store.add_page_snapshot(
        document_id=document_id,
        url="https://example.test/orphan",
        content_hash=snapshot.content_hash,
        storage_path=snapshot.storage_path,
    )
    snapshot_path = service._snapshot_writer.resolve(snapshot.storage_path)

    with monkeypatch.context() as cleanup_failure:
        cleanup_failure.setattr(
            service._snapshot_writer,
            "delete_unreferenced",
            lambda *_args, **_kwargs: {"deleted": 0, "warnings": ["snapshot_unlink_failed"]},
        )
        deleted = service.delete_library_document(document_id)

    assert deleted["status"] == "deleted_with_cleanup_warning"
    assert deleted["warnings"] == ["snapshot_unlink_failed"]
    assert snapshot_path.is_file()
    service.store.close()

    restarted = BaselineService(
        provider=ProviderSettings(),
        db_path=db_path,
        snapshot_root=snapshot_root,
    )
    try:
        assert not snapshot_path.exists()
    finally:
        restarted.store.close()


def test_delete_tombstone_blocks_stale_writers_but_allows_explicit_reimport(tmp_path):
    client, service = _client_with_service(tmp_path)
    content = _docx_bytes("T56 stale writer protection marker")
    imported = client.post(
        "/api/v1/library/documents/import",
        files={"file": ("race.docx", content, "application/octet-stream")},
    ).json()["document"]
    document_id = imported["id"]
    stored = service.store.get_document(document_id)
    assert stored is not None
    library_uri = str(stored["url"])
    stale_prepared = service._indexer.prepare_document(
        Document(
            url=library_uri,
            title="Stale explicit import",
            text="stale explicit import must not return",
            content_type="docx",
            metadata={
                "source": "local_import",
                "origin": "local",
                "artifact_type": "library_document",
            },
        ),
        include_embeddings=False,
    )
    marker_seen_before_delete = service.store.get_document_deletion_marker(library_uri)
    assert marker_seen_before_delete is None

    deleted = client.delete(f"/api/v1/library/documents/{document_id}")
    assert deleted.status_code == 200

    with pytest.raises(DocumentWriteRejected):
        service._indexer.commit_local_import(
            stale_prepared,
            content_sha256=imported["content_sha256"],
            filename="race.docx",
            expected_deletion_marker=marker_seen_before_delete,
        )
    with pytest.raises(DocumentWriteRejected):
        service.store.upsert_document_with_chunks(
            url=library_uri,
            title="Stale import",
            text="stale import must not return",
            chunks=[("stale import must not return", None)],
        )
    assert (
        service.store.replace_document_chunks_if_present(
            expected_document_id=document_id,
            url=library_uri,
            text="stale rebuild must not return",
            chunks=[("stale rebuild must not return", None)],
            doc_hash="stale",
        )
        is None
    )
    assert service.store.get_document(document_id) is None

    reimported = client.post(
        "/api/v1/library/documents/import",
        files={"file": ("race.docx", content, "application/octet-stream")},
    )
    assert reimported.status_code == 200, reimported.text
    assert reimported.json()["status"] == "created"
    assert reimported.json()["document"]["id"] != document_id
    assert service.store.get_document(reimported.json()["document"]["id"]) is not None


def test_snapshot_materialization_cannot_late_write_after_delete(tmp_path, monkeypatch):
    _client, service = _client_with_service(tmp_path)
    url = "https://example.test/t56-snapshot-race"
    document_id, _ = service.store.upsert_document_with_chunks(
        url=url,
        title="Snapshot race",
        text="snapshot race evidence",
        chunks=[("snapshot race evidence", None)],
    )
    original_commit = service.store.add_page_snapshot_with_provenance_if_current

    def delete_then_commit(**kwargs):
        deleted = service.store.delete_library_source_document(document_id)
        assert deleted is not None and deleted["deletable"] is True
        return original_commit(**kwargs)

    monkeypatch.setattr(
        service.store,
        "add_page_snapshot_with_provenance_if_current",
        delete_then_commit,
    )

    with pytest.raises(DocumentWriteRejected):
        service.capture_document_snapshot(document_id=document_id, run_id="run-race")

    assert service.store.get_document(document_id) is None
    assert service.store.list_snapshot_storage_paths() == set()
    assert service.store.list_library_provenance_for_document(document_id) == []
    assert list((tmp_path / "snapshots").rglob("*.txt")) == []


def test_snapshot_cleanup_and_new_materialization_are_serialized(tmp_path, monkeypatch):
    _client, service = _client_with_service(tmp_path)
    monkeypatch.setattr(service, "_run_knowledge_graph_refresh_safe", lambda **_kwargs: None)
    deleted_id, _ = service.store.upsert_document_with_chunks(
        url="https://example.test/t56-cleanup-lock",
        title="Cleanup lock",
        text="cleanup lock evidence",
        chunks=[("cleanup lock evidence", None)],
    )
    retained_id, _ = service.store.upsert_document_with_chunks(
        url="https://example.test/t56-materialize-lock",
        title="Materialize lock",
        text="materialize lock evidence",
        chunks=[("materialize lock evidence", None)],
    )
    service.capture_document_snapshot(document_id=deleted_id)
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    capture_finished = threading.Event()
    original_delete = service._snapshot_writer.delete_unreferenced

    def blocked_delete(*args, **kwargs):
        cleanup_entered.set()
        assert release_cleanup.wait(timeout=2)
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(service._snapshot_writer, "delete_unreferenced", blocked_delete)
    delete_thread = threading.Thread(
        target=service.delete_library_document,
        args=(deleted_id,),
    )

    def capture() -> None:
        service.capture_document_snapshot(document_id=retained_id)
        capture_finished.set()

    capture_thread = threading.Thread(target=capture)
    delete_thread.start()
    assert cleanup_entered.wait(timeout=1)
    capture_thread.start()
    assert not capture_finished.wait(timeout=0.1)
    release_cleanup.set()
    delete_thread.join(timeout=2)
    capture_thread.join(timeout=2)

    assert not delete_thread.is_alive()
    assert not capture_thread.is_alive()
    assert capture_finished.is_set()
    retained_snapshots = service.list_source_snapshots(retained_id)["snapshots"]
    assert len(retained_snapshots) == 1
    retained_paths = service.store.list_snapshot_storage_paths()
    assert len(retained_paths) == 1
    assert service._snapshot_writer.resolve(next(iter(retained_paths))).is_file()


def test_delete_rejects_hidden_and_derived_library_documents(tmp_path):
    client, service = _client_with_service(tmp_path)
    hidden = service._indexer.index_document(
        Document(
            url="https://example.test/hidden",
            title="Hidden candidate",
            text="hidden candidate",
        ),
        adoption_run_id="run-hidden",
    )
    note = service.add_library_note({"content": "note content"})
    provenance = service.store.list_library_provenance(
        artifact_type="note",
        artifact_id=note["note_id"],
    )
    note_document_id = provenance[0]["document_id"]
    library_indexer = LibraryIndexer(service.store, indexer=service._indexer)
    report = library_indexer.index_report(
        report_id="report-t56",
        title="Protected report",
        body_md="# Protected report\n\nReport content.",
        mode="deep",
    )
    memory = library_indexer.index_memory(
        memory_id="memory-t56",
        namespace="workspace",
        scope_key="default",
        content="Protected memory content.",
    )

    hidden_response = client.delete(f"/api/v1/library/documents/{hidden['document_id']}")
    note_response = client.delete(f"/api/v1/library/documents/{note_document_id}")
    report_response = client.delete(f"/api/v1/library/documents/{report['document_id']}")
    memory_response = client.delete(f"/api/v1/library/documents/{memory['document_id']}")

    assert hidden_response.status_code == 409
    assert hidden_response.json()["error"]["code"] == "DOCUMENT_NOT_DELETABLE"
    assert note_response.status_code == 409
    assert note_response.json()["error"]["code"] == "DOCUMENT_NOT_DELETABLE"
    assert report_response.status_code == 409
    assert report_response.json()["error"]["code"] == "DOCUMENT_NOT_DELETABLE"
    assert memory_response.status_code == 409
    assert memory_response.json()["error"]["code"] == "DOCUMENT_NOT_DELETABLE"
    assert service.store.get_document(hidden["document_id"]) is not None
    assert service.store.get_document(note_document_id) is not None
    assert service.store.get_document(report["document_id"]) is not None
    assert service.store.get_document(memory["document_id"]) is not None


@pytest.mark.parametrize("origin", ["local", "external"])
def test_delete_removes_current_graph_projection_but_preserves_history(tmp_path, origin):
    client, service = _client_with_service(tmp_path)
    if origin == "local":
        document = client.post(
            "/api/v1/library/documents/import",
            files={
                "file": (
                    "graph.docx",
                    _docx_bytes("T56 graph projection evidence."),
                    "application/octet-stream",
                )
            },
        ).json()["document"]
        document_id = document["id"]
        artifact_type = "library_document"
        citation_url = None
    else:
        document_id, _ = service.store.upsert_document_with_chunks(
            url="https://example.test/t56-delete",
            title="T56 external source",
            text="T56 external graph projection evidence.",
            chunks=[("T56 external graph projection evidence.", None)],
        )
        artifact_type = "web_document"
        citation_url = "https://example.test/t56-delete"

    session_id = service.store.create_session(title=f"T56 {origin}", workspace="default")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="T56 graph")
    service.store.add_citation(
        run_id=run_id,
        document_id=document_id,
        url=citation_url,
        title=f"T56 {origin} evidence",
        snippet="T56 graph projection evidence.",
        score=1.0,
        origin=origin,
        artifact_type=artifact_type,
        artifact_id=document_id,
        provenance_refs=[f"{origin}:{artifact_type}:{document_id}"],
    )
    before = KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    before_nodes = {
        row["node_key"]
        for row in service.store._conn.execute(
            "SELECT node_key FROM kg_nodes WHERE version_id=?",
            (before["version_id"],),
        ).fetchall()
    }
    assert f"source:{document_id}" in before_nodes

    response = client.delete(f"/api/v1/library/documents/{document_id}")
    assert response.status_code == 200, response.text
    after = KnowledgeGraphBuilder(service.store).rebuild(workspace_id="default")
    after_nodes = {
        row["node_key"]
        for row in service.store._conn.execute(
            "SELECT node_key FROM kg_nodes WHERE version_id=?",
            (after["version_id"],),
        ).fetchall()
    }

    assert f"source:{document_id}" not in after_nodes
    assert service.store.list_citations(run_id)
    historical = service.store._conn.execute(
        "SELECT COUNT(*) AS count FROM kg_graph_versions WHERE id=?",
        (before["version_id"],),
    ).fetchone()
    assert historical["count"] == 1


def test_import_response_and_store_metadata_are_secret_free(tmp_path):
    client, service = _client_with_service(tmp_path)
    result = client.post(
        "/api/v1/library/documents/import",
        files={
            "file": (
                "metadata.xlsx",
                _xlsx_bytes(visible="safe metadata"),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert result.status_code == 200
    body = result.json()
    row = service.store.get_document(body["document"]["id"])
    metadata = json.loads(row["metadata"])
    serialized = json.dumps(body, sort_keys=True)

    assert metadata["source"] == "local_import"
    assert metadata["artifact_type"] == "library_document"
    assert "content" not in body["document"]
    assert "storage_path" not in serialized


def test_docker_api_runtime_includes_declared_document_parsers():
    requirements = (
        Path(__file__).resolve().parents[2] / "docker" / "requirements-api.txt"
    ).read_text(encoding="utf-8").lower()

    assert "pypdf" in requirements
    assert "openpyxl" in requirements
