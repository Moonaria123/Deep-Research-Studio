"""T46 citation schema migration and rich persistence contracts."""

from __future__ import annotations

import sqlite3

from deepresearch_core.baseline.citations import Citation
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.store import Store


def _create_legacy_citation_db(path: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE citations (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            document_id TEXT,
            url TEXT,
            title TEXT,
            snippet TEXT,
            score REAL,
            created_at REAL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO citations(
            id, run_id, document_id, url, title, snippet, score, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "cit_legacy",
            "run_legacy",
            "doc_legacy",
            "https://example.test/legacy",
            "Legacy source",
            "Legacy evidence",
            0.75,
            1_725_000_000.0,
        ),
    )
    connection.commit()
    connection.close()


def test_legacy_citation_table_is_migrated_and_backfilled_idempotently(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    _create_legacy_citation_db(str(db_path))

    store = Store(db_path)
    first = store.list_citations("run_legacy")
    columns = {
        row["name"] for row in store._conn.execute("PRAGMA table_info(citations)").fetchall()  # noqa: SLF001
    }
    store.close()

    reopened = Store(db_path)
    second = reopened.list_citations("run_legacy")
    reopened.close()

    assert {
        "origin",
        "artifact_type",
        "artifact_id",
        "grounding_eligible",
        "provenance_refs",
        "source_status",
        "published_at",
        "updated_at",
        "effective_at",
        "retrieved_at",
    }.issubset(columns)
    assert first == second
    assert len(first) == 1
    assert first[0]["origin"] == "external"
    assert first[0]["artifact_type"] == "web_document"
    assert first[0]["artifact_id"] == "doc_legacy"
    assert first[0]["grounding_eligible"] is True
    assert first[0]["provenance_refs"] == []
    assert first[0]["source_status"] == "active"
    assert first[0]["retrieved_at"] == first[0]["created_at"]


def test_rich_internal_citation_round_trips_without_creating_library_source(tmp_path):
    store = Store(tmp_path / "rich.sqlite3")
    sources_before = store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]  # noqa: SLF001

    store.add_citation(
        run_id="run_memory",
        document_id="",
        url="",
        title="Archived memory",
        snippet="The internal launch target was 2026-07-01.",
        score=0.9,
        origin="memory",
        artifact_type="memory",
        artifact_id="mem_1",
        grounding_eligible=True,
        provenance_refs=["memory:memory:mem_1"],
        source_status="archived",
        effective_at=1_751_328_000.0,
        retrieved_at=1_752_364_800.0,
    )

    citation = store.list_citations("run_memory")[0]
    sources_after = store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]  # noqa: SLF001
    store.close()

    assert citation["url"] == ""
    assert citation["origin"] == "memory"
    assert citation["artifact_type"] == "memory"
    assert citation["artifact_id"] == "mem_1"
    assert citation["source_status"] == "archived"
    assert citation["provenance_refs"] == ["memory:memory:mem_1"]
    assert citation["effective_at"] == 1_751_328_000.0
    assert citation["retrieved_at"] == 1_752_364_800.0
    assert sources_after == sources_before


def test_orchestrator_persists_all_rich_citation_fields():
    store = Store(":memory:")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._store = store  # noqa: SLF001 - narrow persistence integration contract
    citation = Citation(
        id="c1",
        document_id="",
        url="",
        title="Library note",
        snippet="Approved target.",
        score=0.8,
        origin="local",
        artifact_type="library_note",
        artifact_id="note_7",
        grounding_eligible=True,
        provenance_refs=("local:library_note:note_7",),
        source_status="active",
        published_at=100.0,
        updated_at=110.0,
        effective_at=90.0,
        retrieved_at=120.0,
    )

    orchestrator._persist_citations("run_local", [citation])  # noqa: SLF001

    persisted = store.list_citations("run_local")[0]
    store.close()

    assert persisted["origin"] == "local"
    assert persisted["artifact_type"] == "library_note"
    assert persisted["artifact_id"] == "note_7"
    assert persisted["provenance_refs"] == ["local:library_note:note_7"]
    assert persisted["published_at"] == 100.0
    assert persisted["updated_at"] == 110.0
    assert persisted["effective_at"] == 90.0
    assert persisted["retrieved_at"] == 120.0
