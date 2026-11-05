"""SQLite persistence for the baseline (10 tables + FTS5 + job store).

Uses only the standard-library ``sqlite3`` module so the store works out of the box.
FTS5 is used when the local SQLite build supports it; otherwise lexical search falls
back to ``LIKE``. Secrets are never stored here -- only secret-free provider status.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .deep_checkpoint import (
    CheckpointAlreadyContinued,
    CheckpointStateConflict,
    checkpoint_integrity,
    decode_deep_checkpoint,
)

JsonDict = dict[str, Any]
_SEARCH_USAGE_POLICY_REVISION = "_search_usage_policy_revision"
_RUN_ENQUEUE_CONFIRMED = "enqueue_confirmed"
_RUN_ENQUEUE_WORKER_EXPIRED = "_enqueue_worker_expired"
_DOCUMENT_TOMBSTONE_ARTIFACT_TYPE = "_document_delete_tombstone"


class StoreEncryptionError(RuntimeError):
    """Raised when an encryption key is supplied but cannot be honored.

    Covers the two unsafe situations the store must never resolve by silently
    falling back to a plaintext database (DR-ITER-023):
      * no SQLCipher driver is installed, so the key cannot encrypt anything;
      * the key is wrong (or the file is plaintext), so it cannot unlock the data.
    """


class SessionStateConflict(RuntimeError):
    """Raised when a session changed before an inherited run could be frozen."""


class DocumentWriteRejected(RuntimeError):
    """Raised when a late writer targets a document explicitly deleted by the user."""


def _now() -> float:
    return time.time()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _document_tombstone_key(url: str) -> str:
    return hashlib.sha256(str(url or "").encode("utf-8")).hexdigest()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT, workspace TEXT DEFAULT 'default',
    created_at REAL, updated_at REAL, metadata TEXT DEFAULT '{}',
    pinned INTEGER NOT NULL DEFAULT 0, mode_default TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
    created_at REAL, metadata TEXT DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, session_id TEXT, mode TEXT, query TEXT, status TEXT,
    budget TEXT DEFAULT '{}', result TEXT, error TEXT,
    created_at REAL, updated_at REAL, metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, seq INTEGER,
    type TEXT, payload TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, url TEXT UNIQUE, title TEXT, content_type TEXT,
    text TEXT, hash TEXT, created_at REAL, metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT, chunk_index INTEGER,
    text TEXT, embedding TEXT, token_count INTEGER, created_at REAL,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);
CREATE TABLE IF NOT EXISTS citations (
    id TEXT PRIMARY KEY, run_id TEXT, document_id TEXT, url TEXT, title TEXT,
    snippet TEXT, score REAL, created_at REAL,
    origin TEXT NOT NULL DEFAULT 'external',
    artifact_type TEXT NOT NULL DEFAULT 'web_document', artifact_id TEXT,
    grounding_eligible INTEGER NOT NULL DEFAULT 1,
    provenance_refs TEXT NOT NULL DEFAULT '[]',
    source_status TEXT NOT NULL DEFAULT 'active',
    published_at REAL, updated_at REAL, effective_at REAL, retrieved_at REAL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, namespace TEXT, scope_key TEXT, content TEXT,
    salience REAL DEFAULT 1.0, uses INTEGER DEFAULT 0,
    created_at REAL, last_used_at REAL, metadata TEXT DEFAULT '{}',
    archived_at REAL, pinned INTEGER NOT NULL DEFAULT 0, expires_at REAL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, title TEXT, body_md TEXT, mode TEXT,
    parent_report_id TEXT, session_id TEXT, run_id TEXT,
    created_at REAL, metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS page_snapshots (
    id TEXT PRIMARY KEY, document_id TEXT, url TEXT, captured_at REAL,
    content_hash TEXT, storage_path TEXT, mime_type TEXT,
    size_bytes INTEGER DEFAULT 0, status TEXT, error TEXT,
    run_id TEXT, metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS library_provenance (
    id TEXT PRIMARY KEY, artifact_type TEXT, artifact_id TEXT,
    session_id TEXT, run_id TEXT, document_id TEXT, url TEXT,
    created_at REAL, metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS kg_graph_versions (
    id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0,
    started_at REAL, finished_at REAL, activated_at REAL,
    base_snapshot_json TEXT DEFAULT '{}', stats_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS kg_nodes (
    id TEXT PRIMARY KEY, version_id TEXT NOT NULL, node_type TEXT NOT NULL,
    node_key TEXT NOT NULL, title TEXT, summary TEXT,
    attrs_json TEXT DEFAULT '{}', confidence REAL DEFAULT 1.0,
    visible INTEGER NOT NULL DEFAULT 1, created_at REAL,
    FOREIGN KEY(version_id) REFERENCES kg_graph_versions(id)
);
CREATE TABLE IF NOT EXISTS kg_edges (
    id TEXT PRIMARY KEY, version_id TEXT NOT NULL, edge_type TEXT NOT NULL,
    src_node_id TEXT NOT NULL, dst_node_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0, confidence REAL DEFAULT 1.0,
    visible INTEGER NOT NULL DEFAULT 1, attrs_json TEXT DEFAULT '{}', created_at REAL,
    FOREIGN KEY(version_id) REFERENCES kg_graph_versions(id),
    FOREIGN KEY(src_node_id) REFERENCES kg_nodes(id),
    FOREIGN KEY(dst_node_id) REFERENCES kg_nodes(id)
);
CREATE TABLE IF NOT EXISTS kg_provenance (
    id TEXT PRIMARY KEY, version_id TEXT NOT NULL,
    owner_kind TEXT NOT NULL, owner_id TEXT NOT NULL,
    source_kind TEXT NOT NULL, source_id TEXT NOT NULL,
    span_json TEXT DEFAULT '{}', evidence_json TEXT DEFAULT '{}', created_at REAL,
    FOREIGN KEY(version_id) REFERENCES kg_graph_versions(id)
);
CREATE TABLE IF NOT EXISTS kg_build_jobs (
    id TEXT PRIMARY KEY, version_id TEXT, workspace_id TEXT NOT NULL DEFAULT 'default',
    backend TEXT NOT NULL, backend_job_id TEXT, status TEXT NOT NULL,
    stage TEXT, progress REAL DEFAULT 0.0, started_at REAL, finished_at REAL,
    error_code TEXT, error_message TEXT, stats_json TEXT DEFAULT '{}',
    FOREIGN KEY(version_id) REFERENCES kg_graph_versions(id)
);
CREATE TABLE IF NOT EXISTS kg_wiki_snapshots (
    id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL DEFAULT 'default',
    graph_version_id TEXT, status TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0, stale_reason TEXT,
    started_at REAL, finished_at REAL, activated_at REAL,
    stats_json TEXT DEFAULT '{}', error_code TEXT, error_message TEXT,
    FOREIGN KEY(graph_version_id) REFERENCES kg_graph_versions(id)
);
CREATE TABLE IF NOT EXISTS kg_wiki_topics (
    id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, topic_key TEXT NOT NULL,
    title TEXT NOT NULL, summary TEXT, depth INTEGER DEFAULT 0, parent_id TEXT,
    session_ids_json TEXT DEFAULT '[]', source_message_ids_json TEXT DEFAULT '[]',
    attrs_json TEXT DEFAULT '{}', created_at REAL,
    FOREIGN KEY(snapshot_id) REFERENCES kg_wiki_snapshots(id),
    FOREIGN KEY(parent_id) REFERENCES kg_wiki_topics(id)
);
CREATE TABLE IF NOT EXISTS kg_wiki_tree_edges (
    id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL,
    src_topic_id TEXT NOT NULL, dst_topic_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'parent_child', attrs_json TEXT DEFAULT '{}', created_at REAL,
    FOREIGN KEY(snapshot_id) REFERENCES kg_wiki_snapshots(id),
    FOREIGN KEY(src_topic_id) REFERENCES kg_wiki_topics(id),
    FOREIGN KEY(dst_topic_id) REFERENCES kg_wiki_topics(id)
);
CREATE TABLE IF NOT EXISTS kg_wiki_cross_links (
    id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL,
    src_topic_id TEXT NOT NULL, dst_topic_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL, evidence_refs_json TEXT DEFAULT '[]',
    attrs_json TEXT DEFAULT '{}', created_at REAL,
    FOREIGN KEY(snapshot_id) REFERENCES kg_wiki_snapshots(id),
    FOREIGN KEY(src_topic_id) REFERENCES kg_wiki_topics(id),
    FOREIGN KEY(dst_topic_id) REFERENCES kg_wiki_topics(id)
);
CREATE TABLE IF NOT EXISTS kg_wiki_refresh_jobs (
    id TEXT PRIMARY KEY, snapshot_id TEXT, graph_version_id TEXT,
    workspace_id TEXT NOT NULL DEFAULT 'default', backend TEXT NOT NULL,
    backend_job_id TEXT, status TEXT NOT NULL, stage TEXT,
    progress REAL DEFAULT 0.0, started_at REAL, finished_at REAL,
    attempted_sessions INTEGER DEFAULT 0, succeeded_sessions INTEGER DEFAULT 0,
    skipped_sessions INTEGER DEFAULT 0, failed_sessions INTEGER DEFAULT 0,
    error_code TEXT, error_message TEXT, stats_json TEXT DEFAULT '{}',
    FOREIGN KEY(snapshot_id) REFERENCES kg_wiki_snapshots(id),
    FOREIGN KEY(graph_version_id) REFERENCES kg_graph_versions(id)
);
CREATE TABLE IF NOT EXISTS kg_job_events (
    job_kind TEXT NOT NULL,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_kind, job_id, seq)
);
CREATE TABLE IF NOT EXISTS memory_dream_runs (
    id TEXT PRIMARY KEY, workspace_key TEXT, user_key TEXT, status TEXT,
    started_at REAL, finished_at REAL, memories_created INTEGER DEFAULT 0,
    error TEXT, metadata TEXT DEFAULT '{}', backend TEXT NOT NULL DEFAULT 'thread',
    backend_job_id TEXT, updated_at REAL, cancel_requested_at REAL
);
CREATE TABLE IF NOT EXISTS post_turn_memory_runs (
    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, workspace_key TEXT NOT NULL,
    user_key TEXT NOT NULL, status TEXT NOT NULL, checked_at REAL NOT NULL,
    provider_ref TEXT, error_code TEXT, embedding_status TEXT
);
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY, llm_base_url TEXT, llm_model TEXT, embed_model TEXT,
    ollama_base_url TEXT, has_llm_key INTEGER DEFAULT 0, brave_ready INTEGER DEFAULT 0,
    firecrawl_ready INTEGER DEFAULT 0, updated_at REAL
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, value TEXT, expires_at REAL
);
CREATE TABLE IF NOT EXISTS deep_checkpoints (
    run_id TEXT PRIMARY KEY,
    lineage_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    disposition TEXT NOT NULL DEFAULT 'ready',
    claimed_child_run_id TEXT,
    claim_idempotency_key TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_citations_run ON citations(run_id);
CREATE INDEX IF NOT EXISTS idx_memories_ns ON memories(namespace, scope_key);
CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(run_id);
CREATE INDEX IF NOT EXISTS idx_reports_session ON reports(session_id);
CREATE INDEX IF NOT EXISTS idx_reports_parent ON reports(parent_report_id);
CREATE INDEX IF NOT EXISTS idx_page_snapshots_document ON page_snapshots(document_id);
CREATE INDEX IF NOT EXISTS idx_page_snapshots_url ON page_snapshots(url);
CREATE INDEX IF NOT EXISTS idx_library_provenance_artifact ON library_provenance(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_library_provenance_session ON library_provenance(session_id);
CREATE INDEX IF NOT EXISTS idx_library_provenance_run ON library_provenance(run_id);
CREATE INDEX IF NOT EXISTS idx_library_provenance_document ON library_provenance(document_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_active_version ON kg_graph_versions(workspace_id) WHERE is_active=1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_nodes_key ON kg_nodes(version_id, node_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_edges_key ON kg_edges(version_id, edge_type, src_node_id, dst_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_jobs_workspace_status ON kg_build_jobs(workspace_id, status, started_at);
CREATE INDEX IF NOT EXISTS idx_kg_provenance_owner ON kg_provenance(version_id, owner_kind, owner_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_wiki_active_snapshot ON kg_wiki_snapshots(workspace_id) WHERE is_active=1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_wiki_topic_key ON kg_wiki_topics(snapshot_id, topic_key);
CREATE INDEX IF NOT EXISTS idx_kg_wiki_topics_snapshot ON kg_wiki_topics(snapshot_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_kg_wiki_tree_edges_snapshot ON kg_wiki_tree_edges(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_kg_wiki_cross_links_snapshot ON kg_wiki_cross_links(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_kg_wiki_jobs_workspace_status ON kg_wiki_refresh_jobs(workspace_id, status, started_at);
CREATE INDEX IF NOT EXISTS idx_kg_job_events_cursor ON kg_job_events(job_kind, job_id, seq);
CREATE INDEX IF NOT EXISTS idx_memory_dream_runs_workspace ON memory_dream_runs(workspace_key, started_at);
CREATE INDEX IF NOT EXISTS idx_deep_checkpoints_lineage ON deep_checkpoints(lineage_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_deep_checkpoints_disposition ON deep_checkpoints(disposition, updated_at);
"""


@dataclass(frozen=True)
class ChunkRow:
    """A stored chunk with optional embedding."""

    id: int
    document_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None


class Store:
    """Thread-safe SQLite store for sessions, runs, documents, citations, memory."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        encryption_key: str | None = None,
        read_only: bool = False,
    ) -> None:
        self._path = str(db_path)
        self._lock = threading.RLock()
        self._encrypted = False
        self._read_only = bool(read_only)
        if self._read_only and self._path == ":memory:":
            raise ValueError("read-only Store requires an existing database file")
        if self._read_only and not Path(self._path).is_file():
            raise FileNotFoundError(f"read-only database does not exist: {self._path}")
        self._row_factory = sqlite3.Row
        self._conn = self._connect(encryption_key)
        self._conn.row_factory = self._row_factory
        self._conn.execute("PRAGMA foreign_keys=ON")
        if self._read_only:
            # Inventory callers must not acquire a write transaction, create a WAL,
            # or run the additive schema bootstrap on a real deployment volume.
            self._conn.execute("PRAGMA query_only=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._fts5 = bool(
                self._conn.execute(
                    "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
                ).fetchone()[0]
            )
            return
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Cross-process write contention (api + rq worker share one DB file): the
        # in-process RLock cannot serialize writers across processes, so let SQLite
        # queue a contended writer for up to 5s instead of failing fast with
        # "database is locked". synchronous=NORMAL is safe and faster under WAL.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._fts5 = self._detect_fts5()
        self._migrate()

    @property
    def encrypted(self) -> bool:
        """Return whether the database is opened through SQLCipher."""

        return self._encrypted

    def _connect(self, encryption_key: str | None):
        """Open a connection, using SQLCipher when an encryption key is supplied.

        When no key is given, opens an ordinary stdlib ``sqlite3`` connection
        (plaintext) — the default for Docker and local development.

        When a key *is* given we never silently degrade to plaintext (DR-ITER-023):
        if no SQLCipher driver is installed, or the key does not actually unlock the
        database, we raise :class:`StoreEncryptionError` and refuse to continue.
        """

        if not encryption_key:
            if self._read_only:
                database_uri = f"{Path(self._path).resolve().as_uri()}?mode=ro"
                return sqlite3.connect(
                    database_uri,
                    uri=True,
                    check_same_thread=False,
                )
            return sqlite3.connect(self._path, check_same_thread=False)

        conn = self._open_encrypted()
        # SQLCipher requires the key before any other statement.
        conn.execute(f"PRAGMA key = {self._quote_sqlcipher_key(encryption_key)}")
        # Verify the key actually unlocks the database. A wrong key — or a plaintext
        # file opened as if encrypted — fails this probe, so we fail fast instead of
        # silently exposing/overwriting data.
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except Exception as error:  # noqa: BLE001 - any driver error means the key failed
            conn.close()
            raise StoreEncryptionError(
                "DEEPRESEARCH_DB_KEY was rejected by SQLCipher (wrong key, or the "
                "database is not SQLCipher-encrypted). Refusing to continue to avoid "
                "exposing plaintext data."
            ) from error
        self._encrypted = True
        return conn

    def _open_encrypted(self):
        """Return a SQLCipher connection, or fail fast if no driver is installed."""

        import importlib

        for mod_name in ("sqlcipher3", "pysqlcipher3.dbapi2"):
            try:
                driver = importlib.import_module(mod_name)
            except Exception:  # noqa: BLE001 - try the next candidate driver
                continue
            self._row_factory = getattr(driver, "Row", sqlite3.Row)
            return driver.connect(self._path, check_same_thread=False)
        raise StoreEncryptionError(
            "DEEPRESEARCH_DB_KEY is set but no SQLCipher driver (sqlcipher3 / "
            "pysqlcipher3) is installed. Install one, or unset the key — refusing "
            "to store data unencrypted when encryption was requested."
        )

    @staticmethod
    def _quote_sqlcipher_key(encryption_key: str) -> str:
        """Return a SQLCipher pragma string literal for a passphrase."""

        return "'" + encryption_key.replace("'", "''") + "'"

    @property
    def fts5_available(self) -> bool:
        """Return whether FTS5 lexical search is available."""

        return self._fts5

    def close(self) -> None:
        """Close the underlying connection."""

        with self._lock:
            self._conn.close()

    # --- schema ---------------------------------------------------------------

    def _detect_fts5(self) -> bool:
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
            self._conn.execute("DROP TABLE IF EXISTS _fts_probe")
            return True
        except Exception as error:  # noqa: BLE001 - SQLCipher drivers use their own error classes
            if error.__class__.__name__ != "OperationalError":
                raise
            return False

    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            if self._fts5:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, chunk_id UNINDEXED)"
                )
            # Additive vector-index migration: CREATE TABLE IF NOT EXISTS never adds
            # columns to a pre-existing table, so back-fill the embedding columns for
            # semantic recall over memories and conversation history.
            self._ensure_column("memories", "embedding", "TEXT")
            self._ensure_column("memories", "archived_at", "REAL")
            self._ensure_column("memories", "pinned", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memories", "expires_at", "REAL")
            self._ensure_column("messages", "embedding", "TEXT")
            self._ensure_column("sessions", "archived_at", "REAL")
            self._ensure_column("sessions", "pinned", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("sessions", "mode_default", "TEXT")
            self._ensure_column("runs", "metadata", "TEXT DEFAULT '{}'")
            self._ensure_column("citations", "origin", "TEXT NOT NULL DEFAULT 'external'")
            self._ensure_column(
                "citations", "artifact_type", "TEXT NOT NULL DEFAULT 'web_document'"
            )
            self._ensure_column("citations", "artifact_id", "TEXT")
            self._ensure_column(
                "citations", "grounding_eligible", "INTEGER NOT NULL DEFAULT 1"
            )
            self._ensure_column(
                "citations", "provenance_refs", "TEXT NOT NULL DEFAULT '[]'"
            )
            self._ensure_column(
                "citations", "source_status", "TEXT NOT NULL DEFAULT 'active'"
            )
            self._ensure_column("citations", "published_at", "REAL")
            self._ensure_column("citations", "updated_at", "REAL")
            self._ensure_column("citations", "effective_at", "REAL")
            self._ensure_column("citations", "retrieved_at", "REAL")
            self._ensure_column("memory_dream_runs", "backend", "TEXT NOT NULL DEFAULT 'thread'")
            self._ensure_column("memory_dream_runs", "backend_job_id", "TEXT")
            self._ensure_column("memory_dream_runs", "updated_at", "REAL")
            self._ensure_column("memory_dream_runs", "cancel_requested_at", "REAL")
            self._conn.execute(
                "UPDATE memory_dream_runs SET updated_at=COALESCE(finished_at, started_at) "
                "WHERE updated_at IS NULL"
            )
            self._conn.execute(
                "UPDATE citations SET "
                "origin=COALESCE(NULLIF(origin, ''), 'external'), "
                "artifact_type=COALESCE(NULLIF(artifact_type, ''), 'web_document'), "
                "artifact_id=COALESCE(NULLIF(artifact_id, ''), document_id, ''), "
                "grounding_eligible=COALESCE(grounding_eligible, 1), "
                "provenance_refs=COALESCE(NULLIF(provenance_refs, ''), '[]'), "
                "source_status=COALESCE(NULLIF(source_status, ''), 'active'), "
                "retrieved_at=COALESCE(retrieved_at, created_at) "
                "WHERE origin IS NULL OR origin='' OR artifact_type IS NULL OR artifact_type='' "
                "OR artifact_id IS NULL OR artifact_id='' OR grounding_eligible IS NULL "
                "OR provenance_refs IS NULL OR provenance_refs='' "
                "OR source_status IS NULL OR source_status='' OR retrieved_at IS NULL"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(namespace, scope_key, archived_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_pinned_expiry ON memories(namespace, scope_key, pinned, expires_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_pinned_updated ON sessions(archived_at, pinned, updated_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_dream_workspace_status "
                "ON memory_dream_runs(workspace_key, status, updated_at)"
            )
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        """Idempotently add ``column`` to ``table`` (guarded by PRAGMA table_info)."""

        existing = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # --- sessions / messages --------------------------------------------------

    def create_session(self, *, title: str = "", workspace: str = "default", metadata: JsonDict | None = None) -> str:
        """Create a session and return its id."""

        sid = _uid("sess")
        ts = _now()
        stored_metadata = dict(metadata or {})
        if "search_usage_policy" in stored_metadata:
            stored_metadata.setdefault(_SEARCH_USAGE_POLICY_REVISION, 1)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id,title,workspace,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?)",
                (sid, title, workspace, ts, ts, json.dumps(stored_metadata)),
            )
            self._conn.commit()
        return sid

    def get_session(self, session_id: str) -> JsonDict | None:
        """Return a session row as a dict, or None."""

        row = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return _row_to_dict(row)

    def list_sessions(self, *, limit: int = 50, archived: bool = False) -> list[JsonDict]:
        """List sessions newest-first; active (``archived=False``) or archived only."""

        if archived:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE archived_at IS NOT NULL "
                "ORDER BY archived_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE archived_at IS NULL "
                "ORDER BY pinned DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_session_memory_groups(
        self,
        *,
        workspace_key: str = "default",
        session_id: str | None = None,
        query: str = "",
        include_archived_sessions: bool = False,
        include_archived_memories: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> JsonDict:
        """List non-empty Session memory groups without recall side effects."""

        conditions = ["s.workspace=?", "m.namespace='session'", "m.scope_key=s.id"]
        params: list[Any] = [workspace_key]
        if not include_archived_sessions:
            conditions.append("s.archived_at IS NULL")
        if not include_archived_memories:
            conditions.append("m.archived_at IS NULL")
        if session_id:
            conditions.append("s.id=?")
            params.append(session_id)
        normalized_query = str(query or "").strip().lower()
        if normalized_query:
            escaped = normalized_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("(LOWER(s.title) LIKE ? ESCAPE '\\' OR LOWER(m.content) LIKE ? ESCAPE '\\')")
            params.extend((f"%{escaped}%", f"%{escaped}%"))
        where = " AND ".join(conditions)
        total_row = self._conn.execute(
            f"SELECT COUNT(DISTINCT s.id) AS n FROM sessions s JOIN memories m ON m.scope_key=s.id WHERE {where}",
            params,
        ).fetchone()
        total = int(total_row["n"] or 0) if total_row else 0
        group_rows = self._conn.execute(
            "SELECT s.id AS session_id, s.title AS session_title, s.updated_at AS session_updated_at, "
            "s.archived_at AS session_archived_at FROM sessions s "
            "JOIN memories m ON m.scope_key=s.id "
            f"WHERE {where} GROUP BY s.id "
            "ORDER BY CASE WHEN s.archived_at IS NULL THEN 0 ELSE 1 END, "
            "CASE WHEN s.archived_at IS NULL THEN s.updated_at END DESC, "
            "CASE WHEN s.archived_at IS NOT NULL THEN s.archived_at END DESC, "
            "LOWER(s.title), s.id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        session_ids = [str(row["session_id"]) for row in group_rows]
        memories_by_session: dict[str, list[JsonDict]] = {sid: [] for sid in session_ids}
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            archival_filter = "" if include_archived_memories else "AND archived_at IS NULL "
            memory_rows = self._conn.execute(
                "SELECT * FROM memories WHERE namespace='session' "
                f"AND scope_key IN ({placeholders}) {archival_filter}"
                "ORDER BY scope_key, CASE WHEN archived_at IS NULL THEN 0 ELSE 1 END, created_at DESC",
                session_ids,
            ).fetchall()
            for memory_row in memory_rows:
                memory = _strip_embedding(_decode_metadata(_row_to_dict(memory_row))) or {}
                memories_by_session.setdefault(str(memory.get("scope_key") or ""), []).append(memory)
        groups = [
            {
                "session_id": str(row["session_id"]),
                "session_title": str(row["session_title"] or "Untitled"),
                "session_updated_at": row["session_updated_at"],
                "session_archived_at": row["session_archived_at"],
                "memories": memories_by_session.get(str(row["session_id"]), []),
            }
            for row in group_rows
        ]
        return {
            "sessions": groups,
            "total": total,
            "limit": limit,
            "offset": offset,
            "truncated": offset + len(groups) < total,
        }

    def update_session(
        self,
        session_id: str,
        *,
        pinned: bool | None = None,
        mode_default: str | None = None,
        update_pinned: bool = False,
        update_mode_default: bool = False,
        metadata_patch: JsonDict | None = None,
        metadata_remove: Iterable[str] = (),
    ) -> JsonDict | None:
        """Update active session preferences and return the row."""

        updates = ["updated_at=?"]
        params: list[Any] = [_now()]
        if update_pinned:
            updates.append("pinned=?")
            params.append(1 if pinned else 0)
        if update_mode_default:
            updates.append("mode_default=?")
            params.append(mode_default)
        with self._lock:
            remove_keys = tuple(metadata_remove)
            if metadata_patch is not None or remove_keys:
                row = self._conn.execute(
                    "SELECT metadata FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
                current = _safe_json(row["metadata"]) if row and row["metadata"] else {}
                metadata = dict(current) if isinstance(current, dict) else {}
                touches_search_usage = (
                    "search_usage_policy" in (metadata_patch or {})
                    or "search_usage_policy" in remove_keys
                )
                metadata.update(metadata_patch or {})
                for key in remove_keys:
                    metadata.pop(str(key), None)
                if touches_search_usage:
                    metadata[_SEARCH_USAGE_POLICY_REVISION] = _search_usage_policy_revision(current) + 1
                updates.append("metadata=?")
                params.append(json.dumps(metadata))
            params.append(session_id)
            cur = self._conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id=? AND archived_at IS NULL",
                params,
            )
            self._conn.commit()
        if cur.rowcount <= 0:
            return None
        return self.get_session(session_id)

    def create_session_and_run(
        self,
        *,
        title: str,
        mode: str,
        query: str,
        budget: JsonDict | None = None,
        session_metadata: JsonDict | None = None,
        run_metadata: JsonDict | None = None,
    ) -> tuple[str, str]:
        """Atomically create a session and its first queued run."""

        sid = _uid("sess")
        rid = _uid("run")
        ts = _now()
        stored_session_metadata = dict(session_metadata or {})
        if "search_usage_policy" in stored_session_metadata:
            stored_session_metadata.setdefault(_SEARCH_USAGE_POLICY_REVISION, 1)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT INTO sessions(id,title,workspace,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?)",
                    (sid, title, "default", ts, ts, json.dumps(stored_session_metadata)),
                )
                self._conn.execute(
                    "INSERT INTO runs(id,session_id,mode,query,status,budget,created_at,updated_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        rid,
                        sid,
                        mode,
                        query,
                        "queued",
                        json.dumps(budget or {}),
                        ts,
                        ts,
                        json.dumps(run_metadata or {}),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return sid, rid

    def update_session_and_create_run(
        self,
        session_id: str,
        *,
        mode: str,
        query: str,
        budget: JsonDict | None = None,
        run_metadata: JsonDict | None = None,
        search_usage_policy: JsonDict | None = None,
        update_search_usage_policy: bool = False,
        expected_session_metadata: str | None = None,
        expected_session_updated_at: float | None = None,
    ) -> JsonDict | None:
        """Atomically update one session override and create a queued run."""

        rid = _uid("run")
        ts = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT metadata,updated_at FROM sessions WHERE id=? AND archived_at IS NULL",
                    (session_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                if expected_session_metadata is not None and (
                    str(row["metadata"] or "") != expected_session_metadata
                    or float(row["updated_at"] or 0.0) != float(expected_session_updated_at or 0.0)
                ):
                    self._conn.rollback()
                    raise SessionStateConflict("session changed while freezing search usage")
                metadata = _safe_json(row["metadata"]) if row["metadata"] else {}
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                previous_metadata = dict(metadata)
                previous_updated_at = float(row["updated_at"] or 0.0)
                if update_search_usage_policy:
                    if search_usage_policy is None:
                        metadata.pop("search_usage_policy", None)
                    else:
                        metadata["search_usage_policy"] = dict(search_usage_policy)
                    metadata[_SEARCH_USAGE_POLICY_REVISION] = _search_usage_policy_revision(previous_metadata) + 1
                self._conn.execute(
                    "UPDATE sessions SET updated_at=?,metadata=? WHERE id=?",
                    (ts, json.dumps(metadata), session_id),
                )
                self._conn.execute(
                    "INSERT INTO runs(id,session_id,mode,query,status,budget,created_at,updated_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        rid,
                        session_id,
                        mode,
                        query,
                        "queued",
                        json.dumps(budget or {}),
                        ts,
                        ts,
                        json.dumps(run_metadata or {}),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "run_id": rid,
            "previous_session_metadata": previous_metadata,
            "previous_session_updated_at": previous_updated_at,
            "applied_search_usage_policy_revision": _search_usage_policy_revision(metadata),
        }

    def rollback_unenqueued_run(
        self,
        run_id: str,
        session_id: str,
        *,
        created_session: bool,
        previous_session_metadata: str | None = None,
        previous_session_updated_at: float | None = None,
        restore_search_usage_policy: bool = False,
        expected_search_usage_policy_revision: int | None = None,
    ) -> bool:
        """Compensate a failed queue enqueue without leaving partial T50 state.

        The run and session transaction must commit before an out-of-process queue
        can see it. If enqueue itself fails, this method removes the still-queued
        run and either removes its brand-new session or restores only the
        session-local search policy while preserving unrelated concurrent metadata.
        """

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                run = self._conn.execute(
                    "SELECT status,created_at FROM runs WHERE id=? AND session_id=?",
                    (run_id, session_id),
                ).fetchone()
                if run is None or str(run["status"] or "") != "queued":
                    self._conn.rollback()
                    return False
                self._conn.execute("DELETE FROM run_events WHERE run_id=?", (run_id,))
                self._conn.execute("DELETE FROM citations WHERE run_id=?", (run_id,))
                self._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
                if created_session:
                    remaining = self._conn.execute(
                        "SELECT COUNT(*) AS n FROM runs WHERE session_id=?", (session_id,)
                    ).fetchone()
                    messages = self._conn.execute(
                        "SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,)
                    ).fetchone()
                    if int(remaining["n"] or 0) == 0 and int(messages["n"] or 0) == 0:
                        self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                else:
                    row = self._conn.execute(
                        "SELECT metadata,updated_at FROM sessions WHERE id=?", (session_id,)
                    ).fetchone()
                    if row is not None:
                        current = _safe_json(row["metadata"]) if row["metadata"] else {}
                        current_metadata = dict(current) if isinstance(current, dict) else {}
                        run_created_at = float(run["created_at"] or 0.0)
                        current_updated_at = float(row["updated_at"] or 0.0)
                        restore_timestamp = current_updated_at == run_created_at
                        if restore_search_usage_policy:
                            previous = (
                                _safe_json(previous_session_metadata)
                                if previous_session_metadata
                                else {}
                            )
                            previous_metadata = dict(previous) if isinstance(previous, dict) else {}
                            current_revision = _search_usage_policy_revision(current_metadata)
                            if (
                                expected_search_usage_policy_revision is None
                                or current_revision != expected_search_usage_policy_revision
                            ):
                                # A concurrent policy update owns both the metadata and
                                # its timestamp. Never compensate across that revision.
                                restore_timestamp = False
                            else:
                                if "search_usage_policy" in previous_metadata:
                                    current_metadata["search_usage_policy"] = previous_metadata[
                                        "search_usage_policy"
                                    ]
                                else:
                                    current_metadata.pop("search_usage_policy", None)
                                if _SEARCH_USAGE_POLICY_REVISION in previous_metadata:
                                    current_metadata[_SEARCH_USAGE_POLICY_REVISION] = previous_metadata[
                                        _SEARCH_USAGE_POLICY_REVISION
                                    ]
                                else:
                                    current_metadata.pop(_SEARCH_USAGE_POLICY_REVISION, None)
                        restored_updated_at = (
                            previous_session_updated_at
                            if previous_session_updated_at is not None
                            and restore_timestamp
                            else current_updated_at
                        )
                        self._conn.execute(
                            "UPDATE sessions SET metadata=?,updated_at=? WHERE id=?",
                            (json.dumps(current_metadata), restored_updated_at, session_id),
                        )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def enqueue_confirmation_state(self, run_id: str) -> str:
        """Return the persisted API-to-worker enqueue gate state for one run.

        Runs created before T50 do not contain the marker and remain claimable for
        backward compatibility. New background runs explicitly persist ``False``
        until the API confirms queue acceptance.
        """

        with self._lock:
            row = self._conn.execute(
                "SELECT status,metadata FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            return "missing"
        status = str(row["status"] or "")
        if status != "queued":
            return status or "missing"
        metadata = _safe_json(row["metadata"]) if row["metadata"] else {}
        if isinstance(metadata, dict) and metadata.get(_RUN_ENQUEUE_CONFIRMED) is False:
            if metadata.get(_RUN_ENQUEUE_WORKER_EXPIRED) is True:
                return "expired"
            return "pending"
        return "confirmed"

    def confirm_run_enqueued(self, run_id: str) -> bool:
        """Open the worker execution gate after queue acceptance is confirmed."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status,metadata FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or str(row["status"] or "") != "queued":
                    self._conn.rollback()
                    return False
                current = _safe_json(row["metadata"]) if row["metadata"] else {}
                metadata = dict(current) if isinstance(current, dict) else {}
                if metadata.get(_RUN_ENQUEUE_WORKER_EXPIRED) is True:
                    self._conn.rollback()
                    return False
                metadata[_RUN_ENQUEUE_CONFIRMED] = True
                self._conn.execute(
                    "UPDATE runs SET metadata=? WHERE id=? AND status='queued'",
                    (json.dumps(metadata), run_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def expire_enqueue_confirmation_wait(self, run_id: str) -> str:
        """Atomically settle a worker/API confirmation race at the wait deadline.

        The worker records that it has stopped waiting but deliberately leaves the
        run queued. The API still owns terminalization and session-policy rollback
        because only it has the previous session snapshot. If the API already
        confirmed first, the worker receives ``confirmed`` and may continue.
        """

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status,metadata FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return "missing"
                status = str(row["status"] or "")
                if status != "queued":
                    self._conn.rollback()
                    return status or "missing"
                current = _safe_json(row["metadata"]) if row["metadata"] else {}
                metadata = dict(current) if isinstance(current, dict) else {}
                if metadata.get(_RUN_ENQUEUE_CONFIRMED) is not False:
                    self._conn.rollback()
                    return "confirmed"
                metadata[_RUN_ENQUEUE_WORKER_EXPIRED] = True
                self._conn.execute(
                    "UPDATE runs SET metadata=? WHERE id=? AND status='queued'",
                    (json.dumps(metadata), run_id),
                )
                self._conn.commit()
                return "expired"
            except Exception:
                self._conn.rollback()
                raise

    def abort_unconfirmed_enqueue(
        self,
        run_id: str,
        session_id: str,
        *,
        previous_session_metadata: str | None = None,
        previous_session_updated_at: float | None = None,
        restore_search_usage_policy: bool = False,
        expected_search_usage_policy_revision: int | None = None,
    ) -> bool:
        """Terminalize an ambiguously enqueued run so a late worker cannot execute it."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                run = self._conn.execute(
                    "SELECT status,created_at,metadata FROM runs WHERE id=? AND session_id=?",
                    (run_id, session_id),
                ).fetchone()
                if run is None or str(run["status"] or "") != "queued":
                    self._conn.rollback()
                    return False
                run_metadata = _safe_json(run["metadata"]) if run["metadata"] else {}
                if (
                    not isinstance(run_metadata, dict)
                    or run_metadata.get(_RUN_ENQUEUE_CONFIRMED) is not False
                ):
                    self._conn.rollback()
                    return False
                error_payload = {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Queue acceptance could not be confirmed; the run was safely stopped.",
                    }
                }
                now = _now()
                self._conn.execute(
                    "UPDATE runs SET status='error',error=?,updated_at=? WHERE id=?",
                    (json.dumps(error_payload), now, run_id),
                )
                self._conn.execute(
                    "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
                    (run_id, 1, "error", json.dumps(error_payload), now),
                )
                if restore_search_usage_policy:
                    row = self._conn.execute(
                        "SELECT metadata,updated_at FROM sessions WHERE id=?",
                        (session_id,),
                    ).fetchone()
                    if row is not None:
                        current = _safe_json(row["metadata"]) if row["metadata"] else {}
                        current_metadata = dict(current) if isinstance(current, dict) else {}
                        if _search_usage_policy_revision(current_metadata) == int(
                            expected_search_usage_policy_revision or 0
                        ):
                            previous = (
                                _safe_json(previous_session_metadata)
                                if previous_session_metadata
                                else {}
                            )
                            previous_metadata = dict(previous) if isinstance(previous, dict) else {}
                            if "search_usage_policy" in previous_metadata:
                                current_metadata["search_usage_policy"] = previous_metadata[
                                    "search_usage_policy"
                                ]
                            else:
                                current_metadata.pop("search_usage_policy", None)
                            if _SEARCH_USAGE_POLICY_REVISION in previous_metadata:
                                current_metadata[_SEARCH_USAGE_POLICY_REVISION] = previous_metadata[
                                    _SEARCH_USAGE_POLICY_REVISION
                                ]
                            else:
                                current_metadata.pop(_SEARCH_USAGE_POLICY_REVISION, None)
                            current_updated_at = float(row["updated_at"] or 0.0)
                            restored_updated_at = (
                                previous_session_updated_at
                                if previous_session_updated_at is not None
                                and current_updated_at == float(run["created_at"] or 0.0)
                                else current_updated_at
                            )
                            self._conn.execute(
                                "UPDATE sessions SET metadata=?,updated_at=? WHERE id=?",
                                (json.dumps(current_metadata), restored_updated_at, session_id),
                            )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def is_session_archived(self, session_id: str) -> bool:
        """Return whether the session exists and is archived."""

        row = self._conn.execute(
            "SELECT archived_at FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not row:
            return False
        return row["archived_at"] is not None

    def is_session_active(self, session_id: str) -> bool:
        """Return whether ``session_id`` exists and is not archived."""

        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE id=? AND archived_at IS NULL",
            (session_id,),
        ).fetchone()
        return row is not None

    def list_missing_embedding_rows(
        self,
        session_id: str,
        *,
        after_created_at: float | None = None,
        after_id: str | None = None,
        limit: int = 32,
    ) -> list[JsonDict]:
        """List missing message/session-memory embeddings in stable page order.

        Session memories remain eligible even when the memory row itself is
        TTL-archived. The owning session must still be active.
        """

        bounded_limit = max(1, min(int(limit), 256))
        cursor_sql = ""
        params: list[Any] = [session_id, session_id]
        if after_created_at is not None:
            cursor_sql = (
                "WHERE (created_at>? OR (created_at=? AND id>?)) "
            )
            params.extend(
                (
                    float(after_created_at),
                    float(after_created_at),
                    str(after_id or ""),
                )
            )
        params.append(bounded_limit)
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind,id,session_id,content,created_at FROM ("
                "SELECT 'message' AS kind,m.id AS id,m.session_id AS session_id,"
                "m.content AS content,m.created_at AS created_at "
                "FROM messages m JOIN sessions s ON s.id=m.session_id "
                "WHERE m.session_id=? AND s.archived_at IS NULL "
                "AND m.role IN ('user','assistant') AND m.embedding IS NULL "
                "UNION ALL "
                "SELECT 'memory' AS kind,mem.id AS id,mem.scope_key AS session_id,"
                "mem.content AS content,mem.created_at AS created_at "
                "FROM memories mem JOIN sessions s ON s.id=mem.scope_key "
                "WHERE mem.namespace='session' AND mem.scope_key=? "
                "AND s.archived_at IS NULL AND mem.embedding IS NULL"
                f") missing {cursor_sql}"
                "ORDER BY created_at ASC,id ASC LIMIT ?",
                params,
            ).fetchall()
        result: list[JsonDict] = []
        for row in rows:
            content = str(row["content"] or "")
            result.append(
                {
                    "kind": str(row["kind"]),
                    "id": str(row["id"]),
                    "session_id": str(row["session_id"]),
                    "content": content,
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "created_at": float(row["created_at"] or 0.0),
                }
            )
        return result

    def list_sessions_missing_embeddings(
        self,
        *,
        after_session_id: str | None = None,
        limit: int = 100,
    ) -> list[JsonDict]:
        """Return active sessions with message/session-memory embedding gaps."""

        bounded_limit = max(1, min(int(limit), 1000))
        cursor_sql = "AND s.id>? " if after_session_id else ""
        params: list[Any] = [str(after_session_id)] if after_session_id else []
        params.append(bounded_limit)
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id AS session_id,"
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id "
                " AND m.role IN ('user','assistant') AND m.embedding IS NULL) AS messages,"
                "(SELECT COUNT(*) FROM memories mem WHERE mem.namespace='session' "
                " AND mem.scope_key=s.id AND mem.embedding IS NULL) AS memories "
                "FROM sessions s WHERE s.archived_at IS NULL "
                "AND (EXISTS (SELECT 1 FROM messages m WHERE m.session_id=s.id "
                " AND m.role IN ('user','assistant') AND m.embedding IS NULL) "
                "OR EXISTS (SELECT 1 FROM memories mem WHERE mem.namespace='session' "
                " AND mem.scope_key=s.id AND mem.embedding IS NULL)) "
                f"{cursor_sql}ORDER BY s.id ASC LIMIT ?",
                params,
            ).fetchall()
        return [
            {
                "session_id": str(row["session_id"]),
                "messages": int(row["messages"] or 0),
                "memories": int(row["memories"] or 0),
            }
            for row in rows
        ]

    def compare_and_set_embeddings(
        self,
        updates: Iterable[tuple[str, str, str, list[float]]],
    ) -> list[str]:
        """Persist one validated embedding batch with content-aware CAS."""

        applied: list[str] = []
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for kind, row_id, expected_content, embedding in updates:
                    payload = json.dumps(embedding)
                    if kind == "message":
                        cursor = self._conn.execute(
                            "UPDATE messages SET embedding=? "
                            "WHERE id=? AND content=? AND embedding IS NULL "
                            "AND EXISTS (SELECT 1 FROM sessions s "
                            "WHERE s.id=messages.session_id AND s.archived_at IS NULL)",
                            (payload, row_id, expected_content),
                        )
                    elif kind == "memory":
                        cursor = self._conn.execute(
                            "UPDATE memories SET embedding=? "
                            "WHERE id=? AND namespace='session' AND content=? "
                            "AND embedding IS NULL "
                            "AND EXISTS (SELECT 1 FROM sessions s "
                            "WHERE s.id=memories.scope_key AND s.archived_at IS NULL)",
                            (payload, row_id, expected_content),
                        )
                    else:
                        raise ValueError(f"unsupported embedding row kind: {kind}")
                    if cursor.rowcount > 0:
                        applied.append(str(row_id))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return applied

    def compare_and_set_embedding(
        self,
        kind: str,
        row_id: str,
        expected_content: str,
        embedding: list[float],
    ) -> bool:
        """Compatibility wrapper for one content-aware embedding CAS."""

        return bool(
            self.compare_and_set_embeddings(
                [(kind, row_id, expected_content, embedding)]
            )
        )

    def compare_and_set_workspace_memory_embedding(
        self, memory_id: str, expected_content: str, embedding: list[float]
    ) -> bool:
        """Attach a vector to one unchanged workspace memory."""

        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET embedding=? WHERE id=? AND namespace='workspace' "
                "AND content=? AND embedding IS NULL",
                (json.dumps(embedding), memory_id, expected_content),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def workspace_memory_has_embedding(self, memory_id: str, expected_content: str) -> bool:
        """Return whether the unchanged workspace memory already has a vector."""

        row = self._conn.execute(
            "SELECT 1 FROM memories WHERE id=? AND namespace='workspace' "
            "AND content=? AND embedding IS NOT NULL",
            (memory_id, expected_content),
        ).fetchone()
        return row is not None

    def archive_session(self, session_id: str) -> bool:
        """Mark a session archived; returns False when the session does not exist."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET archived_at=? WHERE id=? AND archived_at IS NULL",
                (_now(), session_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def restore_session(self, session_id: str) -> bool:
        """Clear archival on a session; returns False when not found or not archived."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET archived_at=NULL WHERE id=? AND archived_at IS NOT NULL",
                (session_id,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Cascade-delete a session and all session-scoped data; returns False if missing."""

        with self._lock:
            row = self._conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                return False
            run_rows = self._conn.execute(
                "SELECT id FROM runs WHERE session_id=?", (session_id,)
            ).fetchall()
            run_ids = [str(r["id"]) for r in run_rows]
            for rid in run_ids:
                self._conn.execute("DELETE FROM run_events WHERE run_id=?", (rid,))
                self._conn.execute("DELETE FROM citations WHERE run_id=?", (rid,))
                self._conn.execute("DELETE FROM deep_checkpoints WHERE run_id=?", (rid,))
                self._conn.execute("DELETE FROM post_turn_memory_runs WHERE run_id=?", (rid,))
            self._conn.execute("DELETE FROM runs WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.execute(
                "DELETE FROM memories WHERE namespace=? AND scope_key=?",
                ("session", session_id),
            )
            workspace_rows = self._conn.execute(
                "SELECT * FROM memories WHERE namespace=?",
                ("workspace",),
            ).fetchall()
            for memory_row in workspace_rows:
                data = _decode_metadata(_row_to_dict(memory_row)) or {}
                metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                if metadata.get("source") in {"session_distill_v1", "post_turn_memory_v1"} and metadata.get("session_id") == session_id:
                    self._conn.execute("DELETE FROM memories WHERE id=?", (str(data.get("id") or ""),))
            self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: JsonDict | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Append a message to a session (optionally with a content embedding)."""

        mid = _uid("msg")
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(id,session_id,role,content,created_at,metadata,embedding) VALUES (?,?,?,?,?,?,?)",
                (
                    mid,
                    session_id,
                    role,
                    content,
                    _now(),
                    json.dumps(metadata or {}),
                    json.dumps(embedding) if embedding else None,
                ),
            )
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))
            self._conn.commit()
        return mid

    def search_messages_semantic(
        self, query_vec: list[float], *, session_id: str = "", limit: int = 10
    ) -> list[JsonDict]:
        """Return messages ranked by cosine similarity to ``query_vec``.

        ``session_id`` empty => search across all conversations (cross-session);
        non-empty => restrict to that conversation (intra-session). Messages
        without a stored embedding are skipped (graceful: callers fall back to
        lexical/recency ordering when this returns nothing).
        """

        from .retrieval import cosine_similarity

        if not query_vec:
            return []
        if session_id:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id=? AND embedding IS NOT NULL", (session_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM messages WHERE embedding IS NOT NULL").fetchall()
        scored: list[tuple[float, JsonDict]] = []
        for r in rows:
            vec = _safe_json(r["embedding"])
            if not isinstance(vec, list):
                continue
            data = _row_to_dict(r) or {}
            data["score"] = cosine_similarity(query_vec, vec)
            data.pop("embedding", None)
            scored.append((data["score"], data))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [data for _, data in scored[:limit]]

    def list_messages(self, session_id: str) -> list[JsonDict]:
        """List messages for a session in order."""

        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC", (session_id,)
        ).fetchall()
        return [_strip_embedding(_decode_metadata(_row_to_dict(r))) for r in rows]

    # --- runs / run_events (job store) ---------------------------------------

    def create_run(
        self,
        *,
        session_id: str,
        mode: str,
        query: str,
        budget: JsonDict | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Create a run in ``queued`` state and return its id."""

        rid = _uid("run")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs(id,session_id,mode,query,status,budget,created_at,updated_at,metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, session_id, mode, query, "queued", json.dumps(budget or {}), ts, ts, json.dumps(metadata or {})),
            )
            self._conn.commit()
        return rid

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        result: JsonDict | None = None,
        error: JsonDict | None = None,
        metadata_patch: JsonDict | None = None,
    ) -> None:
        """Update run status/result/error and merge optional metadata fields."""

        sets = ["updated_at=?"]
        args: list[Any] = [_now()]
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if result is not None:
            sets.append("result=?")
            args.append(json.dumps(result))
        if error is not None:
            sets.append("error=?")
            args.append(json.dumps(error))
        with self._lock:
            if metadata_patch is not None:
                row = self._conn.execute("SELECT metadata FROM runs WHERE id=?", (run_id,)).fetchone()
                current = _safe_json(row["metadata"]) if row and row["metadata"] else {}
                if not isinstance(current, dict):
                    current = {}
                current.update(metadata_patch)
                sets.append("metadata=?")
                args.append(json.dumps(current))
            args.append(run_id)
            self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", args)
            self._conn.commit()

    def transition_run(
        self,
        run_id: str,
        *,
        expected_statuses: set[str],
        status: str,
        result: JsonDict | None = None,
        error: JsonDict | None = None,
        metadata_patch: JsonDict | None = None,
        citation_rows: Iterable[JsonDict] | None = None,
        events: Iterable[tuple[str, JsonDict]] | None = None,
    ) -> bool:
        """Atomically transition a run only while it is in an expected state.

        T49 uses this fence for success/error/cancel races so a provider response or
        RQ callback that arrives after a terminal transition cannot rewrite history.
        """

        allowed = {str(value) for value in expected_statuses if str(value)}
        if not allowed:
            return False
        pending_citations = [dict(value) for value in (citation_rows or ())]
        pending_events = [(str(event_type), dict(payload)) for event_type, payload in (events or ())]
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status, metadata FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or str(row["status"] or "") not in allowed:
                    self._conn.rollback()
                    return False
                sets = ["status=?", "updated_at=?"]
                args: list[Any] = [status, _now()]
                if result is not None:
                    sets.append("result=?")
                    args.append(json.dumps(result))
                    sets.append("error=NULL")
                if error is not None:
                    sets.append("error=?")
                    args.append(json.dumps(error))
                    sets.append("result=NULL")
                if metadata_patch is not None:
                    current = _safe_json(row["metadata"]) if row["metadata"] else {}
                    if not isinstance(current, dict):
                        current = {}
                    current.update(metadata_patch)
                    sets.append("metadata=?")
                    args.append(json.dumps(current))
                args.append(run_id)
                self._conn.execute(
                    f"UPDATE runs SET {', '.join(sets)} WHERE id=?", args
                )
                for citation in pending_citations:
                    self._insert_citation_locked(run_id, citation)
                if pending_events:
                    next_seq = int(
                        self._conn.execute(
                            "SELECT COALESCE(MAX(seq),0)+1 AS seq FROM run_events WHERE run_id=?",
                            (run_id,),
                        ).fetchone()["seq"]
                    )
                    for event_type, payload in pending_events:
                        self._conn.execute(
                            "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
                            (run_id, next_seq, event_type, json.dumps(payload), _now()),
                        )
                        next_seq += 1
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def update_run_verification(
        self,
        run_id: str,
        *,
        verification: JsonDict,
        confidence_label: str,
        confidence_warning: JsonDict | None,
        conflicts: list[JsonDict] | None = None,
    ) -> bool:
        """Atomically attach advisory verification to an already delivered report.

        The report body, citations, terminal status, and an existing risk
        acknowledgement are immutable across this late verification write.
        """

        terminal_statuses = {
            "completed",
            "completed_degraded",
            "partial_timeout",
        }
        incoming = dict(verification or {})
        incoming_status = str(incoming.get("status") or "")
        if incoming_status not in {"completed", "failed"}:
            return False
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status,result FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or str(row["status"] or "") not in terminal_statuses:
                    self._conn.rollback()
                    return False
                result = _safe_json(row["result"]) if row["result"] else {}
                if not isinstance(result, dict) or not result:
                    self._conn.rollback()
                    return False
                current_verification = result.get("verification")
                current_verification = (
                    dict(current_verification)
                    if isinstance(current_verification, dict)
                    else {}
                )
                current_status = str(current_verification.get("status") or "pending")
                if current_status in {"completed", "failed"}:
                    if current_verification == incoming:
                        self._conn.rollback()
                        return True
                    self._conn.rollback()
                    return False

                warning = dict(confidence_warning) if confidence_warning is not None else None
                existing_warning = result.get("confidence_warning")
                existing_warning = (
                    dict(existing_warning) if isinstance(existing_warning, dict) else {}
                )
                if warning is not None and bool(existing_warning.get("acknowledged")):
                    warning["acknowledged"] = True
                    if existing_warning.get("acknowledged_at") is not None:
                        warning["acknowledged_at"] = existing_warning["acknowledged_at"]

                answer = result.get("answer")
                answer = dict(answer) if isinstance(answer, dict) else {}
                result["verification"] = incoming
                answer["confidence_label"] = str(confidence_label or "uncertain")
                if warning is None:
                    result.pop("confidence_warning", None)
                else:
                    warning.setdefault("acknowledged", False)
                    result["confidence_warning"] = warning
                if conflicts is not None:
                    answer["conflicts"] = [dict(value) for value in conflicts]
                result["answer"] = answer

                now = _now()
                self._conn.execute(
                    "UPDATE runs SET result=?,updated_at=? WHERE id=?",
                    (json.dumps(result), now, run_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def acknowledge_run_confidence(self, run_id: str) -> JsonDict | None:
        """Persist an idempotent acceptance of one report's confidence warning."""

        terminal_statuses = {
            "completed",
            "completed_degraded",
            "partial_timeout",
        }
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status,result FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or str(row["status"] or "") not in terminal_statuses:
                    self._conn.rollback()
                    return None
                result = _safe_json(row["result"]) if row["result"] else {}
                if not isinstance(result, dict):
                    self._conn.rollback()
                    return None
                verification = result.get("verification")
                verification = dict(verification) if isinstance(verification, dict) else {}
                verification_status = str(verification.get("status") or "")
                answer = result.get("answer")
                answer = dict(answer) if isinstance(answer, dict) else {}
                is_legacy_uncertain = (
                    not verification_status
                    and str(answer.get("confidence_label") or "") == "uncertain"
                )
                if verification_status not in {"completed", "failed"} and not is_legacy_uncertain:
                    self._conn.rollback()
                    return None
                warning = result.get("confidence_warning")
                warning = dict(warning) if isinstance(warning, dict) else {}
                if not warning and is_legacy_uncertain:
                    warning = {
                        "reasons": ["legacy_low_confidence"],
                        "acknowledged": False,
                    }
                elif not warning:
                    self._conn.rollback()
                    return None
                acknowledged_at = warning.get("acknowledged_at")
                if bool(warning.get("acknowledged")) and acknowledged_at is not None:
                    self._conn.rollback()
                    return {
                        "acknowledged": True,
                        "acknowledged_at": acknowledged_at,
                    }
                acknowledged_at = _now()
                warning["acknowledged"] = True
                warning["acknowledged_at"] = acknowledged_at
                result["confidence_warning"] = warning
                self._conn.execute(
                    "UPDATE runs SET result=?,updated_at=? WHERE id=?",
                    (json.dumps(result), acknowledged_at, run_id),
                )
                self._conn.commit()
                return {
                    "acknowledged": True,
                    "acknowledged_at": acknowledged_at,
                }
            except Exception:
                self._conn.rollback()
                raise

    def claim_queued_run(self, run_id: str) -> bool:
        """Atomically claim a queued run only after API enqueue confirmation."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status,metadata FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or str(row["status"] or "") != "queued":
                    self._conn.rollback()
                    return False
                metadata = _safe_json(row["metadata"]) if row["metadata"] else {}
                if isinstance(metadata, dict) and metadata.get(_RUN_ENQUEUE_CONFIRMED) is False:
                    self._conn.rollback()
                    return False
                cur = self._conn.execute(
                    "UPDATE runs SET status='running', updated_at=? WHERE id=? AND status='queued'",
                    (_now(), run_id),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except Exception:
                self._conn.rollback()
                raise

    def get_latest_run_for_session(self, session_id: str) -> tuple[str, str] | None:
        """Return the most recent run ``(id, mode)`` for a session, or ``None``."""

        with self._lock:
            row = self._conn.execute(
                "SELECT id, mode FROM runs WHERE session_id=? ORDER BY rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return str(row["id"]), str(row["mode"])

    def get_latest_run_id(self, session_id: str) -> str | None:
        """Return the most recent run id for a session, or None."""

        latest = self.get_latest_run_for_session(session_id)
        return latest[0] if latest else None

    def get_run(self, run_id: str) -> JsonDict | None:
        """Return a run row (result/error/budget parsed) or None."""

        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        data = _row_to_dict(row)
        if data:
            for key in ("budget", "result", "error", "metadata"):
                if data.get(key):
                    data[key] = _safe_json(data[key])
        return data

    def append_event(self, run_id: str, event_type: str, payload: JsonDict) -> int:
        """Append an SSE-style run event; returns its sequence number."""

        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS seq FROM run_events WHERE run_id=?", (run_id,)
            ).fetchone()
            seq = int(row["seq"])
            self._conn.execute(
                "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
                (run_id, seq, event_type, json.dumps(payload), _now()),
            )
            self._conn.commit()
        return seq

    def list_events(self, run_id: str, *, after_seq: int = 0) -> list[JsonDict]:
        """List run events after a sequence number (for resume)."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT seq,type,payload,created_at FROM run_events WHERE run_id=? AND seq>? ORDER BY seq ASC",
                (run_id, after_seq),
            ).fetchall()
        out = []
        for r in rows:
            out.append({"seq": r["seq"], "type": r["type"], "payload": _safe_json(r["payload"]), "created_at": r["created_at"]})
        return out

    # --- Deep checkpoints ------------------------------------------------------

    def request_run_pause(self, run_id: str) -> JsonDict:
        """Persist a cooperative Deep pause request and its SSE event atomically."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT mode,status,metadata FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return {"run_id": run_id, "status": "not_found", "accepted": False}
                mode = str(row["mode"] or "")
                status = str(row["status"] or "")
                if mode != "deep":
                    self._conn.rollback()
                    return {
                        "run_id": run_id,
                        "status": status,
                        "accepted": False,
                        "reason": "mode_not_supported",
                    }
                metadata = _safe_json(row["metadata"]) if row["metadata"] else {}
                if isinstance(metadata, dict) and bool(
                    metadata.get("deep_checkpoint_pause_fenced")
                ):
                    self._conn.rollback()
                    return {
                        "run_id": run_id,
                        "status": status,
                        "accepted": False,
                        "reason": "final_writer_started",
                    }
                if status in {"pausing", "paused"}:
                    self._conn.rollback()
                    return {"run_id": run_id, "status": status, "accepted": True}
                if status != "running":
                    self._conn.rollback()
                    return {
                        "run_id": run_id,
                        "status": status,
                        "accepted": False,
                        "reason": "run_not_pausable",
                    }
                changed_at = _now()
                self._conn.execute(
                    "UPDATE runs SET status='pausing',updated_at=? "
                    "WHERE id=? AND status='running'",
                    (changed_at, run_id),
                )
                self._append_event_locked(
                    run_id,
                    "pause_requested",
                    {"status": "pausing"},
                    created_at=changed_at,
                )
                self._conn.commit()
                return {"run_id": run_id, "status": "pausing", "accepted": True}
            except Exception:
                self._conn.rollback()
                raise

    def rollback_run_pause(self, run_id: str, *, reason: str) -> bool:
        """Return a failed user Pause request to running without hiding the failure."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                changed = self._conn.execute(
                    "UPDATE runs SET status='running',updated_at=? "
                    "WHERE id=? AND mode='deep' AND status='pausing'",
                    (_now(), run_id),
                )
                if changed.rowcount <= 0:
                    self._conn.rollback()
                    return False
                self._append_event_locked(
                    run_id,
                    "checkpoint_save_failed",
                    {"reason": str(reason or "checkpoint_write_failed")},
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def fence_deep_final_writer(self, run_id: str) -> bool:
        """Atomically prevent new Pause requests once final generation may begin."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT mode,status,metadata FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if (
                    row is None
                    or str(row["mode"] or "") != "deep"
                    or str(row["status"] or "") != "running"
                ):
                    self._conn.rollback()
                    return False
                metadata = _safe_json(row["metadata"]) if row["metadata"] else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["deep_checkpoint_pause_fenced"] = True
                self._conn.execute(
                    "UPDATE runs SET metadata=?,updated_at=? WHERE id=? AND status='running'",
                    (json.dumps(metadata), _now(), run_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def save_deep_checkpoint(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage: str,
        schema_version: int,
        payload_json: str,
        payload_sha256: str,
    ) -> JsonDict:
        """Atomically replace the latest valid auto-checkpoint for a running Deep run."""

        decoded = decode_deep_checkpoint(payload_json, payload_sha256)
        if decoded.parent_run_id != run_id or decoded.lineage_id != lineage_id:
            raise CheckpointStateConflict("checkpoint identity does not match the target run")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                run = self._conn.execute(
                    "SELECT mode,status FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if (
                    run is None
                    or str(run["mode"] or "") != "deep"
                    or str(run["status"] or "") not in {"running", "pausing"}
                ):
                    self._conn.rollback()
                    raise CheckpointStateConflict("run is not checkpointable")
                checkpoint = self._conn.execute(
                    "SELECT revision,created_at,disposition FROM deep_checkpoints WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if checkpoint is not None and str(checkpoint["disposition"] or "") != "ready":
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint has already been disposed")
                revision = int(checkpoint["revision"] or 0) + 1 if checkpoint else 1
                created_at = float(checkpoint["created_at"]) if checkpoint else _now()
                updated_at = _now()
                self._conn.execute(
                    """
                    INSERT INTO deep_checkpoints(
                        run_id,lineage_id,revision,schema_version,stage,payload_json,
                        payload_sha256,disposition,claimed_child_run_id,
                        claim_idempotency_key,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,'ready',NULL,NULL,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        lineage_id=excluded.lineage_id,
                        revision=excluded.revision,
                        schema_version=excluded.schema_version,
                        stage=excluded.stage,
                        payload_json=excluded.payload_json,
                        payload_sha256=excluded.payload_sha256,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        lineage_id,
                        revision,
                        int(schema_version),
                        stage,
                        payload_json,
                        payload_sha256,
                        created_at,
                        updated_at,
                    ),
                )
                self._append_event_locked(
                    run_id,
                    "checkpoint_saved",
                    {
                        "stage": stage,
                        "revision": revision,
                        "automatic": True,
                    },
                    created_at=updated_at,
                )
                self._conn.commit()
                return self.get_deep_checkpoint(run_id) or {}
            except Exception:
                self._conn.rollback()
                raise

    def save_deep_checkpoint_and_pause(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage: str,
        schema_version: int,
        payload_json: str,
        payload_sha256: str,
    ) -> JsonDict | None:
        """Commit a safe checkpoint and ``pausing -> paused`` as one transaction."""

        decoded = decode_deep_checkpoint(payload_json, payload_sha256)
        if decoded.parent_run_id != run_id or decoded.lineage_id != lineage_id:
            raise CheckpointStateConflict("checkpoint identity does not match the target run")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                run = self._conn.execute(
                    "SELECT mode,status FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if (
                    run is None
                    or str(run["mode"] or "") != "deep"
                    or str(run["status"] or "") != "pausing"
                ):
                    self._conn.rollback()
                    return None
                checkpoint = self._conn.execute(
                    "SELECT revision,created_at,disposition FROM deep_checkpoints WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if checkpoint is not None and str(checkpoint["disposition"] or "") != "ready":
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint has already been disposed")
                revision = int(checkpoint["revision"] or 0) + 1 if checkpoint else 1
                created_at = float(checkpoint["created_at"]) if checkpoint else _now()
                updated_at = _now()
                self._conn.execute(
                    """
                    INSERT INTO deep_checkpoints(
                        run_id,lineage_id,revision,schema_version,stage,payload_json,
                        payload_sha256,disposition,claimed_child_run_id,
                        claim_idempotency_key,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,'ready',NULL,NULL,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        lineage_id=excluded.lineage_id,
                        revision=excluded.revision,
                        schema_version=excluded.schema_version,
                        stage=excluded.stage,
                        payload_json=excluded.payload_json,
                        payload_sha256=excluded.payload_sha256,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        lineage_id,
                        revision,
                        int(schema_version),
                        stage,
                        payload_json,
                        payload_sha256,
                        created_at,
                        updated_at,
                    ),
                )
                changed = self._conn.execute(
                    "UPDATE runs SET status='paused',result=NULL,error=NULL,updated_at=? "
                    "WHERE id=? AND status='pausing'",
                    (updated_at, run_id),
                )
                if changed.rowcount <= 0:
                    self._conn.rollback()
                    return None
                self._append_event_locked(
                    run_id,
                    "checkpoint_saved",
                    {
                        "stage": stage,
                        "revision": revision,
                        "automatic": False,
                    },
                    created_at=updated_at,
                )
                self._append_event_locked(
                    run_id,
                    "workflow_end",
                    {"status": "paused"},
                    created_at=updated_at,
                )
                self._conn.commit()
                return self.get_deep_checkpoint(run_id) or {}
            except Exception:
                self._conn.rollback()
                raise

    def get_deep_checkpoint(
        self,
        run_id: str,
        *,
        include_payload: bool = False,
    ) -> JsonDict | None:
        """Return a secret-safe checkpoint summary and optional decoded payload."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM deep_checkpoints WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row) or {}
        integrity = checkpoint_integrity(data.get("payload_json"), data.get("payload_sha256"))
        result: JsonDict = {
            "run_id": str(data.get("run_id") or ""),
            "lineage_id": str(data.get("lineage_id") or ""),
            "revision": int(data.get("revision") or 0),
            "schema_version": int(data.get("schema_version") or 0),
            "stage": str(data.get("stage") or ""),
            "disposition": str(data.get("disposition") or ""),
            "claimed_child_run_id": data.get("claimed_child_run_id"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "integrity": integrity,
        }
        if include_payload and integrity == "valid":
            result["payload"] = decode_deep_checkpoint(
                str(data.get("payload_json") or ""),
                str(data.get("payload_sha256") or ""),
            ).to_dict()
        return result

    def get_deep_checkpoint_claim(
        self,
        run_id: str,
        *,
        idempotency_key: str,
    ) -> JsonDict | None:
        """Return an existing child only when the caller repeats the same claim key."""

        with self._lock:
            row = self._conn.execute(
                "SELECT claimed_child_run_id,claim_idempotency_key,disposition "
                "FROM deep_checkpoints WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if (
                row is None
                or str(row["disposition"] or "") != "claimed"
                or str(row["claim_idempotency_key"] or "") != str(idempotency_key)
                or not str(row["claimed_child_run_id"] or "")
            ):
                return None
            child = self._conn.execute(
                "SELECT id,session_id,status FROM runs WHERE id=?",
                (str(row["claimed_child_run_id"]),),
            ).fetchone()
        if child is None:
            return None
        return {
            "run_id": str(child["id"]),
            "session_id": str(child["session_id"]),
            "status": str(child["status"]),
        }

    def claim_deep_checkpoint_and_create_child(
        self,
        *,
        run_id: str,
        expected_revision: int,
        idempotency_key: str,
        mode: str,
        query: str,
        budget: JsonDict,
        metadata: JsonDict,
    ) -> JsonDict:
        """Claim one checkpoint and create its sole child run in one transaction."""

        key = str(idempotency_key or "").strip()
        if not key or len(key) > 200:
            raise ValueError("idempotency_key must be a bounded non-empty string")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                checkpoint = self._conn.execute(
                    "SELECT * FROM deep_checkpoints WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                parent = self._conn.execute(
                    "SELECT session_id,status,mode FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if checkpoint is None or parent is None:
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint is unavailable")
                claimed_key = str(checkpoint["claim_idempotency_key"] or "")
                claimed_child = str(checkpoint["claimed_child_run_id"] or "")
                if str(checkpoint["disposition"] or "") == "claimed":
                    if claimed_key == key and claimed_child:
                        child = self._conn.execute(
                            "SELECT id,session_id,status FROM runs WHERE id=?",
                            (claimed_child,),
                        ).fetchone()
                        self._conn.rollback()
                        if child is None:
                            raise CheckpointStateConflict("claimed child is missing")
                        return {
                            "run_id": str(child["id"]),
                            "session_id": str(child["session_id"]),
                            "status": str(child["status"]),
                            "idempotent_replay": True,
                        }
                    self._conn.rollback()
                    raise CheckpointAlreadyContinued("checkpoint already continued")
                if str(checkpoint["disposition"] or "") != "ready":
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint is not resumable")
                if int(checkpoint["revision"] or 0) != int(expected_revision):
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint revision changed")
                if str(parent["status"] or "") != "paused" or str(parent["mode"] or "") != "deep":
                    self._conn.rollback()
                    raise CheckpointStateConflict("parent run is not paused")
                session_id = str(parent["session_id"] or "")
                session = self._conn.execute(
                    "SELECT archived_at FROM sessions WHERE id=?",
                    (session_id,),
                ).fetchone()
                if session is None or session["archived_at"] is not None:
                    self._conn.rollback()
                    raise CheckpointStateConflict("session is not active")
                child_run_id = _uid("run")
                now = _now()
                child_metadata = dict(metadata)
                child_metadata.setdefault("checkpoint_parent_run_id", run_id)
                child_metadata.setdefault("checkpoint_revision", int(expected_revision))
                child_metadata.setdefault("lineage_id", str(checkpoint["lineage_id"] or ""))
                self._conn.execute(
                    "INSERT INTO runs(id,session_id,mode,query,status,budget,created_at,updated_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        child_run_id,
                        session_id,
                        mode,
                        query,
                        "queued",
                        json.dumps(budget or {}),
                        now,
                        now,
                        json.dumps(child_metadata),
                    ),
                )
                self._conn.execute(
                    "UPDATE deep_checkpoints SET disposition='claimed',"
                    "claimed_child_run_id=?,claim_idempotency_key=?,updated_at=? "
                    "WHERE run_id=? AND revision=? AND disposition='ready'",
                    (child_run_id, key, now, run_id, int(expected_revision)),
                )
                self._conn.commit()
                return {
                    "run_id": child_run_id,
                    "session_id": session_id,
                    "status": "queued",
                    "idempotent_replay": False,
                }
            except Exception:
                self._conn.rollback()
                raise

    def rollback_deep_checkpoint_child_claim(
        self,
        *,
        run_id: str,
        child_run_id: str,
        idempotency_key: str,
    ) -> bool:
        """Undo a child claim only when enqueue failed before the child started."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                child = self._conn.execute(
                    "SELECT status FROM runs WHERE id=?",
                    (child_run_id,),
                ).fetchone()
                checkpoint = self._conn.execute(
                    "SELECT disposition,claimed_child_run_id,claim_idempotency_key "
                    "FROM deep_checkpoints WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if (
                    child is None
                    or str(child["status"] or "") != "queued"
                    or checkpoint is None
                    or str(checkpoint["disposition"] or "") != "claimed"
                    or str(checkpoint["claimed_child_run_id"] or "") != child_run_id
                    or str(checkpoint["claim_idempotency_key"] or "") != str(idempotency_key)
                ):
                    self._conn.rollback()
                    return False
                self._conn.execute("DELETE FROM runs WHERE id=?", (child_run_id,))
                self._conn.execute(
                    "UPDATE deep_checkpoints SET disposition='ready',"
                    "claimed_child_run_id=NULL,claim_idempotency_key=NULL,updated_at=? "
                    "WHERE run_id=?",
                    (_now(), run_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def abandon_deep_checkpoint(
        self,
        run_id: str,
        *,
        expected_revision: int | None = None,
    ) -> JsonDict | None:
        """Mark an unclaimed checkpoint abandoned while retaining its audit record."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                checkpoint = self._conn.execute(
                    "SELECT revision,disposition FROM deep_checkpoints WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if checkpoint is None:
                    self._conn.rollback()
                    return None
                if (
                    expected_revision is not None
                    and int(checkpoint["revision"] or 0) != int(expected_revision)
                ):
                    self._conn.rollback()
                    raise CheckpointStateConflict("checkpoint revision changed")
                disposition = str(checkpoint["disposition"] or "")
                if disposition == "claimed":
                    self._conn.rollback()
                    raise CheckpointAlreadyContinued("checkpoint already continued")
                if disposition != "abandoned":
                    self._conn.execute(
                        "UPDATE deep_checkpoints SET disposition='abandoned',updated_at=? "
                        "WHERE run_id=? AND disposition='ready'",
                        (_now(), run_id),
                    )
                self._conn.commit()
                return self.get_deep_checkpoint(run_id)
            except Exception:
                self._conn.rollback()
                raise

    def _append_event_locked(
        self,
        run_id: str,
        event_type: str,
        payload: JsonDict,
        *,
        created_at: float | None = None,
    ) -> int:
        """Append an event inside an already-owned Store transaction."""

        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS seq FROM run_events WHERE run_id=?",
            (run_id,),
        ).fetchone()
        seq = int(row["seq"])
        self._conn.execute(
            "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
            (run_id, seq, event_type, json.dumps(payload), created_at or _now()),
        )
        return seq

    # --- documents / chunks ---------------------------------------------------

    def upsert_document(
        self,
        *,
        url: str,
        title: str,
        text: str,
        content_type: str = "web",
        doc_hash: str = "",
        metadata: JsonDict | None = None,
        quarantine_library_candidate: bool = False,
    ) -> str:
        """Insert or update a document by URL; returns its id."""

        document_id, _ = self.upsert_document_with_chunks(
            url=url,
            title=title,
            text=text,
            chunks=(),
            content_type=content_type,
            doc_hash=doc_hash,
            metadata=metadata,
            quarantine_library_candidate=quarantine_library_candidate,
        )
        return document_id

    def upsert_document_with_chunks(
        self,
        *,
        url: str,
        title: str,
        text: str,
        chunks: Iterable[tuple[str, list[float] | None]],
        content_type: str = "web",
        doc_hash: str = "",
        metadata: JsonDict | None = None,
        quarantine_library_candidate: bool = False,
        expected_deletion_marker: str | None = None,
        revive_deleted_identity: bool = False,
    ) -> tuple[str, list[int]]:
        """Replace a document and its FTS chunks in one durable transaction."""

        chunk_rows = list(chunks)
        with self._lock:
            try:
                # Read/merge/write under one reserved transaction.  This makes
                # Library visibility monotonic across API and worker connections.
                self._conn.execute("BEGIN IMMEDIATE")
                deletion_marker = self._document_tombstone_marker_locked(url)
                if deletion_marker is not None:
                    if not revive_deleted_identity or deletion_marker != expected_deletion_marker:
                        raise DocumentWriteRejected("document was deleted before this write committed")
                    self._conn.execute(
                        "DELETE FROM library_provenance WHERE artifact_type=? AND id=?",
                        (_DOCUMENT_TOMBSTONE_ARTIFACT_TYPE, deletion_marker),
                    )
                existing = self._conn.execute(
                    "SELECT id,metadata FROM documents WHERE url=?",
                    (url,),
                ).fetchone()
                incoming_metadata = dict(metadata or {})
                incoming_metadata.pop("_library_visible", None)
                incoming_metadata.pop("_library_pending_runs", None)
                current_metadata: JsonDict = {}
                if existing is not None:
                    decoded = _safe_json(existing["metadata"])
                    current_metadata = dict(decoded) if isinstance(decoded, dict) else {}
                merged_metadata = {**current_metadata, **incoming_metadata}
                merged_metadata.pop("_library_pending_runs", None)
                if quarantine_library_candidate:
                    # A fresh candidate is hidden, but an already-visible URL
                    # can never be hidden again by a later run.
                    merged_metadata["_library_visible"] = (
                        False
                        if existing is None or current_metadata.get("_library_visible") is False
                        else True
                    )
                else:
                    merged_metadata.pop("_library_visible", None)

                if existing is None:
                    did = _uid("doc")
                    self._conn.execute(
                        "INSERT INTO documents(id,url,title,content_type,text,hash,created_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            did,
                            url,
                            title,
                            content_type,
                            text,
                            doc_hash,
                            _now(),
                            json.dumps(merged_metadata),
                        ),
                    )
                else:
                    did = existing["id"]
                    existing_chunks = self._conn.execute(
                        "SELECT id FROM chunks WHERE document_id=?",
                        (did,),
                    ).fetchall()
                    existing_chunk_ids = [int(row["id"]) for row in existing_chunks]
                    if self._fts5 and existing_chunk_ids:
                        placeholders = ",".join("?" for _ in existing_chunk_ids)
                        self._conn.execute(
                            f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                            existing_chunk_ids,
                        )
                    self._conn.execute("DELETE FROM chunks WHERE document_id=?", (did,))
                    self._conn.execute(
                        "UPDATE documents SET title=?,text=?,content_type=?,hash=?,metadata=? WHERE id=?",
                        (title, text, content_type, doc_hash, json.dumps(merged_metadata), did),
                    )

                chunk_ids: list[int] = []
                for chunk_index, (chunk_text, embedding) in enumerate(chunk_rows):
                    cursor = self._conn.execute(
                        "INSERT INTO chunks(document_id,chunk_index,text,embedding,token_count,created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            did,
                            chunk_index,
                            chunk_text,
                            json.dumps(embedding) if embedding else None,
                            len(chunk_text.split()),
                            _now(),
                        ),
                    )
                    chunk_id = int(cursor.lastrowid)
                    chunk_ids.append(chunk_id)
                    if self._fts5:
                        self._conn.execute(
                            "INSERT INTO chunks_fts(text,chunk_id) VALUES (?,?)",
                            (chunk_text, chunk_id),
                        )
                self._conn.commit()
                return did, chunk_ids
            except Exception:
                self._conn.rollback()
                raise

    def insert_local_document_with_chunks_if_absent(
        self,
        *,
        url: str,
        title: str,
        text: str,
        chunks: Iterable[tuple[str, list[float] | None]],
        content_type: str,
        doc_hash: str,
        metadata: JsonDict,
        expected_deletion_marker: str | None = None,
    ) -> tuple[bool, str, list[int], JsonDict]:
        """Insert one immutable local import, or return its existing identity unchanged."""

        chunk_rows = list(chunks)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                deletion_marker = self._document_tombstone_marker_locked(url)
                if deletion_marker is not None:
                    if deletion_marker != expected_deletion_marker:
                        raise DocumentWriteRejected("document was deleted before this write committed")
                    self._conn.execute(
                        "DELETE FROM library_provenance WHERE artifact_type=? AND id=?",
                        (_DOCUMENT_TOMBSTONE_ARTIFACT_TYPE, deletion_marker),
                    )
                existing = self._conn.execute(
                    "SELECT * FROM documents WHERE url=?",
                    (url,),
                ).fetchone()
                if existing is not None:
                    existing_row = _row_to_dict(existing) or {}
                    existing_metadata = _safe_json(existing_row.get("metadata"))
                    if not isinstance(existing_metadata, dict) or existing_metadata.get("source") != "local_import":
                        raise ValueError("local import identity collides with a non-import document")
                    chunk_rows_existing = self._conn.execute(
                        "SELECT id FROM chunks WHERE document_id=? ORDER BY chunk_index ASC,id ASC",
                        (existing_row["id"],),
                    ).fetchall()
                    self._conn.rollback()
                    existing_row["metadata"] = existing_metadata
                    return (
                        False,
                        str(existing_row["id"]),
                        [int(row["id"]) for row in chunk_rows_existing],
                        existing_row,
                    )

                document_id = _uid("doc")
                stored_metadata = {
                    **dict(metadata),
                    "source": "local_import",
                    "artifact_type": "library_document",
                    "artifact_id": document_id,
                    "document_id": document_id,
                }
                created_at = _now()
                self._conn.execute(
                    "INSERT INTO documents(id,url,title,content_type,text,hash,created_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        url,
                        title,
                        content_type,
                        text,
                        doc_hash,
                        created_at,
                        json.dumps(stored_metadata),
                    ),
                )
                chunk_ids: list[int] = []
                for chunk_index, (chunk_text, embedding) in enumerate(chunk_rows):
                    cursor = self._conn.execute(
                        "INSERT INTO chunks(document_id,chunk_index,text,embedding,token_count,created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            document_id,
                            chunk_index,
                            chunk_text,
                            json.dumps(embedding) if embedding else None,
                            len(chunk_text.split()),
                            created_at,
                        ),
                    )
                    chunk_id = int(cursor.lastrowid)
                    chunk_ids.append(chunk_id)
                    if self._fts5:
                        self._conn.execute(
                            "INSERT INTO chunks_fts(text,chunk_id) VALUES (?,?)",
                            (chunk_text, chunk_id),
                        )
                self._conn.execute(
                    "INSERT INTO library_provenance"
                    "(id,artifact_type,artifact_id,session_id,run_id,document_id,url,created_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        _uid("prov"),
                        "library_document",
                        document_id,
                        None,
                        None,
                        document_id,
                        None,
                        created_at,
                        json.dumps({"source": "local_import"}),
                    ),
                )
                self._conn.commit()
                row = {
                    "id": document_id,
                    "url": url,
                    "title": title,
                    "content_type": content_type,
                    "hash": doc_hash,
                    "created_at": created_at,
                    "metadata": stored_metadata,
                }
                return True, document_id, chunk_ids, row
            except Exception:
                self._conn.rollback()
                raise

    def replace_document_chunks_if_present(
        self,
        *,
        expected_document_id: str,
        url: str,
        text: str,
        chunks: Iterable[tuple[str, list[float] | None]],
        doc_hash: str,
    ) -> list[int] | None:
        """Replace chunks only while the original document identity still exists."""

        chunk_rows = list(chunks)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._has_document_tombstone_locked(url):
                    self._conn.rollback()
                    return None
                existing = self._conn.execute(
                    "SELECT id FROM documents WHERE id=? AND url=?",
                    (expected_document_id, url),
                ).fetchone()
                if existing is None:
                    self._conn.rollback()
                    return None
                old_chunks = self._conn.execute(
                    "SELECT id FROM chunks WHERE document_id=?",
                    (expected_document_id,),
                ).fetchall()
                old_chunk_ids = [int(row["id"]) for row in old_chunks]
                if self._fts5 and old_chunk_ids:
                    placeholders = ",".join("?" for _ in old_chunk_ids)
                    self._conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                        old_chunk_ids,
                    )
                self._conn.execute("DELETE FROM chunks WHERE document_id=?", (expected_document_id,))
                self._conn.execute(
                    "UPDATE documents SET text=?,hash=? WHERE id=?",
                    (text, doc_hash, expected_document_id),
                )
                chunk_ids: list[int] = []
                for chunk_index, (chunk_text, embedding) in enumerate(chunk_rows):
                    cursor = self._conn.execute(
                        "INSERT INTO chunks(document_id,chunk_index,text,embedding,token_count,created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            expected_document_id,
                            chunk_index,
                            chunk_text,
                            json.dumps(embedding) if embedding else None,
                            len(chunk_text.split()),
                            _now(),
                        ),
                    )
                    chunk_id = int(cursor.lastrowid)
                    chunk_ids.append(chunk_id)
                    if self._fts5:
                        self._conn.execute(
                            "INSERT INTO chunks_fts(text,chunk_id) VALUES (?,?)",
                            (chunk_text, chunk_id),
                        )
                self._conn.commit()
                return chunk_ids
            except Exception:
                self._conn.rollback()
                raise

    def add_chunk(self, *, document_id: str, chunk_index: int, text: str, embedding: list[float] | None = None) -> int:
        """Store a chunk (and index it in FTS5 when available); returns its id."""

        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO chunks(document_id,chunk_index,text,embedding,token_count,created_at) VALUES (?,?,?,?,?,?)",
                (document_id, chunk_index, text, json.dumps(embedding) if embedding else None, len(text.split()), _now()),
            )
            chunk_id = int(cur.lastrowid)
            if self._fts5:
                self._conn.execute("INSERT INTO chunks_fts(text,chunk_id) VALUES (?,?)", (text, chunk_id))
            self._conn.commit()
        return chunk_id

    def get_chunk(self, chunk_id: int) -> ChunkRow | None:
        """Return a chunk by id."""

        row = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return _row_to_chunk(row)

    def get_document(self, document_id: str) -> JsonDict | None:
        """Return a document row by id, or None."""

        row = self._conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return _row_to_dict(row)

    def promote_document_to_library(self, document_id: str) -> bool:
        """Make one grounded external retrieval candidate visible in Library."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT url,metadata FROM documents WHERE id=?",
                    (document_id,),
                ).fetchone()
                if row is None or not str(row["url"] or "").lower().startswith(("http://", "https://")):
                    self._conn.rollback()
                    return False
                metadata = _safe_json(row["metadata"])
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                if metadata.get("_library_visible") is not False:
                    self._conn.rollback()
                    return False
                metadata["_library_visible"] = True
                metadata.pop("_library_pending_runs", None)
                self._conn.execute(
                    "UPDATE documents SET metadata=? WHERE id=?",
                    (json.dumps(metadata), document_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def list_documents(self, *, limit: int = 200) -> list[JsonDict]:
        """List indexed documents newest-first with their chunk counts (no full text)."""

        rows = self._conn.execute(
            "SELECT d.id, d.url, d.title, d.content_type, d.created_at, "
            "COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def has_document_content_types(
        self,
        content_types: Iterable[str],
        *,
        require_embedding: bool = False,
    ) -> bool:
        """Return whether any requested document type can serve the selected channel."""

        values = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in content_types
                if str(value).strip()
            )
        )
        if not values:
            return False
        placeholders = ",".join("?" for _ in values)
        if require_embedding:
            statement = (
                "SELECT 1 FROM documents d JOIN chunks c ON c.document_id=d.id "
                f"WHERE d.content_type IN ({placeholders}) AND c.embedding IS NOT NULL LIMIT 1"
            )
        else:
            statement = f"SELECT 1 FROM documents WHERE content_type IN ({placeholders}) LIMIT 1"
        with self._lock:
            return self._conn.execute(statement, values).fetchone() is not None

    def list_external_documents(self, *, limit: int = 200) -> list[JsonDict]:
        """List user-visible external and local-import Library Sources."""

        rows = self._conn.execute(
            "SELECT d.id, d.url, d.title, d.content_type, d.created_at, d.metadata, "
            "COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.created_at DESC"
        ).fetchall()
        visible: list[JsonDict] = []
        for raw in rows:
            row = _row_to_dict(raw) or {}
            metadata = _safe_json(row.pop("metadata", None))
            if isinstance(metadata, dict) and metadata.get("_library_visible") is False:
                continue
            metadata = metadata if isinstance(metadata, dict) else {}
            url = str(row.get("url") or "")
            local_import = (
                metadata.get("source") == "local_import"
                and metadata.get("artifact_type") == "library_document"
                and url.startswith("library://local/")
            )
            external = url.lower().startswith(("http://", "https://"))
            if not local_import and not external:
                continue
            row.update(
                {
                    "url": None if local_import else url,
                    "origin": "local" if local_import else "external",
                    "artifact_type": "library_document" if local_import else "web_document",
                    "artifact_id": str(row.get("id") or ""),
                    "filename": str(metadata.get("filename") or "") or None,
                    "deletable": True,
                    "content_sha256": str(metadata.get("content_sha256") or "") or None,
                }
            )
            visible.append(row)
            if len(visible) >= limit:
                break
        return visible

    def get_document_deletion_marker(self, url: str) -> str | None:
        """Return the current deletion marker used by explicit import/index CAS."""

        with self._lock:
            return self._document_tombstone_marker_locked(url)

    def delete_library_source_document(self, document_id: str) -> JsonDict | None:
        """Delete one authorized current Library Source and retain an anti-resurrection marker."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                raw = self._conn.execute(
                    "SELECT * FROM documents WHERE id=?",
                    (document_id,),
                ).fetchone()
                if raw is None:
                    self._conn.rollback()
                    return None
                row = _row_to_dict(raw) or {}
                metadata = _safe_json(row.get("metadata"))
                metadata = metadata if isinstance(metadata, dict) else {}
                url = str(row.get("url") or "")
                visible = metadata.get("_library_visible") is not False
                local_import = (
                    metadata.get("source") == "local_import"
                    and metadata.get("artifact_type") == "library_document"
                    and url.startswith("library://local/")
                )
                external = visible and url.lower().startswith(("http://", "https://"))
                if not local_import and not external:
                    self._conn.rollback()
                    return {
                        "document_id": document_id,
                        "deletable": False,
                    }

                chunk_rows = self._conn.execute(
                    "SELECT id FROM chunks WHERE document_id=?",
                    (document_id,),
                ).fetchall()
                chunk_ids = [int(item["id"]) for item in chunk_rows]
                snapshot_rows = self._conn.execute(
                    "SELECT storage_path FROM page_snapshots WHERE document_id=?",
                    (document_id,),
                ).fetchall()
                snapshot_paths = sorted(
                    {
                        str(item["storage_path"])
                        for item in snapshot_rows
                        if str(item["storage_path"] or "")
                    }
                )
                if self._fts5 and chunk_ids:
                    placeholders = ",".join("?" for _ in chunk_ids)
                    self._conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                        chunk_ids,
                    )
                self._conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                snapshots_deleted = self._conn.execute(
                    "DELETE FROM page_snapshots WHERE document_id=?",
                    (document_id,),
                ).rowcount
                provenance_deleted = self._conn.execute(
                    "DELETE FROM library_provenance WHERE document_id=? OR "
                    "(artifact_id=? AND artifact_type IN ('source','library_document'))",
                    (document_id, document_id),
                ).rowcount
                self._conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
                tombstone_id = _document_tombstone_key(url)
                self._conn.execute(
                    "DELETE FROM library_provenance WHERE artifact_type=? AND artifact_id=?",
                    (_DOCUMENT_TOMBSTONE_ARTIFACT_TYPE, tombstone_id),
                )
                self._conn.execute(
                    "INSERT INTO library_provenance"
                    "(id,artifact_type,artifact_id,session_id,run_id,document_id,url,created_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        _uid("prov"),
                        _DOCUMENT_TOMBSTONE_ARTIFACT_TYPE,
                        tombstone_id,
                        None,
                        None,
                        None,
                        None,
                        _now(),
                        json.dumps({"deleted_document_id": document_id}),
                    ),
                )
                self._conn.commit()
                return {
                    "document_id": document_id,
                    "deletable": True,
                    "origin": "local" if local_import else "external",
                    "artifact_type": "library_document" if local_import else "web_document",
                    "url": url,
                    "chunks_deleted": len(chunk_ids),
                    "snapshots_deleted": int(snapshots_deleted),
                    "provenance_deleted": int(provenance_deleted),
                    "snapshot_paths": snapshot_paths,
                }
            except Exception:
                self._conn.rollback()
                raise

    def list_snapshot_storage_paths(self) -> set[str]:
        """Return all snapshot paths still referenced by current database rows."""

        rows = self._conn.execute(
            "SELECT DISTINCT storage_path FROM page_snapshots WHERE storage_path IS NOT NULL AND storage_path<>''"
        ).fetchall()
        return {str(row["storage_path"]) for row in rows}

    def list_library_provenance_for_document(self, document_id: str) -> list[JsonDict]:
        """List public provenance rows for one current source document."""

        rows = self._conn.execute(
            "SELECT * FROM library_provenance WHERE document_id=? AND artifact_type<>? "
            "ORDER BY created_at DESC",
            (document_id, _DOCUMENT_TOMBSTONE_ARTIFACT_TYPE),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(row)) for row in rows]

    def _document_tombstone_marker_locked(self, url: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM library_provenance WHERE artifact_type=? AND artifact_id=? LIMIT 1",
            (_DOCUMENT_TOMBSTONE_ARTIFACT_TYPE, _document_tombstone_key(url)),
        ).fetchone()
        return str(row["id"]) if row is not None else None

    def _has_document_tombstone_locked(self, url: str) -> bool:
        return self._document_tombstone_marker_locked(url) is not None

    def _require_current_document_locked(
        self,
        document_id: str,
        *,
        expected_url: str | None = None,
    ) -> str:
        row = self._conn.execute(
            "SELECT url FROM documents WHERE id=?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise DocumentWriteRejected("document no longer exists")
        current_url = str(row["url"] or "")
        if expected_url is not None and current_url != expected_url:
            raise DocumentWriteRejected("document identity changed")
        if self._has_document_tombstone_locked(current_url):
            raise DocumentWriteRejected("document was deleted before this write committed")
        return current_url

    def iter_documents_with_text(self, *, document_ids: list[str] | None = None) -> list[JsonDict]:
        """Return stored documents including text for index rebuild."""

        if document_ids is not None:
            ids = [str(item) for item in document_ids if str(item)]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            rows = self._conn.execute(
                f"SELECT * FROM documents WHERE id IN ({placeholders}) ORDER BY created_at DESC",
                ids,
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [_row_to_dict(r) for r in rows]

    def iter_chunks_with_embeddings(self) -> Iterable[ChunkRow]:
        """Yield all chunks that have stored embeddings (for vector search)."""

        rows = self._conn.execute("SELECT * FROM chunks WHERE embedding IS NOT NULL").fetchall()
        for r in rows:
            chunk = _row_to_chunk(r)
            if chunk:
                yield chunk

    def embedding_index_signature(self) -> tuple[int, int]:
        """Return ``(count, max_id)`` of chunks with embeddings.

        A cheap staleness key for derived ANN indexes (DR-ITER-020): when this changes
        the index must be rebuilt from the store, which stays the single source of truth.
        """

        row = self._conn.execute(
            "SELECT count(*) AS n, COALESCE(MAX(id), 0) AS m FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        return (int(row["n"]), int(row["m"])) if row else (0, 0)

    def search_chunks_lexical(self, query: str, *, limit: int = 10) -> list[tuple[int, float]]:
        """Return (chunk_id, score) lexical matches; higher score = better."""

        query = (query or "").strip()
        if not query:
            return []
        if self._fts5:
            try:
                rows = self._conn.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (_fts_query(query), limit),
                ).fetchall()
                # bm25 is lower-is-better; invert to higher-is-better.
                return [(int(r["chunk_id"]), -float(r["score"])) for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE text LIKE ? LIMIT ?", (like, limit)
        ).fetchall()
        return [(int(r["id"]), 1.0) for r in rows]

    # --- reports --------------------------------------------------------------

    def add_report(
        self,
        *,
        title: str,
        body_md: str,
        mode: str,
        parent_report_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Store a canonical Library report."""

        report_id = _uid("rep")
        with self._lock:
            self._conn.execute(
                "INSERT INTO reports"
                "(id,title,body_md,mode,parent_report_id,session_id,run_id,created_at,metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    report_id,
                    title,
                    body_md,
                    mode,
                    parent_report_id,
                    session_id,
                    run_id,
                    _now(),
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return report_id

    def get_report(self, report_id: str) -> JsonDict | None:
        """Return a report with full Markdown body."""

        row = self._conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return _decode_metadata(_row_to_dict(row))

    def list_reports(self, *, limit: int = 100) -> list[JsonDict]:
        """List reports newest-first without full Markdown bodies."""

        rows = self._conn.execute(
            "SELECT id,title,mode,parent_report_id,session_id,run_id,created_at,metadata,"
            "substr(body_md,1,240) AS body_preview "
            "FROM reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    def list_child_reports(self, parent_report_id: str) -> list[JsonDict]:
        """List child reports for a parent report."""

        rows = self._conn.execute(
            "SELECT id,title,mode,parent_report_id,session_id,run_id,created_at,metadata,"
            "substr(body_md,1,240) AS body_preview "
            "FROM reports WHERE parent_report_id=? ORDER BY created_at ASC",
            (parent_report_id,),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    # --- snapshots ------------------------------------------------------------

    def add_page_snapshot(
        self,
        *,
        document_id: str | None,
        url: str,
        content_hash: str,
        storage_path: str,
        mime_type: str = "text/plain",
        size_bytes: int = 0,
        status: str = "ok",
        error: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Store a page snapshot metadata row."""

        snapshot_id = _uid("snap")
        with self._lock:
            self._conn.execute(
                "INSERT INTO page_snapshots"
                "(id,document_id,url,captured_at,content_hash,storage_path,mime_type,size_bytes,status,error,run_id,metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    document_id,
                    url,
                    _now(),
                    content_hash,
                    storage_path,
                    mime_type,
                    size_bytes,
                    status,
                    error,
                    run_id,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return snapshot_id

    def add_page_snapshot_with_provenance_if_current(
        self,
        *,
        document_id: str,
        expected_url: str,
        url: str,
        content_hash: str,
        storage_path: str,
        mime_type: str = "text/plain",
        size_bytes: int = 0,
        status: str = "ok",
        error: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Atomically attach a snapshot and provenance only to the expected live document."""

        snapshot_id = _uid("snap")
        created_at = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._require_current_document_locked(
                    document_id,
                    expected_url=expected_url,
                )
                self._conn.execute(
                    "INSERT INTO page_snapshots"
                    "(id,document_id,url,captured_at,content_hash,storage_path,mime_type,size_bytes,status,error,run_id,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        document_id,
                        url,
                        created_at,
                        content_hash,
                        storage_path,
                        mime_type,
                        size_bytes,
                        status,
                        error,
                        run_id,
                        json.dumps(metadata or {}),
                    ),
                )
                self._conn.execute(
                    "INSERT INTO library_provenance"
                    "(id,artifact_type,artifact_id,session_id,run_id,document_id,url,created_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        _uid("prov"),
                        "snapshot",
                        snapshot_id,
                        None,
                        run_id,
                        document_id,
                        url,
                        created_at,
                        json.dumps(metadata or {}),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return snapshot_id

    def get_page_snapshot(self, snapshot_id: str) -> JsonDict | None:
        """Return a snapshot metadata row."""

        row = self._conn.execute("SELECT * FROM page_snapshots WHERE id=?", (snapshot_id,)).fetchone()
        return _decode_metadata(_row_to_dict(row))

    def list_page_snapshots_for_document(self, document_id: str, *, limit: int = 20) -> list[JsonDict]:
        """List snapshots for a document newest-first."""

        rows = self._conn.execute(
            "SELECT * FROM page_snapshots WHERE document_id=? ORDER BY captured_at DESC LIMIT ?",
            (document_id, limit),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    def list_page_snapshots_for_url(self, url: str, *, limit: int = 20) -> list[JsonDict]:
        """List snapshots for a URL newest-first."""

        rows = self._conn.execute(
            "SELECT * FROM page_snapshots WHERE url=? ORDER BY captured_at DESC LIMIT ?",
            (url, limit),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    # --- citations ------------------------------------------------------------

    def add_citation(
        self,
        *,
        run_id: str,
        document_id: str,
        url: str | None,
        title: str,
        snippet: str,
        score: float,
        origin: str = "external",
        artifact_type: str = "web_document",
        artifact_id: str = "",
        grounding_eligible: bool = True,
        provenance_refs: Iterable[str] | None = None,
        source_status: str = "active",
        published_at: float | None = None,
        updated_at: float | None = None,
        effective_at: float | None = None,
        retrieved_at: float | None = None,
    ) -> str:
        """Persist a citation for a run."""

        with self._lock:
            cid = self._insert_citation_locked(
                run_id,
                {
                    "document_id": document_id,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "score": score,
                    "origin": origin,
                    "artifact_type": artifact_type,
                    "artifact_id": artifact_id,
                    "grounding_eligible": grounding_eligible,
                    "provenance_refs": list(provenance_refs or ()),
                    "source_status": source_status,
                    "published_at": published_at,
                    "updated_at": updated_at,
                    "effective_at": effective_at,
                    "retrieved_at": retrieved_at,
                },
            )
            self._conn.commit()
        return cid

    def _insert_citation_locked(self, run_id: str, citation: JsonDict) -> str:
        """Insert one citation using the caller's transaction and Store lock."""

        cid = _uid("cit")
        created_at = _now()
        document_id = str(citation.get("document_id") or "")
        self._conn.execute(
            "INSERT INTO citations("
            "id,run_id,document_id,url,title,snippet,score,created_at,origin,"
            "artifact_type,artifact_id,grounding_eligible,provenance_refs,"
            "source_status,published_at,updated_at,effective_at,retrieved_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                run_id,
                document_id,
                citation.get("url"),
                str(citation.get("title") or ""),
                str(citation.get("snippet") or ""),
                float(citation.get("score") or 0.0),
                created_at,
                str(citation.get("origin") or "external"),
                str(citation.get("artifact_type") or "web_document"),
                str(citation.get("artifact_id") or document_id),
                int(bool(citation.get("grounding_eligible", True))),
                json.dumps(list(citation.get("provenance_refs") or ()), ensure_ascii=False),
                str(citation.get("source_status") or "active"),
                citation.get("published_at"),
                citation.get("updated_at"),
                citation.get("effective_at"),
                citation.get("retrieved_at")
                if citation.get("retrieved_at") is not None
                else created_at,
            ),
        )
        return cid

    def list_citations(self, run_id: str) -> list[JsonDict]:
        """List citations for a run."""

        rows = self._conn.execute("SELECT * FROM citations WHERE run_id=? ORDER BY score DESC", (run_id,)).fetchall()
        citations: list[JsonDict] = []
        for row in rows:
            citation = _row_to_dict(row) or {}
            refs = _safe_json(citation.get("provenance_refs"))
            citation["provenance_refs"] = refs if isinstance(refs, list) else []
            citation["grounding_eligible"] = bool(citation.get("grounding_eligible", 1))
            citation["origin"] = citation.get("origin") or "external"
            citation["artifact_type"] = citation.get("artifact_type") or "web_document"
            citation["artifact_id"] = citation.get("artifact_id") or citation.get("document_id") or ""
            citation["source_status"] = citation.get("source_status") or "active"
            if citation.get("retrieved_at") is None:
                citation["retrieved_at"] = citation.get("created_at")
            citations.append(citation)
        return citations

    # --- library provenance ---------------------------------------------------

    def add_library_provenance(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
        document_id: str | None = None,
        url: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Record where a Library artifact came from."""

        pid = _uid("prov")
        with self._lock:
            self._conn.execute(
                "INSERT INTO library_provenance"
                "(id,artifact_type,artifact_id,session_id,run_id,document_id,url,created_at,metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    pid,
                    artifact_type,
                    artifact_id,
                    session_id,
                    run_id,
                    document_id,
                    url,
                    _now(),
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return pid

    def add_library_provenance_for_current_document(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        document_id: str,
        expected_url: str,
        session_id: str | None = None,
        run_id: str | None = None,
        url: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Attach provenance only while the expected document identity is current."""

        provenance_id = _uid("prov")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._require_current_document_locked(
                    document_id,
                    expected_url=expected_url,
                )
                self._conn.execute(
                    "INSERT INTO library_provenance"
                    "(id,artifact_type,artifact_id,session_id,run_id,document_id,url,created_at,metadata) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        provenance_id,
                        artifact_type,
                        artifact_id,
                        session_id,
                        run_id,
                        document_id,
                        url,
                        _now(),
                        json.dumps(metadata or {}),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return provenance_id

    def list_library_provenance(self, *, artifact_type: str, artifact_id: str) -> list[JsonDict]:
        """List provenance rows for one Library artifact."""

        rows = self._conn.execute(
            "SELECT * FROM library_provenance WHERE artifact_type=? AND artifact_id=? "
            "ORDER BY created_at DESC",
            (artifact_type, artifact_id),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    def list_library_artifacts_for_session(self, session_id: str, *, limit: int = 100) -> list[JsonDict]:
        """List Library artifacts connected to a session."""

        rows = self._conn.execute(
            "SELECT * FROM library_provenance WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [_decode_metadata(_row_to_dict(r)) for r in rows]

    def delete_library_artifact(self, *, artifact_type: str, artifact_id: str) -> int:
        """Delete derived Library documents/chunks/provenance for one artifact."""

        with self._lock:
            try:
                deleted = self._delete_library_artifact_locked(
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return deleted

    def _delete_library_artifact_locked(self, *, artifact_type: str, artifact_id: str) -> int:
        """Delete one derived Library artifact using the caller's transaction."""

        rows = self._conn.execute(
            "SELECT document_id FROM library_provenance WHERE artifact_type=? AND artifact_id=?",
            (artifact_type, artifact_id),
        ).fetchall()
        document_ids = sorted({str(r["document_id"]) for r in rows if r["document_id"]})
        for document_id in document_ids:
            chunk_rows = self._conn.execute(
                "SELECT id FROM chunks WHERE document_id=?",
                (document_id,),
            ).fetchall()
            chunk_ids = [int(r["id"]) for r in chunk_rows]
            if self._fts5 and chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                self._conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
            self._conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            self._conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        cur = self._conn.execute(
            "DELETE FROM library_provenance WHERE artifact_type=? AND artifact_id=?",
            (artifact_type, artifact_id),
        )
        return int(cur.rowcount)

    def _has_library_artifact_locked(self, *, artifact_type: str, artifact_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM library_provenance WHERE artifact_type=? AND artifact_id=? LIMIT 1",
            (artifact_type, artifact_id),
        ).fetchone()
        return row is not None

    def purge_memory_library_documents(self, *, dry_run: bool = True) -> JsonDict:
        """Purge legacy memory-derived Library rows without touching memories."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT document_id FROM library_provenance WHERE artifact_type=?",
                ("memory",),
            ).fetchall()
            document_ids = sorted({str(row["document_id"]) for row in rows if row["document_id"]})
            chunk_count = 0
            for document_id in document_ids:
                chunk_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE document_id=?",
                    (document_id,),
                ).fetchone()
                chunk_count += int(chunk_row["n"] or 0) if chunk_row else 0
            result = {
                "dry_run": bool(dry_run),
                "documents": len(document_ids),
                "chunks": chunk_count,
                "provenance": len(document_ids),
                "document_ids": document_ids,
            }
            if dry_run:
                return result
            try:
                for document_id in document_ids:
                    chunk_rows = self._conn.execute(
                        "SELECT id FROM chunks WHERE document_id=?",
                        (document_id,),
                    ).fetchall()
                    chunk_ids = [int(row["id"]) for row in chunk_rows]
                    if self._fts5 and chunk_ids:
                        placeholders = ",".join("?" for _ in chunk_ids)
                        self._conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
                    self._conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                    self._conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
                self._conn.execute("DELETE FROM library_provenance WHERE artifact_type=?", ("memory",))
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    # --- memory dreaming ------------------------------------------------------

    def create_memory_dream_run_if_idle(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        status: str = "queued",
        backend: str = "thread",
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        """Atomically reserve one active Dream run for a workspace."""

        dream_id = _uid("dream")
        timestamp = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                active = self._conn.execute(
                    "SELECT * FROM memory_dream_runs WHERE workspace_key=? "
                    "AND status IN ('queued','running','cancelling') "
                    "ORDER BY updated_at DESC, started_at DESC LIMIT 1",
                    (workspace_key,),
                ).fetchone()
                if active is not None:
                    self._conn.rollback()
                    return {"created": False, "run": _decode_metadata(_row_to_dict(active))}
                self._conn.execute(
                    "INSERT INTO memory_dream_runs"
                    "(id,workspace_key,user_key,status,started_at,finished_at,memories_created,error,metadata,"
                    "backend,backend_job_id,updated_at,cancel_requested_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        dream_id,
                        workspace_key,
                        user_key,
                        status,
                        timestamp,
                        None,
                        0,
                        None,
                        json.dumps(metadata or {}),
                        backend,
                        None,
                        timestamp,
                        None,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        row = self.get_memory_dream_run(dream_id) or {}
        return {"created": True, "run": row}

    def create_memory_dream_run(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        metadata: JsonDict | None = None,
    ) -> str:
        """Create a Dreaming status row and return its id."""

        dream_id = _uid("dream")
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_dream_runs"
                "(id,workspace_key,user_key,status,started_at,finished_at,memories_created,error,metadata,"
                "backend,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dream_id,
                    workspace_key,
                    user_key,
                    "running",
                    timestamp,
                    None,
                    0,
                    None,
                    json.dumps(metadata or {}),
                    "thread",
                    timestamp,
                ),
            )
            self._conn.commit()
        return dream_id

    def update_memory_dream_run(
        self,
        dream_run_id: str,
        *,
        status: str,
        memories_created: int = 0,
        error: str | None = None,
        metadata: JsonDict | None = None,
        expected_statuses: set[str] | None = None,
        backend_job_id: str | None = None,
        update_backend_job_id: bool = False,
        cancel_requested: bool = False,
    ) -> bool:
        """Conditionally transition a Dream run and merge safe metadata."""

        with self._lock:
            try:
                # The API and RQ worker use separate SQLite connections. Reserve the
                # write lock before reading so a late worker cannot race a cancel.
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT * FROM memory_dream_runs WHERE id=?", (dream_run_id,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                current = _decode_metadata(_row_to_dict(row)) or {}
                if expected_statuses is not None and str(current.get("status") or "") not in expected_statuses:
                    self._conn.rollback()
                    return False
                merged_metadata = current.get("metadata") if isinstance(current.get("metadata"), dict) else {}
                merged_metadata = {**merged_metadata, **(metadata or {})}
                timestamp = _now()
                terminal = status in {"completed", "cancelled", "error"}
                fields = [
                    "status=?",
                    "finished_at=?",
                    "memories_created=?",
                    "error=?",
                    "metadata=?",
                    "updated_at=?",
                ]
                params: list[Any] = [
                    status,
                    timestamp if terminal else None,
                    memories_created,
                    error,
                    json.dumps(merged_metadata),
                    timestamp,
                ]
                if update_backend_job_id:
                    fields.append("backend_job_id=?")
                    params.append(backend_job_id)
                if cancel_requested:
                    fields.append("cancel_requested_at=?")
                    params.append(timestamp)
                params.append(dream_run_id)
                self._conn.execute(
                    f"UPDATE memory_dream_runs SET {', '.join(fields)} WHERE id=?",
                    params,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return True

    def get_memory_dream_run(self, dream_run_id: str) -> JsonDict | None:
        """Return a Dreaming run row by id."""

        row = self._conn.execute("SELECT * FROM memory_dream_runs WHERE id=?", (dream_run_id,)).fetchone()
        data = _row_to_dict(row)
        return _decode_metadata(data) if data is not None else None

    def set_memory_dream_backend_job_id(self, dream_run_id: str, backend_job_id: str) -> bool:
        """Attach the external job id without changing a concurrent run transition."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE memory_dream_runs SET backend_job_id=? WHERE id=?",
                (backend_job_id, dream_run_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_latest_memory_dream_run(self, *, workspace_key: str = "default") -> JsonDict | None:
        """Return the latest Dreaming status row for a workspace."""

        row = self._conn.execute(
            "SELECT * FROM memory_dream_runs WHERE workspace_key=? ORDER BY started_at DESC LIMIT 1",
            (workspace_key,),
        ).fetchone()
        data = _row_to_dict(row)
        return _decode_metadata(data) if data is not None else None

    def upsert_post_turn_memory_run(
        self,
        *,
        run_id: str,
        session_id: str,
        workspace_key: str,
        user_key: str,
        status: str,
        provider_ref: str = "",
        error_code: str | None = None,
        embedding_status: str | None = None,
    ) -> None:
        """Persist the latest secret-free outcome for one post-turn Memory job."""

        with self._lock:
            self._conn.execute(
                "INSERT INTO post_turn_memory_runs"
                "(run_id,session_id,workspace_key,user_key,status,checked_at,provider_ref,error_code,embedding_status) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
                "session_id=excluded.session_id,workspace_key=excluded.workspace_key,user_key=excluded.user_key,"
                "status=excluded.status,checked_at=excluded.checked_at,provider_ref=excluded.provider_ref,"
                "error_code=excluded.error_code,embedding_status=excluded.embedding_status",
                (run_id, session_id, workspace_key, user_key, status, _now(), provider_ref, error_code, embedding_status),
            )
            self._conn.commit()

    def get_latest_post_turn_memory_run(self, *, workspace_key: str) -> JsonDict | None:
        row = self._conn.execute(
            "SELECT * FROM post_turn_memory_runs WHERE workspace_key=? ORDER BY checked_at DESC LIMIT 1",
            (workspace_key,),
        ).fetchone()
        return _row_to_dict(row)

    def get_post_turn_memory_run(self, run_id: str) -> JsonDict | None:
        row = self._conn.execute(
            "SELECT * FROM post_turn_memory_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return _row_to_dict(row)

    def fail_post_turn_memory_run(
        self, *, run_id: str, session_id: str, error_code: str
    ) -> bool:
        """Atomically fail only a nonterminal post-turn job audit."""

        with self._lock:
            cursor = self._conn.execute(
                "UPDATE post_turn_memory_runs SET status='failed',checked_at=?,error_code=? "
                "WHERE run_id=? AND session_id=? AND status IN ('queued','running')",
                (_now(), error_code, run_id, session_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def list_active_memory_dream_runs(self, *, workspace_key: str | None = None) -> list[JsonDict]:
        """Return queued/running/cancelling Dream rows for reconciliation."""

        if workspace_key is None:
            rows = self._conn.execute(
                "SELECT * FROM memory_dream_runs WHERE status IN ('queued','running','cancelling') "
                "ORDER BY updated_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_dream_runs WHERE workspace_key=? "
                "AND status IN ('queued','running','cancelling') ORDER BY updated_at DESC",
                (workspace_key,),
            ).fetchall()
        return [_decode_metadata(_row_to_dict(row)) for row in rows]

    # --- memories -------------------------------------------------------------

    def add_memory(
        self,
        *,
        namespace: str,
        scope_key: str,
        content: str,
        salience: float = 1.0,
        metadata: JsonDict | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Add a memory item (optionally with a content embedding)."""

        mid = _uid("mem")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories(id,namespace,scope_key,content,salience,uses,created_at,last_used_at,metadata,embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    namespace,
                    scope_key,
                    content,
                    salience,
                    0,
                    ts,
                    ts,
                    json.dumps(metadata or {}),
                    json.dumps(embedding) if embedding else None,
                ),
            )
            self._conn.commit()
        return mid

    def iter_memories_with_embeddings(
        self,
        *,
        namespace: str,
        scope_key: str,
        limit: int = 200,
        include_archived: bool = False,
    ) -> list[JsonDict]:
        """Return memories in a namespace/scope that have stored embeddings.

        Each dict includes the parsed ``embedding`` list so callers can rank by
        cosine similarity. Memories without embeddings are excluded.
        """

        now = _now()
        if include_archived:
            archival_filter = (
                "AND (archived_at IS NOT NULL OR pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
        else:
            archival_filter = (
                "AND archived_at IS NULL AND (pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE namespace=? AND scope_key=? AND embedding IS NOT NULL "
            f"{archival_filter}"
            "ORDER BY last_used_at DESC LIMIT ?",
            (namespace, scope_key, now, limit),
        ).fetchall()
        out: list[JsonDict] = []
        for r in rows:
            vec = _safe_json(r["embedding"])
            if not isinstance(vec, list):
                continue
            data = _row_to_dict(r) or {}
            data["embedding"] = vec
            out.append(data)
        return out

    def get_memory(self, memory_id: str) -> JsonDict | None:
        """Return one memory row by id."""

        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        data = _row_to_dict(row)
        return _strip_embedding(_decode_metadata(data)) if data is not None else None

    def list_memories(
        self,
        *,
        namespace: str,
        scope_key: str,
        limit: int = 20,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> list[JsonDict]:
        """List memories for a namespace/scope ordered by salience then recency."""

        now = _now()
        if archived_only:
            archival_filter = "AND archived_at IS NOT NULL "
            params: tuple[Any, ...] = (namespace, scope_key, limit)
        elif include_archived:
            archival_filter = (
                "AND (archived_at IS NOT NULL OR pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
            params = (namespace, scope_key, now, limit)
        else:
            archival_filter = (
                "AND archived_at IS NULL AND (pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
            params = (namespace, scope_key, now, limit)
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE namespace=? AND scope_key=? "
            f"{archival_filter}"
            "ORDER BY salience DESC, last_used_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_strip_embedding(_decode_metadata(_row_to_dict(r))) for r in rows]

    def list_all_memories(
        self,
        *,
        limit: int = 1000,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> list[JsonDict]:
        """List memories across namespaces for Library index backfill."""

        now = _now()
        if archived_only:
            archival_filter = "WHERE archived_at IS NOT NULL "
            params: tuple[Any, ...] = (limit,)
        elif include_archived:
            archival_filter = (
                "WHERE archived_at IS NOT NULL OR pinned=1 OR expires_at IS NULL OR expires_at>? "
            )
            params = (now, limit)
        else:
            archival_filter = (
                "WHERE archived_at IS NULL AND (pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
            params = (now, limit)
        rows = self._conn.execute(
            f"SELECT * FROM memories {archival_filter}ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_strip_embedding(_decode_metadata(_row_to_dict(r))) for r in rows]

    def count_memories(self, *, namespace: str, scope_key: str, include_archived: bool = False) -> int:
        """Return the number of memories in a namespace/scope (DR-ITER-057)."""

        now = _now()
        if include_archived:
            archival_filter = (
                "AND (archived_at IS NOT NULL OR pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
        else:
            archival_filter = (
                "AND archived_at IS NULL AND (pinned=1 OR expires_at IS NULL OR expires_at>?) "
            )
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE namespace=? AND scope_key=? "
            f"{archival_filter}",
            (namespace, scope_key, now),
        ).fetchone()
        return int(row[0]) if row else 0

    def sweep_expired_memories(
        self,
        *,
        now: float | None = None,
        limit: int = 100,
        namespace: str | None = None,
        scope_key: str | None = None,
    ) -> JsonDict:
        """Archive a stable bounded batch of expired, unpinned memories."""

        checked_at = float(_now() if now is None else now)
        batch_limit = max(1, min(int(limit), 500))
        where = ["archived_at IS NULL", "pinned=0", "expires_at IS NOT NULL", "expires_at<=?"]
        params: list[Any] = [checked_at]
        if namespace is not None:
            where.append("namespace=?")
            params.append(namespace)
        if scope_key is not None:
            where.append("scope_key=?")
            params.append(scope_key)
        clause = " AND ".join(where)
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id FROM memories WHERE {clause} ORDER BY expires_at ASC, id ASC LIMIT ?",
                    (*params, batch_limit),
                ).fetchall()
                memory_ids: list[str] = []
                for row in rows:
                    memory_id = str(row["id"])
                    cur = self._conn.execute(
                        "UPDATE memories SET archived_at=? "
                        "WHERE id=? AND archived_at IS NULL AND pinned=0 "
                        "AND expires_at IS NOT NULL AND expires_at<=?",
                        (checked_at, memory_id, checked_at),
                    )
                    if cur.rowcount <= 0:
                        continue
                    if self._has_library_artifact_locked(artifact_type="memory", artifact_id=memory_id):
                        self._delete_library_artifact_locked(artifact_type="memory", artifact_id=memory_id)
                    memory_ids.append(memory_id)
                remaining_row = self._conn.execute(
                    f"SELECT COUNT(*) FROM memories WHERE {clause}",
                    params,
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "checked_at": checked_at,
            "scanned": len(rows),
            "archived": len(memory_ids),
            "memory_ids": memory_ids,
            "remaining": int(remaining_row[0]) if remaining_row else 0,
        }

    def count_expired_memories(
        self,
        *,
        now: float | None = None,
        namespace: str | None = None,
        scope_key: str | None = None,
    ) -> int:
        """Count expired, unpinned active memories for one optional scope."""

        checked_at = float(_now() if now is None else now)
        where = ["archived_at IS NULL", "pinned=0", "expires_at IS NOT NULL", "expires_at<=?"]
        params: list[Any] = [checked_at]
        if namespace is not None:
            where.append("namespace=?")
            params.append(namespace)
        if scope_key is not None:
            where.append("scope_key=?")
            params.append(scope_key)
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
        return int(row[0]) if row else 0

    def archive_memory(self, memory_id: str) -> bool:
        """Mark one memory archived; returns False when missing or already archived."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET archived_at=? WHERE id=? AND archived_at IS NULL AND pinned=0",
                (_now(), memory_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def archive_memory_and_purge_library(self, memory_id: str) -> bool:
        """Archive a memory and purge its Library artifact in one transaction."""

        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE memories SET archived_at=? WHERE id=? AND archived_at IS NULL AND pinned=0",
                    (_now(), memory_id),
                )
                if cur.rowcount <= 0:
                    self._conn.rollback()
                    return False
                if self._has_library_artifact_locked(artifact_type="memory", artifact_id=memory_id):
                    self._delete_library_artifact_locked(artifact_type="memory", artifact_id=memory_id)
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        salience: float | None = None,
        metadata: JsonDict | None = None,
        pinned: bool | None = None,
        expires_at: float | None = None,
        update_content: bool = False,
        update_salience: bool = False,
        update_metadata: bool = False,
        update_pinned: bool = False,
        update_expires_at: bool = False,
    ) -> JsonDict | None:
        """Update an active memory row and return it, or None when missing/archived."""

        with self._lock:
            try:
                updated = self._update_memory_locked(
                    memory_id,
                    content=content,
                    salience=salience,
                    metadata=metadata,
                    pinned=pinned,
                    expires_at=expires_at,
                    update_content=update_content,
                    update_salience=update_salience,
                    update_metadata=update_metadata,
                    update_pinned=update_pinned,
                    update_expires_at=update_expires_at,
                )
                if updated is None:
                    self._conn.rollback()
                    return None
                self._conn.commit()
                return updated
            except Exception:
                self._conn.rollback()
                raise

    def update_memory_and_purge_library(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        salience: float | None = None,
        metadata: JsonDict | None = None,
        pinned: bool | None = None,
        expires_at: float | None = None,
        update_content: bool = False,
        update_salience: bool = False,
        update_metadata: bool = False,
        update_pinned: bool = False,
        update_expires_at: bool = False,
    ) -> JsonDict | None:
        """Update an active memory and purge its Library artifact in one transaction."""

        with self._lock:
            try:
                updated = self._update_memory_locked(
                    memory_id,
                    content=content,
                    salience=salience,
                    metadata=metadata,
                    pinned=pinned,
                    expires_at=expires_at,
                    update_content=update_content,
                    update_salience=update_salience,
                    update_metadata=update_metadata,
                    update_pinned=update_pinned,
                    update_expires_at=update_expires_at,
                )
                if updated is None:
                    self._conn.rollback()
                    return None
                if self._has_library_artifact_locked(artifact_type="memory", artifact_id=memory_id):
                    self._delete_library_artifact_locked(artifact_type="memory", artifact_id=memory_id)
                self._conn.commit()
                return updated
            except Exception:
                self._conn.rollback()
                raise

    def batch_update_memory_expiry(
        self,
        *,
        namespace: str,
        scope_key: str,
        expires_at: float | None,
        dry_run: bool = False,
    ) -> JsonDict:
        """Batch-update expiry for active, unpinned memories in one namespace/scope."""

        with self._lock:
            try:
                counts = self._conn.execute(
                    """
                    SELECT
                      SUM(CASE WHEN archived_at IS NULL AND pinned = 0 THEN 1 ELSE 0 END) AS would_update,
                      SUM(CASE WHEN archived_at IS NULL AND pinned = 1 THEN 1 ELSE 0 END) AS skipped_pinned,
                      SUM(CASE WHEN archived_at IS NOT NULL THEN 1 ELSE 0 END) AS skipped_archived
                    FROM memories
                    WHERE namespace=? AND scope_key=?
                    """,
                    (namespace, scope_key),
                ).fetchone()
                would_update = int(counts["would_update"] or 0) if counts else 0
                skipped_pinned = int(counts["skipped_pinned"] or 0) if counts else 0
                skipped_archived = int(counts["skipped_archived"] or 0) if counts else 0
                updated = 0
                if not dry_run and would_update > 0:
                    cur = self._conn.execute(
                        """
                        UPDATE memories
                        SET expires_at=?, last_used_at=?
                        WHERE namespace=? AND scope_key=? AND archived_at IS NULL AND pinned = 0
                        """,
                        (expires_at, _now(), namespace, scope_key),
                    )
                    updated = int(cur.rowcount or 0)
                self._conn.commit()
                return {
                    "namespace": namespace,
                    "dry_run": bool(dry_run),
                    "would_update": would_update,
                    "updated": updated,
                    "skipped_pinned": skipped_pinned,
                    "skipped_archived": skipped_archived,
                    "failed": 0,
                }
            except Exception:
                self._conn.rollback()
                raise

    def _update_memory_locked(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        salience: float | None = None,
        metadata: JsonDict | None = None,
        pinned: bool | None = None,
        expires_at: float | None = None,
        update_content: bool = False,
        update_salience: bool = False,
        update_metadata: bool = False,
        update_pinned: bool = False,
        update_expires_at: bool = False,
    ) -> JsonDict | None:
        now = _now()
        updates = ["last_used_at=?"]
        params: list[Any] = [now]
        if update_content:
            updates.append("content=?")
            params.append(content or "")
        if update_salience:
            updates.append("salience=?")
            params.append(max(0.0, float(salience or 0.0)))
        if update_metadata:
            updates.append("metadata=?")
            params.append(json.dumps(metadata or {}))
        if update_pinned:
            updates.append("pinned=?")
            params.append(1 if pinned else 0)
            if pinned:
                updates.append("expires_at=NULL")
        if update_expires_at and not (update_pinned and pinned):
            updates.append("expires_at=?")
            params.append(expires_at)
        params.append(memory_id)
        params.append(now)
        cur = self._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} "
            "WHERE id=? AND archived_at IS NULL "
            "AND (pinned=1 OR expires_at IS NULL OR expires_at>?)",
            params,
        )
        if cur.rowcount <= 0:
            return None
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        data = _row_to_dict(row)
        return _strip_embedding(_decode_metadata(data)) if data is not None else None

    def restore_memory(self, memory_id: str) -> bool:
        """Clear archival on one memory; returns False when missing or not archived."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET archived_at=NULL, expires_at=NULL "
                "WHERE id=? AND archived_at IS NOT NULL",
                (memory_id,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one memory by id."""

        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def delete_memory_and_purge_library(self, memory_id: str) -> bool:
        """Delete a memory and purge its Library artifact in one transaction."""

        with self._lock:
            try:
                cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                if cur.rowcount <= 0:
                    self._conn.rollback()
                    return False
                if self._has_library_artifact_locked(artifact_type="memory", artifact_id=memory_id):
                    self._delete_library_artifact_locked(artifact_type="memory", artifact_id=memory_id)
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def merge_memories_atomic(
        self,
        *,
        namespace: str,
        scope_key: str,
        content: str,
        salience: float,
        metadata: JsonDict | None,
        source_ids: list[str],
    ) -> str:
        """Insert a merged memory, archive sources, and purge source Library artifacts atomically."""

        merged_id = _uid("mem")
        ts = _now()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO memories(id,namespace,scope_key,content,salience,uses,created_at,last_used_at,metadata,embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        merged_id,
                        namespace,
                        scope_key,
                        content,
                        salience,
                        0,
                        ts,
                        ts,
                        json.dumps(metadata or {}),
                        None,
                    ),
                )
                for memory_id in source_ids:
                    cur = self._conn.execute(
                        "UPDATE memories SET archived_at=? WHERE id=? AND archived_at IS NULL",
                        (ts, memory_id),
                    )
                    if cur.rowcount <= 0:
                        raise ValueError(f"source memory could not be archived: {memory_id}")
                    if self._has_library_artifact_locked(artifact_type="memory", artifact_id=memory_id):
                        self._delete_library_artifact_locked(artifact_type="memory", artifact_id=memory_id)
                self._conn.commit()
                return merged_id
            except Exception:
                self._conn.rollback()
                raise

    def update_memory_salience(self, memory_id: str, salience: float) -> bool:
        """Update one memory's salience without touching usage counters."""

        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET salience=?, last_used_at=? WHERE id=?",
                (max(0.0, salience), _now(), memory_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def upsert_workspace_session_memory(
        self,
        *,
        session_id: str,
        workspace_key: str,
        content: str,
        metadata: JsonDict | None = None,
        salience: float = 2.5,
    ) -> JsonDict:
        """Create or update the current workspace summary memory for one session."""

        payload = {
            **(metadata or {}),
            "source": "session_distill_v1",
            "session_id": session_id,
            "workspace_key": workspace_key,
        }
        return self._upsert_memory_by_metadata(
            namespace="workspace",
            scope_key=workspace_key,
            source="session_distill_v1",
            match_key="session_id",
            match_value=session_id,
            content=content,
            salience=salience,
            metadata=payload,
        )

    def upsert_post_turn_memory(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_key: str,
        content: str,
        metadata: JsonDict | None = None,
        salience: float = 2.0,
    ) -> JsonDict:
        """Create or update the current turn-level workspace memory for one run."""

        payload = {
            **(metadata or {}),
            "source": "post_turn_memory_v1",
            "session_id": session_id,
            "run_id": run_id,
            "workspace_key": workspace_key,
        }
        match_key = "run_id" if run_id else "content_hash"
        match_value = str(payload.get(match_key) or "")
        if not match_value:
            match_key = "session_id"
            match_value = session_id
        return self._upsert_memory_by_metadata(
            namespace="workspace",
            scope_key=workspace_key,
            source="post_turn_memory_v1",
            match_key=match_key,
            match_value=match_value,
            content=content,
            salience=salience,
            metadata=payload,
        )

    def add_workspace_memory_fact_once(
        self,
        *,
        workspace_key: str,
        content: str,
        content_hash: str,
        metadata: JsonDict | None = None,
        salience: float = 2.2,
    ) -> JsonDict:
        """Insert an ADD-only memory_fact_v1 row unless this fact hash already exists."""

        payload = {
            **(metadata or {}),
            "source": "memory_fact_v1",
            "schema_version": "memory_fact_v1",
            "workspace_key": workspace_key,
            "content_hash": content_hash,
        }
        ts = _now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE namespace=? AND scope_key=? AND archived_at IS NULL",
                ("workspace", workspace_key),
            ).fetchall()
            for row in rows:
                data = _decode_metadata(_row_to_dict(row)) or {}
                meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                if meta.get("source") == "memory_fact_v1" and meta.get("content_hash") == content_hash:
                    return {"memory_id": str(data.get("id") or ""), "created": False, "memory": _strip_embedding(data)}
            memory_id = _uid("mem")
            self._conn.execute(
                "INSERT INTO memories(id,namespace,scope_key,content,salience,uses,created_at,last_used_at,metadata,embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    memory_id,
                    "workspace",
                    workspace_key,
                    content,
                    max(0.0, float(salience)),
                    0,
                    ts,
                    ts,
                    json.dumps(payload),
                    None,
                ),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            data = _strip_embedding(_decode_metadata(_row_to_dict(row))) if row else {}
            return {"memory_id": memory_id, "created": True, "memory": data}

    def upsert_user_profile_memory(
        self,
        *,
        user_key: str,
        content: str,
        sections: list[JsonDict],
        metadata: JsonDict | None = None,
        salience: float = 3.0,
    ) -> JsonDict:
        """Create or update the current multi-section user profile memory."""

        payload = {
            **(metadata or {}),
            "source": "user_profile_v1",
            "user_key": user_key,
            "sections": sections,
        }
        return self._upsert_memory_by_metadata(
            namespace="user",
            scope_key=user_key,
            source="user_profile_v1",
            match_key="source",
            match_value="user_profile_v1",
            content=content,
            salience=salience,
            metadata=payload,
        )

    def _upsert_memory_by_metadata(
        self,
        *,
        namespace: str,
        scope_key: str,
        source: str,
        match_key: str,
        match_value: str,
        content: str,
        salience: float,
        metadata: JsonDict,
    ) -> JsonDict:
        ts = _now()
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE namespace=? AND scope_key=? AND archived_at IS NULL",
                    (namespace, scope_key),
                ).fetchall()
                existing_id = ""
                for row in rows:
                    data = _decode_metadata(_row_to_dict(row)) or {}
                    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                    if meta.get("source") == source and meta.get(match_key) == match_value:
                        existing_id = str(data.get("id") or "")
                        break
                if existing_id:
                    self._conn.execute(
                        "UPDATE memories SET embedding=CASE WHEN content=? THEN embedding ELSE NULL END, "
                        "content=?, salience=?, last_used_at=?, metadata=? WHERE id=?",
                        (content, content, max(0.0, float(salience)), ts, json.dumps(metadata), existing_id),
                    )
                    memory_id = existing_id
                    created = False
                else:
                    memory_id = _uid("mem")
                    self._conn.execute(
                        "INSERT INTO memories(id,namespace,scope_key,content,salience,uses,created_at,last_used_at,metadata,embedding) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            memory_id,
                            namespace,
                            scope_key,
                            content,
                            max(0.0, float(salience)),
                            0,
                            ts,
                            ts,
                            json.dumps(metadata),
                            None,
                        ),
                    )
                    created = True
                self._conn.commit()
                row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                data = _strip_embedding(_decode_metadata(_row_to_dict(row))) if row else {}
                return {"memory_id": memory_id, "created": created, "memory": data}
            except Exception:
                self._conn.rollback()
                raise

    def touch_memory(self, memory_id: str, *, salience: float | None = None) -> None:
        """Mark a memory as used (and optionally update salience)."""

        with self._lock:
            if salience is None:
                self._conn.execute(
                    "UPDATE memories SET uses=uses+1, last_used_at=? WHERE id=?", (_now(), memory_id)
                )
            else:
                self._conn.execute(
                    "UPDATE memories SET uses=uses+1, last_used_at=?, salience=? WHERE id=?",
                    (_now(), salience, memory_id),
                )
            self._conn.commit()

    def delete_memories(self, *, namespace: str, scope_key: str) -> int:
        """Delete memories in a namespace/scope; returns the deleted count."""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE namespace=? AND scope_key=?", (namespace, scope_key)
            )
            self._conn.commit()
        return cur.rowcount

    # --- providers (secret-free) ---------------------------------------------

    def set_provider_status(self, status: JsonDict, *, provider_id: str = "default") -> None:
        """Persist secret-free provider readiness/config."""

        with self._lock:
            self._conn.execute(
                "INSERT INTO providers(id,llm_base_url,llm_model,embed_model,ollama_base_url,has_llm_key,brave_ready,firecrawl_ready,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "llm_base_url=excluded.llm_base_url, llm_model=excluded.llm_model, embed_model=excluded.embed_model, "
                "ollama_base_url=excluded.ollama_base_url, has_llm_key=excluded.has_llm_key, brave_ready=excluded.brave_ready, "
                "firecrawl_ready=excluded.firecrawl_ready, updated_at=excluded.updated_at",
                (
                    provider_id,
                    str(status.get("llm_base_url", "")),
                    str(status.get("llm_model", "")),
                    str(status.get("embed_model", "")),
                    str(status.get("ollama_base_url", "")),
                    int(bool(status.get("has_llm_key"))),
                    int(bool(status.get("brave_ready"))),
                    int(bool(status.get("firecrawl_ready"))),
                    _now(),
                ),
            )
            self._conn.commit()

    def get_provider_status(self, *, provider_id: str = "default") -> JsonDict | None:
        """Return secret-free provider status, or None."""

        row = self._conn.execute("SELECT * FROM providers WHERE id=?", (provider_id,)).fetchone()
        return _row_to_dict(row)

    # --- cache ----------------------------------------------------------------

    def cache_set(self, key: str, value: JsonDict, *, ttl_s: float = 3600.0) -> None:
        """Set a cache entry with a TTL."""

        with self._lock:
            self._conn.execute(
                "INSERT INTO cache(key,value,expires_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, expires_at=excluded.expires_at",
                (key, json.dumps(value), _now() + ttl_s),
            )
            self._conn.commit()

    def cache_get(self, key: str) -> JsonDict | None:
        """Return a non-expired cache entry, or None."""

        row = self._conn.execute("SELECT value, expires_at FROM cache WHERE key=?", (key,)).fetchone()
        if not row or float(row["expires_at"]) < _now():
            return None
        return _safe_json(row["value"])

    def cache_purge(self, *, scope: str = "all") -> int:
        """Delete retrieval cache entries for the requested scope."""

        with self._lock:
            if scope == "all":
                cur = self._conn.execute("DELETE FROM cache")
            elif scope == "search":
                cur = self._conn.execute("DELETE FROM cache WHERE key LIKE ?", ("search:%",))
            elif scope == "fetch":
                cur = self._conn.execute("DELETE FROM cache WHERE key LIKE ?", ("fetch:%",))
            else:
                raise ValueError(f"unsupported cache scope: {scope}")
            self._conn.commit()
        return int(cur.rowcount)

    # --- import ---------------------------------------------------------------

    def import_file_history(self, history_dir: str | Path) -> int:
        """Import legacy JSON report files into sessions/documents; returns count."""

        directory = Path(history_dir)
        if not directory.exists():
            return 0
        count = 0
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            query = str(data.get("query") or data.get("user_query") or path.stem)
            answer = str(data.get("report") or data.get("answer") or data.get("content") or "")
            sid = self.create_session(title=query[:80], metadata={"imported_from": path.name})
            self.add_message(sid, "user", query)
            if answer:
                self.add_message(sid, "assistant", answer)
            count += 1
        return count


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free text."""

    terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    return " OR ".join(terms) if terms else query


def _row_to_dict(row: sqlite3.Row | None) -> JsonDict | None:
    return {k: row[k] for k in row.keys()} if row is not None else None


def _decode_metadata(data: JsonDict | None) -> JsonDict:
    if data is None:
        return {}
    metadata = data.get("metadata")
    if isinstance(metadata, str):
        parsed = _safe_json(metadata)
        data["metadata"] = parsed if isinstance(parsed, dict) else {}
    elif metadata is None:
        data["metadata"] = {}
    return data


def _strip_embedding(data: JsonDict | None) -> JsonDict | None:
    """Drop the raw ``embedding`` column from a row dict (keeps API payloads lean)."""

    if data is not None:
        data.pop("embedding", None)
    return data


def _row_to_chunk(row: sqlite3.Row | None) -> ChunkRow | None:
    if row is None:
        return None
    embedding = _safe_json(row["embedding"]) if row["embedding"] else None
    return ChunkRow(
        id=int(row["id"]),
        document_id=str(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        embedding=embedding if isinstance(embedding, list) else None,
    )


def _safe_json(value: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _search_usage_policy_revision(metadata: Any) -> int:
    """Return the non-negative policy revision embedded in session metadata."""

    if not isinstance(metadata, dict):
        return 0
    try:
        return max(0, int(metadata.get(_SEARCH_USAGE_POLICY_REVISION) or 0))
    except (TypeError, ValueError):
        return 0
