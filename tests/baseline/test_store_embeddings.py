"""Store-level vector-index migration + semantic helpers (memories + messages)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from deepresearch_core.baseline.store import Store


def test_embedding_columns_added_to_memories_and_messages():
    store = Store(":memory:")
    mem_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    msg_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(messages)").fetchall()}
    assert "embedding" in mem_cols
    assert "embedding" in msg_cols


def test_legacy_db_gets_embedding_columns_backfilled(tmp_path: Path):
    db = tmp_path / "legacy.db"
    # Simulate a pre-migration database without embedding columns.
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, namespace TEXT, scope_key TEXT, content TEXT, "
        "salience REAL, uses INTEGER, created_at REAL, last_used_at REAL, metadata TEXT)"
    )
    conn.execute(
        "CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
        "created_at REAL, metadata TEXT)"
    )
    conn.commit()
    conn.close()

    store = Store(db)  # opening runs the additive migration
    mem_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    msg_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(messages)").fetchall()}
    mem_indexes = {r["name"] for r in store._conn.execute("PRAGMA index_list(memories)").fetchall()}
    assert "embedding" in mem_cols and "embedding" in msg_cols
    assert "archived_at" in mem_cols
    assert "idx_memories_archived" in mem_indexes


def test_iter_memories_with_embeddings_filters_and_parses():
    store = Store(":memory:")
    store.add_memory(namespace="session", scope_key="s1", content="with vec", embedding=[1.0, 0.0])
    store.add_memory(namespace="session", scope_key="s1", content="no vec")
    rows = store.iter_memories_with_embeddings(namespace="session", scope_key="s1")
    assert len(rows) == 1
    assert rows[0]["content"] == "with vec"
    assert rows[0]["embedding"] == [1.0, 0.0]


def test_search_messages_semantic_intra_session_ranks_by_cosine():
    store = Store(":memory:")
    sid = store.create_session(title="t")
    store.add_message(sid, "user", "apples", embedding=[1.0, 0.0])
    store.add_message(sid, "user", "oranges", embedding=[0.0, 1.0])
    hits = store.search_messages_semantic([1.0, 0.0], session_id=sid, limit=5)
    assert hits[0]["content"] == "apples"
    assert hits[0]["score"] > hits[1]["score"]
    assert "embedding" not in hits[0]


def test_search_messages_semantic_cross_session_spans_conversations():
    store = Store(":memory:")
    s1 = store.create_session(title="a")
    s2 = store.create_session(title="b")
    store.add_message(s1, "user", "alpha", embedding=[1.0, 0.0])
    store.add_message(s2, "user", "beta", embedding=[0.9, 0.1])
    hits = store.search_messages_semantic([1.0, 0.0], session_id="", limit=5)
    contents = {h["content"] for h in hits}
    assert contents == {"alpha", "beta"}


def test_search_messages_semantic_skips_messages_without_embeddings():
    store = Store(":memory:")
    sid = store.create_session(title="t")
    store.add_message(sid, "user", "no embedding here")
    assert store.search_messages_semantic([1.0, 0.0], session_id=sid) == []


def test_list_messages_does_not_leak_embedding():
    store = Store(":memory:")
    sid = store.create_session(title="t")
    store.add_message(sid, "user", "hello", embedding=[1.0, 2.0])
    msgs = store.list_messages(sid)
    assert msgs and "embedding" not in msgs[0]
