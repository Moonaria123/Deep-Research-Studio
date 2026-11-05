"""Inventory or explicitly enqueue missing semantic continuity work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch_core.baseline.job_backend import redis_url  # noqa: E402
from deepresearch_core.baseline.semantic_enrichment import (  # noqa: E402
    MAX_BATCH_ROWS,
    MissingEmbeddingRow,
    RQSemanticEnrichmentBackend,
    SemanticEnrichmentRequest,
    build_embedding_batch,
)
from deepresearch_core.baseline.store import Store  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory missing semantic continuity rows; enqueue only with --enqueue."
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("DEEPRESEARCH_DB_PATH", "data/baseline.sqlite3"),
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Submit secret-free session requests to the dedicated maintenance queue.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=1000,
        help="Maximum number of active sessions to inspect in this invocation.",
    )
    parser.add_argument(
        "--resume-cursor",
        default="",
        help="Resume after this opaque session id.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _inventory(
    store: Store,
    *,
    max_sessions: int,
    resume_cursor: str,
) -> tuple[dict[str, object], list[str]]:
    remaining = max(0, int(max_sessions))
    cursor = str(resume_cursor or "")
    sessions: list[dict[str, object]] = []
    while remaining > 0:
        page_size = min(100, remaining)
        rows = store.list_sessions_missing_embeddings(
            after_session_id=cursor or None,
            limit=page_size,
        )
        if not rows:
            break
        sessions.extend(rows)
        cursor = str(rows[-1]["session_id"])
        remaining -= len(rows)
        if len(rows) < page_size:
            break

    message_count = sum(int(row["messages"]) for row in sessions)
    memory_count = sum(int(row["memories"]) for row in sessions)
    estimated_batches = sum(
        _estimate_session_batches(store, str(row["session_id"]))
        for row in sessions
    )
    payload: dict[str, object] = {
        "sessions": len(sessions),
        "messages": message_count,
        "memories": memory_count,
        "estimated_batches": estimated_batches,
        "resume_cursor": cursor or None,
        "enqueue_requested": False,
        "enqueued": 0,
    }
    return payload, [str(row["session_id"]) for row in sessions]


def _estimate_session_batches(store: Store, session_id: str) -> int:
    """Mirror the real row/byte batch planner without emitting row content."""

    cursor_created_at: float | None = None
    cursor_id: str | None = None
    batches = 0
    while True:
        values = store.list_missing_embedding_rows(
            session_id,
            after_created_at=cursor_created_at,
            after_id=cursor_id,
            limit=MAX_BATCH_ROWS,
        )
        if not values:
            return batches
        rows = [MissingEmbeddingRow.from_value(value) for value in values]
        batch_rows, _, _ = build_embedding_batch(rows)
        if not batch_rows:
            # The real job will fail this malformed row rather than spin. Count the
            # attempted batch once so inventory remains bounded and conservative.
            return batches + 1
        batches += 1
        cursor_created_at = batch_rows[-1].created_at
        cursor_id = batch_rows[-1].id


def _enqueue(session_ids: Sequence[str]) -> int:
    try:
        import redis as redis_lib
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("--enqueue requires the existing redis package") from error

    connection = redis_lib.Redis.from_url(
        redis_url(),
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )
    connection.ping()
    backend = RQSemanticEnrichmentBackend(connection)
    for session_id in session_ids:
        backend.enqueue(
            SemanticEnrichmentRequest(
                session_id=session_id,
                trigger="backlog",
            )
        )
    return len(session_ids)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded, dry-run-by-default continuity inventory."""

    args = _parser().parse_args(argv)
    if args.max_sessions < 1:
        raise SystemExit("--max-sessions must be positive")

    store = Store(
        Path(args.db_path).expanduser(),
        encryption_key=str(os.environ.get("DEEPRESEARCH_DB_KEY") or "").strip() or None,
        read_only=True,
    )
    try:
        result, session_ids = _inventory(
            store,
            max_sessions=args.max_sessions,
            resume_cursor=args.resume_cursor,
        )
    finally:
        store.close()

    if args.enqueue:
        result["enqueue_requested"] = True
        result["enqueued"] = _enqueue(session_ids)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        mode = "enqueue" if args.enqueue else "dry-run"
        print(
            "Semantic continuity "
            f"{mode}: sessions={result['sessions']}, messages={result['messages']}, "
            f"memories={result['memories']}, batches={result['estimated_batches']}, "
            f"enqueued={result['enqueued']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
