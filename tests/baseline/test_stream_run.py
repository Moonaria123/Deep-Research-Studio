"""SSE stream robustness: keepalive, idle timeout, terminal-status fallback (DR-ITER-058)."""

from __future__ import annotations

import json

from deepresearch_core.baseline import BaselineService
from deepresearch_core.settings import ProviderSettings

KEEPALIVE = b": keepalive\n\n"


def _service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def _make_run(service: BaselineService, status: str = "running") -> str:
    store = service._store
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="fast", query="q")
    store.update_run(rid, status=status)
    return rid


def _frames(service: BaselineService, rid: str, **kwargs) -> list[bytes]:
    return list(service.stream_run(rid, **kwargs))


def test_stream_replays_events_and_stops_at_terminal_event():
    service = _service()
    rid = _make_run(service)
    store = service._store
    store.append_event(rid, "node_status", {"message": "working"})
    store.update_run(rid, status="completed", result={"answer": "a"})
    store.append_event(rid, "workflow_end", {"status": "completed"})

    frames = _frames(service, rid, poll_s=0.01, timeout_s=2)
    assert b"event: node_status" in frames[0]
    assert b"event: workflow_end" in frames[-1]
    assert len(frames) == 2


def test_stream_synthesizes_error_for_terminal_run_without_event():
    # A killed worker can leave status=error (failure callback) -- or even just a
    # terminal row with no terminal event. The stream must still end explicitly.
    service = _service()
    rid = _make_run(service)
    payload = {"error": {"code": "INTERNAL_ERROR", "message": "run terminated"}}
    service._store.update_run(rid, status="error", error=payload)

    frames = _frames(service, rid, poll_s=0.01, timeout_s=2)
    assert len(frames) == 1
    assert frames[0].startswith(b"event: error\n")
    body = json.loads(frames[0].split(b"data: ", 1)[1].strip())
    assert body["error"]["message"] == "run terminated"


def test_stream_synthesizes_workflow_end_for_completed_run_without_event():
    service = _service()
    rid = _make_run(service)
    service._store.update_run(rid, status="completed", result={"answer": "a"})

    frames = _frames(service, rid, poll_s=0.01, timeout_s=2)
    assert frames == [b'event: workflow_end\ndata: {"status": "completed"}\n\n']


def test_stream_synthesizes_workflow_end_for_partial_timeout_without_event():
    service = _service()
    rid = _make_run(service)
    service._store.update_run(rid, status="partial_timeout", result={"answer": "partial"})

    frames = _frames(service, rid, poll_s=0.01, timeout_s=2)
    body = json.loads(frames[0].split(b"data: ", 1)[1].strip())
    assert frames[0].startswith(b"event: workflow_end\n")
    assert body == {"status": "partial_timeout", "code": "RUN_TIMEOUT"}


def test_stream_synthesizes_workflow_end_for_completed_degraded_without_event():
    service = _service()
    rid = _make_run(service)
    service._store.update_run(
        rid,
        status="completed_degraded",
        result={"answer": "bounded", "degraded_reasons": ["temporal_coverage_unknown"]},
    )

    frames = _frames(service, rid, poll_s=0.01, timeout_s=2)
    body = json.loads(frames[0].split(b"data: ", 1)[1].strip())
    assert frames[0].startswith(b"event: workflow_end\n")
    assert body == {"status": "completed_degraded"}


def test_stream_emits_keepalive_while_idle_then_times_out():
    service = _service()
    rid = _make_run(service, status="running")

    frames = _frames(service, rid, poll_s=0.01, timeout_s=0.3, keepalive_s=0.05)
    # No events ever arrive: the stream sends keepalive comments, then ends on the
    # idle timeout without fabricating a terminal event for a still-running run.
    assert frames and all(frame == KEEPALIVE for frame in frames)
