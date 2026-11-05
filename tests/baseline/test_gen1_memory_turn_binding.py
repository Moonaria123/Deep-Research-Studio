"""Delayed Memory jobs must capture their own completed turn."""

import hashlib

import pytest

from deepresearch_core.baseline.memory_distillation import MemoryDistillationError, MemoryDistillationService
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM


def _distiller(store):
    llm = FakeLLM(scripted={
        "Capture the last completed DeepResearch turn": '{"summary":"Tides follow lunar gravity."}',
    })
    return MemoryDistillationService(store=store, memory_llm=lambda: llm), llm


def _turn(store, session_id, run_id, marker):
    return [
        store.add_message(session_id, role, f"{marker}-{role}", metadata={"run_id": run_id})
        for role in ("user", "assistant")
    ]


def test_delayed_job_uses_its_run_messages_and_watermark(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    try:
        session_id = store.create_session(title="QA turn binding")
        first_ids = _turn(store, session_id, "run-a", "FIRST_TURN")
        _turn(store, session_id, "run-b", "FUTURE_TURN")
        distiller, llm = _distiller(store)

        result = distiller.refresh_post_turn(session_id=session_id, run_id="run-a")

        prompt = llm.calls[0][-1]["content"]
        assert "FIRST_TURN-user" in prompt and "FIRST_TURN-assistant" in prompt
        assert "FUTURE_TURN" not in prompt
        row = store.list_memories(namespace="workspace", scope_key="default")[0]
        assert result["memory_id"] == row["id"]
        assert row["metadata"]["last_run_id"] == "run-a"
        assert row["metadata"]["last_message_id"] == first_ids[-1]
        digest = hashlib.sha256()
        for mid, role in zip(first_ids, ("user", "assistant")):
            digest.update(mid.encode())
            digest.update(f"FIRST_TURN-{role}".encode())
        assert row["metadata"]["content_hash"] == digest.hexdigest()[:16]
    finally:
        store.close()


@pytest.mark.parametrize("wrong_run", ["missing-run", "other-session-run"])
def test_unmatched_run_cannot_capture_another_turn(tmp_path, wrong_run):
    store = Store(tmp_path / "memory.sqlite3")
    try:
        session_id = store.create_session(title="QA selected session")
        other_session = store.create_session(title="QA other session")
        _turn(store, session_id, "correct-run", "SELECTED")
        _turn(store, other_session, "other-session-run", "OTHER")
        distiller, llm = _distiller(store)

        with pytest.raises(MemoryDistillationError) as exc:
            distiller.refresh_post_turn(session_id=session_id, run_id=wrong_run)

        assert exc.value.error_code == "POST_TURN_SOURCE_MISSING"
        assert llm.calls == []
        assert store.list_memories(namespace="workspace", scope_key="default") == []
    finally:
        store.close()


def test_manual_capture_without_run_retains_latest_turn_fallback(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    try:
        session_id = store.create_session(title="QA manual capture")
        store.add_message(session_id, "user", "Legacy untagged message")
        store.add_message(session_id, "assistant", "Legacy answer")
        distiller, llm = _distiller(store)

        result = distiller.refresh_post_turn(session_id=session_id)

        assert result["status"] == "completed"
        assert "Legacy untagged message" in llm.calls[0][-1]["content"]
    finally:
        store.close()
