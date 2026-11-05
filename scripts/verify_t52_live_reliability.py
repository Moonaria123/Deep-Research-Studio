"""Opt-in deployed T52 Live reliability verifier.

Each invocation runs exactly one locked scenario through the public FastAPI run
surface and the configured RQ worker.  The verifier records only secret-free
terminal, SSE, grounding, citation-binding, deadline, and writer diagnostic
evidence.  It never reads provider credentials and is inert without the
explicit ``--confirm-live-check`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch_core.baseline.evidence_eligibility import (  # noqa: E402
    output_shape_valid,
    parse_research_intent,
)


SUCCESS_STATUSES = {"completed", "completed_degraded"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {"partial_timeout", "error", "cancelled"}
CASES: dict[str, dict[str, Any]] = {
    "l1": {
        "mode": "fast",
        "thinking_effort": None,
        "timeout_s": 60,
        "query": "What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
        "anchors": ("fts5", "full-text"),
    },
    "l2": {
        "mode": "fast",
        "thinking_effort": "off",
        "timeout_s": 60,
        "query": "What is Python pathlib.Path? Use one authoritative source and answer in two concise sentences.",
        "anchors": ("pathlib", "path"),
    },
    "l3": {
        "mode": "multiple",
        "thinking_effort": None,
        "timeout_s": 60,
        "query": (
            "Compare SQLite FTS5 and PostgreSQL full-text search using authoritative sources. "
            "Answer in exactly 4 concise sentences with citations."
        ),
        "anchors": ("sqlite", "postgresql", "full-text"),
    },
    "l4": {
        "mode": "multiple",
        "thinking_effort": "off",
        "timeout_s": 60,
        "query": (
            "Compare SQLite FTS5 and PostgreSQL full-text search using authoritative sources. "
            "Answer in exactly 4 concise sentences with citations."
        ),
        "anchors": ("sqlite", "postgresql", "full-text"),
    },
    "l5": {
        "mode": "deep",
        "thinking_effort": None,
        "timeout_s": 180,
        "query": (
            "Write a concise research report explaining SQLite FTS5, its MATCH query syntax, "
            "and its ranking capabilities using authoritative SQLite documentation. "
            "Use exactly 6 concise sentences with citations."
        ),
        "anchors": ("fts5", "match", ("rank", "bm25")),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one opt-in deployed T52 Live scenario.")
    parser.add_argument("--confirm-live-check", action="store_true")
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--api-origin", default="http://127.0.0.1:18787")
    parser.add_argument("--model", default="poe:gpt-5.4")
    parser.add_argument("--existing-run-id", default="")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("API response was not an object")
    return parsed


def _answer_is_relevant(text: str, anchors: Sequence[str | Sequence[str]]) -> bool:
    """Require every topic while allowing explicit domain-equivalent anchor terms."""

    normalized = str(text or "").casefold()
    for raw_group in anchors:
        group = (raw_group,) if isinstance(raw_group, str) else tuple(raw_group)
        alternatives = tuple(str(value).casefold() for value in group if str(value))
        if not alternatives or not any(value in normalized for value in alternatives):
            return False
    return True


def _read_sse(url: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
        content_type = response.headers.get("Content-Type", "")
    frames = [frame for frame in body.replace("\r\n", "\n").split("\n\n") if "event:" in frame]
    event_types: list[str] = []
    event_ids: list[int] = []
    workflow_end_statuses: list[str] = []
    answer_delta_bytes = 0
    answer_resets = 0
    for frame in frames:
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        event_type = fields.get("event", "")
        if event_type:
            event_types.append(event_type)
        if fields.get("id", "").isdigit():
            event_ids.append(int(fields["id"]))
        if event_type == "answer_reset":
            answer_resets += 1
        data_text = fields.get("data", "")
        if event_type == "answer_delta" and data_text:
            try:
                data = json.loads(data_text)
                answer_delta_bytes += len(str(data.get("delta") or "").encode("utf-8"))
            except json.JSONDecodeError:
                pass
        if event_type == "workflow_end" and data_text:
            try:
                data = json.loads(data_text)
                if data.get("status"):
                    workflow_end_statuses.append(str(data["status"]))
            except json.JSONDecodeError:
                pass
    return {
        "content_type": content_type,
        "event_count": len(event_types),
        "event_types": event_types,
        "first_seq": event_ids[0] if event_ids else None,
        "last_seq": event_ids[-1] if event_ids else None,
        "strictly_increasing_ids": event_ids == sorted(set(event_ids)),
        "workflow_end_count": event_types.count("workflow_end"),
        "workflow_end_statuses": workflow_end_statuses,
        "answer_delta_count": event_types.count("answer_delta"),
        "answer_delta_bytes": answer_delta_bytes,
        "answer_reset_count": answer_resets,
    }


def _error_code(run: dict[str, Any]) -> str:
    error = run.get("error")
    if not isinstance(error, dict):
        return ""
    envelope = error.get("error")
    return str(envelope.get("code") or "") if isinstance(envelope, dict) else ""


def _writer_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
    for owner in (run.get("result"), run.get("metadata")):
        if isinstance(owner, dict) and isinstance(owner.get("writer_diagnostics"), dict):
            return dict(owner["writer_diagnostics"])
    return {}


def _deadline_summary(run: dict[str, Any]) -> dict[str, Any]:
    for owner in (run.get("result"), run.get("metadata")):
        if isinstance(owner, dict) and isinstance(owner.get("deadline_summary"), dict):
            return dict(owner["deadline_summary"])
    return {}


def _stream_metrics(run: dict[str, Any]) -> dict[str, Any]:
    for owner in (run.get("result"), run.get("metadata")):
        if isinstance(owner, dict) and isinstance(owner.get("stream_metrics"), dict):
            return dict(owner["stream_metrics"])
    return {}


def _citation_binding(answer: dict[str, Any]) -> dict[str, Any]:
    citations = [item for item in answer.get("citations") or [] if isinstance(item, dict)]
    citation_ids = {str(item.get("id") or "") for item in citations if item.get("id")}
    claim_map = answer.get("claim_map") if isinstance(answer.get("claim_map"), dict) else {}
    bound_ids = {
        str(evidence_id)
        for values in claim_map.values()
        if isinstance(values, list)
        for evidence_id in values
        if evidence_id
    }
    return {
        "citation_ids": sorted(citation_ids),
        "bound_evidence_ids": sorted(bound_ids),
        "claim_count": len(claim_map),
        "valid": bool(citation_ids and claim_map) and bound_ids <= citation_ids,
    }


def _provider_success(diagnostics: dict[str, Any], *, expected_provider: str) -> dict[str, Any]:
    attempts = [dict(item) for item in diagnostics.get("attempts") or [] if isinstance(item, dict)]
    providers = [str(item.get("provider") or "") for item in attempts]
    final_provider = str(diagnostics.get("provider") or "")
    cross_backend = any(str(item.get("fallback_scope") or "") == "cross_backend" for item in attempts)
    return {
        "expected_provider": expected_provider,
        "final_provider": final_provider,
        "attempt_providers": providers,
        "cross_backend_fallback": cross_backend,
        "valid": bool(attempts) and final_provider == expected_provider and not cross_backend and all(
            provider == expected_provider for provider in providers
        ),
    }


def _safe_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "classification",
        "provider",
        "model",
        "finish_reason",
        "visible_chars",
        "usage",
        "fallback_scope",
        "call_options",
        "attempts",
        "writer_started_at",
        "deadline_at",
    }
    return {key: value for key, value in diagnostics.items() if key in allowed}


def _artifact_path(evidence_dir: str, case_id: str) -> Path:
    directory = Path(evidence_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case_id}.json"
    if path.exists():
        raise FileExistsError(f"evidence already exists: {path.name}")
    return path


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    spec = CASES[args.case]
    base = args.api_origin.rstrip("/") + "/api/v1"
    llm_status = _request_json("GET", f"{base}/providers/llm")
    roles = llm_status.get("roles") if isinstance(llm_status.get("roles"), dict) else {}
    started_at = time.time()
    existing_run_id = str(args.existing_run_id or "").strip()
    if existing_run_id:
        run_id = existing_run_id
        run = _request_json("GET", f"{base}/runs/{run_id}", timeout_s=30.0)
        if run.get("query") != spec["query"] or run.get("mode") != spec["mode"]:
            raise RuntimeError("existing run does not match the locked case")
        session_id = str(run.get("session_id") or "")
    else:
        payload: dict[str, Any] = {
            "query": spec["query"],
            "mode": spec["mode"],
            "llm_model": args.model,
            "library_scope": "off",
            "memory_policy": "off",
        }
        if spec["thinking_effort"] is not None:
            payload["thinking_effort"] = spec["thinking_effort"]
        created = _request_json("POST", f"{base}/runs", payload=payload, timeout_s=30.0)
        run_id = str(created.get("run_id") or created.get("id") or "")
        session_id = str(created.get("session_id") or "")
        if not run_id or not session_id:
            raise RuntimeError("POST /runs returned incomplete identifiers")
    sse = _read_sse(
        f"{base}/runs/{run_id}/stream?after_seq=0",
        timeout_s=float(spec["timeout_s"]) + 30.0,
    )
    run = _request_json("GET", f"{base}/runs/{run_id}", timeout_s=30.0)
    terminal_at = time.time()
    if existing_run_id:
        started_at = float(run.get("created_at") or started_at)
        terminal_at = float(run.get("updated_at") or terminal_at)
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
    text = str(answer.get("text") or "")
    anchors = tuple(spec["anchors"])
    binding = _citation_binding(answer)
    diagnostics = _writer_diagnostics(run)
    provider = _provider_success(diagnostics, expected_provider="poe")
    deadline = _deadline_summary(run)
    stream_metrics = _stream_metrics(run)
    status = str(run.get("status") or "")
    shape_valid = output_shape_valid(parse_research_intent(str(spec["query"])), text)
    citations = [item for item in answer.get("citations") or [] if isinstance(item, dict)]
    sources = [item for item in result.get("sources") or [] if isinstance(item, dict)]
    deadline_met = bool(deadline.get("deadline_met"))
    workflow_end_valid = (
        sse["workflow_end_count"] == 1
        and sse["workflow_end_statuses"] == [status]
        and bool(sse["strictly_increasing_ids"])
    )
    grounded = bool(text.strip() and citations and float(answer.get("coverage") or 0.0) > 0.0)
    relevant = _answer_is_relevant(text, anchors)
    ok = all(
        (
            status in SUCCESS_STATUSES,
            grounded,
            relevant,
            shape_valid,
            bool(binding["valid"]),
            bool(provider["valid"]),
            deadline_met,
            workflow_end_valid,
            int(sse["answer_delta_count"]) > 0,
        )
    )
    return {
        "schema": "t52-live-v1",
        "case": args.case,
        "ok": ok,
        "status": status,
        "error_code": _error_code(run),
        "run_id": run_id,
        "session_id": session_id,
        "mode": spec["mode"],
        "thinking_effort": spec["thinking_effort"] or "configured_default",
        "requested_model": args.model,
        "runtime_default_model": roles.get("default"),
        "replayed_existing_run": bool(existing_run_id),
        "elapsed_ms": round((terminal_at - started_at) * 1000.0, 3),
        "answer": {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "visible_chars": len(text),
            "coverage": answer.get("coverage"),
            "citation_count": len(citations),
            "source_count": len(sources),
            "shape_valid": shape_valid,
            "relevant": relevant,
            "citation_binding": binding,
            "citations": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "origin": item.get("origin"),
                    "artifact_type": item.get("artifact_type"),
                    "search_provider": item.get("search_provider"),
                    "fetch_provider": item.get("fetch_provider"),
                    "grounding_eligible": item.get("grounding_eligible"),
                }
                for item in citations
            ],
        },
        "provider_success": provider,
        "writer_diagnostics": _safe_diagnostics(diagnostics),
        "deadline_summary": deadline,
        "stream_metrics": stream_metrics,
        "sse": sse,
        "degraded_reasons": result.get("degraded_reasons") or [],
        "retrieval_outcomes": result.get("retrieval_outcomes") or [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.confirm_live_check:
        payload = {
            "check": "t52_live_reliability",
            "case": args.case,
            "ok": False,
            "status": "skipped",
            "reason": "confirm_live_check_required",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["status"])
        return 0
    try:
        path = _artifact_path(args.evidence_dir, args.case)
        payload = run_case(args)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["evidence_file"] = str(path)
        exit_code = 0 if payload["ok"] else 1
    except Exception as error:  # noqa: BLE001 - verifier must emit a bounded diagnostic
        payload = {
            "check": "t52_live_reliability",
            "case": args.case,
            "ok": False,
            "status": "failed",
            "reason": f"{type(error).__name__}: {str(error)[:300]}",
        }
        exit_code = 1
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else str(payload["status"]))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
