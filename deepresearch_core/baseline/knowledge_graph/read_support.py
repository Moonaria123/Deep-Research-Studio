"""Public DTO conversion helpers for bounded Knowledge Graph reads."""

from __future__ import annotations

import json
from typing import Any, Iterable


JsonDict = dict[str, Any]
KG_PROVENANCE_BYTES_MAX = 16 * 1024


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def public_version(value: JsonDict | None) -> JsonDict | None:
    if not value:
        return None
    return {
        "id": str(value.get("id") or ""),
        "status": str(value.get("status") or ""),
        "activated_at": float(value.get("activated_at") or 0.0),
    }


def public_node(row: Any, *, query: str = "") -> JsonDict:
    attrs = json_object(row["attrs_json"])
    title = str(row["title"] or "")
    summary = str(row["summary"] or "")
    key = str(row["node_key"] or "")
    payload: JsonDict = {
        "id": str(row["id"] or ""),
        "type": str(row["node_type"] or ""),
        "key": key,
        "title": title,
        "summary": summary,
        "attrs": attrs,
        "confidence": float(row["confidence"] or 0.0),
        "visible": bool(row["visible"]),
    }
    if query:
        lowered_title = title.lower()
        lowered_key = key.lower()
        lowered_summary = summary.lower()
        if query in {lowered_title, lowered_key}:
            score = 4.0
        elif lowered_title.startswith(query) or lowered_key.startswith(query):
            score = 3.0
        elif query in lowered_title or query in lowered_key:
            score = 2.0
        elif query in lowered_summary:
            score = 1.0
        else:
            score = 0.5
        payload["match_score"] = score
    return payload


def public_edge(row: Any) -> JsonDict:
    return {
        "id": str(row["id"] or ""),
        "type": str(row["edge_type"] or ""),
        "source": str(row["src_node_id"] or ""),
        "target": str(row["dst_node_id"] or ""),
        "weight": float(row["weight"] or 0.0),
        "confidence": float(row["confidence"] or 0.0),
        "visible": bool(row["visible"]),
        "attrs": json_object(row["attrs_json"]),
    }


def public_provenance(row: Any) -> JsonDict:
    return {
        "id": str(row["id"] or ""),
        "owner_kind": str(row["owner_kind"] or ""),
        "source_kind": str(row["source_kind"] or ""),
        "source_id": str(row["source_id"] or ""),
        "span": json_object(row["span_json"]),
        "evidence": json_object(row["evidence_json"]),
        "created_at": float(row["created_at"] or 0.0),
    }


def bounded_provenance(rows: Iterable[Any], *, limit: int) -> tuple[list[JsonDict], bool]:
    selected: list[JsonDict] = []
    truncated = False
    for index, row in enumerate(rows):
        if index >= limit:
            truncated = True
            break
        item = public_provenance(row)
        candidate = [*selected, item]
        encoded = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(encoded.encode("utf-8")) > KG_PROVENANCE_BYTES_MAX:
            truncated = True
            break
        selected.append(item)
    return selected, truncated


def public_job(value: JsonDict) -> JsonDict:
    return {
        "id": str(value.get("id") or ""),
        "status": str(value.get("status") or ""),
        "stage": str(value.get("stage") or ""),
        "progress": float(value.get("progress") or 0.0),
        "started_at": float(value.get("started_at") or 0.0),
        "finished_at": float(value.get("finished_at") or 0.0),
        "error_code": str(value.get("error_code") or ""),
    }


def json_object(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
