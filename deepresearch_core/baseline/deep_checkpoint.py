"""Secret-safe, versioned codec for resumable Deep Research checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

JsonDict = dict[str, Any]

CHECKPOINT_SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024
CHECKPOINT_STAGES = frozenset({"plan_ready", "round_complete", "final_evidence_ready"})

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "lineage_id",
        "parent_run_id",
        "request",
        "research_intent",
        "plan",
        "pending_tasks",
        "completed_sections",
        "evidence_snapshot",
        "retrieval_state",
        "temporal_coverage",
        "conflicts",
        "evidence_eligibility",
        "grounding",
        "remediation",
        "phase_audit",
        "budget_usage",
    }
)
_REQUIRED_FIELDS = _TOP_LEVEL_FIELDS
_DENIED_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "authorization",
        "auth_header",
        "client_secret",
        "password",
        "prompt",
        "raw_prompt",
        "raw_response",
        "reasoning",
        "hidden_reasoning",
        "chain_of_thought",
        "process_object",
    }
)
_SECRET_VALUE_RE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"\bsk-[A-Za-z0-9_-]{12,}|"
    r"\b(?:ghp|xox[baprs])_[A-Za-z0-9_-]{12,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


class CheckpointCodecError(ValueError):
    """Raised when checkpoint content is unsafe, malformed, or not canonical."""


class CheckpointAlreadyContinued(RuntimeError):
    """Raised when another idempotency key already claimed a checkpoint."""


class CheckpointStateConflict(RuntimeError):
    """Raised when a checkpoint revision or disposition changed concurrently."""


@dataclass(frozen=True)
class EncodedDeepCheckpoint:
    """Canonical serialized checkpoint ready for a short SQLite transaction."""

    payload_json: str
    payload_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DeepCheckpointV1:
    """Allowlisted checkpoint state saved only at a Deep safe boundary."""

    schema_version: int
    stage: str
    lineage_id: str
    parent_run_id: str
    request: JsonDict
    research_intent: JsonDict
    plan: JsonDict
    pending_tasks: list[JsonDict]
    completed_sections: list[JsonDict]
    evidence_snapshot: list[JsonDict]
    retrieval_state: JsonDict
    temporal_coverage: JsonDict
    conflicts: list[JsonDict]
    evidence_eligibility: list[JsonDict]
    grounding: JsonDict
    remediation: JsonDict
    phase_audit: list[JsonDict]
    budget_usage: JsonDict

    @classmethod
    def from_dict(cls, value: Any) -> "DeepCheckpointV1":
        """Validate an untrusted JSON-like value against the v1 allowlist."""

        if not isinstance(value, dict):
            raise CheckpointCodecError("checkpoint payload must be an object")
        unknown = set(value) - _TOP_LEVEL_FIELDS
        missing = _REQUIRED_FIELDS - set(value)
        if unknown:
            raise CheckpointCodecError("checkpoint payload contains unknown fields")
        if missing:
            raise CheckpointCodecError("checkpoint payload is missing required fields")
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCodecError("unsupported checkpoint schema version")
        stage = str(value.get("stage") or "")
        if stage not in CHECKPOINT_STAGES:
            raise CheckpointCodecError("checkpoint stage is not a safe boundary")
        lineage_id = _bounded_identifier(value.get("lineage_id"), "lineage_id")
        parent_run_id = _bounded_identifier(value.get("parent_run_id"), "parent_run_id")
        _validate_secret_safe(value)
        normalized = _json_clone(value)
        return cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            stage=stage,
            lineage_id=lineage_id,
            parent_run_id=parent_run_id,
            request=_require_dict(normalized, "request"),
            research_intent=_require_dict(normalized, "research_intent"),
            plan=_require_dict(normalized, "plan"),
            pending_tasks=_require_dict_list(normalized, "pending_tasks"),
            completed_sections=_require_dict_list(normalized, "completed_sections"),
            evidence_snapshot=_require_dict_list(normalized, "evidence_snapshot"),
            retrieval_state=_require_dict(normalized, "retrieval_state"),
            temporal_coverage=_require_dict(normalized, "temporal_coverage"),
            conflicts=_require_dict_list(normalized, "conflicts"),
            evidence_eligibility=_require_dict_list(normalized, "evidence_eligibility"),
            grounding=_require_dict(normalized, "grounding"),
            remediation=_require_dict(normalized, "remediation"),
            phase_audit=_require_dict_list(normalized, "phase_audit"),
            budget_usage=_require_dict(normalized, "budget_usage"),
        )

    def to_dict(self) -> JsonDict:
        """Return a detached JSON-compatible representation."""

        return _json_clone(
            {
                "schema_version": self.schema_version,
                "stage": self.stage,
                "lineage_id": self.lineage_id,
                "parent_run_id": self.parent_run_id,
                "request": self.request,
                "research_intent": self.research_intent,
                "plan": self.plan,
                "pending_tasks": self.pending_tasks,
                "completed_sections": self.completed_sections,
                "evidence_snapshot": self.evidence_snapshot,
                "retrieval_state": self.retrieval_state,
                "temporal_coverage": self.temporal_coverage,
                "conflicts": self.conflicts,
                "evidence_eligibility": self.evidence_eligibility,
                "grounding": self.grounding,
                "remediation": self.remediation,
                "phase_audit": self.phase_audit,
                "budget_usage": self.budget_usage,
            }
        )


def encode_deep_checkpoint(value: DeepCheckpointV1 | JsonDict) -> EncodedDeepCheckpoint:
    """Return canonical JSON and its SHA-256 after re-validating the payload."""

    checkpoint = value if isinstance(value, DeepCheckpointV1) else DeepCheckpointV1.from_dict(value)
    payload = checkpoint.to_dict()
    _validate_secret_safe(payload)
    try:
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCodecError("checkpoint payload is not valid JSON") from error
    payload_bytes = payload_json.encode("utf-8")
    if len(payload_bytes) > MAX_CHECKPOINT_BYTES:
        raise CheckpointCodecError("checkpoint payload exceeds the 4 MiB limit")
    return EncodedDeepCheckpoint(
        payload_json=payload_json,
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        size_bytes=len(payload_bytes),
    )


def decode_deep_checkpoint(payload_json: str, payload_sha256: str) -> DeepCheckpointV1:
    """Verify hash, schema, safety, and canonical form before returning a checkpoint."""

    if not isinstance(payload_json, str):
        raise CheckpointCodecError("checkpoint payload must be JSON text")
    payload_bytes = payload_json.encode("utf-8")
    if len(payload_bytes) > MAX_CHECKPOINT_BYTES:
        raise CheckpointCodecError("checkpoint payload exceeds the 4 MiB limit")
    actual_hash = hashlib.sha256(payload_bytes).hexdigest()
    if not isinstance(payload_sha256, str) or actual_hash != payload_sha256.lower():
        raise CheckpointCodecError("checkpoint payload hash mismatch")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise CheckpointCodecError("checkpoint payload is malformed JSON") from error
    checkpoint = DeepCheckpointV1.from_dict(payload)
    canonical = encode_deep_checkpoint(checkpoint)
    if canonical.payload_json != payload_json:
        raise CheckpointCodecError("checkpoint payload is not canonical")
    return checkpoint


def checkpoint_integrity(payload_json: Any, payload_sha256: Any) -> str:
    """Return ``valid`` or ``invalid`` without exposing codec details."""

    try:
        decode_deep_checkpoint(str(payload_json or ""), str(payload_sha256 or ""))
    except CheckpointCodecError:
        return "invalid"
    return "valid"


def _bounded_identifier(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 200:
        raise CheckpointCodecError(f"{field_name} must be a bounded non-empty identifier")
    return text


def _require_dict(value: JsonDict, field_name: str) -> JsonDict:
    item = value.get(field_name)
    if not isinstance(item, dict):
        raise CheckpointCodecError(f"{field_name} must be an object")
    return item


def _require_dict_list(value: JsonDict, field_name: str) -> list[JsonDict]:
    items = value.get(field_name)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise CheckpointCodecError(f"{field_name} must be an array of objects")
    return items


def _validate_secret_safe(value: Any, *, key: str = "") -> None:
    normalized_key = str(key or "").strip().lower()
    if normalized_key in _DENIED_KEYS or normalized_key.endswith(("_api_key", "_token", "_secret")):
        raise CheckpointCodecError("checkpoint contains a forbidden sensitive field")
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _validate_secret_safe(child_value, key=str(child_key))
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_secret_safe(child, key=key)
        return
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise CheckpointCodecError("checkpoint contains credential-like material")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise CheckpointCodecError("checkpoint contains a non-JSON process object")


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise CheckpointCodecError("checkpoint payload is not valid JSON") from error
