"""Memory lifecycle operations for managed local memories."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from .errors import BaselineError, ErrorCode
from .memory import MemoryLayer
from .store import Store

JsonDict = dict[str, Any]

_MANAGED_NAMESPACES = {MemoryLayer.WORKSPACE.value, MemoryLayer.USER.value}
_LIFECYCLE_NAMESPACES = {*_MANAGED_NAMESPACES, MemoryLayer.SESSION.value}


class MemoryLifecycleService:
    """Manage local memories with operation-specific namespace boundaries."""

    def __init__(
        self,
        store: Store,
        *,
        index_memory: Callable[..., JsonDict | None],
    ) -> None:
        self._store = store
        self._index_memory = index_memory

    def list_managed(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        limit: int = 100,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> list[JsonDict]:
        """List workspace/user memories for lifecycle UI surfaces."""

        for namespace, scope_key in (
            (MemoryLayer.WORKSPACE.value, workspace_key),
            (MemoryLayer.USER.value, user_key),
        ):
            if scope_key:
                self._store.sweep_expired_memories(
                    namespace=namespace,
                    scope_key=scope_key,
                    limit=min(max(1, int(limit)), 500),
                )
        rows: list[JsonDict] = []
        for namespace, scope_key in (
            (MemoryLayer.WORKSPACE.value, workspace_key),
            (MemoryLayer.USER.value, user_key),
        ):
            if not scope_key:
                continue
            rows.extend(
                self._store.list_memories(
                    namespace=namespace,
                    scope_key=scope_key,
                    limit=limit,
                    include_archived=include_archived,
                    archived_only=archived_only,
                )
            )
        rows.sort(
            key=lambda row: float(row.get("archived_at") or row.get("last_used_at") or row.get("created_at") or 0),
            reverse=True,
        )
        return [_public_memory(row) for row in rows[:limit]]

    def archive_memory(self, memory_id: str) -> JsonDict:
        """Archive a memory and remove any legacy derived Library index."""

        memory = self._lifecycle_memory(memory_id)
        if memory.get("archived_at") is not None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory is already archived")
        if bool(memory.get("pinned")):
            raise BaselineError(ErrorCode.MEMORY_PINNED, "已 Pin，请先取消 Pin 再归档")
        if not self._store.archive_memory_and_purge_library(memory_id):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory could not be archived")
        return {"memory_id": memory_id, "archived": True}

    def restore_memory(self, memory_id: str) -> JsonDict:
        """Restore an archived memory."""

        memory = self._lifecycle_memory(memory_id)
        if memory.get("archived_at") is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory is not archived")
        if not self._store.restore_memory(memory_id):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory could not be restored")
        return {"memory_id": memory_id, "archived": False, "library_index_warning": None}

    def patch_memory(self, memory_id: str, payload: JsonDict | None) -> JsonDict:
        """Update one active workspace/user memory and purge legacy derived Library context."""

        updates = _validate_patch_payload(payload)
        memory = self._managed_memory(memory_id)
        if memory.get("archived_at") is not None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "archived memories cannot be patched")
        requested_expiry = updates.get("expires_at")
        pinning = updates.get("pinned") is True
        remains_pinned = bool(memory.get("pinned")) and updates.get("pinned") is not False
        if requested_expiry is not None and (pinning or remains_pinned):
            raise BaselineError(ErrorCode.MEMORY_PINNED, "已 Pin，请先取消 Pin 再设置 TTL")

        needs_reindex = "content" in updates or "metadata" in updates
        updater = self._store.update_memory_and_purge_library if needs_reindex else self._store.update_memory
        updated = updater(
            memory_id,
            content=updates.get("content"),
            salience=updates.get("salience"),
            metadata=updates.get("metadata"),
            pinned=updates.get("pinned"),
            expires_at=updates.get("expires_at"),
            update_content="content" in updates,
            update_salience="salience" in updates,
            update_metadata="metadata" in updates,
            update_pinned="pinned" in updates,
            update_expires_at="expires_at" in updates,
        )
        if updated is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory could not be patched")

        return {
            "memory_id": memory_id,
            "memory": _public_memory(updated),
            "library_index_warning": None,
        }

    def batch_expiry(self, payload: JsonDict | None = None) -> JsonDict:
        """Apply one expiry value to active, unpinned workspace/user memories."""

        request = _validate_batch_expiry_payload(payload)
        self._store.sweep_expired_memories(
            namespace=request["namespace"],
            scope_key=request["scope_key"],
            limit=500,
        )
        return self._store.batch_update_memory_expiry(
            namespace=request["namespace"],
            scope_key=request["scope_key"],
            expires_at=request["expires_at"],
            dry_run=request["dry_run"],
        )

    def cleanup_expired(self, payload: JsonDict | None = None) -> JsonDict:
        """Explicitly sweep bounded expired Memory batches for selected scopes."""

        request = payload or {}
        allowed = {"limit", "namespace", "scope_key", "workspace_key", "user_key"}
        unknown = sorted(set(request) - allowed)
        if unknown:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown cleanup fields: {', '.join(unknown)}",
            )
        try:
            limit = int(request.get("limit", 100))
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "limit must be an integer") from error
        if not 1 <= limit <= 500:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "limit must be between 1 and 500")

        explicit_scope = "namespace" in request or "scope_key" in request
        if explicit_scope:
            if "workspace_key" in request or "user_key" in request:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "namespace/scope_key cannot be combined with workspace_key/user_key",
                )
            namespace = str(request.get("namespace") or "").strip()
            scope_key = str(request.get("scope_key") or "").strip()
            if namespace not in _MANAGED_NAMESPACES:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "namespace must be workspace or user",
                )
            if not scope_key:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "scope_key is required")
            targets = [(namespace, scope_key)]
        else:
            workspace_key = str(request.get("workspace_key", "default") or "").strip()
            user_key = str(request.get("user_key", "default") or "").strip()
            targets = [
                (namespace, scope_key)
                for namespace, scope_key in (
                    (MemoryLayer.WORKSPACE.value, workspace_key),
                    (MemoryLayer.USER.value, user_key),
                )
                if scope_key
            ]
        if not targets:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "at least one cleanup scope is required")

        checked_at = time.time()
        results = []
        remaining_limit = limit
        for namespace, scope_key in targets:
            if remaining_limit <= 0:
                continue
            result = self._store.sweep_expired_memories(
                now=checked_at,
                namespace=namespace,
                scope_key=scope_key,
                limit=remaining_limit,
            )
            results.append(result)
            remaining_limit -= int(result.get("archived") or 0)
        memory_ids = [str(memory_id) for item in results for memory_id in item.get("memory_ids", [])]
        return {
            "checked_at": checked_at,
            "scanned": sum(int(item.get("scanned") or 0) for item in results),
            "archived": len(memory_ids),
            "memory_ids": memory_ids,
            "remaining": sum(
                self._store.count_expired_memories(
                    now=checked_at,
                    namespace=namespace,
                    scope_key=scope_key,
                )
                for namespace, scope_key in targets
            ),
        }

    def delete_memory(self, memory_id: str) -> JsonDict:
        """Permanently delete a memory and any legacy derived Library index."""

        self._lifecycle_memory(memory_id)
        if not self._store.delete_memory_and_purge_library(memory_id):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory could not be deleted")
        return {"memory_id": memory_id, "deleted": True}

    def _managed_memory(self, memory_id: str) -> JsonDict:
        return self._memory_in_namespaces(
            memory_id,
            _MANAGED_NAMESPACES,
            message="only workspace/user memories can be edited or configured",
        )

    def _lifecycle_memory(self, memory_id: str) -> JsonDict:
        return self._memory_in_namespaces(
            memory_id,
            _LIFECYCLE_NAMESPACES,
            message="only workspace/user/session memories support lifecycle actions",
        )

    def _memory_in_namespaces(
        self,
        memory_id: str,
        allowed: set[str],
        *,
        message: str,
    ) -> JsonDict:
        memory = self._store.get_memory(str(memory_id or "").strip())
        if not memory:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory not found")
        namespace = str(memory.get("namespace") or "")
        if namespace not in allowed:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                message,
                details={"namespace": namespace},
            )
        return memory


def _validate_patch_payload(payload: JsonDict | None) -> JsonDict:
    if not isinstance(payload, dict) or not payload:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "memory patch payload is required")
    allowed = {"content", "salience", "metadata", "pinned", "expires_at"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "unsupported memory patch field", details={"fields": unknown})

    updates: JsonDict = {}
    if "content" in payload:
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise BaselineError(ErrorCode.INVALID_REQUEST, "content must be a non-empty string")
        updates["content"] = content.strip()
    if "salience" in payload:
        salience = payload.get("salience")
        if isinstance(salience, bool):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "salience must be a non-negative number")
        try:
            value = float(salience)
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "salience must be a non-negative number") from error
        if not math.isfinite(value) or value < 0:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "salience must be a non-negative number")
        updates["salience"] = value
    if "metadata" in payload:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "metadata must be an object")
        updates["metadata"] = dict(metadata)
    if "pinned" in payload:
        pinned = payload.get("pinned")
        if not isinstance(pinned, bool):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "pinned must be a boolean")
        updates["pinned"] = pinned
    if "expires_at" in payload:
        expires_at = payload.get("expires_at")
        if expires_at is None:
            updates["expires_at"] = None
        else:
            if isinstance(expires_at, bool):
                raise BaselineError(ErrorCode.INVALID_REQUEST, "expires_at must be a timestamp or null")
            try:
                value = float(expires_at)
            except (TypeError, ValueError) as error:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "expires_at must be a timestamp or null") from error
            if not math.isfinite(value) or value <= 0:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "expires_at must be a timestamp or null")
            updates["expires_at"] = value
    if not updates:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "memory patch payload is required")
    return updates


def _validate_batch_expiry_payload(payload: JsonDict | None) -> JsonDict:
    if not isinstance(payload, dict):
        raise BaselineError(ErrorCode.INVALID_REQUEST, "memory batch expiry payload is required")
    namespace = str(payload.get("namespace") or "").strip().lower()
    if namespace not in _MANAGED_NAMESPACES:
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            "namespace must be workspace or user",
            details={"namespace": namespace},
        )
    if "expires_at" not in payload:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "expires_at is required")
    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise BaselineError(ErrorCode.INVALID_REQUEST, "dry_run must be a boolean")
    scope_field = "workspace_key" if namespace == MemoryLayer.WORKSPACE.value else "user_key"
    scope_key = str(payload.get(scope_field) or "default").strip()
    if not scope_key:
        raise BaselineError(ErrorCode.INVALID_REQUEST, f"{scope_field} is required")
    expires_at = _validate_patch_payload({"expires_at": payload.get("expires_at")})["expires_at"]
    return {
        "namespace": namespace,
        "scope_key": scope_key,
        "expires_at": expires_at,
        "dry_run": dry_run,
    }


def _public_memory(row: JsonDict) -> JsonDict:
    namespace = str(row.get("namespace") or row.get("layer") or "")
    salience = float(row.get("salience") or 0.0)
    return {
        **row,
        "layer": namespace,
        "effective_salience": float(row.get("effective_salience") or salience),
        "pinned": bool(row.get("pinned")),
        "expires_at": row.get("expires_at"),
    }
