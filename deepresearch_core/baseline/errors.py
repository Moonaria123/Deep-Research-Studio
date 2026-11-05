"""Unified error codes and payloads for the baseline API surface."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable, user-facing error codes returned by ``/api/v1``."""

    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
    GROUNDING_REQUIRED = "GROUNDING_REQUIRED"
    STREAM_LIMIT_EXCEEDED = "STREAM_LIMIT_EXCEEDED"
    APP_BUDGET_EXCEEDED = "APP_BUDGET_EXCEEDED"
    SEARCH_PROVIDER_DOWN = "SEARCH_PROVIDER_DOWN"
    SEARCH_PROVIDER_STATE_CONFLICT = "SEARCH_PROVIDER_STATE_CONFLICT"
    CREDENTIAL_SOURCE_READ_ONLY = "CREDENTIAL_SOURCE_READ_ONLY"
    RUN_TIMEOUT = "RUN_TIMEOUT"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_NOT_PAUSABLE = "RUN_NOT_PAUSABLE"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"
    CHECKPOINT_STATE_CONFLICT = "CHECKPOINT_STATE_CONFLICT"
    CHECKPOINT_ALREADY_CONTINUED = "CHECKPOINT_ALREADY_CONTINUED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_ARCHIVED = "SESSION_ARCHIVED"
    MEMORY_PINNED = "MEMORY_PINNED"
    MEMORY_DREAM_RUNNING = "MEMORY_DREAM_RUNNING"
    MEMORY_DREAM_NOT_FOUND = "MEMORY_DREAM_NOT_FOUND"
    KNOWLEDGE_GRAPH_BUILD_RUNNING = "KNOWLEDGE_GRAPH_BUILD_RUNNING"
    KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING = "KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_DELETABLE = "DOCUMENT_NOT_DELETABLE"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    DOCUMENT_TYPE_UNSUPPORTED = "DOCUMENT_TYPE_UNSUPPORTED"
    DOCUMENT_ENCRYPTED = "DOCUMENT_ENCRYPTED"
    DOCUMENT_LIMIT_EXCEEDED = "DOCUMENT_LIMIT_EXCEEDED"
    DOCUMENT_NO_TEXT = "DOCUMENT_NO_TEXT"
    DOCUMENT_PARSE_FAILED = "DOCUMENT_PARSE_FAILED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# HTTP status mapping for surfaces that speak HTTP.
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.PROVIDER_NOT_CONFIGURED: 409,
    ErrorCode.GROUNDING_REQUIRED: 422,
    ErrorCode.STREAM_LIMIT_EXCEEDED: 422,
    ErrorCode.APP_BUDGET_EXCEEDED: 409,
    ErrorCode.SEARCH_PROVIDER_DOWN: 502,
    ErrorCode.SEARCH_PROVIDER_STATE_CONFLICT: 409,
    ErrorCode.CREDENTIAL_SOURCE_READ_ONLY: 409,
    ErrorCode.RUN_TIMEOUT: 504,
    ErrorCode.RUN_CANCELLED: 409,
    ErrorCode.RUN_NOT_FOUND: 404,
    ErrorCode.RUN_NOT_PAUSABLE: 409,
    ErrorCode.CHECKPOINT_NOT_FOUND: 404,
    ErrorCode.CHECKPOINT_INVALID: 422,
    ErrorCode.CHECKPOINT_STATE_CONFLICT: 409,
    ErrorCode.CHECKPOINT_ALREADY_CONTINUED: 409,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_ARCHIVED: 410,
    ErrorCode.MEMORY_PINNED: 409,
    ErrorCode.MEMORY_DREAM_RUNNING: 409,
    ErrorCode.MEMORY_DREAM_NOT_FOUND: 404,
    ErrorCode.KNOWLEDGE_GRAPH_BUILD_RUNNING: 409,
    ErrorCode.KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING: 409,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.DOCUMENT_NOT_DELETABLE: 409,
    ErrorCode.DOCUMENT_TOO_LARGE: 413,
    ErrorCode.DOCUMENT_TYPE_UNSUPPORTED: 415,
    ErrorCode.DOCUMENT_ENCRYPTED: 422,
    ErrorCode.DOCUMENT_LIMIT_EXCEEDED: 422,
    ErrorCode.DOCUMENT_NO_TEXT: 422,
    ErrorCode.DOCUMENT_PARSE_FAILED: 422,
    ErrorCode.UPSTREAM_ERROR: 502,
    ErrorCode.INTERNAL_ERROR: 500,
}


class BaselineError(Exception):
    """Application error carrying a stable code and safe message.

    ``message`` and ``details`` must never contain secrets; callers are
    responsible for redaction before constructing the error.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def http_status(self) -> int:
        """Return the HTTP status code for this error."""

        return HTTP_STATUS.get(self.code, 500)

    def to_payload(self) -> dict[str, Any]:
        """Return the wire payload for this error."""

        return error_payload(self.code, self.message, details=self.details or None)


def error_payload(
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified error envelope: ``{"error": {code, message, details?}}``."""

    body: dict[str, Any] = {"code": code.value, "message": message}
    if details:
        body["details"] = details
    return {"error": body}
