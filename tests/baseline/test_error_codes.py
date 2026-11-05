"""DR-ITER-073: stable T10 error-code contract."""

from __future__ import annotations

import json

from deepresearch_core.baseline.errors import BaselineError, ErrorCode, HTTP_STATUS


def test_t10_error_codes_have_spec_http_mappings() -> None:
    assert HTTP_STATUS[ErrorCode.APP_BUDGET_EXCEEDED] == 409
    assert HTTP_STATUS[ErrorCode.SEARCH_PROVIDER_DOWN] == 502
    assert HTTP_STATUS[ErrorCode.RUN_TIMEOUT] == 504


def test_timeout_error_payload_is_secret_free() -> None:
    payload = BaselineError(ErrorCode.RUN_TIMEOUT, "Run timed out before evidence was available.").to_payload()

    assert payload == {
        "error": {
            "code": "RUN_TIMEOUT",
            "message": "Run timed out before evidence was available.",
        }
    }
    assert "sk-" not in json.dumps(payload).lower()
