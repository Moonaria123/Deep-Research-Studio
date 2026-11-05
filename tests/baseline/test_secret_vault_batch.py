"""Batch credential resolution keeps vault access bounded to one call."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from pathlib import Path

import pytest

from deepresearch_core.secret_vault import (
    SecretVault,
    SecretVaultError,
    make_secret_ref,
    resolve_config_secrets,
)


class _Rows:
    def __init__(self, values: dict[str, str], calls: list[tuple[str, tuple[str, ...]]]) -> None:
        self._values = values
        self._calls = calls
        self._parameters: tuple[str, ...] = ()

    def execute(self, sql: str, parameters: tuple[str, ...]):
        self._calls.append((sql, parameters))
        self._parameters = parameters
        return self

    def fetchall(self) -> list[tuple[str, str]]:
        return [
            (reference, self._values[reference])
            for reference in self._parameters
            if reference in self._values
        ]


def _reference(value: str | None = None) -> dict[str, str]:
    return {"secret_ref": value or make_secret_ref()}


def _fake_sqlcipher(
    monkeypatch: pytest.MonkeyPatch,
    vault: SecretVault,
    values: dict[str, str],
) -> tuple[list[bool], list[tuple[str, tuple[str, ...]]], list[bool]]:
    read_only_calls: list[bool] = []
    query_calls: list[tuple[str, tuple[str, ...]]] = []
    closed: list[bool] = []

    @contextmanager
    def connection(*, read_only: bool = False):
        read_only_calls.append(read_only)
        try:
            yield _Rows(values, query_calls)
        finally:
            closed.append(True)

    monkeypatch.setattr(vault, "_uses_dpapi", lambda: False)
    monkeypatch.setattr(vault, "_sqlcipher_connection", connection)
    return read_only_calls, query_calls, closed


def test_resolve_config_references_with_one_read_and_deduplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _reference()
    second = _reference()
    values = {
        first["secret_ref"]: "first-secret",
        second["secret_ref"]: "second-secret",
    }
    vault = SecretVault(tmp_path / "config.json")
    read_only_calls, query_calls, closed = _fake_sqlcipher(monkeypatch, vault, values)
    monkeypatch.setattr(
        "deepresearch_core.secret_vault.SecretVault",
        lambda _config_path: vault,
    )

    runtime, protected, migrated = resolve_config_secrets(
        tmp_path / "config.json",
        {
            "provider": {"api_key": first},
            "profiles": [
                {"api_key": second},
                {"api_key": first},
            ],
        },
        read_only=True,
    )

    assert runtime["provider"]["api_key"] == "first-secret"
    assert runtime["profiles"] == [
        {"api_key": "second-secret"},
        {"api_key": "first-secret"},
    ]
    assert protected["provider"]["api_key"] == first
    assert migrated is False
    assert read_only_calls == [True]
    assert len(query_calls) == 1
    assert query_calls[0][1] == (first["secret_ref"], second["secret_ref"])
    assert closed == [True]


def test_get_many_empty_and_invalid_refs_do_not_open_vault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = SecretVault(tmp_path / "config.json")
    read_only_calls, query_calls, closed = _fake_sqlcipher(monkeypatch, vault, {})

    assert vault.get_many([]) == {}
    with pytest.raises(SecretVaultError, match="invalid"):
        vault.get_many([{"secret_ref": "not-a-reference"}])

    assert read_only_calls == []
    assert query_calls == []
    assert closed == []


def test_missing_reference_fails_closed_and_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    present = _reference()
    missing = _reference()
    vault = SecretVault(tmp_path / "config.json")
    _read_only_calls, _query_calls, closed = _fake_sqlcipher(
        monkeypatch,
        vault,
        {present["secret_ref"]: "present-secret"},
    )

    with pytest.raises(SecretVaultError, match="missing") as exc_info:
        vault.get_many([present, missing])

    assert "present-secret" not in str(exc_info.value)
    assert closed == [True]


def test_each_batch_observes_current_vault_values_without_process_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()
    values = {reference["secret_ref"]: "before-rotation"}
    vault = SecretVault(tmp_path / "config.json")
    read_only_calls, query_calls, closed = _fake_sqlcipher(monkeypatch, vault, values)

    assert vault.get_many([reference])[reference["secret_ref"]] == "before-rotation"
    values[reference["secret_ref"]] = "after-rotation"
    assert vault.get_many([reference])[reference["secret_ref"]] == "after-rotation"

    assert read_only_calls == [False, False]
    assert len(query_calls) == 2
    assert closed == [True, True]


def test_dpapi_batch_reads_vault_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch_core import secure_config

    first = _reference()
    second = _reference()
    encoded = {
        first["secret_ref"]: base64.b64encode(b"first-secret").decode("ascii"),
        second["secret_ref"]: base64.b64encode(b"second-secret").decode("ascii"),
    }
    reads: list[bool] = []
    vault = SecretVault(tmp_path / "config.json")
    monkeypatch.setattr(vault, "_uses_dpapi", lambda: True)
    monkeypatch.setattr(
        vault,
        "_dpapi_read",
        lambda: reads.append(True) or dict(encoded),
    )
    monkeypatch.setattr(secure_config, "dpapi_unprotect", lambda raw: raw)

    assert vault.get_many([first, second, first]) == {
        first["secret_ref"]: "first-secret",
        second["secret_ref"]: "second-secret",
    }
    assert reads == [True]


def test_batch_failure_does_not_log_or_echo_plaintext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference = _reference()
    vault = SecretVault(tmp_path / "config.json")
    _fake_sqlcipher(
        monkeypatch,
        vault,
        {reference["secret_ref"]: "never-log-this-secret"},
    )

    with pytest.raises(SecretVaultError):
        vault.get_many([reference, _reference()])

    assert "never-log-this-secret" not in caplog.text
