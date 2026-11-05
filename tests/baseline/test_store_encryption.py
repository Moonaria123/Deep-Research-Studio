"""DR-ITER-023: the store must never silently fall back to plaintext.

These tests assume no SQLCipher driver is installed (the default CI/dev env), which is
exactly the situation in which a supplied key must fail fast rather than write plaintext.
"""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest

from deepresearch_core.baseline.store import Store, StoreEncryptionError


def _sqlcipher_available() -> bool:
    for mod_name in ("sqlcipher3", "pysqlcipher3.dbapi2"):
        try:
            importlib.import_module(mod_name)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def test_no_key_opens_plaintext_and_reports_unencrypted(tmp_path: Path):
    store = Store(tmp_path / "plain.sqlite3")
    assert store.encrypted is False
    # A plaintext file is readable by the stdlib driver without a key.
    store.create_session(title="hello")
    store.close()
    conn = sqlite3.connect(tmp_path / "plain.sqlite3")
    assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    conn.close()


def test_sqlcipher_key_pragma_escapes_single_quotes():
    assert Store._quote_sqlcipher_key("abc'def") == "'abc''def'"


def test_fts5_detection_handles_sqlcipher_driver_operational_error():
    OperationalError = type("OperationalError", (Exception,), {})

    class Conn:
        def execute(self, statement: str):
            raise OperationalError("no such module: fts5")

    store = object.__new__(Store)
    store._conn = Conn()
    assert Store._detect_fts5(store) is False


def test_encrypted_connection_uses_driver_row_factory(monkeypatch, tmp_path: Path):
    class Conn:
        pass

    class Driver:
        Row = object()

        @staticmethod
        def connect(path: str, *, check_same_thread: bool):
            assert path.endswith("enc.sqlite3")
            assert check_same_thread is False
            return Conn()

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "sqlcipher3":
            raise ModuleNotFoundError(name)
        if name == "pysqlcipher3.dbapi2":
            return Driver
        return real_import_module(name)

    store = object.__new__(Store)
    store._path = str(tmp_path / "enc.sqlite3")
    store._row_factory = sqlite3.Row
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert isinstance(Store._open_encrypted(store), Conn)
    assert store._row_factory is Driver.Row


@pytest.mark.skipif(
    _sqlcipher_available(),
    reason="SQLCipher is installed; the missing-driver fail-fast path cannot be exercised",
)
def test_key_without_sqlcipher_driver_fails_fast(tmp_path: Path):
    # Requesting encryption with no driver must raise, not silently store plaintext.
    with pytest.raises(StoreEncryptionError) as excinfo:
        Store(tmp_path / "enc.sqlite3", encryption_key="hunter2")
    assert "SQLCipher" in str(excinfo.value)
    # And it must not have created/left a usable plaintext database behind.
    leaked = tmp_path / "enc.sqlite3"
    if leaked.exists():
        conn = sqlite3.connect(leaked)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert tables == [], "encryption was requested but plaintext tables were written"
