"""Read-only health checks used by the public Docker launcher."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepresearch_core.secret_vault import SecretVault, SecretVaultError, is_secret_ref

DATA = Path("/data")
PROTECTED_FIELDS = {
    "api_key",
    "llm_api_key",
    "brave_api_key",
    "firecrawl_api_key",
    "exa_api_key",
    "tavily_api_key",
}


def open_query_only(path: Path) -> sqlite3.Connection:
    """Open in the source lock domain while prohibiting logical writes."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def backup_and_check(source_path: Path, target_path: Path) -> None:
    source = open_query_only(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise SystemExit("database quick_check failed")
    finally:
        target.close()
        source.close()


def _collect_references(value: object, *, field: str = "") -> list[str]:
    if field in PROTECTED_FIELDS:
        if value in (None, ""):
            return []
        if is_secret_ref(value):
            return [str(value["secret_ref"])]  # type: ignore[index]
        if isinstance(value, str):
            raise SecretVaultError("Plaintext provider credential requires protected migration")
        raise SecretVaultError("Provider credential reference is invalid")
    references: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            references.extend(_collect_references(item, field=str(key).strip().lower()))
    elif isinstance(value, list):
        for item in value:
            references.extend(_collect_references(item))
    return references


def validate_vault_references(config_path: Path, protected: object) -> None:
    references = list(dict.fromkeys(_collect_references(protected)))
    if not references:
        return
    vault = SecretVault(config_path)
    if not vault.path.is_file():
        raise SecretVaultError("Provider credential vault is unavailable")
    from pysqlcipher3 import dbapi2 as sqlcipher

    database_key = hmac.new(
        vault._root_key(), b"deepresearch/provider-vault/v1", hashlib.sha256  # noqa: SLF001
    ).hexdigest()
    connection = sqlcipher.connect(f"file:{vault.path}?mode=ro", uri=True)
    try:
        connection.execute(f"PRAGMA key = \"x'{database_key}'\"")
        connection.execute("PRAGMA query_only = ON")
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'provider_secrets'"
        ).fetchone()
        if table is None:
            raise SecretVaultError("Provider credential vault has an invalid shape")
        placeholders = ",".join("?" for _ in references)
        rows = connection.execute(
            f"SELECT secret_ref FROM provider_secrets WHERE secret_ref IN ({placeholders})",
            tuple(references),
        ).fetchall()
        if {str(row[0]) for row in rows} != set(references):
            raise SecretVaultError("Provider credential reference is missing from the vault")
    finally:
        connection.close()


def main() -> None:
    config_path = DATA / "config.json"
    source_database = DATA / "baseline.sqlite3"
    scratch = Path("/tmp/deep-research-studio-doctor")
    scratch.mkdir(mode=0o700)
    database_path = scratch / "baseline.sqlite3"
    if not source_database.is_file():
        raise SystemExit("installation data is incomplete")
    backup_and_check(source_database, database_path)
    if config_path.is_file():
        protected = json.loads(config_path.read_text(encoding="utf-8"))
        validate_vault_references(config_path, protected)
    print("doctor: database and credential references are readable")


if __name__ == "__main__":
    main()
