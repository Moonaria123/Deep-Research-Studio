"""Encrypted, reference-based storage for provider credentials.

Configuration JSON stores only opaque references.  Windows uses the current
user's DPAPI protection; Docker/Linux uses the existing SQLCipher runtime and a
root key supplied through a mounted, memory-backed file.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


VAULT_KEY_FILE_ENV = "DEEPRESEARCH_VAULT_KEY_FILE"
VAULT_PATH_ENV = "DEEPRESEARCH_VAULT_PATH"
SECRET_REF_VERSION = "vault-v1"


class SecretVaultError(RuntimeError):
    """Raised when encrypted credential storage cannot be used safely."""


def make_secret_ref() -> str:
    return f"{SECRET_REF_VERSION}:{uuid.uuid4()}"


def is_secret_ref(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"secret_ref"}:
        return False
    raw = value.get("secret_ref")
    if not isinstance(raw, str) or not raw.startswith(f"{SECRET_REF_VERSION}:"):
        return False
    try:
        parsed = uuid.UUID(raw.split(":", 1)[1])
    except (ValueError, AttributeError):
        return False
    return str(parsed) == raw.split(":", 1)[1]


class SecretVault:
    """Store and retrieve immutable credential values by opaque reference."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = Path(config_path)
        explicit = str(os.environ.get(VAULT_PATH_ENV) or "").strip()
        if os.name == "nt":
            self.path = Path(explicit) if explicit else self._config_path.with_name(
                f".{self._config_path.name}.provider-secrets.dpapi.json"
            )
        else:
            self.path = Path(explicit) if explicit else self._config_path.with_name(
                f".{self._config_path.name}.provider-secrets.sqlite3"
            )

    @staticmethod
    def _uses_dpapi() -> bool:
        return os.name == "nt"

    def put(self, value: str) -> dict[str, str]:
        text = str(value or "")
        if not text:
            raise SecretVaultError("Refusing to store an empty provider credential")
        reference = make_secret_ref()
        try:
            if os.name == "nt":
                self._dpapi_put(reference, text)
            else:
                with self._sqlcipher_connection() as connection:
                    connection.execute(
                        "INSERT INTO provider_secrets(secret_ref, secret_value) VALUES (?, ?)",
                        (reference, text),
                    )
                    connection.commit()
        except SecretVaultError:
            raise
        except Exception as error:
            raise SecretVaultError("Provider credential could not be stored") from error
        return {"secret_ref": reference}

    def get(self, reference: object, *, read_only: bool = False) -> str:
        values = self.get_many([reference], read_only=read_only)
        secret_ref = str(reference["secret_ref"])  # type: ignore[index]
        return values[secret_ref]

    def get_many(
        self,
        references: list[object] | tuple[object, ...],
        *,
        read_only: bool = False,
    ) -> dict[str, str]:
        """Resolve references using one vault read scoped to this call."""

        secret_refs: list[str] = []
        for reference in references:
            if not is_secret_ref(reference):
                raise SecretVaultError("Provider credential reference is invalid")
            secret_ref = str(reference["secret_ref"])  # type: ignore[index]
            if secret_ref not in secret_refs:
                secret_refs.append(secret_ref)
        if not secret_refs:
            return {}
        try:
            if self._uses_dpapi():
                return self._dpapi_get_many(secret_refs)
            with self._sqlcipher_connection(read_only=read_only) as connection:
                placeholders = ",".join("?" for _ in secret_refs)
                rows = connection.execute(
                    "SELECT secret_ref, secret_value FROM provider_secrets "
                    f"WHERE secret_ref IN ({placeholders})",
                    tuple(secret_refs),
                ).fetchall()
        except SecretVaultError:
            raise
        except Exception as error:
            raise SecretVaultError("Provider credential could not be read") from error
        values = {str(row[0]): str(row[1]) for row in rows}
        if any(secret_ref not in values for secret_ref in secret_refs):
            raise SecretVaultError("Provider credential reference is missing from the vault")
        return {secret_ref: values[secret_ref] for secret_ref in secret_refs}

    def _dpapi_read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SecretVaultError("Provider credential vault is unreadable") from error
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "deepresearch_provider_vault"
            or payload.get("version") != 1
        ):
            raise SecretVaultError("Provider credential vault has an unsupported format")
        secrets = payload.get("secrets")
        valid_items = isinstance(secrets, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in secrets.items()
        )
        if not valid_items:
            raise SecretVaultError("Provider credential vault has an invalid shape")
        return dict(secrets)

    def _dpapi_put(self, reference: str, value: str) -> None:
        from .secure_config import dpapi_protect

        try:
            secrets = self._dpapi_read()
            secrets[reference] = base64.b64encode(dpapi_protect(value.encode("utf-8"))).decode("ascii")
            payload = {"kind": "deepresearch_provider_vault", "version": 1, "secrets": secrets}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except SecretVaultError:
            raise
        except Exception as error:
            raise SecretVaultError("Provider credential could not be stored") from error
        finally:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink()

    def _dpapi_get(self, reference: str) -> str:
        return self._dpapi_get_many([reference])[reference]

    def _dpapi_get_many(self, references: list[str]) -> dict[str, str]:
        from .secure_config import dpapi_unprotect

        encoded_values = self._dpapi_read()
        if any(not encoded_values.get(reference) for reference in references):
            raise SecretVaultError("Provider credential reference is missing from the vault")
        try:
            return {
                reference: dpapi_unprotect(
                    base64.b64decode(encoded_values[reference].encode("ascii"), validate=True)
                ).decode("utf-8")
                for reference in references
            }
        except Exception as error:
            raise SecretVaultError("Provider credential vault could not be decrypted") from error

    def _root_key(self) -> bytes:
        key_path_text = str(os.environ.get(VAULT_KEY_FILE_ENV) or "").strip()
        if not key_path_text:
            raise SecretVaultError(f"{VAULT_KEY_FILE_ENV} is required for encrypted provider credentials")
        key_path = Path(key_path_text)
        try:
            info = key_path.stat()
        except OSError as error:
            raise SecretVaultError("Provider vault key file is unavailable") from error
        if not stat.S_ISREG(info.st_mode):
            raise SecretVaultError("Provider vault key path must be a regular file")
        if info.st_size > 65:
            raise SecretVaultError("Provider vault key file must contain exactly 32 bytes")
        try:
            raw = key_path.read_bytes().strip()
        except OSError as error:
            raise SecretVaultError("Provider vault key file is unavailable") from error
        if len(raw) != 64:
            raise SecretVaultError("Provider vault key file must contain exactly 32 bytes")
        try:
            key = bytes.fromhex(raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise SecretVaultError("Provider vault key file must contain 32 bytes encoded as hexadecimal") from error
        if len(key) != 32:
            raise SecretVaultError("Provider vault key file must contain exactly 32 bytes")
        return key

    @contextmanager
    def _sqlcipher_connection(self, *, read_only: bool = False):  # noqa: ANN202
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher
        except ImportError as error:
            raise SecretVaultError("SQLCipher is required for the provider credential vault") from error
        root_key = self._root_key()
        database_key = hmac.new(root_key, b"deepresearch/provider-vault/v1", hashlib.sha256).hexdigest()
        if read_only and not self.path.is_file():
            raise SecretVaultError("Provider credential vault is unavailable")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = None
        try:
            connection = sqlcipher.connect(str(self.path))
            connection.execute(f"PRAGMA key = \"x'{database_key}'\"")
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
            if read_only:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'provider_secrets'"
                ).fetchone()
                if table is None:
                    raise SecretVaultError("Provider credential vault has an invalid shape")
            else:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS provider_secrets ("
                    "secret_ref TEXT PRIMARY KEY, secret_value TEXT NOT NULL)"
                )
            if os.name != "nt" and not read_only:
                os.chmod(self.path, 0o600)
        except Exception as error:
            try:
                if connection is not None:
                    connection.close()
            except Exception:
                pass
            raise SecretVaultError("Provider credential vault could not be opened") from error
        try:
            yield connection
        finally:
            connection.close()


def validate_vault_key(config_path: Path) -> None:
    """Fail fast when the platform's provider-vault prerequisite is unavailable.

    This does not create or open a vault.  Docker/Linux validates only the
    memory-backed root-key file; Windows relies on the current user's DPAPI.
    """

    if os.name != "nt":
        SecretVault(config_path)._root_key()


def protect_config_secrets(
    config_path: Path,
    value: Any,
    *,
    previous_plain: Any = None,
    previous_protected: Any = None,
) -> Any:
    """Replace credential fields with refs, retaining unchanged existing refs."""

    vault = SecretVault(config_path)

    def walk(current: Any, old_plain: Any, old_protected: Any, *, field: str = "") -> Any:
        if field in {"api_key", "llm_api_key", "brave_api_key", "firecrawl_api_key", "exa_api_key", "tavily_api_key"}:
            if current in (None, ""):
                return current
            if is_secret_ref(current):
                return current
            if not isinstance(current, str):
                raise SecretVaultError("Provider credential must be a string or secret reference")
            if current == old_plain and is_secret_ref(old_protected):
                return old_protected
            return vault.put(current)
        if isinstance(current, dict):
            old_plain_map = old_plain if isinstance(old_plain, dict) else {}
            old_protected_map = old_protected if isinstance(old_protected, dict) else {}
            return {
                str(key): walk(
                    item,
                    old_plain_map.get(key),
                    old_protected_map.get(key),
                    field=str(key).strip().lower(),
                )
                for key, item in current.items()
            }
        if isinstance(current, list):
            old_plain_list = old_plain if isinstance(old_plain, list) else []
            old_protected_list = old_protected if isinstance(old_protected, list) else []
            return [
                walk(
                    item,
                    old_plain_list[index] if index < len(old_plain_list) else None,
                    old_protected_list[index] if index < len(old_protected_list) else None,
                )
                for index, item in enumerate(current)
            ]
        return current

    return walk(value, previous_plain, previous_protected)


def resolve_config_secrets(
    config_path: Path,
    value: Any,
    *,
    migrate: bool = True,
    read_only: bool = False,
) -> tuple[Any, Any, bool]:
    """Return runtime plaintext, normalized protected data, and migration flag."""

    vault = SecretVault(config_path)
    references: list[object] = []

    def collect_references(current: Any, *, field: str = "") -> None:
        if field in {"api_key", "llm_api_key", "brave_api_key", "firecrawl_api_key", "exa_api_key", "tavily_api_key"}:
            if is_secret_ref(current):
                references.append(current)
            return
        if isinstance(current, dict):
            for key, item in current.items():
                collect_references(item, field=str(key).strip().lower())
        elif isinstance(current, list):
            for item in current:
                collect_references(item)

    collect_references(value)
    resolved = vault.get_many(references, read_only=read_only)

    def walk(current: Any, *, field: str = "") -> tuple[Any, Any, bool]:
        if field in {"api_key", "llm_api_key", "brave_api_key", "firecrawl_api_key", "exa_api_key", "tavily_api_key"}:
            if current in (None, ""):
                return current, current, False
            if is_secret_ref(current):
                return resolved[str(current["secret_ref"])], current, False
            if isinstance(current, dict):
                raise SecretVaultError("Provider credential reference is invalid")
            if not isinstance(current, str):
                raise SecretVaultError("Provider credential must be a string or secret reference")
            if not migrate:
                return current, current, False
            protected = vault.put(current)
            return current, protected, True
        if isinstance(current, dict):
            plain: dict[str, Any] = {}
            protected: dict[str, Any] = {}
            migrated = False
            for key, item in current.items():
                item_plain, item_protected, item_migrated = walk(item, field=str(key).strip().lower())
                plain[str(key)] = item_plain
                protected[str(key)] = item_protected
                migrated = migrated or item_migrated
            return plain, protected, migrated
        if isinstance(current, list):
            plain_list: list[Any] = []
            protected_list: list[Any] = []
            migrated = False
            for item in current:
                item_plain, item_protected, item_migrated = walk(item)
                plain_list.append(item_plain)
                protected_list.append(item_protected)
                migrated = migrated or item_migrated
            return plain_list, protected_list, migrated
        return current, current, False

    return walk(value)
