"""Provider credentials persist as encrypted references, never raw JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deepresearch_core.config import load_provider_config, save_provider_config
from deepresearch_core.secret_vault import SecretVault, SecretVaultError, make_secret_ref, validate_vault_key
from deepresearch_core.settings import ProviderSettings


@pytest.fixture(autouse=True)
def isolated_vault(
    isolated_provider_environment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Use a fast reversible protector; production DPAPI itself has separate tests."""

    from deepresearch_core import secure_config

    def put_with_test_dpapi(vault: SecretVault, value: str) -> dict[str, str]:
        reference = make_secret_ref()
        vault._dpapi_put(reference, str(value))  # noqa: SLF001
        return {"secret_ref": reference}

    monkeypatch.setattr(SecretVault, "_uses_dpapi", staticmethod(lambda: True))
    monkeypatch.setattr(SecretVault, "put", put_with_test_dpapi)
    monkeypatch.setenv("DEEPRESEARCH_VAULT_PATH", str(tmp_path / "vault.dpapi.json"))
    monkeypatch.setattr(secure_config, "dpapi_protect", lambda raw: b"protected:" + raw[::-1])
    monkeypatch.setattr(
        secure_config,
        "dpapi_unprotect",
        lambda raw: raw.removeprefix(b"protected:")[::-1],
    )


def test_save_and_restart_roundtrip_has_no_raw_secret(tmp_path: Path):
    path = tmp_path / "config.json"
    save_provider_config(
        path,
        ProviderSettings(
            llm_base_url="https://example.test/v1",
            llm_api_key="llm-canary",
            llm_model="model",
            brave_api_key="brave-canary",
        ),
    )

    persisted = path.read_text(encoding="utf-8")
    assert "llm-canary" not in persisted
    assert "brave-canary" not in persisted
    raw = json.loads(persisted)
    assert set(raw["provider"]["llm_api_key"]) == {"secret_ref"}
    assert set(raw["provider"]["brave_api_key"]) == {"secret_ref"}

    reloaded = load_provider_config(path)
    assert reloaded.llm_api_key == "llm-canary"
    assert reloaded.brave_api_key == "brave-canary"


def test_legacy_plaintext_migrates_once_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": {"llm_api_key": "legacy-canary"}}),
        encoding="utf-8",
    )

    assert load_provider_config(path).llm_api_key == "legacy-canary"
    first = path.read_text(encoding="utf-8")
    assert "legacy-canary" not in first
    assert load_provider_config(path).llm_api_key == "legacy-canary"
    assert path.read_text(encoding="utf-8") == first


def test_read_only_legacy_load_does_not_migrate_or_create_sidecars(tmp_path: Path):
    path = tmp_path / "config.json"
    original = json.dumps({"provider": {"llm_api_key": "read-only-canary"}}).encode()
    path.write_bytes(original)

    loaded = load_provider_config(path, read_only=True)

    assert loaded.llm_api_key == "read-only-canary"
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_read_only_missing_ref_does_not_create_vault(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "provider": {
                    "llm_api_key": {
                        "secret_ref": "vault-v1:00000000-0000-0000-0000-000000000000"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SecretVaultError, match="missing"):
        load_provider_config(path, read_only=True)
    assert list(tmp_path.iterdir()) == [path]


def test_missing_reference_fails_closed(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": {"llm_api_key": {"secret_ref": "vault-v1:00000000-0000-0000-0000-000000000000"}}}),
        encoding="utf-8",
    )

    with pytest.raises(SecretVaultError, match="missing"):
        load_provider_config(path)


def test_invalid_reference_shape_fails_closed(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": {"llm_api_key": {"secret_ref": "wrong:ref"}}}),
        encoding="utf-8",
    )

    with pytest.raises(SecretVaultError, match="invalid"):
        load_provider_config(path)


def test_decryption_failure_is_redacted_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from deepresearch_core import secure_config

    path = tmp_path / "config.json"
    save_provider_config(path, ProviderSettings(llm_api_key="wrong-key-canary"))
    monkeypatch.setattr(
        secure_config,
        "dpapi_unprotect",
        lambda _raw: (_ for _ in ()).throw(OSError("sensitive backend detail")),
    )

    with pytest.raises(SecretVaultError) as exc_info:
        load_provider_config(path)
    assert "sensitive backend detail" not in str(exc_info.value)
    assert "wrong-key-canary" not in str(exc_info.value)


def test_failed_config_replace_leaves_legacy_file_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from deepresearch_core import config as config_module

    path = tmp_path / "config.json"
    original = json.dumps({"provider": {"llm_api_key": "migration-canary"}})
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        config_module,
        "_write_json_config_unlocked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(OSError, match="disk unavailable"):
        load_provider_config(path)
    assert path.read_text(encoding="utf-8") == original


def test_root_key_parser_rejects_wrong_length_without_echoing_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    key_path = tmp_path / "vault-key"
    key_path.write_text("too-short-secret", encoding="ascii")
    monkeypatch.setenv("DEEPRESEARCH_VAULT_KEY_FILE", str(key_path))

    with pytest.raises(SecretVaultError) as exc_info:
        SecretVault(tmp_path / "config.json")._root_key()  # noqa: SLF001
    assert "too-short-secret" not in str(exc_info.value)


def test_validate_vault_key_prerequisite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.name != "nt":
        key_path = tmp_path / "vault-key"
        key_path.write_text("00" * 32, encoding="ascii")
        monkeypatch.setenv("DEEPRESEARCH_VAULT_KEY_FILE", str(key_path))
    validate_vault_key(tmp_path / "config.json")


@pytest.mark.skipif(os.name == "nt", reason="Linux/Docker root-key prerequisite")
def test_validate_vault_key_rejects_missing_linux_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DEEPRESEARCH_VAULT_KEY_FILE", raising=False)

    with pytest.raises(SecretVaultError, match="required"):
        validate_vault_key(tmp_path / "config.json")
