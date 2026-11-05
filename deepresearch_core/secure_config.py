import base64
import ctypes
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .config import PROVIDER_FIELDS
from .settings import ProviderSettings


SECURE_CONFIG_ENV_VAR = "DEEPRESEARCH_SECURE_CONFIG"
SECURE_CONFIG_KIND = "deepresearch_windows_secure_provider"
SECURE_CONFIG_VERSION = 1
Protector = Callable[[bytes], bytes]
Unprotector = Callable[[bytes], bytes]


def secure_config_supported() -> bool:
    """Return whether OS-backed secure config storage is available."""

    return os.name == "nt"


def default_secure_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the default Windows secure provider config path."""

    env = environ if environ is not None else os.environ
    explicit = str(env.get(SECURE_CONFIG_ENV_VAR) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    appdata = str(env.get("APPDATA") or "").strip()
    if appdata:
        return Path(appdata).expanduser() / "DeepResearch" / "secure-provider.dpapi.json"
    return Path.home() / "AppData" / "Roaming" / "DeepResearch" / "secure-provider.dpapi.json"


def save_secure_provider_config(path: Path, provider: ProviderSettings, *, protector: Protector | None = None) -> None:
    """Save provider settings encrypted for the current Windows user."""

    protect = protector or dpapi_protect
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted: dict[str, str] = {}
    for field in PROVIDER_FIELDS:
        value = str(getattr(provider, field) or "").strip()
        if value:
            encrypted[field] = base64.b64encode(protect(value.encode("utf-8"))).decode("ascii")
    payload = {
        "kind": SECURE_CONFIG_KIND,
        "version": SECURE_CONFIG_VERSION,
        "provider": encrypted,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_secure_provider_config(path: Path, *, unprotector: Unprotector | None = None) -> ProviderSettings:
    """Load provider settings encrypted for the current Windows user."""

    if not path.exists():
        return ProviderSettings()
    unprotect = unprotector or dpapi_unprotect
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Invalid DeepResearch secure config JSON") from error
    if not isinstance(data, dict) or data.get("kind") != SECURE_CONFIG_KIND:
        raise ValueError("Invalid DeepResearch secure config shape")
    provider = data.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("Invalid DeepResearch secure provider config")
    values: dict[str, str] = {}
    for field in PROVIDER_FIELDS:
        encoded = str(provider.get(field) or "").strip()
        if not encoded:
            values[field] = ""
            continue
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
            values[field] = unprotect(raw).decode("utf-8")
        except Exception as error:
            raise ValueError("Invalid DeepResearch secure provider config") from error
    return ProviderSettings(**values)


def secure_config_setup_payload(
    path: Path,
    *,
    source_config_path: Path,
    provider: ProviderSettings,
    imported: bool,
    supported: bool | None = None,
) -> dict[str, Any]:
    """Return a secret-free secure config setup payload."""

    exists = path.exists()
    return {
        "tool": "deepresearch_secure_config",
        "supported": secure_config_supported() if supported is None else bool(supported),
        "secure_config_path": str(path),
        "source_config_path": str(source_config_path),
        "exists": bool(exists or imported),
        "imported": bool(imported),
        "provider_status": provider.safe_status(),
        "missing_required": provider.missing_required(),
        "writes_files": bool(imported),
        "next_steps": [
            "Use --secure-config with Windows doctor or start commands to load this protected config.",
            "After confirming the secure config works, remove the plaintext source config only if you no longer need it.",
            "Do not paste provider secrets into chat, logs, or command arguments.",
        ],
    }


def dpapi_protect(raw: bytes) -> bytes:
    """Protect bytes for the current Windows user with DPAPI."""

    if not secure_config_supported():
        raise RuntimeError("Windows secure config requires DPAPI on Windows")
    return _crypt_protect_data(raw)


def dpapi_unprotect(raw: bytes) -> bytes:
    """Unprotect bytes for the current Windows user with DPAPI."""

    if not secure_config_supported():
        raise RuntimeError("Windows secure config requires DPAPI on Windows")
    return _crypt_unprotect_data(raw)


def _crypt_protect_data(raw: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_buffer = ctypes.create_string_buffer(raw)
    input_blob = _DataBlob(len(raw), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte)))
    output_blob = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "DeepResearch provider config",
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _crypt_unprotect_data(raw: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_buffer = ctypes.create_string_buffer(raw)
    input_blob = _DataBlob(len(raw), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte)))
    output_blob = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


class _DataBlob(ctypes.Structure):
    """Windows DATA_BLOB used by DPAPI."""

    _fields_ = [
        ("cbData", ctypes.c_ulong),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]
