"""Build, unlock, and validate the local Deep Research Studio Web stack."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker" / "compose.gen1.yml"
BINDING_FILE = ".deepresearch-key-binding.json"
BINDING_CONTEXT = b"deep-research-studio/data-binding/v1\0"
PROJECT_DEFAULT = "deep-research-studio"
sys.path.insert(0, str(ROOT))

from deepresearch_core.secure_config import dpapi_protect, dpapi_unprotect  # noqa: E402


class StartupError(RuntimeError):
    """A bounded operator-facing failure that does not expose private diagnostics."""


def _fail(message: str) -> NoReturn:
    raise StartupError(message)


def _docker(args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(["docker", *args], input=input_bytes, capture_output=True, check=False)
    if result.returncode:
        _fail(f"Docker {args[0]} operation failed (exit {result.returncode}).")
    return result.stdout


def _compose(project: str, args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    return _docker(["compose", "-p", project, "-f", str(COMPOSE_FILE), *args], input_bytes=input_bytes)


def _resolve_data_dir(value: Path | None) -> Path:
    if value is None:
        _fail("--data-dir is required and must name an absolute host path.")
    if not value.is_absolute():
        _fail("The data directory must be an absolute host path.")
    try:
        result = value.resolve(strict=False)
    except OSError as error:
        raise StartupError("The data directory could not be resolved.") from error
    if result.exists() and (result.is_symlink() or not result.is_dir()):
        _fail("The data path must be a real directory, not a file or symbolic link.")
    return result


def _has_entries(path: Path) -> bool:
    try:
        return path.is_dir() and next(path.iterdir(), None) is not None
    except OSError as error:
        raise StartupError("The data directory could not be inspected.") from error


def _state_dir(project: str) -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            _fail("LOCALAPPDATA is required on Windows.")
        return Path(base) / "DeepResearchStudio" / project
    base = Path(os.environ["XDG_STATE_HOME"]) if os.environ.get("XDG_STATE_HOME") else Path.home() / ".local" / "state"
    return base / "deep-research-studio" / project


def _linux_mount_type(path: Path) -> str:
    """Return the most specific Linux mount type containing path."""
    try:
        resolved = path.resolve(strict=False)
        entries: list[tuple[int, str]] = []
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mount = Path(fields[1].replace("\\040", " ")).resolve(strict=False)
            try:
                resolved.relative_to(mount)
            except ValueError:
                continue
            entries.append((len(mount.parts), fields[2].lower()))
        return max(entries, default=(0, "unknown"))[1]
    except OSError:
        return "unknown"


def _validate_linux_private_path(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
    if path.is_symlink() or not path.is_dir():
        _fail("The Linux key directory must be a real directory.")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        _fail("The Linux key directory must be owned by the current user with mode 0700.")
    if _linux_mount_type(path) in {"9p", "drvfs", "vfat", "exfat", "ntfs", "fuseblk"}:
        _fail("Store the Linux key under the WSL/Linux home filesystem, not a Windows or removable mount.")


def _key_path(project: str) -> Path:
    state = _state_dir(project)
    return state / ("vault-root.dpapi" if os.name == "nt" else "vault-root.key")


def _create_key(project: str) -> bytes:
    path = _key_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        _validate_linux_private_path(path.parent, create=True)
    key = secrets.token_bytes(32)
    payload = dpapi_protect(key) if os.name == "nt" else key
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(path, 0o600)
    except FileExistsError as error:
        raise StartupError("The protected root key already exists; initialization was refused.") from error
    return key


def _read_key(project: str) -> bytes:
    path = _key_path(project)
    if not path.exists():
        _fail("The protected root key is missing. Restore it, or use --initialize only with an empty data directory.")
    if path.is_symlink() or not path.is_file():
        _fail("The protected root key must be a regular file, not a symbolic link.")
    if os.name != "nt":
        _validate_linux_private_path(path.parent, create=False)
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            _fail("The Linux root key must be owned by the current user with mode 0600.")
    try:
        key = dpapi_unprotect(path.read_bytes()) if os.name == "nt" else path.read_bytes()
    except Exception as error:
        raise StartupError("The protected root key could not be opened for this user.") from error
    if len(key) != 32:
        _fail("The protected root key has an invalid format.")
    return key


def _binding_tag(key: bytes, salt: bytes, state: str) -> str:
    return hmac.new(key, BINDING_CONTEXT + state.encode("ascii") + b"\0" + salt, hashlib.sha256).hexdigest()


def _write_binding(data_dir: Path, key: bytes) -> None:
    salt = secrets.token_bytes(32)
    payload = {
        "schema": "deep-research-studio-key-binding/v2",
        "state": "initializing",
        "salt": salt.hex(),
        "tag": _binding_tag(key, salt, "initializing"),
    }
    path = data_dir / BINDING_FILE
    temporary = data_dir / f".{BINDING_FILE}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_binding(data_dir: Path, key: bytes) -> str:
    path = data_dir / BINDING_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema")
        state = payload.get("state", "ready")
        if schema not in {"deep-research-studio-key-binding/v1", "deep-research-studio-key-binding/v2"}:
            raise ValueError
        if state not in {"initializing", "ready"}:
            raise ValueError
        if schema.endswith("/v1") and state != "ready":
            raise ValueError
        salt = bytes.fromhex(payload["salt"])
        expected = (
            hmac.new(key, BINDING_CONTEXT + salt, hashlib.sha256).hexdigest()
            if schema.endswith("/v1")
            else _binding_tag(key, salt, state)
        )
        if len(salt) != 32 or not hmac.compare_digest(expected, payload["tag"]):
            raise ValueError
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StartupError("The root key does not match this data directory, or its binding record is invalid.") from error
    return state


def _mark_binding_ready(data_dir: Path, key: bytes) -> None:
    path = data_dir / BINDING_FILE
    state = _verify_binding(data_dir, key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if state == "ready" and payload.get("schema") == "deep-research-studio-key-binding/v2":
        return
    payload["state"] = "ready"
    payload["schema"] = "deep-research-studio-key-binding/v2"
    payload["tag"] = _binding_tag(key, bytes.fromhex(payload["salt"]), "ready")
    temporary = data_dir / f".{BINDING_FILE}.{os.getpid()}.ready.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_existing_data(data_dir: Path) -> None:
    # config.json is created only after the user saves their first setting.
    required = ("baseline.sqlite3", BINDING_FILE)
    if not data_dir.is_dir() or any(not (data_dir / name).is_file() for name in required):
        _fail("Existing installation data is incomplete. Restore the data directory and matching root key.")


def _set_runtime_environment(data_dir: Path) -> None:
    os.environ["DEEPRESEARCH_DATA_DIR"] = data_dir.as_posix()
    os.environ.setdefault("DEEPRESEARCH_API_IMAGE", "deep-research-studio-api:1.0.0")
    os.environ.setdefault("DEEPRESEARCH_WEB_IMAGE", "deep-research-studio-web:1.0.0")
    os.environ.setdefault("DEEPRESEARCH_REDIS_IMAGE", "redis:7-alpine@sha256:8b81dd37ff027bec4e516d41acfbe9fe2460070dc6d4a4570a2ac5b9d59df065")


def _build(project: str) -> None:
    _compose(project, ["build", "--pull", "api", "web"])
    _compose(project, ["pull", "redis"])


def _verify_images_available() -> None:
    for image in (os.environ["DEEPRESEARCH_API_IMAGE"], os.environ["DEEPRESEARCH_WEB_IMAGE"], os.environ["DEEPRESEARCH_REDIS_IMAGE"]):
        _docker(["image", "inspect", image])


def _provision_key(project: str, key: bytes) -> None:
    _compose(project, ["up", "-d", "--no-build", "key-holder", "redis"])
    holder = _compose(project, ["ps", "-q", "key-holder"]).decode("ascii").strip()
    if not holder:
        _fail("The memory-backed key holder did not start.")
    provision = """import hmac,os,sys
from pathlib import Path
root='/run/deepresearch-secrets'
if not any(p[1]==root and p[2]=='tmpfs' for p in (line.split() for line in Path('/proc/mounts').read_text().splitlines())): raise SystemExit(21)
value=sys.stdin.buffer.read(66).strip()
if len(value)!=64 or len(bytes.fromhex(value.decode('ascii')))!=32: raise SystemExit(22)
path=Path(root)/'vault.key'
if path.exists() and not hmac.compare_digest(path.read_bytes().strip(),value): raise SystemExit(23)
if not path.exists():
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 with os.fdopen(fd,'wb') as handle: handle.write(value+b'\\n'); handle.flush(); os.fsync(handle.fileno())
"""
    _docker(["exec", "-i", holder, "python", "-c", provision], input_bytes=key.hex().encode("ascii"))


def _doctor(project: str, data_dir: Path, key: bytes) -> None:
    """Run the doctor in an exact, self-removing container without starting the stack."""
    name = f"{project}-doctor-{os.getpid()}"
    bootstrap = """import os,runpy,sys
from pathlib import Path
value=sys.stdin.buffer.read(66).strip()
if len(value)!=64 or len(bytes.fromhex(value.decode('ascii')))!=32: raise SystemExit(22)
path=Path('/run/deepresearch-secrets/vault.key')
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'wb') as handle: handle.write(value+b'\\n'); handle.flush(); os.fsync(handle.fileno())
runpy.run_path('/app/scripts/release_doctor.py',run_name='__main__')
"""
    args = [
        "run", "--name", name, "--rm", "-i", "--network", "none", "--read-only",
        "--mount", f"type=bind,src={data_dir.as_posix()},dst=/data",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16777216,mode=0700",
        "--tmpfs", "/run/deepresearch-secrets:rw,noexec,nosuid,size=1048576,mode=0700",
        "-e", "DEEPRESEARCH_VAULT_KEY_FILE=/run/deepresearch-secrets/vault.key",
        os.environ["DEEPRESEARCH_API_IMAGE"], "python", "-c", bootstrap,
    ]
    try:
        _docker(args, input_bytes=key.hex().encode("ascii"))
    finally:
        probe = subprocess.run(["docker", "container", "inspect", name], capture_output=True, check=False)
        if probe.returncode == 0:
            subprocess.run(["docker", "container", "rm", "-f", name], capture_output=True, check=False)


def run(project: str, *, data_dir: Path, initialize: bool, build: bool, check_only: bool) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", project):
        _fail("Invalid Compose project name.")
    resolved = _resolve_data_dir(data_dir)
    if initialize and check_only:
        _fail("--initialize and --check are mutually exclusive.")
    key_exists = _key_path(project).exists()
    _set_runtime_environment(resolved)
    if build:
        _build(project)
    else:
        _verify_images_available()
    if initialize:
        binding_exists = (resolved / BINDING_FILE).is_file()
        if key_exists or _has_entries(resolved):
            if not (key_exists and binding_exists):
                _fail("Initialization requires empty data, or a matching interrupted installation.")
            key = _read_key(project)
            if _verify_binding(resolved, key) != "initializing":
                _fail("This installation is already initialized; start it without --initialize.")
        else:
            _docker(["info", "--format", "{{.OSType}}"])
            resolved.mkdir(parents=True, exist_ok=True)
            key = _create_key(project)
            _write_binding(resolved, key)
    else:
        _validate_existing_data(resolved)
        key = _read_key(project)
        _verify_binding(resolved, key)
    if check_only:
        _doctor(project, resolved, key)
        del key
        print(f"{project}: data, key binding, database, and credential vault checks passed.")
        return
    _provision_key(project, key)
    del key
    _compose(project, ["up", "-d", "--no-build", "--wait", "--wait-timeout", "180"])
    key = _read_key(project)
    _doctor(project, resolved, key)
    _mark_binding_ready(resolved, key)
    del key
    print(f"{project}: healthy at http://127.0.0.1:{os.environ.get('DEEPRESEARCH_WEB_PORT', '15175')}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT_DEFAULT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true", help="Create a new key and installation in an empty directory.")
    parser.add_argument("--build", action="store_true", help="Build public API/Web images and pull Redis before startup.")
    parser.add_argument("--check", action="store_true", help="Run read-only data and credential-vault checks without starting the app.")
    args = parser.parse_args()
    if args.initialize and args.check:
        parser.error("--initialize and --check are mutually exclusive")
    try:
        run(args.project, data_dir=args.data_dir, initialize=args.initialize, build=args.build, check_only=args.check)
    except StartupError as error:
        print(f"Deep Research Studio startup failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Deep Research Studio startup failed: {type(error).__name__}. Check Docker, data, and the protected key.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
