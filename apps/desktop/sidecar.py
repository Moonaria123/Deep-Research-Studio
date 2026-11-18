"""Run the isolated Windows desktop API or its bounded document parser worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading


API_HOST = "127.0.0.1"
API_PORT = 18789
TAURI_ORIGIN = "http://tauri.localhost"
DEV_ORIGIN = "http://127.0.0.1:5173"
READY_MESSAGE = "DEEPRESEARCH_DESKTOP_READY"
PROFILE_SCHEMA = "deep-research-studio.desktop-profile/v1"
_PRIVATE_PREFIXES = (
    "DEEPRESEARCH_", "OPENAI_", "ANTHROPIC_", "BRAVE_", "FIRECRAWL_",
    "TAVILY_", "EXA_", "POE_", "VOLCENGINE_", "ARK_", "OTEL_",
)


def drive_type(path: Path) -> int:
    """Query the resolved drive rather than trusting a local-looking junction."""

    import ctypes

    get_drive_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint
    return int(get_drive_type(path.anchor))


def prepare_profile(data_dir: str) -> Path:
    """Keep desktop state and credentials separate from every Web installation."""

    development = os.environ.get("DEEPRESEARCH_DESKTOP_DEV") == "1"
    root = Path(data_dir)
    if not data_dir or not root.is_absolute() or str(root).startswith("\\\\"):
        raise ValueError("Desktop data directory must be an absolute local path")
    root = root.resolve()
    if str(root).startswith("\\\\") or (os.name == "nt" and drive_type(root) not in {2, 3, 6}):
        raise ValueError("Desktop data directory must resolve to a local drive")
    marker = root / "desktop-profile.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != {"schema": PROFILE_SCHEMA}:
            raise ValueError("Unsupported desktop data profile")
    elif root.exists() and any(root.iterdir()):
        # An existing Web/T67/unknown data directory must never be adopted implicitly.
        raise ValueError("Desktop data directory is not an initialized desktop profile")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"schema": PROFILE_SCHEMA}, handle)
            handle.write("\n")

    # A GUI launch must not pick up another checkout's credentials or runtime paths.
    for key in list(os.environ):
        if key.upper().startswith(_PRIVATE_PREFIXES):
            os.environ.pop(key, None)
    os.environ.update({
        "DEEPRESEARCH_DESKTOP": "1",
        "DEEPRESEARCH_DESKTOP_DATA_DIR": str(root),
        "DEEPRESEARCH_CONFIG": str(root / "config.json"),
        "DEEPRESEARCH_DB_PATH": str(root / "baseline.sqlite3"),
        "DEEPRESEARCH_SECURE_CONFIG": str(root / "secure-provider.dpapi.json"),
        "DEEPRESEARCH_VAULT_PATH": str(root / "provider-secrets.dpapi.json"),
        "DEEPRESEARCH_JOB_BACKEND": "thread",
        "DEEPRESEARCH_VECTOR_BACKEND": "numpy",
        "DEEPRESEARCH_LOCAL_WEB_GUARD": "1",
        "DEEPRESEARCH_WEB_ORIGINS": ",".join((TAURI_ORIGIN, DEV_ORIGIN)) if development else TAURI_ORIGIN,
        "DEEPRESEARCH_API_HOST": API_HOST,
        "DEEPRESEARCH_API_PORT": str(API_PORT),
        "DEEPRESEARCH_AUTH_MODE": "off",
    })
    return root


def reserve_listener() -> socket.socket:
    """Bind before opening the database so a second launch cannot touch state."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((API_HOST, API_PORT))
        return listener
    except BaseException:
        listener.close()
        raise


def run_api(data_dir: str) -> int:
    """Signal readiness only after the owned server has successfully started."""

    with reserve_listener() as listener:
        prepare_profile(data_dir)
        import uvicorn
        from deepresearch_core.baseline import create_app
        from deepresearch_core.desktop_lifecycle import (
            recover_interrupted_desktop_state,
            shutdown_desktop_service,
        )
        from run_api import build_service

        service = build_service()
        recover_interrupted_desktop_state(service)
        app = create_app(service, cors_origins=tuple(os.environ["DEEPRESEARCH_WEB_ORIGINS"].split(",")))
        # The desktop adapter shuts down all owned workers under one deadline after
        # HTTP draining; do not spend a separate five seconds on the shared handler.
        app.router.on_shutdown.remove(service.shutdown)

        class DesktopServer(uvicorn.Server):
            async def startup(self, sockets=None):
                await super().startup(sockets=sockets)
                if self.started:
                    print(READY_MESSAGE, flush=True)

        server = DesktopServer(uvicorn.Config(
            app, host=API_HOST, port=API_PORT, loop="asyncio", http="h11", ws="none",
            access_log=False, log_level="warning", timeout_graceful_shutdown=5,
        ))

        def watch_shell() -> None:
            # EOF also covers an unexpected shell exit. The Windows job is the hard
            # containment boundary; this pipe permits normal SQLite/service cleanup.
            if sys.stdin is not None:
                sys.stdin.readline()
            server.should_exit = True

        threading.Thread(target=watch_shell, name="desktop-parent", daemon=True).start()
        try:
            server.run(sockets=[listener])
        finally:
            outcome = shutdown_desktop_service(service, timeout_s=5.0)
            if outcome.safe_to_close_store:
                service.store.close()
        return 0 if server.started else 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--document-worker"]:
        from deepresearch_core.baseline.local_document_import import _main

        return _main(args[1:])
    if args or os.name != "nt":
        print("DESKTOP_STARTUP_FAILED: Windows desktop launch required", file=sys.stderr)
        return 1
    data_dir = os.environ.get("DEEPRESEARCH_DESKTOP_DATA_DIR", "")
    if not data_dir:
        print("DESKTOP_STARTUP_FAILED: desktop profile is missing", file=sys.stderr)
        return 1
    try:
        return run_api(data_dir)
    except OSError:
        print("DESKTOP_STARTUP_FAILED: local port or profile is unavailable", file=sys.stderr)
    except Exception:
        # Provider/config/parser diagnostics must not expose credentials in a GUI log.
        print("DESKTOP_STARTUP_FAILED: local service could not start safely", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
