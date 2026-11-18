"""Exercise the actual Windows sidecar offline, preserving isolated test evidence."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile


ORIGIN = "http://tauri.localhost"
BASE = "http://127.0.0.1:18789/api/v1"
FIXTURE_SECRET = "desktop-offline-synthetic-value-not-a-provider-key"


def request(path, *, method="GET", payload=None, headers=None, raw=None):
    body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    actual = {"Origin": ORIGIN, "Sec-Fetch-Site": "cross-site", **(headers or {})}
    if payload is not None:
        actual["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=actual)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def document_fixture() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>''')
        archive.writestr("_rels/.rels", '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>''')
        archive.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Offline desktop import fixture. All data stays in the isolated smoke profile.</w:t></w:r></w:p></w:body></w:document>''')
    return buffer.getvalue()


def start(command: list[str], profile: Path):
    environment = dict(os.environ)
    environment["DEEPRESEARCH_DESKTOP_DATA_DIR"] = str(profile)
    process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return process


def ready(process, timeout=60):
    lines = queue.Queue()

    def read():
        for line in process.stdout:
            lines.put(line.strip())
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.25)
        except queue.Empty:
            continue
        if line == "DEEPRESEARCH_DESKTOP_READY":
            return
        if line is None:
            break
    raise RuntimeError("Sidecar did not signal readiness")


def stop(process):
    if process.poll() is None:
        try:
            process.stdin.write("shutdown\n")
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            process.wait(timeout=10)
            raise RuntimeError("Sidecar required forced process-tree cleanup")
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = output / "profile"
    root = Path(__file__).resolve().parents[1]
    command = [str(args.executable.resolve())] if args.executable else [sys.executable, str(root / "apps/desktop/sidecar.py")]
    results = {"kind": "packaged" if args.executable else "source", "checks": [], "passed": False}
    if args.executable:
        results["exe_sha256"] = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    process = None

    def passed(name):
        results["checks"].append(name)

    try:
        started = time.monotonic()
        process = start(command, profile)
        ready(process)
        results["cold_start_seconds"] = round(time.monotonic() - started, 3)
        code, health, headers = request("/health")
        assert code == 200 and headers["Access-Control-Allow-Origin"] == ORIGIN
        assert health.get("job_backend") == "thread", "Desktop must use thread jobs"
        passed("ready_health_and_exact_cors")
        assert request("/providers/llm", headers={"Origin": "http://untrusted.invalid"})[0] == 403
        assert request("/health", headers={"Host": "rebinding.invalid"})[0] == 403
        passed("host_and_origin_rejection")

        duplicate = start(command, output / "duplicate-profile")
        stdout, _ = duplicate.communicate(timeout=30)
        assert duplicate.returncode != 0 and "DEEPRESEARCH_DESKTOP_READY" not in stdout
        assert not (output / "duplicate-profile").exists()
        passed("second_launch_rejected_before_data_open")

        code, session, _ = request("/sessions", method="POST", payload={"title": "Desktop offline smoke"})
        assert code == 200
        session_id = session["session_id"]
        code, registry, _ = request("/providers/llm", method="PUT", payload={
            "id": "offline-fixture", "label": "Offline smoke fixture", "type": "openai",
            "base_url": "http://127.0.0.1:1/v1", "api_key": FIXTURE_SECRET,
            "models": ["offline-fixture"], "enabled": False,
        })
        assert code == 200 and FIXTURE_SECRET not in json.dumps(registry)
        for path in (profile / "config.json", profile / "provider-secrets.dpapi.json"):
            assert FIXTURE_SECRET.encode() not in path.read_bytes()
        passed("provider_dpapi_save_without_plaintext_or_network_probe")

        boundary = "DesktopOfflineFixtureBoundary"
        content = document_fixture()
        multipart = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"fixture.docx\"\r\nContent-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\r\n\r\n".encode()
                     + content + f"\r\n--{boundary}--\r\n".encode())
        code, imported, _ = request("/library/documents/import", method="POST", raw=multipart,
                                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert code == 200 and imported.get("status") == "created", (
            f"Document import failed: HTTP {code}, code={imported.get('error', {}).get('code', 'unknown')}"
        )
        passed("document_import_through_packaged_worker")
        assert stop(process) == 0
        process = None
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", 18789)) != 0
        passed("graceful_exit_releases_port")

        process = start(command, profile)
        ready(process)
        code, sessions, _ = request("/sessions")
        assert code == 200 and any(item["id"] == session_id for item in sessions["sessions"])
        code, registry, _ = request("/providers/llm")
        assert code == 200 and any(p["id"] == "offline-fixture" and p["has_key"] for p in registry["providers"])
        passed("restart_preserves_session_and_decrypts_provider")
        # A closed parent pipe must stop the service without an HTTP shutdown API.
        process.stdin.close()
        assert process.wait(timeout=15) == 0
        process = None
        passed("parent_pipe_eof_stops_sidecar")
        results["passed"] = True
    except Exception as error:
        results["error_type"] = type(error).__name__
        results["error"] = str(error) if isinstance(error, (AssertionError, RuntimeError)) else "Smoke verification failed"
    finally:
        if process is not None:
            try:
                stop(process)
            except Exception:
                results["cleanup_failed"] = True
                results["passed"] = False
        (output / "result.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
