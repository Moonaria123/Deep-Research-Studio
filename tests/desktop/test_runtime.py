"""Desktop profile isolation, browser boundaries, and frozen worker contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from apps.desktop import sidecar


@pytest.fixture
def desktop_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    root = sidecar.prepare_profile(str(tmp_path / "desktop"))
    return root


def test_profile_does_not_adopt_web_paths_or_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {
        "DEEPRESEARCH_CONFIG": str(tmp_path / "web" / "config.json"),
        "DEEPRESEARCH_DB_PATH": str(tmp_path / "web" / "baseline.sqlite3"),
        "DEEPRESEARCH_JOB_BACKEND": "rq",
        "DEEPRESEARCH_VAULT_KEY_FILE": "old-web-vault",
        "DEEPRESEARCH_LLM_API_KEY": "synthetic-inherited-test-value",
        "OPENAI_API_KEY": "synthetic-inherited-test-value",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://untrusted.invalid",
        "SYSTEMROOT": "C:\\Windows",
    })
    root = sidecar.prepare_profile(str(tmp_path / "desktop"))
    assert os.environ["DEEPRESEARCH_CONFIG"] == str(root / "config.json")
    assert os.environ["DEEPRESEARCH_DB_PATH"] == str(root / "baseline.sqlite3")
    assert os.environ["DEEPRESEARCH_JOB_BACKEND"] == "thread"
    assert os.environ["DEEPRESEARCH_WEB_ORIGINS"] == sidecar.TAURI_ORIGIN
    assert "DEEPRESEARCH_LLM_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
    assert "DEEPRESEARCH_VAULT_KEY_FILE" not in os.environ
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
    assert os.environ["SYSTEMROOT"] == "C:\\Windows"
    assert not (tmp_path / "web").exists()


def test_reopen_preserves_profile_and_business_data(desktop_profile):
    marker = desktop_profile / "desktop-profile.json"
    previous = marker.read_bytes()
    data = desktop_profile / "baseline.sqlite3"
    data.write_bytes(b"synthetic business data")
    assert sidecar.prepare_profile(str(desktop_profile)) == desktop_profile
    assert marker.read_bytes() == previous
    assert data.read_bytes() == b"synthetic business data"


def test_development_origin_requires_explicit_shell_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("DEEPRESEARCH_DESKTOP_DEV", "1")
    root = sidecar.prepare_profile(str(tmp_path / "desktop"))
    assert os.environ["DEEPRESEARCH_WEB_ORIGINS"].split(",") == [sidecar.TAURI_ORIGIN, sidecar.DEV_ORIGIN]
    sidecar.prepare_profile(str(root))
    assert os.environ["DEEPRESEARCH_WEB_ORIGINS"] == sidecar.TAURI_ORIGIN


@pytest.mark.parametrize("marker", [None, "{}", '{"schema":"unknown/v3"}', "invalid"])
def test_unknown_or_corrupt_profiles_fail_without_modification(tmp_path, marker):
    existing = tmp_path / "baseline.sqlite3"
    existing.write_bytes(b"existing Web or unknown data")
    if marker is not None:
        (tmp_path / "desktop-profile.json").write_text(marker, encoding="utf-8")
    with pytest.raises(ValueError):
        sidecar.prepare_profile(str(tmp_path))
    assert existing.read_bytes() == b"existing Web or unknown data"


@pytest.mark.parametrize("path", ["", "relative-profile", "\\\\server\\share\\profile"])
def test_profile_requires_absolute_local_directory(path):
    with pytest.raises(ValueError):
        sidecar.prepare_profile(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive type validation")
@pytest.mark.parametrize("drive_kind", [0, 1, 4, 5])
def test_profile_rejects_remote_unknown_or_read_only_drives(monkeypatch, tmp_path, drive_kind):
    monkeypatch.setattr(sidecar, "drive_type", lambda _: drive_kind)
    with pytest.raises(ValueError, match="local drive"):
        sidecar.prepare_profile(str(tmp_path / "desktop"))
    assert not (tmp_path / "desktop").exists()


@pytest.mark.skipif(os.name != "nt", reason="Desktop uses Windows DPAPI")
def test_desktop_credentials_are_dpapi_protected_and_reopen(desktop_profile):
    from deepresearch_core.config import load_provider_config, save_provider_config
    from deepresearch_core.settings import ProviderSettings

    config = desktop_profile / "config.json"
    value = "desktop-synthetic-credential-for-offline-test"
    save_provider_config(config, ProviderSettings(brave_api_key=value))
    assert value not in config.read_text(encoding="utf-8")
    assert value not in (desktop_profile / "provider-secrets.dpapi.json").read_text(encoding="utf-8")
    sidecar.prepare_profile(str(desktop_profile))
    assert load_provider_config(config).brave_api_key == value


def test_exact_desktop_origin_can_cross_site_but_other_browser_origins_cannot(desktop_profile):
    from fastapi.testclient import TestClient
    from deepresearch_core.baseline import BaselineService, create_app
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:",
                              snapshot_root=desktop_profile / "snapshots")
    with TestClient(create_app(service, cors_origins=(sidecar.TAURI_ORIGIN,)),
                    base_url="http://127.0.0.1:18789") as client:
        headers = {"Origin": sidecar.TAURI_ORIGIN, "Sec-Fetch-Site": "cross-site"}
        response = client.get("/api/v1/providers/llm", headers=headers)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == sidecar.TAURI_ORIGIN
        preflight = client.options("/api/v1/providers/llm", headers={
            **headers, "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        })
        assert preflight.status_code == 200
        for origin in ("http://evil.invalid", "http://tauri.localhost.evil.invalid", "null"):
            bad = {"Origin": origin, "Sec-Fetch-Site": "cross-site"}
            assert client.get("/api/v1/providers/llm", headers=bad).status_code == 403
            assert client.put("/api/v1/providers/llm", headers=bad, json={}).status_code == 403
        assert client.get("/api/v1/health", headers={**headers, "Host": "evil.invalid"}).status_code == 403
    service.store.close()


def test_frozen_parser_reenters_worker_without_starting_server(monkeypatch):
    from deepresearch_core.baseline import local_document_import as parser

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({"ok": True, "text": "fixture text"}))

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("DEEPRESEARCH_LLM_API_KEY", "synthetic-parent-only")
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "test-extraction")
    result = parser.parse_local_document(b"%PDF-1.4\n", filename="fixture.pdf", mime_type="application/pdf")
    assert result.text == "fixture text"
    command, options = calls[0]
    assert command[:3] == [sys.executable, "--document-worker", "--worker"]
    assert "DEEPRESEARCH_LLM_API_KEY" not in options["env"]
    assert options["env"]["_PYI_APPLICATION_HOME_DIR"] == "test-extraction"
    assert "PYTHONPATH" not in options["env"]
    assert options["timeout"] == 20.0
    assert options["stdin"] == subprocess.DEVNULL


def test_worker_dispatch_does_not_initialize_profile(monkeypatch):
    from deepresearch_core.baseline import local_document_import as parser

    seen = []
    monkeypatch.setattr(parser, "_main", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(sidecar, "prepare_profile", lambda *_: pytest.fail("worker must not open profile"))
    assert sidecar.main(["--document-worker", "--worker", "fixture.pdf", "--kind", "pdf"]) == 0
    assert seen == [["--worker", "fixture.pdf", "--kind", "pdf"]]
