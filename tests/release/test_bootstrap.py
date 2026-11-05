from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_launcher", ROOT / "scripts" / "start_gen1_web.py")
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
DOCTOR_SPEC = importlib.util.spec_from_file_location("release_doctor", ROOT / "scripts" / "release_doctor.py")
assert DOCTOR_SPEC and DOCTOR_SPEC.loader
doctor = importlib.util.module_from_spec(DOCTOR_SPEC)
DOCTOR_SPEC.loader.exec_module(doctor)


def test_binding_accepts_matching_key_and_rejects_other_key(tmp_path: Path):
    key = b"a" * 32
    launcher._write_binding(tmp_path, key)
    assert launcher._verify_binding(tmp_path, key) == "initializing"

    with pytest.raises(launcher.StartupError, match="does not match"):
        launcher._verify_binding(tmp_path, b"b" * 32)


def test_binding_rejects_tampering(tmp_path: Path):
    launcher._write_binding(tmp_path, b"a" * 32)
    binding = tmp_path / launcher.BINDING_FILE
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["tag"] = "0" * 64
    binding.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(launcher.StartupError, match="binding record"):
        launcher._verify_binding(tmp_path, b"a" * 32)


def test_binding_state_is_authenticated(tmp_path: Path):
    launcher._write_binding(tmp_path, b"a" * 32)
    binding = tmp_path / launcher.BINDING_FILE
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["state"] = "ready"
    binding.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(launcher.StartupError, match="binding record"):
        launcher._verify_binding(tmp_path, b"a" * 32)


def test_legacy_binding_only_accepts_ready_and_is_upgraded(tmp_path: Path):
    key = b"a" * 32
    salt = b"s" * 32
    binding = tmp_path / launcher.BINDING_FILE
    legacy = {
        "schema": "deep-research-studio-key-binding/v1",
        "state": "ready",
        "salt": salt.hex(),
        "tag": launcher.hmac.new(key, launcher.BINDING_CONTEXT + salt, launcher.hashlib.sha256).hexdigest(),
    }
    binding.write_text(json.dumps(legacy), encoding="utf-8")
    assert launcher._verify_binding(tmp_path, key) == "ready"

    legacy["state"] = "initializing"
    binding.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(launcher.StartupError, match="binding record"):
        launcher._verify_binding(tmp_path, key)

    legacy["state"] = "ready"
    binding.write_text(json.dumps(legacy), encoding="utf-8")
    launcher._mark_binding_ready(tmp_path, key)
    upgraded = json.loads(binding.read_text(encoding="utf-8"))
    assert upgraded["schema"].endswith("/v2")
    assert launcher._verify_binding(tmp_path, key) == "ready"


def test_data_directory_is_explicit_absolute_and_not_a_symlink(tmp_path: Path):
    with pytest.raises(launcher.StartupError, match="required"):
        launcher._resolve_data_dir(None)
    with pytest.raises(launcher.StartupError, match="absolute"):
        launcher._resolve_data_dir(Path("relative"))

    target = tmp_path / "data"
    target.mkdir()
    assert launcher._resolve_data_dir(target) == target.resolve()


def test_existing_installation_requires_complete_data(tmp_path: Path):
    (tmp_path / launcher.BINDING_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(launcher.StartupError, match="incomplete"):
        launcher._validate_existing_data(tmp_path)


def test_existing_installation_does_not_require_unsaved_config(tmp_path: Path):
    (tmp_path / launcher.BINDING_FILE).write_text("{}", encoding="utf-8")
    (tmp_path / "baseline.sqlite3").write_bytes(b"placeholder")
    launcher._validate_existing_data(tmp_path)


def test_public_compose_has_local_ports_and_no_host_specific_default():
    compose = (ROOT / "docker" / "compose.gen1.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:${DEEPRESEARCH_WEB_PORT:-15175}:80" in compose
    assert "127.0.0.1:${DEEPRESEARCH_API_PORT:-18789}:8787" in compose
    assert "DEEPRESEARCH_DATA_DIR:?" in compose
    assert "D:/" not in compose
    assert "read_only: true" in compose
    assert "network_mode: none" in compose
    assert "doctor:" not in compose


def test_check_only_uses_one_shot_isolated_container():
    source = (ROOT / "scripts" / "start_gen1_web.py").read_text(encoding="utf-8")
    assert '"--rm", "-i", "--network", "none", "--read-only"' in source
    assert '"container", "rm", "-f", name' in source
    assert "_doctor(project, resolved, key)" in source


def test_doctor_query_only_backup_preserves_source(tmp_path: Path):
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "snapshot.sqlite3"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
    connection.execute("INSERT INTO sample VALUES ('kept')")
    connection.commit()
    connection.close()
    before_hash = hashlib.sha256(source.read_bytes()).digest()

    read_only = doctor.open_query_only(source)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        read_only.execute("INSERT INTO sample VALUES ('blocked')")
    read_only.close()
    doctor.backup_and_check(source, target)

    assert hashlib.sha256(source.read_bytes()).digest() == before_hash
    with sqlite3.connect(source) as verified:
        assert verified.execute("SELECT value FROM sample").fetchall() == [("kept",)]
    with sqlite3.connect(target) as snapshot:
        assert snapshot.execute("SELECT value FROM sample").fetchall() == [("kept",)]


def test_doctor_rejects_plaintext_credentials_and_collects_references():
    with pytest.raises(doctor.SecretVaultError, match="Plaintext"):
        doctor._collect_references({"providers": [{"api_key": "not-protected"}]})
    reference = {"secret_ref": "vault-v1:00000000-0000-0000-0000-000000000001"}
    assert doctor._collect_references({"llm_api_key": reference}) == [reference["secret_ref"]]
    assert doctor._collect_references({"api_key": ""}) == []


def test_failed_build_precedes_any_initialization_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key_path = tmp_path / "state" / "vault-root.key"
    monkeypatch.setattr(launcher, "_key_path", lambda _project: key_path)
    monkeypatch.setattr(launcher, "_set_runtime_environment", lambda _path: None)
    monkeypatch.setattr(launcher, "_build", lambda _project: (_ for _ in ()).throw(launcher.StartupError("build failed")))

    data = tmp_path / "data"
    with pytest.raises(launcher.StartupError, match="build failed"):
        launcher.run("release-test", data_dir=data, initialize=True, build=True, check_only=False)
    assert not data.exists()
    assert not key_path.exists()


def test_interrupted_initialization_can_resume_without_replacing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key = b"k" * 32
    key_path = tmp_path / "state" / "vault-root.key"
    data = tmp_path / "data"
    created = 0

    def create_key(_project: str) -> bytes:
        nonlocal created
        created += 1
        key_path.parent.mkdir(parents=True)
        key_path.write_bytes(b"sealed")
        return key

    monkeypatch.setattr(launcher, "_key_path", lambda _project: key_path)
    monkeypatch.setattr(launcher, "_create_key", create_key)
    monkeypatch.setattr(launcher, "_read_key", lambda _project: key)
    monkeypatch.setattr(launcher, "_set_runtime_environment", lambda _path: None)
    monkeypatch.setattr(launcher, "_verify_images_available", lambda: None)
    monkeypatch.setattr(launcher, "_docker", lambda *_args, **_kwargs: b"linux")
    monkeypatch.setattr(launcher, "_compose", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(launcher, "_doctor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "_provision_key", lambda *_args, **_kwargs: (_ for _ in ()).throw(launcher.StartupError("port conflict")))

    with pytest.raises(launcher.StartupError, match="port conflict"):
        launcher.run("release-test", data_dir=data, initialize=True, build=False, check_only=False)
    assert created == 1
    assert launcher._verify_binding(data, key) == "initializing"

    monkeypatch.setattr(launcher, "_provision_key", lambda *_args, **_kwargs: None)
    launcher.run("release-test", data_dir=data, initialize=True, build=False, check_only=False)
    assert created == 1
    assert launcher._verify_binding(data, key) == "ready"


def test_initialize_and_check_are_rejected_before_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key_path = tmp_path / "state" / "vault-root.key"
    monkeypatch.setattr(launcher, "_key_path", lambda _project: key_path)
    with pytest.raises(launcher.StartupError, match="mutually exclusive"):
        launcher.run("release-test", data_dir=tmp_path / "data", initialize=True, build=False, check_only=True)
    assert not key_path.exists()


def test_public_dockerfile_excludes_internal_operations_scripts():
    dockerfile = (ROOT / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    assert "release_doctor.py" in dockerfile
    assert "authorized_cleanup" not in dockerfile
    assert "backfill_semantic" not in dockerfile
    assert "verify_t5" not in dockerfile
