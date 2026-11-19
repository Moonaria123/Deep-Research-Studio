from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "apps" / "desktop" / "scripts" / "build_sidecar.py"
SPEC = importlib.util.spec_from_file_location("desktop_build_sidecar", SCRIPT)
assert SPEC and SPEC.loader
build_sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_sidecar)


def test_build_command_is_onefile_and_has_no_runtime_data_inputs(tmp_path: Path) -> None:
    command = build_sidecar.build_command(
        Path("venv/python.exe"),
        root=ROOT,
        entry=ROOT / "apps" / "desktop" / "sidecar.py",
        work_dir=tmp_path / "work",
        dist_dir=tmp_path / "dist",
        spec_dir=tmp_path / "spec",
    )

    assert "--onefile" in command
    assert "--add-data" not in command
    arguments = {str(item).lower() for item in command}
    for forbidden in ("config.json", ".dpapi", ".sqlite", ".key"):
        assert not any(item.endswith(forbidden) for item in arguments)
    assert "run_api" in command


def test_source_identity_is_stable_and_contains_only_allowlisted_code(tmp_path: Path) -> None:
    desktop = tmp_path / "apps" / "desktop"
    core = tmp_path / "deepresearch_core"
    scripts = desktop / "scripts"
    scripts.mkdir(parents=True)
    core.mkdir()
    (desktop / "sidecar.py").write_text("print('ready')\n", encoding="utf-8")
    (desktop / "requirements-sidecar.txt").write_text("fastapi==1\n", encoding="utf-8")
    (desktop / "requirements-build.txt").write_text("PyInstaller==1\n", encoding="utf-8")
    (scripts / "build_sidecar.py").write_text("# builder\n", encoding="utf-8")
    (tmp_path / "run_api.py").write_text("# api\n", encoding="utf-8")
    (core / "service.py").write_text("# service\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"secret":"not-collected"}', encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "baseline.sqlite3").write_bytes(b"secret data")

    files = build_sidecar.source_files(tmp_path)
    first = build_sidecar.source_identity(tmp_path, files)
    second = build_sidecar.source_identity(tmp_path, reversed(files))

    assert first == second
    paths = {item["path"] for item in first["files"]}
    assert "deepresearch_core/service.py" in paths
    assert "config.json" not in paths
    assert not any(path.startswith("data/") for path in paths)


def test_locked_requirements_accept_only_exact_unconditional_pins(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("PyYAML==6.0.3\npython_multipart==0.0.32  # API form parser\n", encoding="utf-8")
    assert build_sidecar.locked_requirements([requirements]) == {
        "pyyaml": "6.0.3",
        "python-multipart": "0.0.32",
    }

    requirements.write_text("fastapi>=0.100\n", encoding="utf-8")
    try:
        build_sidecar.locked_requirements([requirements])
    except ValueError as error:
        assert "exact pin" in str(error)
    else:
        raise AssertionError("an unpinned desktop requirement was accepted")


def test_changed_sources_are_rejected_before_publication() -> None:
    initial = {"sha256": "a" * 64, "file_count": 1, "files": []}
    final = {"sha256": "b" * 64, "file_count": 1, "files": []}

    build_sidecar.ensure_source_unchanged(initial, dict(initial))
    try:
        build_sidecar.ensure_source_unchanged(initial, final)
    except RuntimeError as error:
        assert "changed during the build" in str(error)
    else:
        raise AssertionError("a build with changing inputs was accepted")


def test_manifest_records_binary_and_source_hashes_without_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / build_sidecar.BINARY_NAME
    binary.write_bytes(b"MZ reproducible fixture")
    manifest = tmp_path / build_sidecar.MANIFEST_NAME
    monkeypatch.setattr(build_sidecar, "_git_commit", lambda _root: "a" * 40)

    build_sidecar.write_manifest(
        manifest,
        binary=binary,
        root=tmp_path,
        identity={"sha256": "b" * 64, "file_count": 1, "files": [{"path": "sidecar.py", "sha256": "c" * 64}]},
        python_metadata={"version": "3.12.10", "pyinstaller": "6.16.0"},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == "deepresearch.desktop-sidecar-build/v1"
    assert payload["binary"]["sha256"] == build_sidecar.sha256_file(binary)
    assert payload["source"]["commit"] == "a" * 40
    assert str(tmp_path) not in manifest.read_text(encoding="utf-8")
