"""Build and identify the Windows API sidecar without collecting user data.

The build is intentionally rooted in an isolated virtual environment supplied
with ``--python``. PyInstaller follows Python imports from the desktop entry
point; the script never adds the repository's data, config, or runtime folders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence


TARGET_TRIPLE = "x86_64-pc-windows-msvc"
BINARY_NAME = f"deepresearch-api-{TARGET_TRIPLE}.exe"
MANIFEST_NAME = f"deepresearch-api-{TARGET_TRIPLE}.manifest.json"
HIDDEN_IMPORTS = (
    "multipart",
    "numpy",
    "openpyxl",
    "pypdf",
    "run_api",
    "uvicorn.lifespan.on",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
)


def repo_root(script_path: Path | None = None) -> Path:
    """Resolve the repository root from this script's fixed location."""

    path = (script_path or Path(__file__)).resolve()
    return path.parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> tuple[Path, ...]:
    """Return the allowlisted source inputs used to identify a sidecar build."""

    explicit = (
        root / "apps" / "desktop" / "sidecar.py",
        root / "apps" / "desktop" / "requirements-sidecar.txt",
        root / "apps" / "desktop" / "requirements-build.txt",
        root / "apps" / "desktop" / "scripts" / "build_sidecar.py",
        root / "run_api.py",
    )
    package_sources = tuple(sorted((root / "deepresearch_core").rglob("*.py")))
    selected = tuple(path for path in (*explicit, *package_sources) if path.is_file())
    if root / "apps" / "desktop" / "sidecar.py" not in selected:
        raise FileNotFoundError("Desktop sidecar entry point is missing")
    return selected


def source_identity(root: Path, files: Iterable[Path]) -> dict[str, object]:
    """Hash paths and contents without placing source text in the manifest."""

    records: list[tuple[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(files):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        file_hash = sha256_file(path)
        records.append((relative, file_hash))
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "sha256": aggregate.hexdigest(),
        "file_count": len(records),
        "files": [{"path": path, "sha256": digest} for path, digest in records],
    }


def ensure_source_unchanged(initial: dict[str, object], final: dict[str, object]) -> None:
    """Reject a build whose input tree changed while PyInstaller was reading it."""

    if final != initial:
        raise RuntimeError("Sidecar sources changed during the build; discard this build and retry")


def build_command(
    python: Path,
    *,
    root: Path,
    entry: Path,
    work_dir: Path,
    dist_dir: Path,
    spec_dir: Path,
) -> list[str]:
    """Construct the fixed, secret-free PyInstaller command."""

    command = [
        str(python),
        "-m",
        "PyInstaller",
        "--onefile",
        "--clean",
        "--noconfirm",
        "--name",
        "deepresearch-api",
        "--paths",
        str(root),
        "--workpath",
        str(work_dir),
        "--distpath",
        str(dist_dir),
        "--specpath",
        str(spec_dir),
    ]
    for module in HIDDEN_IMPORTS:
        command.extend(("--hidden-import", module))
    command.append(str(entry))
    return command


def locked_requirements(paths: Iterable[Path]) -> dict[str, str]:
    """Read the simple, exact pins used by the desktop build."""

    locked: dict[str, str] = {}
    for path in paths:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "==" not in line or any(marker in line for marker in (";", "[", "]")):
                raise ValueError(f"Desktop requirement must be an unconditional exact pin: {line}")
            name, version = line.split("==", 1)
            normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
            if not normalized or not version.strip():
                raise ValueError(f"Invalid desktop requirement: {line}")
            locked[normalized] = version.strip()
    return locked


def _python_metadata(python: Path, *, expected: dict[str, str]) -> dict[str, object]:
    probe = (
        "import importlib.metadata,json,sys,PyInstaller;"
        "print(json.dumps({'version':sys.version.split()[0],"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'pyinstaller':PyInstaller.__version__,"
        "'distributions':{name:importlib.metadata.version(name) for name in sys.argv[1:]}}))"
    )
    completed = subprocess.run(
        [str(python), "-c", probe, *sorted(expected)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    metadata = json.loads(completed.stdout)
    if Path(str(metadata["prefix"])).resolve() == Path(str(metadata["base_prefix"])).resolve():
        raise RuntimeError("--python must point to an isolated virtual environment")
    if str(metadata["version"]).split(".")[:2] != ["3", "12"]:
        raise RuntimeError("The sidecar must be built with Python 3.12")
    installed = metadata["distributions"]
    mismatches = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in expected.items()
        if installed.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"Build environment does not match locked requirements: {mismatches}")
    return metadata


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def write_manifest(
    path: Path,
    *,
    binary: Path,
    root: Path,
    identity: dict[str, object],
    python_metadata: dict[str, object],
) -> None:
    payload = {
        "schema": "deepresearch.desktop-sidecar-build/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "target_triple": TARGET_TRIPLE,
        "binary": {
            "name": binary.name,
            "bytes": binary.stat().st_size,
            "sha256": sha256_file(binary),
        },
        "source": {
            "commit": _git_commit(root),
            **identity,
        },
        "builder": {
            "python": python_metadata["version"],
            "pyinstaller": python_metadata["pyinstaller"],
            "platform": platform.platform(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Python 3.12 executable in an isolated venv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Tauri binary directory (defaults to apps/desktop/src-tauri/binaries)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("The Gen1 sidecar must be built on 64-bit Windows")

    python = args.python.resolve()
    if not python.is_file():
        raise SystemExit(f"Build Python does not exist: {python}")
    requirements = (
        root / "apps" / "desktop" / "requirements-sidecar.txt",
        root / "apps" / "desktop" / "requirements-build.txt",
    )
    python_metadata = _python_metadata(python, expected=locked_requirements(requirements))
    entry = root / "apps" / "desktop" / "sidecar.py"
    files = source_files(root)
    identity = source_identity(root, files)

    build_root = root / ".tmp" / "desktop-sidecar-build"
    work_dir = build_root / "work"
    dist_dir = build_root / "dist"
    spec_dir = build_root / "spec"
    for directory in (work_dir, dist_dir, spec_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log_path = build_root / "pyinstaller.log"
    command = build_command(
        python,
        root=root,
        entry=entry,
        work_dir=work_dir,
        dist_dir=dist_dir,
        spec_dir=spec_dir,
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"PyInstaller failed with exit code {completed.returncode}; see {log_path}")

    final_identity = source_identity(root, source_files(root))
    try:
        ensure_source_unchanged(identity, final_identity)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    built = dist_dir / "deepresearch-api.exe"
    if not built.is_file() or built.stat().st_size == 0:
        raise SystemExit(f"PyInstaller did not produce the expected binary: {built}")

    output_dir = (args.output_dir or root / "apps" / "desktop" / "src-tauri" / "binaries").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / BINARY_NAME
    temporary = output_dir / f".{BINARY_NAME}.tmp"
    shutil.copy2(built, temporary)
    temporary.replace(target)
    manifest = output_dir / MANIFEST_NAME
    write_manifest(
        manifest,
        binary=target,
        root=root,
        identity=identity,
        python_metadata=python_metadata,
    )
    print(f"Sidecar: {target}")
    print(f"SHA-256: {sha256_file(target)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
