"""Safe local text snapshots for Library sources.

T8b stores sanitized text snapshots only. The helper owns path containment so API
code never needs to trust caller-provided filesystem paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from .errors import BaselineError, ErrorCode


_SNAPSHOT_LOCKS: dict[str, threading.RLock] = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class SnapshotWriteResult:
    """Metadata for a written snapshot file."""

    url: str
    content_hash: str
    storage_path: str
    mime_type: str
    size_bytes: int


class SnapshotWriter:
    """Write and read bounded text snapshots under one root directory."""

    def __init__(self, root: str | Path, *, max_chars: int = 500_000) -> None:
        self._root = Path(root).expanduser().resolve()
        self._max_chars = max(1, int(max_chars))

    @contextmanager
    def operation_lock(self) -> Iterator[None]:
        """Serialize snapshot file/Store binding operations across service processes."""

        key = str(self._root)
        with _SNAPSHOT_LOCKS_GUARD:
            local_lock = _SNAPSHOT_LOCKS.setdefault(key, threading.RLock())
        with local_lock:
            self._root.mkdir(parents=True, exist_ok=True)
            lock_path = self._root / ".snapshots.lock"
            with lock_path.open("a+b") as handle:
                _lock_file(handle)
                try:
                    yield
                finally:
                    _unlock_file(handle)

    def write_text_snapshot(self, *, url: str, title: str = "", text: str) -> SnapshotWriteResult:
        """Sanitize text, write it under the snapshot root, and return metadata."""

        body = _sanitize_text(text, max_chars=self._max_chars)
        digest = hashlib.sha256(f"{url}\n{body}".encode("utf-8")).hexdigest()
        rel = Path(digest[:2]) / f"{digest}.txt"
        target = self.resolve(str(rel))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return SnapshotWriteResult(
            url=url,
            content_hash=digest,
            storage_path=rel.as_posix(),
            mime_type="text/plain",
            size_bytes=target.stat().st_size,
        )

    def read_text(self, storage_path: str) -> str:
        """Read a snapshot after validating path containment."""

        return self.resolve(storage_path).read_text(encoding="utf-8")

    def resolve(self, storage_path: str) -> Path:
        """Resolve a relative snapshot path and reject escapes."""

        raw = Path(str(storage_path or ""))
        if raw.is_absolute() or any(part == ".." for part in raw.parts):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "invalid snapshot path")
        resolved = (self._root / raw).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "invalid snapshot path") from exc
        return resolved

    def delete_unreferenced(
        self,
        storage_paths: Iterable[str],
        *,
        referenced_paths: set[str],
    ) -> dict[str, object]:
        """Delete safe regular files no longer referenced by snapshot rows."""

        deleted = 0
        warnings: list[str] = []
        for storage_path in sorted({str(item) for item in storage_paths if str(item)}):
            if storage_path in referenced_paths:
                continue
            target = self._safe_unlink_target(storage_path)
            if not target.exists() and not target.is_symlink():
                continue
            try:
                info = target.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                    warnings.append("snapshot_not_safe_to_unlink")
                    continue
                target.unlink()
                deleted += 1
                self._prune_empty_parents(target.parent)
            except OSError:
                warnings.append("snapshot_unlink_failed")
        return {"deleted": deleted, "warnings": sorted(set(warnings))}

    def sweep_orphans(self, *, referenced_paths: set[str], limit: int = 4096) -> dict[str, object]:
        """Run a bounded, non-following sweep for unreferenced snapshot text files."""

        if not self._root.exists():
            return {"scanned": 0, "deleted": 0, "warnings": []}
        scanned = 0
        candidates: list[str] = []
        warnings: list[str] = []
        for raw_root, directories, files in os.walk(self._root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if not (Path(raw_root) / name).is_symlink()
            ]
            for filename in sorted(files):
                if scanned >= max(0, int(limit)):
                    warnings.append("snapshot_sweep_limit_reached")
                    break
                scanned += 1
                path = Path(raw_root) / filename
                if path.suffix.lower() != ".txt":
                    continue
                try:
                    relative = path.relative_to(self._root).as_posix()
                except ValueError:
                    warnings.append("snapshot_path_outside_root")
                    continue
                if relative not in referenced_paths:
                    candidates.append(relative)
            if scanned >= max(0, int(limit)):
                break
        cleanup = self.delete_unreferenced(candidates, referenced_paths=referenced_paths)
        warnings.extend(str(item) for item in cleanup["warnings"])
        return {
            "scanned": scanned,
            "deleted": int(cleanup["deleted"]),
            "warnings": sorted(set(warnings)),
        }

    def _prune_empty_parents(self, parent: Path) -> None:
        current = parent
        while current != self._root:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _safe_unlink_target(self, storage_path: str) -> Path:
        raw = Path(str(storage_path or ""))
        if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "invalid snapshot path")
        parent = (self._root / raw).parent.resolve()
        try:
            parent.relative_to(self._root)
        except ValueError as exc:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "invalid snapshot path") from exc
        return parent / raw.name


def _sanitize_text(text: str, *, max_chars: int) -> str:
    """Return a bounded plain-text snapshot body."""

    body = str(text or "")
    body = re.sub(r"(?is)<script\b.*?</script>", " ", body)
    body = re.sub(r"(?is)<style\b.*?</style>", " ", body)
    body = "".join(ch if ch in "\n\t" or ord(ch) >= 32 else " " for ch in body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()[:max_chars]


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
