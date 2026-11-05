"""Bounded, local-only parsing for user-imported PDF and OOXML documents."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from .errors import BaselineError, ErrorCode

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_CHARS = 250_000
MAX_PDF_PAGES = 500
MAX_OOXML_ENTRIES = 4096
MAX_OOXML_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_OOXML_COMPRESSION_RATIO = 100
MAX_XLSX_WORKSHEETS = 100
MAX_XLSX_VISITED_CELLS = 200_000
DEFAULT_PARSE_TIMEOUT_S = 20.0

_MIME_TYPES = {
    "pdf": {"application/pdf", "application/octet-stream", ""},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
        "",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "",
    },
}
_BIDI_CONTROLS = {
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}
_FILESYSTEM_UNSAFE_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


@dataclass(frozen=True)
class ParsedLocalDocument:
    """Secret-free result returned by the isolated parser."""

    content_type: str
    text: str
    warnings: tuple[str, ...]


class _WorkerError(Exception):
    """Structured parser failure safe to serialize across the subprocess boundary."""

    def __init__(self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def sanitize_import_filename(filename: str, *, fallback_extension: str = "") -> str:
    """Return a bounded display filename without path or bidi-control ambiguity."""

    normalized = unicodedata.normalize("NFKC", str(filename or ""))
    leaf = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    leaf = "".join(
        "_" if ch in _FILESYSTEM_UNSAFE_CHARS else ch
        for ch in leaf
        if ch not in _BIDI_CONTROLS and (ch == "\t" or ord(ch) >= 32)
    )
    leaf = " ".join(leaf.split()).strip(" .")
    if not leaf:
        leaf = f"document{fallback_extension}"
    if Path(leaf).stem.rstrip(" .").lower() in _WINDOWS_RESERVED_STEMS:
        leaf = f"_{leaf}"
    if len(leaf) > 180:
        suffix = Path(leaf).suffix[:16]
        leaf = f"{Path(leaf).stem[: max(1, 180 - len(suffix))]}{suffix}"
    return leaf


def sanitize_import_title(title: str, *, filename: str) -> str:
    """Return a bounded display title, defaulting to the sanitized filename stem."""

    normalized = unicodedata.normalize("NFKC", str(title or ""))
    normalized = "".join(
        ch
        for ch in normalized
        if ch not in _BIDI_CONTROLS and (ch == "\t" or ord(ch) >= 32)
    )
    normalized = " ".join(normalized.split()).strip()
    return (normalized or Path(filename).stem or filename)[:200]


def detect_local_document_type(content: bytes, *, filename: str, mime_type: str) -> str:
    """Validate extension, MIME, signature, and OOXML package identity."""

    extension = Path(str(filename or "")).suffix.lower().lstrip(".")
    if extension not in {"pdf", "docx", "xlsx"}:
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Only PDF, DOCX, and XLSX documents are supported.",
        )
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if normalized_mime not in _MIME_TYPES[extension]:
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Document MIME type does not match the file extension.",
        )
    if extension == "pdf":
        if not content.startswith(b"%PDF-"):
            raise BaselineError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "Document signature does not match PDF.",
            )
        return extension
    if not content.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Document signature does not match an OOXML package.",
        )
    package_type = _inspect_ooxml_package(content)
    if package_type != extension:
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "OOXML package structure does not match the file extension.",
        )
    return extension


def parse_local_document(
    content: bytes,
    *,
    filename: str,
    mime_type: str,
    timeout_s: float = DEFAULT_PARSE_TIMEOUT_S,
) -> ParsedLocalDocument:
    """Parse one bounded local file in a killable, credential-minimized subprocess."""

    if len(content) > MAX_UPLOAD_BYTES:
        raise BaselineError(
            ErrorCode.DOCUMENT_TOO_LARGE,
            "Document exceeds the 10 MiB import limit.",
            details={"max_bytes": MAX_UPLOAD_BYTES},
        )
    kind = detect_local_document_type(content, filename=filename, mime_type=mime_type)
    safe_filename = sanitize_import_filename(filename, fallback_extension=f".{kind}")
    with tempfile.TemporaryDirectory(prefix="deepresearch-import-") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        try:
            temp_dir.chmod(0o700)
        except OSError:
            pass
        source = temp_dir / safe_filename
        descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        command = [
            sys.executable,
            "-m",
            "deepresearch_core.baseline.local_document_import",
            "--worker",
            str(source),
            "--kind",
            kind,
        ]
        environment = _minimal_subprocess_environment()
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(0.1, float(timeout_s)),
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise BaselineError(
                ErrorCode.DOCUMENT_PARSE_FAILED,
                "Document parsing exceeded the 20 second limit.",
                details={"reason": "parser_timeout"},
            ) from exc
        if completed.returncode != 0 or len(completed.stdout) > 2 * 1024 * 1024:
            raise BaselineError(
                ErrorCode.DOCUMENT_PARSE_FAILED,
                "Document parsing failed.",
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BaselineError(
                ErrorCode.DOCUMENT_PARSE_FAILED,
                "Document parser returned an invalid result.",
            ) from exc
    if not isinstance(payload, dict):
        raise BaselineError(ErrorCode.DOCUMENT_PARSE_FAILED, "Document parser returned an invalid result.")
    if not payload.get("ok"):
        raw_code = str(payload.get("code") or ErrorCode.DOCUMENT_PARSE_FAILED.value)
        try:
            code = ErrorCode(raw_code)
        except ValueError:
            code = ErrorCode.DOCUMENT_PARSE_FAILED
        details = payload.get("details")
        raise BaselineError(
            code,
            str(payload.get("message") or "Document parsing failed."),
            details=dict(details) if isinstance(details, dict) else None,
        )
    text = str(payload.get("text") or "").strip()
    if not text:
        raise BaselineError(ErrorCode.DOCUMENT_NO_TEXT, "Document contains no extractable visible text.")
    warnings = payload.get("warnings")
    return ParsedLocalDocument(
        content_type=kind,
        text=text[:MAX_EXTRACTED_CHARS],
        warnings=tuple(str(item) for item in warnings) if isinstance(warnings, list) else (),
    )


def _inspect_ooxml_package(content: bytes) -> str:
    try:
        from io import BytesIO

        with zipfile.ZipFile(BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_OOXML_ENTRIES:
                raise BaselineError(
                    ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                    "OOXML package contains too many entries.",
                    details={"max_entries": MAX_OOXML_ENTRIES},
                )
            total_size = 0
            names: set[str] = set()
            for info in infos:
                normalized = info.filename.replace("\\", "/")
                normalized_key = normalized.lower()
                path = PurePosixPath(normalized)
                if (
                    not normalized
                    or normalized.startswith("/")
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or ":" in path.parts[0]
                ):
                    raise BaselineError(
                        ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                        "OOXML package contains an unsafe path.",
                    )
                if normalized_key in names:
                    raise BaselineError(
                        ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                        "OOXML package contains duplicate entries.",
                    )
                if info.flag_bits & 0x1:
                    raise BaselineError(
                        ErrorCode.DOCUMENT_ENCRYPTED,
                        "Encrypted OOXML documents are not supported.",
                    )
                total_size += int(info.file_size)
                if total_size > MAX_OOXML_UNCOMPRESSED_BYTES:
                    raise BaselineError(
                        ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                        "OOXML package exceeds the uncompressed size limit.",
                        details={"max_uncompressed_bytes": MAX_OOXML_UNCOMPRESSED_BYTES},
                    )
                if info.file_size and (
                    not info.compress_size
                    or info.file_size / max(1, info.compress_size) > MAX_OOXML_COMPRESSION_RATIO
                ):
                    raise BaselineError(
                        ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                        "OOXML package exceeds the compression ratio limit.",
                        details={"max_compression_ratio": MAX_OOXML_COMPRESSION_RATIO},
                    )
                names.add(normalized_key)
            if "[content_types].xml" not in names:
                raise BaselineError(
                    ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                    "OOXML content types are missing.",
                )
            content_types = archive.read("[Content_Types].xml").lower()
    except BaselineError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Document is not a valid OOXML package.",
        ) from exc

    macro_markers = (b"macroenabled", b"vbaproject")
    if any(marker in content_types for marker in macro_markers) or any(
        "vbaproject" in name.lower() for name in names
    ):
        raise BaselineError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Macro-enabled documents are not supported.",
        )
    if "word/document.xml" in names:
        dangerous = ("word/embeddings/", "word/activex/", "word/oleobject")
        if any(any(marker in name for marker in dangerous) for name in names):
            raise BaselineError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "DOCX embedded active objects are not supported.",
            )
        return "docx"
    if "xl/workbook.xml" in names:
        dangerous = ("xl/embeddings/", "xl/activex/", "xl/oleobjects/")
        if any(any(marker in name for marker in dangerous) for name in names):
            raise BaselineError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "XLSX embedded active objects are not supported.",
            )
        return "xlsx"
    raise BaselineError(
        ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
        "OOXML package is neither DOCX nor XLSX.",
    )


def _minimal_subprocess_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "VIRTUAL_ENV",
        "PYTHONHOME",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONWARNINGS"] = "ignore"
    environment["PYTHONPATH"] = os.pathsep.join(str(item) for item in sys.path if item)
    return environment


def _parse_pdf(path: Path) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            raise _WorkerError(ErrorCode.DOCUMENT_ENCRYPTED, "Encrypted PDF documents are not supported.")
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise _WorkerError(
                ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                "PDF exceeds the page limit.",
                details={"max_pages": MAX_PDF_PAGES},
            )
        pieces: list[str] = []
        total = 0
        truncated = False
        for page in reader.pages:
            page_text = str(page.extract_text() or "").strip()
            if not page_text:
                continue
            remaining = MAX_EXTRACTED_CHARS - total
            if remaining <= 0:
                truncated = True
                break
            pieces.append(page_text[:remaining])
            total += min(len(page_text), remaining)
            if len(page_text) > remaining:
                truncated = True
                break
    except _WorkerError:
        raise
    except Exception as exc:
        raise _WorkerError(ErrorCode.DOCUMENT_PARSE_FAILED, "PDF parsing failed.") from exc
    text = "\n\n".join(pieces).strip()
    if not text:
        raise _WorkerError(ErrorCode.DOCUMENT_NO_TEXT, "PDF contains no extractable text.")
    return text, ["text_truncated"] if truncated else []


def _parse_docx(path: Path) -> tuple[str, list[str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = {
                name.replace("\\", "/").lower(): name
                for name in archive.namelist()
            }
            part_names = [
                name
                for normalized, name in sorted(names.items())
                if normalized == "word/document.xml"
                or normalized.startswith("word/header")
                or normalized.startswith("word/footer")
            ]
            paragraphs: list[str] = []
            total = 0
            truncated = False
            for part_name in part_names:
                raw_xml = archive.read(part_name)
                if b"<!doctype" in raw_xml.lower() or b"<!entity" in raw_xml.lower():
                    raise _WorkerError(
                        ErrorCode.DOCUMENT_PARSE_FAILED,
                        "DOCX contains unsupported XML declarations.",
                    )
                root = ElementTree.fromstring(raw_xml)
                for paragraph in (
                    element
                    for element in root.iter()
                    if element.tag.rsplit("}", 1)[-1] == "p"
                ):
                    rendered = _docx_paragraph_text(paragraph)
                    if not rendered:
                        continue
                    remaining = MAX_EXTRACTED_CHARS - total
                    if remaining <= 0:
                        truncated = True
                        break
                    paragraphs.append(rendered[:remaining])
                    total += min(len(rendered), remaining)
                    if len(rendered) > remaining:
                        truncated = True
                        break
                if truncated:
                    break
            raw_text = "\n".join(paragraphs)
    except _WorkerError:
        raise
    except Exception as exc:
        raise _WorkerError(ErrorCode.DOCUMENT_PARSE_FAILED, "DOCX parsing failed.") from exc
    text = raw_text.strip()
    if not text:
        raise _WorkerError(ErrorCode.DOCUMENT_NO_TEXT, "DOCX contains no extractable text.")
    return text[:MAX_EXTRACTED_CHARS], ["text_truncated"] if truncated else []


def _docx_paragraph_text(paragraph: ElementTree.Element) -> str:
    rendered: list[str] = []
    for child in paragraph.iter():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag != "r":
            continue
        hidden = any(
            descendant.tag.rsplit("}", 1)[-1] == "vanish"
            and str(
                descendant.attrib.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val",
                    "1",
                )
            ).lower()
            not in {"0", "false", "off"}
            for descendant in child.iter()
        )
        if hidden:
            continue
        for descendant in child.iter():
            descendant_tag = descendant.tag.rsplit("}", 1)[-1]
            if descendant_tag == "t" and descendant.text:
                rendered.append(descendant.text)
            elif descendant_tag == "tab":
                rendered.append("\t")
            elif descendant_tag in {"br", "cr"}:
                rendered.append("\n")
    return " ".join("".join(rendered).split())


def _parse_xlsx(path: Path) -> tuple[str, list[str]]:
    from openpyxl import load_workbook

    visibility, missing_formula_cache = _xlsx_visibility(path)
    try:
        workbook = load_workbook(
            filename=str(path),
            read_only=True,
            data_only=True,
            keep_links=False,
            keep_vba=False,
        )
    except Exception as exc:
        raise _WorkerError(ErrorCode.DOCUMENT_PARSE_FAILED, "XLSX parsing failed.") from exc
    try:
        if len(workbook.worksheets) > MAX_XLSX_WORKSHEETS:
            raise _WorkerError(
                ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                "XLSX exceeds the worksheet limit.",
                details={"max_worksheets": MAX_XLSX_WORKSHEETS},
            )
        lines: list[str] = []
        text_chars = 0
        visited = 0
        truncated = False
        for worksheet in workbook.worksheets:
            if str(getattr(worksheet, "sheet_state", "visible")) != "visible":
                continue
            worksheet_path = str(getattr(worksheet, "_worksheet_path", ""))
            hidden_rows, hidden_columns = visibility.get(worksheet_path, (set(), ()))
            sheet_started = False
            for row_index, row in enumerate(worksheet.iter_rows(), start=1):
                values: list[str] = []
                for column_index, cell in enumerate(row, start=1):
                    visited += 1
                    if visited > MAX_XLSX_VISITED_CELLS:
                        raise _WorkerError(
                            ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                            "XLSX exceeds the visited cell limit.",
                            details={"max_cells": MAX_XLSX_VISITED_CELLS},
                        )
                    if row_index in hidden_rows or any(
                        start <= column_index <= end for start, end in hidden_columns
                    ):
                        continue
                    rendered = _render_cell_value(getattr(cell, "value", None))
                    if rendered:
                        values.append(rendered[:10_000])
                if values:
                    line = "\t".join(values)
                    if not sheet_started:
                        header = f"[Sheet: {worksheet.title}]"
                        header_size = len(header) + (1 if lines else 0)
                        if text_chars + header_size >= MAX_EXTRACTED_CHARS:
                            truncated = True
                            break
                        lines.append(header)
                        text_chars += header_size
                        sheet_started = True
                    remaining = MAX_EXTRACTED_CHARS - text_chars - 1
                    if remaining <= 0:
                        truncated = True
                        break
                    rendered_line = line[:remaining]
                    lines.append(rendered_line)
                    text_chars += len(rendered_line) + 1
                    if len(line) > remaining:
                        truncated = True
                        break
            if truncated:
                break
    except _WorkerError:
        raise
    except Exception as exc:
        raise _WorkerError(ErrorCode.DOCUMENT_PARSE_FAILED, "XLSX parsing failed.") from exc
    finally:
        workbook.close()
    text = "\n".join(lines).strip()
    if not text:
        raise _WorkerError(ErrorCode.DOCUMENT_NO_TEXT, "XLSX contains no visible cached values.")
    warnings: list[str] = []
    if missing_formula_cache:
        warnings.append("formula_cache_missing")
    if truncated:
        warnings.append("text_truncated")
    return text[:MAX_EXTRACTED_CHARS], warnings


def _xlsx_visibility(path: Path) -> tuple[dict[str, tuple[set[int], tuple[tuple[int, int], ...]]], bool]:
    visibility: dict[str, tuple[set[int], tuple[tuple[int, int], ...]]] = {}
    missing_formula_cache = False
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                normalized = name.replace("\\", "/")
                if not normalized.startswith("xl/worksheets/") or not normalized.endswith(".xml"):
                    continue
                raw_xml = archive.read(name)
                if b"<!doctype" in raw_xml.lower() or b"<!entity" in raw_xml.lower():
                    raise _WorkerError(
                        ErrorCode.DOCUMENT_PARSE_FAILED,
                        "XLSX contains unsupported XML declarations.",
                    )
                root = ElementTree.fromstring(raw_xml)
                hidden_rows: set[int] = set()
                hidden_columns: list[tuple[int, int]] = []
                for element in root.iter():
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag == "row" and str(element.attrib.get("hidden") or "").lower() in {"1", "true"}:
                        try:
                            hidden_rows.add(int(element.attrib.get("r") or 0))
                        except ValueError:
                            continue
                    elif tag == "col" and str(element.attrib.get("hidden") or "").lower() in {"1", "true"}:
                        try:
                            hidden_columns.append(
                                (
                                    int(element.attrib.get("min") or 0),
                                    int(element.attrib.get("max") or 0),
                                )
                            )
                        except ValueError:
                            continue
                    elif tag == "c":
                        formula = next(
                            (child for child in element if child.tag.rsplit("}", 1)[-1] == "f"),
                            None,
                        )
                        cached = next(
                            (child for child in element if child.tag.rsplit("}", 1)[-1] == "v"),
                            None,
                        )
                        if formula is not None and (cached is None or cached.text in (None, "")):
                            missing_formula_cache = True
                visibility[normalized] = (hidden_rows, tuple(hidden_columns))
    except Exception as exc:
        raise _WorkerError(ErrorCode.DOCUMENT_PARSE_FAILED, "XLSX structure parsing failed.") from exc
    return visibility, missing_formula_cache


def _render_cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return " ".join(str(value).split())


def _worker_parse(path: Path, kind: str) -> dict[str, Any]:
    try:
        if kind == "pdf":
            text, warnings = _parse_pdf(path)
        elif kind == "docx":
            text, warnings = _parse_docx(path)
        elif kind == "xlsx":
            text, warnings = _parse_xlsx(path)
        else:
            raise _WorkerError(ErrorCode.DOCUMENT_TYPE_UNSUPPORTED, "Unsupported document type.")
        return {"ok": True, "text": text, "warnings": warnings}
    except _WorkerError as exc:
        return {
            "ok": False,
            "code": exc.code.value,
            "message": exc.message,
            "details": exc.details,
        }
    except Exception:
        return {
            "ok": False,
            "code": ErrorCode.DOCUMENT_PARSE_FAILED.value,
            "message": "Document parsing failed.",
        }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--kind", required=True, choices=("pdf", "docx", "xlsx"))
    args = parser.parse_args(argv)
    payload = _worker_parse(Path(args.worker), args.kind)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the parent subprocess
    raise SystemExit(_main())
