import os
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "prototype" / "runtime" / "reports"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "prototype" / "runtime" / "history"


def default_reports_dir() -> Path:
    """Return the shared local Markdown report directory."""

    return DEFAULT_REPORTS_DIR


def default_history_dir() -> Path:
    """Return the shared local answer-history directory."""

    return DEFAULT_HISTORY_DIR


def reports_dir_from_env(environ: Mapping[str, str] | None = None) -> Path:
    """Return the local report directory selected by environment or default."""

    env = environ if environ is not None else os.environ
    value = str(env.get("DEEPRESEARCH_REPORTS_DIR") or "").strip()
    if value:
        return Path(value).expanduser()
    return default_reports_dir()


def history_dir_from_env(environ: Mapping[str, str] | None = None) -> Path:
    """Return the local answer-history directory selected by environment or default."""

    env = environ if environ is not None else os.environ
    value = str(env.get("DEEPRESEARCH_HISTORY_DIR") or "").strip()
    if value:
        return Path(value).expanduser()
    return default_history_dir()
