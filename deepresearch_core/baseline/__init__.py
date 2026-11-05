"""DeepResearch baseline subsystem.

Local-first, BYOK implementation aligned to ``docs/deep-research-report.md``:
provider adapters + model router, SQLite (+FTS5) store, indexing + dual-channel
retrieval (RRF), first-class citations, four research modes with budget layering,
models council, a three-layer memory engine, and a FastAPI ``/api/v1`` gateway.

Every optional dependency (litellm, chromadb, faiss, ollama, fastapi) is imported
lazily so the subsystem runs on the Python standard library alone.
"""

from .budget import Budget, Mode, RunRequest, preset_for
from .errors import BaselineError, ErrorCode, error_payload
from .memory import MemoryEngine, MemoryLayer
from .service import BaselineService

__all__ = [
    "BaselineError",
    "ErrorCode",
    "error_payload",
    "Budget",
    "Mode",
    "RunRequest",
    "preset_for",
    "MemoryEngine",
    "MemoryLayer",
    "BaselineService",
]


def create_app(service: "BaselineService", **kwargs):
    """Lazily build the FastAPI app (imports the web stack only when called)."""

    from .api import create_app as _create_app

    return _create_app(service, **kwargs)
