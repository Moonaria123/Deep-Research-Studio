"""Contract drift checks for the documented baseline API surface."""

from __future__ import annotations

import re
from pathlib import Path

from deepresearch_core.baseline import create_app
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.settings import ProviderSettings


ROOT = Path(__file__).resolve().parents[2]
API_SPEC = ROOT / "docs" / "API_SPEC.md"
API_PREFIX = "/api/v1"
IGNORED_METHODS = {"HEAD", "OPTIONS"}
IGNORED_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _documented_routes() -> set[tuple[str, str]]:
    """Read the route manifest table from docs/API_SPEC.md."""

    content = API_SPEC.read_text(encoding="utf-8")
    routes: set[tuple[str, str]] = set()
    for line in content.splitlines():
        match = re.match(r"^\|\s*`(GET|POST|PUT|PATCH|DELETE)`\s*\|\s*`([^`]+)`\s*\|", line)
        if match:
            routes.add((match.group(1), match.group(2)))
    return routes


def _runtime_routes() -> set[tuple[str, str]]:
    """Inspect the FastAPI route table instead of parsing decorators."""

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    app = create_app(service, auth_mode="off")
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if path in IGNORED_PATHS or not path.startswith(API_PREFIX):
            continue
        for method in getattr(route, "methods", set()) or set():
            if method not in IGNORED_METHODS:
                routes.add((method, path))
    return routes


def test_api_spec_manifest_matches_runtime_routes() -> None:
    """The API spec manifest should document every stable /api/v1 method/path."""

    documented = _documented_routes()
    runtime = _runtime_routes()

    missing = sorted(runtime - documented)
    stale = sorted(documented - runtime)

    assert not missing, f"docs/API_SPEC.md is missing runtime routes: {missing}"
    assert not stale, f"docs/API_SPEC.md documents stale routes: {stale}"
