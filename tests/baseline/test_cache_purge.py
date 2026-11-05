"""DR-ITER-081 cache purge contract tests."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service(tmp_path) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.store.cache_set("search:a", {"value": "search-a"})
    service.store.cache_set("search:b", {"value": "search-b"})
    service.store.cache_set("fetch:a", {"value": "fetch-a"})
    service.store.cache_set("other:a", {"value": "other-a"})
    return service


def test_cache_purge_all_scope(tmp_path):
    service = _service(tmp_path)

    result = service.purge_cache("all")

    assert result == {"purged_count": 4, "scope": "all"}
    assert service.store.cache_get("search:a") is None
    assert service.store.cache_get("fetch:a") is None
    assert service.store.cache_get("other:a") is None


def test_cache_purge_search_scope_preserves_fetch_and_other(tmp_path):
    service = _service(tmp_path)

    result = service.purge_cache("search")

    assert result == {"purged_count": 2, "scope": "search"}
    assert service.store.cache_get("search:a") is None
    assert service.store.cache_get("search:b") is None
    assert service.store.cache_get("fetch:a") == {"value": "fetch-a"}
    assert service.store.cache_get("other:a") == {"value": "other-a"}


def test_cache_purge_fetch_scope_preserves_search_and_other(tmp_path):
    service = _service(tmp_path)

    result = service.purge_cache("fetch")

    assert result == {"purged_count": 1, "scope": "fetch"}
    assert service.store.cache_get("fetch:a") is None
    assert service.store.cache_get("search:a") == {"value": "search-a"}
    assert service.store.cache_get("other:a") == {"value": "other-a"}


def test_cache_purge_rejects_invalid_scope(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(BaselineError) as exc:
        service.purge_cache("documents")

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_cache_purge_api_roundtrip_defaults_to_all(tmp_path):
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    response = client.delete("/api/v1/cache")

    assert response.status_code == 200
    assert response.json() == {"purged_count": 4, "scope": "all"}


def test_cache_purge_api_supports_scope_query(tmp_path):
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    response = client.delete("/api/v1/cache", params={"scope": "fetch"})

    assert response.status_code == 200
    assert response.json() == {"purged_count": 1, "scope": "fetch"}
    assert service.store.cache_get("search:a") == {"value": "search-a"}
