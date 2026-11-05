"""T50 contracts for Exa/Tavily providers and frozen search usage policy."""

from __future__ import annotations

import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request

import pytest

from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline import create_app
from deepresearch_core.baseline.adapters import Document, ResilientSearchAdapter
from deepresearch_core.baseline.deadline import (
    ProviderCallBudget,
    current_provider_call_budget,
    use_provider_call_budget,
)
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.job_backend import JobEnqueueStateUnknown
from deepresearch_core.baseline.search_options import SearchOptions
from deepresearch_core.baseline.search_usage import (
    DEFAULT_SEARCH_USAGE_POLICY,
    normalize_search_usage_policy,
    resolve_search_usage,
)
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.config import (
    load_provider_config,
    load_search_usage_config,
    save_provider_config,
    save_search_usage_config,
)
from deepresearch_core.connectivity import check_named_search_provider_connectivity
from deepresearch_core.settings import ProviderSettings
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.search import SearchClient, SearchSettings
from deepresearch_core.standalone.llm import ProviderRequestError


def _read_json_request(request: Request) -> dict[str, object]:
    return json.loads((request.data or b"{}").decode("utf-8"))


def test_default_policy_and_auto_resolution_are_deterministic() -> None:
    policy = normalize_search_usage_policy(None)
    assert policy == DEFAULT_SEARCH_USAGE_POLICY

    resolved = resolve_search_usage(
        policy,
        source="global",
        configured={"brave": True, "tavily": True, "exa": True, "firecrawl": False},
        requested_locale={"country": "SG", "search_lang": "zh", "ui_language": "zh-Hans-SG"},
    )

    assert resolved.source == "global"
    assert resolved.search_chain == ("brave", "tavily", "exa")
    assert resolved.extract_chain == ("tavily", "exa")
    assert "firecrawl_not_configured" in resolved.reason_codes
    assert resolved.to_dict()["effective_policy"]["fallback_chain"] == [
        "brave",
        "tavily",
        "exa",
        "firecrawl",
    ]
    assert resolved.to_dict()["search_fallback_tail"] == ["cache", "indexed_external"]


def test_policy_normalization_clears_fields_irrelevant_to_selected_mode() -> None:
    normalized = normalize_search_usage_policy(
        {
            "usage_mode": "single",
            "primary_provider": "exa",
            "search_provider": "brave",
            "extract_provider": "tavily",
            "fallback_chain": ["tavily"],
        },
        require_complete=True,
    )

    assert normalized.to_dict() == {
        "usage_mode": "single",
        "primary_provider": "exa",
        "fallback_chain": ["tavily"],
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"usage_mode": "auto", "fallback_chain": ["brave", "brave"]},
            "duplicate",
        ),
        (
            {
                "usage_mode": "dual_role",
                "search_provider": "exa",
                "extract_provider": "brave",
                "fallback_chain": [],
            },
            "extract",
        ),
        (
            {"usage_mode": "single", "fallback_chain": []},
            "primary_provider",
        ),
    ],
)
def test_policy_rejects_invalid_or_ambiguous_contracts(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_search_usage_policy(payload, require_complete=True)


def test_tavily_search_contract_does_not_request_raw_content() -> None:
    seen: dict[str, object] = {}

    def transport(request: Request) -> dict[str, object]:
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["payload"] = _read_json_request(request)
        return {
            "results": [
                {
                    "title": "Tavily result",
                    "url": "https://example.com/tavily",
                    "content": "A concise search snippet.",
                    "published_date": "2026-07-01T00:00:00Z",
                }
            ]
        }

    client = SearchClient(SearchSettings(tavily_api_key="tvly-test"), transport=transport)
    hits = client.tavily_search(
        "query",
        count=3,
        options=SearchOptions(
            country="SG",
            search_lang="zh",
            ui_language="zh-Hans-SG",
            include_domains=("example.com",),
            exclude_domains=("ads.example",),
            freshness="pw",
        ),
    )

    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["headers"]["Authorization"] == "Bearer tvly-test"
    assert seen["payload"] == {
        "query": "query",
        "search_depth": "basic",
        "max_results": 3,
        "country": "singapore",
        "include_domains": ["example.com"],
        "exclude_domains": ["ads.example"],
        "time_range": "week",
    }
    assert hits[0]["url"] == "https://example.com/tavily"
    assert hits[0]["published_at"] == "2026-07-01T00:00:00Z"


def test_exa_search_and_contents_contracts_are_separate() -> None:
    requests: list[Request] = []

    def transport(request: Request) -> dict[str, object]:
        requests.append(request)
        if request.full_url.endswith("/search"):
            return {
                "results": [
                    {
                        "id": "exa-1",
                        "title": "Exa result",
                        "url": "https://example.com/exa",
                        "publishedDate": "2026-07-02T00:00:00Z",
                    }
                ]
            }
        return {
            "results": [
                {
                    "id": "exa-1",
                    "url": "https://example.com/exa",
                    "title": "Exa result",
                    "text": "Fetched Exa body.",
                    "highlights": ["Focused Exa excerpt."],
                }
            ],
            "statuses": [{"id": "exa-1", "status": "success"}],
        }

    client = SearchClient(SearchSettings(exa_api_key="exa-test"), transport=transport)
    hits = client.exa_search(
        "query",
        count=2,
        options=SearchOptions(
            country="SG",
            search_lang="zh",
            include_domains=("example.com",),
            exclude_domains=("ads.example",),
            freshness="pd",
        ),
    )
    document = client.exa_contents(
        "https://example.com/exa",
        max_characters=20_000,
        highlights_query="focused query",
    )

    search_payload = _read_json_request(requests[0])
    contents_payload = _read_json_request(requests[1])
    assert requests[0].full_url == "https://api.exa.ai/search"
    assert dict(requests[0].header_items())["X-api-key"] == "exa-test"
    freshness_boundary = search_payload.pop("startPublishedDate")
    assert isinstance(freshness_boundary, str) and freshness_boundary.endswith("Z")
    assert search_payload == {
        "query": "query",
        "numResults": 2,
        "type": "auto",
        "userLocation": "SG",
        "includeDomains": ["example.com"],
        "excludeDomains": ["ads.example"],
    }
    assert "contents" not in search_payload
    assert requests[1].full_url == "https://api.exa.ai/contents"
    assert contents_payload == {
        "urls": ["https://example.com/exa"],
        "text": {"maxCharacters": 20000},
        "highlights": {"query": "focused query"},
    }
    assert hits[0]["url"] == "https://example.com/exa"
    assert document["markdown"] == "Focused Exa excerpt."


@pytest.mark.parametrize(
    "provider,response",
    [
        (
            "tavily",
            {"results": [{"raw_content": "Unbound body without its source URL."}]},
        ),
        (
            "tavily",
            {
                "results": [
                    {"url": "https://other.example/page", "raw_content": "Wrong page."}
                ]
            },
        ),
        (
            "exa",
            {
                "results": [
                    {
                        "id": "exa-1",
                        "url": "https://target.example/page",
                        "text": "Body without an explicit success status.",
                    }
                ]
            },
        ),
        (
            "exa",
            {
                "results": [
                    {"id": "exa-2", "url": "https://other.example/page", "text": "Wrong page."}
                ],
                "statuses": [{"id": "exa-2", "status": "success"}],
            },
        ),
    ],
)
def test_extract_requires_explicit_target_binding_and_success(
    provider: str,
    response: dict[str, object],
) -> None:
    client = SearchClient(
        SearchSettings(tavily_api_key="tavily-test", exa_api_key="exa-test"),
        transport=lambda request: response,
    )

    with pytest.raises(ProviderRequestError):
        if provider == "tavily":
            client.tavily_extract("https://target.example/page")
        else:
            client.exa_contents("https://target.example/page")


def test_search_usage_config_preserves_other_blocks(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": {"brave_api_key": "secret"}, "llm_roles": {"default": "x"}}),
        encoding="utf-8",
    )
    policy = normalize_search_usage_policy(
        {
            "usage_mode": "dual_role",
            "search_provider": "brave",
            "extract_provider": "exa",
            "fallback_chain": ["exa", "tavily", "firecrawl"],
        },
        require_complete=True,
    )

    save_search_usage_config(path, policy)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert set(persisted["provider"]["brave_api_key"]) == {"secret_ref"}
    assert load_provider_config(path).brave_api_key == "secret"
    assert persisted["llm_roles"] == {"default": "x"}
    assert load_search_usage_config(path) == policy


@pytest.mark.parametrize(
    "contents",
    [
        "{not-json",
        json.dumps({"search_usage": {"usage_mode": "auto", "fallback_chain": ["unknown"]}}),
    ],
)
def test_corrupt_search_usage_config_fails_closed(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_search_usage_config(path)


def test_concurrent_provider_and_policy_saves_preserve_both_blocks(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    policy = normalize_search_usage_policy(
        {
            "usage_mode": "single",
            "primary_provider": "exa",
            "fallback_chain": ["tavily"],
        },
        require_complete=True,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                save_provider_config,
                path,
                ProviderSettings(brave_api_key="brave-concurrent"),
            ),
            executor.submit(save_search_usage_config, path, policy),
        ]
        for future in futures:
            future.result()

    assert load_provider_config(path).brave_api_key == "brave-concurrent"
    assert load_search_usage_config(path) == policy


class _RecordingJobBackend:
    name = "test"

    def __init__(self) -> None:
        self.requests: list[object] = []

    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        return "job-test"

    def status(self) -> dict[str, object]:
        return {"backend": self.name}


class _FailingJobBackend(_RecordingJobBackend):
    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        raise RuntimeError("queue unavailable")


class _ConcurrentPatchFailingJobBackend(_RecordingJobBackend):
    def __init__(self, patch) -> None:  # noqa: ANN001
        super().__init__()
        self._patch = patch

    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        self._patch()
        raise RuntimeError("queue unavailable after concurrent patch")


class _UnknownEnqueueJobBackend(_RecordingJobBackend):
    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        raise JobEnqueueStateUnknown("unknown queue state")


class _ClaimThenUnknownJobBackend(_RecordingJobBackend):
    def __init__(self, store: Store) -> None:
        super().__init__()
        self._store = store
        self.claimed = False

    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        # Reproduce a worker being scheduled before the API receives an ACK.
        self.claimed = self._store.claim_queued_run(run_id)
        raise JobEnqueueStateUnknown("unknown queue state after worker scheduling")


class _ExpireBeforeApiConfirmJobBackend(_RecordingJobBackend):
    def __init__(self, store: Store) -> None:
        super().__init__()
        self._store = store

    def enqueue(self, request, run_id: str) -> str:  # noqa: ANN001, ANN201
        self.requests.append((request, run_id))
        assert self._store.expire_enqueue_confirmation_wait(run_id) == "expired"
        return "job-worker-stopped"


class _PatchBeforeInheritedRunStore(Store):
    def __init__(self, path: Path, policy: dict[str, object]) -> None:
        super().__init__(path)
        self._policy = policy
        self._patched = False

    def update_session_and_create_run(self, session_id: str, **kwargs):  # noqa: ANN003, ANN201
        if not self._patched and kwargs.get("expected_session_metadata") is not None:
            self._patched = True
            self.update_session(
                session_id,
                metadata_patch={"search_usage_policy": self._policy},
            )
        return super().update_session_and_create_run(session_id, **kwargs)


def _service(tmp_path: Path) -> BaselineService:
    service = BaselineService(
        provider=ProviderSettings(
            llm_base_url="https://llm.example.test/v1",
            llm_api_key="llm-secret",
            llm_model="model",
            brave_api_key="brave-secret",
            exa_api_key="exa-secret",
            tavily_api_key="tavily-secret",
        ),
        db_path=tmp_path / "t50.sqlite3",
        config_path=tmp_path / "config.json",
    )
    service._job_backend = _RecordingJobBackend()  # noqa: SLF001 - isolated enqueue contract
    return service


def test_provider_status_is_capability_structured_and_secret_free(tmp_path: Path) -> None:
    service = _service(tmp_path)
    status = service.provider_status()

    assert {
        provider: {
            key: value[key]
            for key in ("configured", "search_ready", "extract_ready")
        }
        for provider, value in status["search_providers"].items()
    } == {
        "brave": {"configured": True, "search_ready": True, "extract_ready": False},
        "tavily": {"configured": True, "search_ready": True, "extract_ready": True},
        "exa": {"configured": True, "search_ready": True, "extract_ready": True},
        "firecrawl": {"configured": False, "search_ready": False, "extract_ready": False},
    }
    assert "secret" not in json.dumps(status)
    service.store.close()


def test_session_override_is_persisted_and_run_policy_is_frozen(tmp_path: Path) -> None:
    service = _service(tmp_path)
    override = {
        "usage_mode": "dual_role",
        "search_provider": "brave",
        "extract_provider": "exa",
        "fallback_chain": ["tavily", "exa"],
    }

    created = service.create_run(
        {"query": "alpha", "mode": "fast", "search_usage_policy": override},
        background=True,
    )
    session = next(item for item in service.list_sessions() if item["id"] == created["session_id"])
    request, _ = service._job_backend.requests[0]  # noqa: SLF001

    assert session["search_usage_policy"] == override
    assert request.resolved_search_usage["source"] == "session"
    assert request.resolved_search_usage["search_chain"] == ["brave", "tavily", "exa"]
    assert request.resolved_search_usage["extract_chain"] == ["exa", "tavily"]

    service.update_search_usage(
        {"usage_mode": "single", "primary_provider": "tavily", "fallback_chain": ["exa"]}
    )
    service.patch_session(created["session_id"], {"search_usage_policy": None})
    next_run = service.create_run(
        {"query": "beta", "mode": "fast", "session_id": created["session_id"]},
        background=True,
    )
    next_request, _ = service._job_backend.requests[1]  # noqa: SLF001

    assert next_run["session_id"] == created["session_id"]
    assert service.list_sessions()[0]["search_usage_policy"] is None
    assert request.resolved_search_usage["effective_policy"] == override
    assert next_request.resolved_search_usage["source"] == "global"
    assert next_request.resolved_search_usage["search_chain"] == ["tavily", "exa"]
    service.store.close()


def test_search_usage_api_rejects_invalid_provider(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(BaselineError) as exc:
        service.update_search_usage({"usage_mode": "auto", "fallback_chain": ["unknown"]})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    service.store.close()


def test_corrupt_session_policy_fails_closed_instead_of_using_global(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session_id = service.create_session(title="Corrupt policy")["session_id"]
    service.store._conn.execute(  # noqa: SLF001 - inject persisted corruption
        "UPDATE sessions SET metadata=? WHERE id=?",
        (json.dumps({"search_usage_policy": {"usage_mode": "single"}}), session_id),
    )
    service.store._conn.commit()  # noqa: SLF001

    with pytest.raises(BaselineError) as exc:
        service.create_run(
            {"query": "must not silently reroute", "mode": "fast", "session_id": session_id},
            background=True,
        )

    assert exc.value.code is ErrorCode.INTERNAL_ERROR
    assert service.store.get_latest_run_for_session(session_id) is None
    service.store.close()


def test_search_usage_save_failure_preserves_effective_policy(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    previous = service.get_search_usage()

    def _fail_save(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
        raise OSError("disk full")

    monkeypatch.setattr("deepresearch_core.baseline.service.save_search_usage_config", _fail_save)

    with pytest.raises(OSError, match="disk full"):
        service.update_search_usage(
            {
                "usage_mode": "single",
                "primary_provider": "exa",
                "fallback_chain": ["tavily"],
            }
        )

    assert service.get_search_usage() == previous
    service.store.close()


def test_search_usage_http_contract_and_session_clear(tmp_path: Path) -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))
    policy = {
        "usage_mode": "dual_role",
        "search_provider": "brave",
        "extract_provider": "tavily",
        "fallback_chain": ["exa", "firecrawl"],
    }

    saved = client.patch("/api/v1/settings/search-usage", json=policy)
    assert saved.status_code == 200
    assert client.get("/api/v1/settings/search-usage").json() == policy

    session_id = service.create_session(title="T50 HTTP")["session_id"]
    override = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"search_usage_policy": policy},
    )
    assert override.status_code == 200
    assert override.json()["session"]["search_usage_policy"] == policy
    cleared = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"search_usage_policy": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["session"]["search_usage_policy"] is None

    invalid = client.patch(
        "/api/v1/settings/search-usage",
        json={"usage_mode": "auto", "fallback_chain": ["unknown"]},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    service.store.close()


def test_tools_test_can_select_one_provider_without_aggregate_probe(tmp_path: Path) -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=tmp_path / "tools.sqlite3")
    client = TestClient(create_app(service))

    response = client.post("/api/v1/tools/test", json={"provider": "tavily"})

    assert response.status_code == 200
    payload = response.json()
    assert {key: payload[key] for key in (
        "enabled", "ok", "status", "provider", "capabilities", "latency_ms", "endpoint_host", "error_message"
    )} == {
        "enabled": True,
        "ok": False,
        "status": "missing_config",
        "provider": "tavily",
        "capabilities": {"search": True, "extract": True},
        "latency_ms": None,
        "endpoint_host": "api.tavily.com",
        "error_message": "",
    }
    service.store.close()


def test_enqueue_failure_rolls_back_new_session_and_queued_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._job_backend = _FailingJobBackend()  # noqa: SLF001 - fault injection

    with pytest.raises(RuntimeError, match="queue unavailable"):
        service.create_run(
            {
                "query": "atomic home",
                "mode": "fast",
                "search_usage_policy": {
                    "usage_mode": "auto",
                    "fallback_chain": ["brave", "tavily", "exa", "firecrawl"],
                },
            }
        )

    assert service.list_sessions() == []
    assert service.store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0  # noqa: SLF001
    service.store.close()


def test_enqueue_failure_restores_existing_session_override(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session_id = service.create_session(title="Existing")["session_id"]
    previous = {
        "usage_mode": "single",
        "primary_provider": "brave",
        "fallback_chain": ["tavily"],
    }
    service.patch_session(session_id, {"search_usage_policy": previous})
    service._job_backend = _FailingJobBackend()  # noqa: SLF001 - fault injection

    with pytest.raises(RuntimeError, match="queue unavailable"):
        service.create_run(
            {
                "query": "atomic follow-up",
                "mode": "fast",
                "session_id": session_id,
                "search_usage_policy": {
                    "usage_mode": "single",
                    "primary_provider": "exa",
                    "fallback_chain": ["tavily"],
                },
            }
        )

    session = next(item for item in service.list_sessions() if item["id"] == session_id)
    assert session["search_usage_policy"] == previous
    assert service.store.get_latest_run_for_session(session_id) is None
    service.store.close()


def test_enqueue_failure_restores_existing_session_timestamp_without_override(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    session_id = service.create_session(title="Existing timestamp")["session_id"]
    before = service.store.get_session(session_id)
    service._job_backend = _FailingJobBackend()  # noqa: SLF001 - fault injection

    with pytest.raises(RuntimeError, match="queue unavailable"):
        service.create_run(
            {"query": "atomic inherited follow-up", "mode": "fast", "session_id": session_id}
        )

    after = service.store.get_session(session_id)
    assert after is not None and before is not None
    assert after["updated_at"] == before["updated_at"]
    assert service.store.get_latest_run_for_session(session_id) is None
    service.store.close()


def test_inherited_policy_is_refrozen_after_concurrent_session_patch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.store.close()
    tavily = {
        "usage_mode": "single",
        "primary_provider": "tavily",
        "fallback_chain": ["exa"],
    }
    store = _PatchBeforeInheritedRunStore(tmp_path / "atomic-inherit.sqlite3", tavily)
    service._store = store  # noqa: SLF001 - inject deterministic CAS race
    session_id = store.create_session(
        title="Concurrent inherit",
        metadata={
            "search_usage_policy": {
                "usage_mode": "single",
                "primary_provider": "brave",
                "fallback_chain": ["exa"],
            }
        },
    )

    service.create_run(
        {"query": "freeze current session", "mode": "fast", "session_id": session_id},
        background=True,
    )

    request, _ = service._job_backend.requests[0]  # noqa: SLF001
    assert request.resolved_search_usage["effective_policy"] == tavily
    assert request.resolved_search_usage["search_chain"][0] == "tavily"
    session = next(item for item in service.list_sessions() if item["id"] == session_id)
    assert session["search_usage_policy"] == tavily
    store.close()


def test_enqueue_compensation_does_not_overwrite_concurrent_policy_patch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session_id = service.create_session(title="Concurrent compensation")["session_id"]
    brave = {
        "usage_mode": "single",
        "primary_provider": "brave",
        "fallback_chain": ["exa"],
    }
    tavily = {
        "usage_mode": "single",
        "primary_provider": "tavily",
        "fallback_chain": ["exa"],
    }
    service.patch_session(session_id, {"search_usage_policy": brave})
    service._job_backend = _ConcurrentPatchFailingJobBackend(  # noqa: SLF001
        lambda: service.patch_session(session_id, {"search_usage_policy": tavily})
    )

    with pytest.raises(RuntimeError, match="concurrent patch"):
        service.create_run(
            {
                "query": "do not overwrite concurrent policy",
                "mode": "fast",
                "session_id": session_id,
                "search_usage_policy": {
                    "usage_mode": "single",
                    "primary_provider": "exa",
                    "fallback_chain": ["brave"],
                },
            }
        )

    session = next(item for item in service.list_sessions() if item["id"] == session_id)
    assert session["search_usage_policy"] == tavily
    assert service.store.get_latest_run_for_session(session_id) is None
    service.store.close()


def test_unknown_enqueue_state_is_terminal_and_cannot_run_late(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._job_backend = _UnknownEnqueueJobBackend()  # noqa: SLF001

    with pytest.raises(JobEnqueueStateUnknown):
        service.create_run({"query": "ambiguous enqueue", "mode": "fast"})

    sessions = service.list_sessions()
    assert len(sessions) == 1
    run_id = str(service.store.get_latest_run_id(sessions[0]["id"]) or "")
    run = service.store.get_run(run_id)
    assert run["status"] == "error"
    assert service.store.claim_queued_run(run_id) is False
    assert [event["type"] for event in service.store.list_events(run_id)] == ["error"]
    service.store.close()


def test_worker_cannot_claim_before_enqueue_confirmation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    backend = _ClaimThenUnknownJobBackend(service.store)
    service._job_backend = backend  # noqa: SLF001 - deterministic lost-ACK race

    with pytest.raises(JobEnqueueStateUnknown):
        service.create_run({"query": "ambiguous scheduling race", "mode": "fast"})

    assert backend.claimed is False
    session = service.list_sessions()[0]
    run_id = str(service.store.get_latest_run_id(session["id"]) or "")
    run = service.store.get_run(run_id)
    assert run is not None
    assert run["status"] == "error"
    assert run["metadata"]["enqueue_confirmed"] is False
    service.store.close()


def test_confirmed_enqueue_opens_claim_gate(tmp_path: Path) -> None:
    store = Store(tmp_path / "enqueue-gate.sqlite3")
    session_id, run_id = store.create_session_and_run(
        title="Gate",
        mode="fast",
        query="confirm before claim",
        run_metadata={"enqueue_confirmed": False},
    )

    assert store.enqueue_confirmation_state(run_id) == "pending"
    assert store.claim_queued_run(run_id) is False
    assert store.confirm_run_enqueued(run_id) is True
    assert store.enqueue_confirmation_state(run_id) == "confirmed"
    assert store.abort_unconfirmed_enqueue(run_id, session_id) is False
    assert store.get_run(run_id)["status"] == "queued"
    assert store.claim_queued_run(run_id) is True
    assert store.get_run(run_id)["session_id"] == session_id
    store.close()


def test_worker_wait_expiry_preserves_api_session_override_compensation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session_id = service.create_session(title="Worker expiry compensation")["session_id"]
    previous = {
        "usage_mode": "single",
        "primary_provider": "brave",
        "fallback_chain": ["tavily"],
    }
    service.patch_session(session_id, {"search_usage_policy": previous})
    service._job_backend = _ExpireBeforeApiConfirmJobBackend(service.store)  # noqa: SLF001

    with pytest.raises(JobEnqueueStateUnknown, match="could not be committed"):
        service.create_run(
            {
                "query": "worker stopped before API confirmation",
                "mode": "fast",
                "session_id": session_id,
                "search_usage_policy": {
                    "usage_mode": "single",
                    "primary_provider": "exa",
                    "fallback_chain": ["tavily"],
                },
            }
        )

    session = next(item for item in service.list_sessions() if item["id"] == session_id)
    run_id = str(service.store.get_latest_run_id(session_id) or "")
    assert session["search_usage_policy"] == previous
    assert service.store.get_run(run_id)["status"] == "error"
    service.store.close()


def test_worker_refreshes_credentials_before_each_frozen_run(monkeypatch) -> None:
    from deepresearch_core.baseline import worker
    from deepresearch_core.baseline.budget import RunRequest

    calls: list[tuple[str, object]] = []

    class _WorkerService:
        def reconfigure(self, provider) -> None:  # noqa: ANN001
            calls.append(("reconfigure", provider))

        def execute_run(self, request, run_id: str) -> None:  # noqa: ANN001
            calls.append(("execute", (request.resolved_search_usage, run_id)))

    exa_settings = ProviderSettings(exa_api_key="fresh-exa")
    tavily_settings = ProviderSettings(tavily_api_key="fresh-tavily")
    providers = iter(
        (
            exa_settings,
            tavily_settings,
            tavily_settings,
        )
    )
    first_request = RunRequest.from_payload(
        {
            "query": "worker refresh one",
            "mode": "fast",
            "deadline_at": time.time() + 60,
            "resolved_search_usage": {
                "source": "session",
                "effective_policy": {
                    "usage_mode": "single",
                    "primary_provider": "exa",
                    "fallback_chain": [],
                },
                "search_chain": ["exa"],
                "extract_chain": ["exa"],
                "reason_codes": [],
            },
        }
    )
    second_request = RunRequest.from_payload(
        {
            "query": "worker refresh two",
            "mode": "fast",
            "deadline_at": time.time() + 60,
            "resolved_search_usage": {
                "source": "global",
                "effective_policy": {
                    "usage_mode": "single",
                    "primary_provider": "tavily",
                    "fallback_chain": [],
                },
                "search_chain": ["tavily"],
                "extract_chain": ["tavily"],
                "reason_codes": [],
            },
        }
    )
    monkeypatch.setattr(worker, "_service", _WorkerService())
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_load_worker_provider_settings", lambda: next(providers))

    worker.run_job(first_request.to_payload(), "run-t50-one")
    worker.run_job(second_request.to_payload(), "run-t50-two")
    worker.run_job(second_request.to_payload(), "run-t50-three")

    assert calls == [
        ("reconfigure", exa_settings),
        ("execute", (first_request.resolved_search_usage, "run-t50-one")),
        ("reconfigure", tavily_settings),
        ("execute", (second_request.resolved_search_usage, "run-t50-two")),
        ("execute", (second_request.resolved_search_usage, "run-t50-three")),
    ]


def test_provider_status_and_rq_payload_do_not_expose_raw_keys(tmp_path: Path, caplog) -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))
    created = service.create_run({"query": "secret scan", "mode": "fast"})
    request, _ = service._job_backend.requests[0]  # noqa: SLF001
    run_id = str(created["run_id"])
    service.store.append_event(run_id, "workflow_end", {"status": "queued"})

    serialized = json.dumps(
        {
            "status": service.provider_status(),
            "payload": request.to_payload(),
            "sessions": service.list_sessions(),
            "run": client.get(f"/api/v1/runs/{run_id}").json(),
            "events": service.store.list_events(run_id),
            "sse": client.get(f"/api/v1/runs/{run_id}/stream").text,
            "sqlite": "\n".join(service.store._conn.iterdump()),  # noqa: SLF001
            "logs": caplog.text,
        }
    )
    for secret in ("llm-secret", "brave-secret", "exa-secret", "tavily-secret"):
        assert secret not in serialized
    service.store.close()


class _RoutingSearchClient:
    settings = SearchSettings(
        brave_api_key="brave",
        tavily_api_key="tavily",
        exa_api_key="exa",
        firecrawl_api_key="firecrawl",
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("brave")
        return []

    def tavily_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("tavily")
        return [{"title": "Tavily", "url": "https://example.com/t", "description": "hit"}]

    def exa_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("exa")
        return [{"title": "Exa", "url": "https://example.com/e", "description": "hit"}]

    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("firecrawl")
        return []

    def tavily_extract(self, url: str):  # noqa: ANN201
        self.calls.append("tavily_extract")
        raise RuntimeError("tavily down")

    def exa_contents(self, url: str):  # noqa: ANN201
        self.calls.append("exa_contents")
        return {"markdown": "Exa body", "raw": {"url": url}, "canonical_url": url}

    def scrape(self, url: str):  # noqa: ANN201
        self.calls.append("firecrawl_extract")
        return {"markdown": "Firecrawl body", "raw": {"url": url}}


class _ScopedFallbackSearchClient(_RoutingSearchClient):
    def tavily_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("tavily")
        return [
            {
                "title": "Generic HTTP troubleshooting",
                "url": "https://example.com/http-errors",
                "description": "A general overview of HTTP error troubleshooting.",
            }
        ]

    def exa_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("exa")
        return [
            {
                "title": "RFC 9110: HTTP Semantics",
                "url": "https://www.ietf.org/archive/id/draft-ietf-httpbis-semantics-19.html",
                "description": "The 404 status code indicates that the origin server did not find a current representation.",
            }
        ]


def test_nonempty_provider_results_fall_through_only_when_all_candidates_are_rejected() -> None:
    store = Store(":memory:")
    client = _ScopedFallbackSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {
        "search_chain": ["tavily", "exa"],
        "extract_chain": ["tavily", "exa"],
        "reason_codes": [],
    }

    result = adapter.search_network(
        "HTTP status 404",
        relevance_query=(
            "According to IETF documentation, what does HTTP status 404 mean? "
            "Answer in two cited sentences."
        ),
        resolved_usage=usage,
    )

    assert client.calls == ["tavily", "exa"]
    assert result.status["irrelevant_providers"] == ["tavily"]
    assert result.status["selected_provider"] == "exa"
    assert [hit.url for hit in result.hits] == [
        "https://www.ietf.org/archive/id/draft-ietf-httpbis-semantics-19.html"
    ]
    store.close()


def test_relevance_aware_cache_read_does_not_replay_rejected_provider_hits() -> None:
    store = Store(":memory:")
    client = _ScopedFallbackSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {
        "search_chain": ["tavily", "exa"],
        "extract_chain": ["tavily", "exa"],
        "reason_codes": [],
    }
    stale = adapter.search_network("HTTP status 404", resolved_usage={**usage, "search_chain": ["tavily"]})
    assert adapter.persist_search_cache(
        "HTTP status 404",
        count=5,
        options=None,
        result=stale,
        resolved_usage=usage,
    )

    cached = adapter.read_search_cache(
        "HTTP status 404",
        relevance_query=(
            "According to IETF documentation, what does HTTP status 404 mean? "
            "Answer in two cited sentences."
        ),
        resolved_usage=usage,
    )

    assert cached is None
    store.close()


def test_resolved_chain_drives_search_fetch_and_cache_versions() -> None:
    store = Store(":memory:")
    client = _RoutingSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {
        "source": "session",
        "effective_policy": {
            "usage_mode": "dual_role",
            "search_provider": "brave",
            "extract_provider": "tavily",
            "fallback_chain": ["tavily", "exa", "firecrawl"],
        },
        "search_chain": ["brave", "tavily", "exa", "firecrawl"],
        "extract_chain": ["tavily", "exa", "firecrawl"],
        "reason_codes": [],
    }

    result = adapter.search_with_outcome("alpha", resolved_usage=usage)
    document = adapter.fetch("https://example.com/t", resolved_usage=usage)

    assert client.calls[:2] == ["brave", "tavily"]
    assert result.status["selected_provider"] == "tavily"
    assert result.hits[0].metadata["search_provider"] == "tavily"
    assert client.calls[-2:] == ["tavily_extract", "exa_contents"]
    assert document.metadata["fetch_provider"] == "exa"
    assert store.cache_get(adapter.search_cache_key("alpha", resolved_usage=usage))["version"] == 4
    assert store.cache_get(adapter.fetch_cache_key("https://example.com/t", resolved_usage=usage))["version"] == 3

    different_usage = {**usage, "search_chain": ["exa", "tavily"]}
    assert adapter.search_cache_key("alpha", resolved_usage=usage) != adapter.search_cache_key(
        "alpha", resolved_usage=different_usage
    )
    store.close()


def test_search_chain_shares_budget_and_falls_through_after_child_timeout() -> None:
    class _Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    class _BudgetedSearchClient(_RoutingSearchClient):
        def __init__(self, clock: _Clock) -> None:
            super().__init__()
            self.clock = clock
            self.timeouts: list[float] = []

        def tavily_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
            self.calls.append("tavily")
            budget = current_provider_call_budget()
            assert budget is not None
            timeout_s = budget.call_timeout_s(60.0)
            self.timeouts.append(timeout_s)
            self.clock.now += timeout_s
            budget.check("tavily_search")
            raise AssertionError("child timeout check must raise")

        def exa_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
            self.calls.append("exa")
            budget = current_provider_call_budget()
            assert budget is not None
            self.timeouts.append(budget.call_timeout_s(60.0))
            return [{"title": "Exa", "url": "https://example.com/e", "description": "hit"}]

    clock = _Clock()
    client = _BudgetedSearchClient(clock)
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    parent_budget = ProviderCallBudget(
        request_deadline=20.0,
        phase_deadline=15.0,
        clock=clock,
    )

    with use_provider_call_budget(parent_budget):
        result = adapter.search_network(
            "SQLite FTS5",
            resolved_usage={"search_chain": ["tavily", "exa"], "reason_codes": []},
        )

    assert client.calls[-2:] == ["tavily", "exa"]
    assert client.timeouts == pytest.approx([7.5, 7.5])
    assert result.status["selected_provider"] == "exa"
    assert result.status["provider_failure_reasons"] == {"tavily": "timeout"}
    store.close()


def test_extract_chain_reserves_budget_for_the_next_provider_after_child_timeout() -> None:
    class _Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    class _BudgetedExtractClient(_RoutingSearchClient):
        def __init__(self, clock: _Clock) -> None:
            super().__init__()
            self.clock = clock
            self.timeouts: list[float] = []

        def tavily_extract(self, url: str):  # noqa: ANN201
            self.calls.append("tavily_extract")
            budget = current_provider_call_budget()
            assert budget is not None
            timeout_s = budget.call_timeout_s(60.0)
            self.timeouts.append(timeout_s)
            self.clock.now += timeout_s
            budget.check("tavily_extract")
            raise AssertionError("child timeout check must raise")

        def exa_contents(self, url: str, **kwargs):  # noqa: ANN003, ANN201
            self.calls.append("exa_contents")
            budget = current_provider_call_budget()
            assert budget is not None
            self.timeouts.append(budget.call_timeout_s(60.0))
            return {"markdown": "Exa fallback body", "raw": {"url": url}, "canonical_url": url}

    clock = _Clock()
    client = _BudgetedExtractClient(clock)
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    parent_budget = ProviderCallBudget(
        request_deadline=20.0,
        phase_deadline=15.0,
        clock=clock,
    )

    with use_provider_call_budget(parent_budget):
        document = adapter.fetch_network(
            "https://example.com/guidance",
            resolved_usage={"extract_chain": ["tavily", "exa"], "reason_codes": []},
        )

    assert client.calls[-2:] == ["tavily_extract", "exa_contents"]
    assert client.timeouts == pytest.approx([10.0, 5.0])
    assert document.metadata["fetch_provider"] == "exa"
    assert document.metadata["provider_failure_reasons"] == {"tavily": "timeout"}
    store.close()


def test_exa_numeric_standard_focus_keeps_highlights_and_uses_bounded_long_text_fallback() -> None:
    class _FocusedExaClient(_RoutingSearchClient):
        def __init__(self) -> None:
            super().__init__()
            self.contents_options: list[dict[str, object]] = []

        def exa_contents(
            self,
            url: str,
            *,
            max_characters: int = 100_000,
            highlights_query: str | None = None,
        ):  # noqa: ANN201
            self.calls.append("exa_contents")
            self.contents_options.append(
                {
                    "max_characters": max_characters,
                    "highlights_query": highlights_query,
                }
            )
            return {
                "markdown": "The 404 status code indicates that the target representation was not found.",
                "raw": {"url": url},
                "canonical_url": url,
            }

    client = _FocusedExaClient()
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    focus = "According to IETF documentation, what does HTTP status 404 mean?"

    document = adapter.fetch_network(
        "https://www.ietf.org/rfc/rfc9110.html",
        relevance_query=focus,
        resolved_usage={"extract_chain": ["exa"], "reason_codes": []},
    )
    adapter.fetch_network(
        "https://www.who.int/health-topics/physical-activity",
        relevance_query="According to WHO, what is physical activity?",
        resolved_usage={"extract_chain": ["exa"], "reason_codes": []},
    )

    assert document.metadata["fetch_provider"] == "exa"
    assert client.contents_options == [
        {
            "max_characters": 500_000,
            "highlights_query": focus,
        },
        {
            "max_characters": 100_000,
            "highlights_query": "According to WHO, what is physical activity?",
        },
    ]
    store.close()


def test_exa_numeric_standard_focus_recovers_target_window_when_highlights_are_off_topic() -> None:
    class _OffTopicHighlightExaClient(_RoutingSearchClient):
        def exa_contents(
            self,
            url: str,
            *,
            max_characters: int = 100_000,
            highlights_query: str | None = None,
        ):  # noqa: ANN201
            self.calls.append("exa_contents")
            return {
                "markdown": "HTTP defines status code 200 and partial content responses.",
                "raw": {
                    "url": url,
                    "text": (
                        "RFC 9110 HTTP Semantics.\n\n"
                        + ("Protocol background. " * 500)
                        + "\n\n15.5.5. 404 Not Found\n\n"
                        "The 404 status code indicates that the origin server did not find "
                        "a current representation for the target resource.\n\n"
                        + ("Appendix material. " * 500)
                    ),
                },
                "canonical_url": url,
            }

    client = _OffTopicHighlightExaClient()
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    document = adapter.fetch_network(
        "https://www.ietf.org/rfc/rfc9110.html",
        relevance_query="According to IETF documentation, what does HTTP status 404 mean?",
        resolved_usage={"extract_chain": ["exa"], "reason_codes": []},
    )

    assert document.metadata["fetch_provider"] == "exa"
    assert "origin server did not find a current representation" in document.text
    assert len(document.text) < 13_000
    store.close()


def test_search_option_application_is_preserved_by_cache() -> None:
    store = Store(":memory:")
    client = _RoutingSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {
        "search_chain": ["tavily"],
        "extract_chain": ["exa"],
        "reason_codes": [],
    }
    options = SearchOptions(
        include_domains=("example.com",),
        exclude_domains=("ads.example",),
        freshness="pm",
    )

    live = adapter.search_with_outcome(
        "alpha",
        options=options,
        resolved_usage=usage,
    )
    cached = adapter.read_search_cache(
        "alpha",
        options=options,
        resolved_usage=usage,
    )

    assert live.status["search_options"] == {
        "requested": {
            "include_domains": ["example.com"],
            "exclude_domains": ["ads.example"],
            "freshness": "pm",
        },
        "applied_fields": ["include_domains", "exclude_domains", "freshness"],
        "ignored_fields": [],
        "reason_codes": [],
    }
    assert cached is not None
    assert cached.status["search_options"] == live.status["search_options"]
    assert all(hit.metadata["content_basis"] == "snippet" for hit in cached.hits)
    store.close()


def test_fetch_cache_key_uses_canonical_url_and_extract_chain() -> None:
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(_RoutingSearchClient(), store=store, retriever=Retriever(store))
    usage = {"extract_chain": ["exa", "tavily"]}

    first = adapter.fetch_cache_key(
        "https://Example.com:443/a?utm_source=campaign&b=2#fragment",
        resolved_usage=usage,
    )
    second = adapter.fetch_cache_key("https://example.com/a?b=2", resolved_usage=usage)
    other_chain = adapter.fetch_cache_key(
        "https://example.com/a?b=2",
        resolved_usage={"extract_chain": ["tavily", "exa"]},
    )

    assert first == second
    assert first != other_chain
    assert first != adapter.fetch_cache_key(
        "https://example.com/a?b=2",
        relevance_query="different research focus",
        resolved_usage=usage,
    )

    adapter.persist_fetch_cache(
        Document(
            url="https://example.com/a?b=2",
            title="Fetched",
            text="Fetched body",
            content_type="web",
            metadata={"fetch_provider": "exa", "content_basis": "fetched"},
        ),
        resolved_usage=usage,
    )
    cached_value = store.cache_get(first)
    assert cached_value["producer_provider"] == "exa"
    assert cached_value["content_basis"] == "fetched"
    cached_document = adapter.read_fetch_cache(
        "https://example.com/a?b=2",
        resolved_usage=usage,
    )
    assert cached_document is not None
    assert cached_document.metadata["fetch_provider"] == "exa"
    assert cached_document.metadata["content_basis"] == "fetched"
    store.close()


class _CircuitSearchClient(_RoutingSearchClient):
    def __init__(self, *, search_status: int = 401, extract_status: int = 402) -> None:
        super().__init__()
        self.search_status = search_status
        self.extract_status = extract_status

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
        self.calls.append("brave")
        raise ProviderRequestError("brave failed", status_code=self.search_status)

    def tavily_extract(self, url: str):  # noqa: ANN201
        self.calls.append("tavily_extract")
        raise ProviderRequestError("tavily failed", status_code=self.extract_status)


def test_auth_and_payment_failures_open_only_the_current_run_circuit() -> None:
    store = Store(":memory:")
    client = _CircuitSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {
        "search_chain": ["brave", "tavily"],
        "extract_chain": ["tavily", "exa"],
        "reason_codes": [],
    }
    circuit: set[str] = set()

    first_search = adapter.search_network("alpha", resolved_usage=usage, provider_circuit=circuit)
    second_search = adapter.search_network("beta", resolved_usage=usage, provider_circuit=circuit)
    first_fetch = adapter.fetch_network(
        "https://example.com/t",
        resolved_usage=usage,
        provider_circuit=circuit,
    )
    second_fetch = adapter.fetch_network(
        "https://example.com/e",
        resolved_usage=usage,
        provider_circuit=circuit,
    )

    assert circuit == {"brave", "tavily"}
    assert first_search.status["provider_failure_reasons"] == {"brave": "auth_failed"}
    assert second_search.status["skipped_providers"] == ["brave"]
    assert client.calls.count("brave") == 1
    assert first_fetch.metadata["fetch_provider"] == "exa"
    assert first_fetch.metadata["provider_failure_reasons"] == {"tavily": "payment_required"}
    assert second_fetch.metadata["skipped_extract_providers"] == ["tavily"]
    assert client.calls.count("tavily_extract") == 1
    store.close()


def test_rate_limit_does_not_open_the_run_circuit() -> None:
    store = Store(":memory:")
    client = _CircuitSearchClient(search_status=429)
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {"search_chain": ["brave", "tavily"], "extract_chain": ["exa"], "reason_codes": []}
    circuit: set[str] = set()

    first = adapter.search_network("alpha", resolved_usage=usage, provider_circuit=circuit)
    second = adapter.search_network("beta", resolved_usage=usage, provider_circuit=circuit)

    assert circuit == set()
    assert first.status["provider_failure_reasons"] == {"brave": "rate_limited"}
    assert second.status["providers_attempted"][:2] == ["brave", "tavily"]
    assert client.calls.count("brave") == 2
    store.close()


def test_forbidden_does_not_open_auth_circuit() -> None:
    store = Store(":memory:")
    client = _CircuitSearchClient(search_status=403)
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    usage = {"search_chain": ["brave", "tavily"], "extract_chain": ["exa"], "reason_codes": []}
    circuit: set[str] = set()

    first = adapter.search_network("alpha", resolved_usage=usage, provider_circuit=circuit)
    second = adapter.search_network("beta", resolved_usage=usage, provider_circuit=circuit)

    assert circuit == set()
    assert first.status["provider_failure_reasons"] == {"brave": "forbidden"}
    assert second.status["providers_attempted"][:2] == ["brave", "tavily"]
    assert client.calls.count("brave") == 2
    store.close()


@pytest.mark.parametrize("status_code", [432, 433])
def test_tavily_credit_status_opens_payment_circuit(status_code: int) -> None:
    store = Store(":memory:")
    client = _CircuitSearchClient(extract_status=status_code)
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    circuit: set[str] = set()

    document = adapter.fetch_network(
        "https://example.com/e",
        resolved_usage={"extract_chain": ["tavily", "exa"], "reason_codes": []},
        provider_circuit=circuit,
    )

    assert circuit == {"tavily"}
    assert document.metadata["provider_failure_reasons"] == {"tavily": "payment_required"}
    store.close()


@pytest.mark.parametrize(
    "status_code, expected",
    [(403, "forbidden"), (432, "payment_required"), (433, "payment_required"), (429, "rate_limited")],
)
def test_named_search_connectivity_preserves_provider_failure_class(
    status_code: int,
    expected: str,
) -> None:
    def opener(request, timeout):  # noqa: ANN001, ANN202, ARG001
        raise urllib.error.HTTPError(request.full_url, status_code, "failure", None, None)

    result = check_named_search_provider_connectivity(
        ProviderSettings(tavily_api_key="tavily-test"),
        provider_id="tavily",
        opener=opener,
    )

    assert result["status"] == expected


def test_deadline_errors_are_not_swallowed_as_provider_fallback() -> None:
    class _DeadlineClient(_RoutingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201
            raise BaselineError(ErrorCode.RUN_TIMEOUT, "deadline")

    store = Store(":memory:")
    adapter = ResilientSearchAdapter(_DeadlineClient(), store=store, retriever=Retriever(store))

    with pytest.raises(BaselineError) as exc:
        adapter.search_network(
            "alpha",
            resolved_usage={"search_chain": ["brave", "tavily"], "reason_codes": []},
        )

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    store.close()
