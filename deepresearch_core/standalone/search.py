import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from .llm import (
    ProviderRequestError,
    RetryPolicy,
    Sleeper,
    default_json_transport,
    join_url,
    request_json_with_retries,
    retry_policy_from_env,
    validate_http_url,
)


JsonDict = Dict[str, Any]
Transport = Callable[[urllib.request.Request], JsonDict]


class MissingSearchCredentialError(ValueError):
    """Raised when search or scrape credentials are missing."""


@dataclass(frozen=True)
class SearchSettings:
    """Settings for the built-in BYOK search and extract providers."""

    brave_api_key: str = ""
    firecrawl_api_key: str = ""
    firecrawl_base_url: str = "https://api.firecrawl.dev"
    exa_api_key: str = ""
    tavily_api_key: str = ""
    country: str = "US"
    search_lang: str = "en"


class SearchClient:
    """Stdlib HTTP client for Brave, Firecrawl, Exa, and Tavily."""

    def __init__(
        self,
        settings: SearchSettings,
        *,
        transport: Optional[Transport] = None,
        retry_policy: Optional[RetryPolicy] = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.settings = settings
        self.transport = transport or default_json_transport
        self.retry_policy = retry_policy or retry_policy_from_env()
        self.sleeper = sleeper

    def require_ready(self) -> None:
        """Validate that search and scrape credentials are present."""

        missing = []
        if not self.search_ready():
            missing.append("Brave API Key")
        if not self.scrape_ready():
            missing.append("Firecrawl API Key")
        if missing:
            raise MissingSearchCredentialError(f"搜索配置未完成：请填写 {', '.join(missing)}。")
        validate_http_url(self.settings.firecrawl_base_url, field_name="firecrawl_base_url")

    def require_search_ready(self) -> None:
        """Validate that web search can run without requiring page scraping."""

        if not self.search_ready():
            raise MissingSearchCredentialError("搜索配置未完成：请填写 Brave API Key。")

    def require_scrape_ready(self) -> None:
        """Validate that page scraping can run."""

        if not self.scrape_ready():
            raise MissingSearchCredentialError("搜索配置未完成：请填写 Firecrawl API Key。")
        validate_http_url(self.settings.firecrawl_base_url, field_name="firecrawl_base_url")

    def search_ready(self) -> bool:
        """Return whether Brave search is configured."""

        return bool(self.settings.brave_api_key.strip())

    def scrape_ready(self) -> bool:
        """Return whether Firecrawl scraping is configured."""

        return bool(self.settings.firecrawl_api_key.strip())

    def provider_ready(self, provider: str, *, capability: str = "search") -> bool:
        """Return whether one named provider is configured for a capability."""

        normalized = str(provider or "").strip().lower()
        configured = {
            "brave": bool(self.settings.brave_api_key.strip()),
            "firecrawl": bool(self.settings.firecrawl_api_key.strip()),
            "exa": bool(self.settings.exa_api_key.strip()),
            "tavily": bool(self.settings.tavily_api_key.strip()),
        }
        if capability == "extract" and normalized == "brave":
            return False
        return bool(configured.get(normalized))

    def require_provider_ready(self, provider: str, *, capability: str = "search") -> None:
        """Raise a secret-free configuration error for one named provider."""

        if self.provider_ready(provider, capability=capability):
            return
        label = str(provider or "").strip().title() or "Search"
        raise MissingSearchCredentialError(f"搜索配置未完成：请填写 {label} API Key。")

    def search(self, query: str, *, count: int = 5, options: Any = None) -> List[JsonDict]:
        """Search Brave and normalize web results."""

        self.require_search_ready()
        rewritten_query = _query_with_domain_filters(query, options)
        locale = _effective_locale(options, self.settings)
        params_payload = {
            "q": rewritten_query,
            "count": max(1, min(count, 20)),
            "country": locale["country"],
            "search_lang": _brave_search_language(locale["search_lang"]),
        }
        if locale.get("ui_language"):
            params_payload["ui_lang"] = locale["ui_language"]
        freshness = _option_freshness(options)
        if freshness:
            params_payload["freshness"] = freshness
        params = urllib.parse.urlencode(
            params_payload
        )
        request = urllib.request.Request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={"Accept": "application/json", "X-Subscription-Token": self.settings.brave_api_key},
            method="GET",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        raw_results = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
        normalized = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                {
                    "id": f"S{index + 1}",
                    "title": str(item.get("title") or url),
                    "url": url,
                    "description": str(item.get("description") or item.get("snippet") or ""),
                    "age": str(item.get("age") or ""),
                    "source": urllib.parse.urlparse(url).netloc,
                    **_source_temporal_fields(item),
                }
            )
        return normalized

    def firecrawl_search(self, query: str, *, count: int = 5, options: Any = None) -> List[JsonDict]:
        """Search Firecrawl's v2 search endpoint and normalize web results."""

        self.require_scrape_ready()
        locale = _effective_locale(options, self.settings)
        payload = {
            "query": _query_with_domain_filters(query, options),
            "limit": max(1, min(count, 100)),
            "sources": ["web"],
            "country": locale["country"],
        }
        request = urllib.request.Request(
            join_url(self.settings.firecrawl_base_url, "/v2/search"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.firecrawl_api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        body = data.get("data", data) if isinstance(data, dict) else {}
        raw_results = body.get("web", []) if isinstance(body, dict) else []
        normalized = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("metadata", {}).get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                {
                    "id": f"F{index + 1}",
                    "title": str(item.get("title") or item.get("metadata", {}).get("title") or url),
                    "url": url,
                    "description": str(item.get("description") or item.get("snippet") or ""),
                    "source": urllib.parse.urlparse(url).netloc,
                    **_source_temporal_fields(item),
                }
            )
        return normalized

    def tavily_search(self, query: str, *, count: int = 5, options: Any = None) -> List[JsonDict]:
        """Search Tavily without requesting raw page content."""

        self.require_provider_ready("tavily")
        locale = _effective_locale(options, self.settings)
        payload: JsonDict = {
            "query": str(query or "").strip(),
            "search_depth": "basic",
            "max_results": max(1, min(count, 20)),
        }
        country = tavily_country_name(locale.get("country"))
        if country:
            payload["country"] = country
        include = _option_domains(options, "include_domains")
        exclude = _option_domains(options, "exclude_domains")
        if include:
            payload["include_domains"] = include
        if exclude:
            payload["exclude_domains"] = exclude
        freshness = _option_freshness(options)
        if freshness in _TAVILY_FRESHNESS:
            payload["time_range"] = _TAVILY_FRESHNESS[freshness]
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.tavily_api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        normalized: list[JsonDict] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                {
                    "id": str(item.get("id") or f"T{index + 1}"),
                    "title": str(item.get("title") or url),
                    "url": url,
                    "description": str(item.get("content") or item.get("snippet") or ""),
                    "source": urllib.parse.urlparse(url).netloc,
                    **_source_temporal_fields(item),
                }
            )
        return normalized

    def exa_search(self, query: str, *, count: int = 5, options: Any = None) -> List[JsonDict]:
        """Search Exa without embedding a Contents request."""

        self.require_provider_ready("exa")
        locale = _effective_locale(options, self.settings)
        payload: JsonDict = {
            "query": str(query or "").strip(),
            "numResults": max(1, min(count, 100)),
            "type": "auto",
        }
        country = str(locale.get("country") or "").strip().upper()
        if len(country) == 2 and country.isascii() and country.isalpha():
            payload["userLocation"] = country
        include = _option_domains(options, "include_domains")
        exclude = _option_domains(options, "exclude_domains")
        if include:
            payload["includeDomains"] = include
        if exclude:
            payload["excludeDomains"] = exclude
        freshness = _option_freshness(options)
        if freshness in _FRESHNESS_DAYS:
            payload["startPublishedDate"] = _freshness_start_iso(freshness)
        request = urllib.request.Request(
            "https://api.exa.ai/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self.settings.exa_api_key,
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        normalized: list[JsonDict] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            highlights = item.get("highlights")
            snippet = " ".join(str(value) for value in highlights) if isinstance(highlights, list) else ""
            normalized.append(
                {
                    "id": str(item.get("id") or f"E{index + 1}"),
                    "title": str(item.get("title") or url),
                    "url": url,
                    "description": str(item.get("description") or snippet),
                    "source": urllib.parse.urlparse(url).netloc,
                    **_source_temporal_fields(item),
                }
            )
        return normalized

    def scrape(self, url: str) -> JsonDict:
        """Scrape one URL with Firecrawl and return normalized markdown."""

        self.require_scrape_ready()
        request = urllib.request.Request(
            join_url(self.settings.firecrawl_base_url, "/v2/scrape"),
            data=json.dumps({"url": url, "formats": ["markdown"], "onlyMainContent": True}, ensure_ascii=False).encode(
                "utf-8"
            ),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.firecrawl_api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        body = data.get("data", data) if isinstance(data, dict) else {}
        markdown = str(body.get("markdown") or body.get("content") or "") if isinstance(body, dict) else ""
        if not markdown.strip():
            raise ProviderRequestError(f"Firecrawl returned empty markdown for {url}")
        return {"markdown": markdown, "raw": body}

    def tavily_extract(self, url: str) -> JsonDict:
        """Extract one selected URL through Tavily as Markdown."""

        self.require_provider_ready("tavily", capability="extract")
        target = str(url or "").strip()
        request = urllib.request.Request(
            "https://api.tavily.com/extract",
            data=json.dumps(
                {
                    "urls": [target],
                    "extract_depth": "basic",
                    "format": "markdown",
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.tavily_api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        failed = data.get("failed_results", []) if isinstance(data, dict) else []
        item = next(
            (
                value
                for value in results
                if isinstance(value, dict) and _same_document_url(value.get("url"), target)
            ),
            None,
        )
        if item is None:
            detail = ""
            failed_item = next(
                (
                    value
                    for value in failed
                    if isinstance(value, dict)
                    and _same_document_url(value.get("url"), target)
                ),
                None,
            ) if isinstance(failed, list) else None
            if failed_item is not None:
                detail = str(failed_item.get("error") or failed_item.get("message") or "")
            raise ProviderRequestError(f"Tavily Extract failed for {target}: {detail or 'empty result'}")
        markdown = str(item.get("raw_content") or "")
        if not markdown.strip():
            raise ProviderRequestError(f"Tavily Extract returned empty raw_content for {target}")
        canonical = str(item.get("url") or "").strip()
        return {"markdown": markdown, "raw": item, "canonical_url": canonical}

    def exa_contents(
        self,
        url: str,
        *,
        max_characters: int = 100_000,
        highlights_query: str | None = None,
    ) -> JsonDict:
        """Retrieve Contents for one URL selected after Exa Search."""

        self.require_provider_ready("exa", capability="extract")
        target = str(url or "").strip()
        payload: JsonDict = {
            "urls": [target],
            "text": {"maxCharacters": max(1, int(max_characters))},
        }
        focus = str(highlights_query or "").strip()
        if focus:
            payload["highlights"] = {"query": focus}
        request = urllib.request.Request(
            "https://api.exa.ai/contents",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self.settings.exa_api_key,
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        statuses = data.get("statuses", []) if isinstance(data, dict) else []
        item = next(
            (
                value
                for value in results
                if isinstance(value, dict) and _same_document_url(value.get("url"), target)
            ),
            None,
        )
        result_id = str(item.get("id") or "").strip() if item is not None else ""
        status_item = next(
            (
                value
                for value in statuses
                if isinstance(value, dict)
                and (
                    (result_id and str(value.get("id") or "").strip() == result_id)
                    or _same_document_url(value.get("url"), target)
                )
            ),
            None,
        ) if isinstance(statuses, list) else None
        status = str((status_item or {}).get("status") or "").strip().lower()
        if item is None or status not in {"success", "completed"}:
            raise ProviderRequestError(f"Exa Contents failed for {target}")
        highlights = item.get("highlights")
        focused_text = "\n\n".join(
            str(value).strip()
            for value in (highlights if isinstance(highlights, list) else [])
            if str(value).strip()
        )
        markdown = focused_text or str(item.get("text") or item.get("content") or "")
        if not markdown.strip():
            raise ProviderRequestError(f"Exa Contents returned empty content for {target}")
        canonical = str(item.get("url") or "").strip()
        return {"markdown": markdown, "raw": item, "canonical_url": canonical}

    def search_and_scrape(self, query: str, *, count: int = 5) -> List[JsonDict]:
        """Search Brave and scrape each result, preserving partial results."""

        results = self.search(query, count=count)
        enriched = []
        for result in results:
            item = dict(result)
            if not self.scrape_ready():
                item["markdown"] = item.get("description", "")
                item["scrape_status"] = "skipped_missing_firecrawl"
                enriched.append(item)
                continue
            try:
                scraped = self.scrape(item["url"])
                item["markdown"] = scraped["markdown"]
                item["scrape_status"] = "ok"
            except Exception as error:
                item["markdown"] = item.get("description", "")
                item["scrape_status"] = "fallback"
                item["scrape_error"] = str(error)
            enriched.append(item)
        return enriched


def _option_freshness(options: Any) -> str:
    value = getattr(options, "freshness", None)
    return str(value or "").strip()


def _same_document_url(left: object, right: object) -> bool:
    """Return whether two HTTP URLs identify the same requested document."""

    return bool(_document_url_identity(left)) and _document_url_identity(left) == _document_url_identity(right)


def _document_url_identity(value: object) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return ""
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not host:
        return ""
    port = parsed.port
    netloc = host
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)))
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _effective_locale(options: Any, settings: SearchSettings) -> JsonDict:
    resolver = getattr(options, "effective_locale", None)
    if callable(resolver):
        resolved = resolver(settings)
        if isinstance(resolved, dict):
            return dict(resolved)
    return {"country": settings.country, "search_lang": settings.search_lang}


def _brave_search_language(value: object) -> str:
    """Translate product locale aliases to Brave's search_lang enum."""

    normalized = str(value or "").strip().lower()
    if normalized == "zh":
        return "zh-hans"
    return normalized


def _source_temporal_fields(item: JsonDict) -> JsonDict:
    """Preserve provider-supplied exact time fields for later normalization."""

    nested = item.get("metadata")
    sources = (item, nested if isinstance(nested, dict) else {})
    aliases = {
        "published_at": ("published_at", "publishedAt", "published_date", "publishedDate", "datePublished"),
        "updated_at": ("updated_at", "updatedAt", "updated_date", "updatedDate", "lastModified"),
        "effective_at": ("effective_at", "effectiveAt", "effective_date", "effectiveDate", "validFrom"),
    }
    result: JsonDict = {}
    for canonical, names in aliases.items():
        value = next(
            (
                source.get(name)
                for source in sources
                for name in names
                if source.get(name) not in (None, "")
            ),
            None,
        )
        if value is not None:
            result[canonical] = value
    return result


def _option_domains(options: Any, field: str) -> list[str]:
    value = getattr(options, field, None)
    if not value:
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def _query_with_domain_filters(query: str, options: Any) -> str:
    include = _option_domains(options, "include_domains")
    exclude = _option_domains(options, "exclude_domains")
    if not include and not exclude:
        return query
    parts = [str(query or "").strip()]
    parts.extend(f"site:{domain}" for domain in include)
    parts.extend(f"-site:{domain}" for domain in exclude)
    return " ".join(part for part in parts if part)


_TAVILY_COUNTRIES = {
    "AE": "united arab emirates",
    "AR": "argentina",
    "AT": "austria",
    "AU": "australia",
    "BE": "belgium",
    "BR": "brazil",
    "CA": "canada",
    "CH": "switzerland",
    "CL": "chile",
    "CN": "china",
    "CO": "colombia",
    "CZ": "czech republic",
    "DE": "germany",
    "DK": "denmark",
    "EG": "egypt",
    "ES": "spain",
    "FI": "finland",
    "FR": "france",
    "GB": "united kingdom",
    "GR": "greece",
    "HK": "hong kong",
    "HU": "hungary",
    "ID": "indonesia",
    "IE": "ireland",
    "IL": "israel",
    "IN": "india",
    "IT": "italy",
    "JP": "japan",
    "KR": "south korea",
    "MX": "mexico",
    "MY": "malaysia",
    "NL": "netherlands",
    "NO": "norway",
    "NZ": "new zealand",
    "PH": "philippines",
    "PL": "poland",
    "PT": "portugal",
    "RO": "romania",
    "RU": "russia",
    "SA": "saudi arabia",
    "SE": "sweden",
    "SG": "singapore",
    "TH": "thailand",
    "TR": "turkey",
    "TW": "taiwan",
    "UA": "ukraine",
    "US": "united states",
    "VN": "vietnam",
    "ZA": "south africa",
}

_TAVILY_FRESHNESS = {
    "pd": "day",
    "pw": "week",
    "pm": "month",
    "py": "year",
}

_FRESHNESS_DAYS = {
    "pd": 1,
    "pw": 7,
    "pm": 30,
    "py": 365,
}


def tavily_country_name(value: object) -> str:
    """Map normalized ISO2 input to Tavily's documented country enum."""

    return _TAVILY_COUNTRIES.get(str(value or "").strip().upper(), "")


def _freshness_start_iso(value: str, *, now: datetime | None = None) -> str:
    """Translate an explicit relative freshness option to Exa's UTC boundary."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    start = current.astimezone(timezone.utc) - timedelta(days=_FRESHNESS_DAYS[value])
    return start.isoformat(timespec="seconds").replace("+00:00", "Z")
