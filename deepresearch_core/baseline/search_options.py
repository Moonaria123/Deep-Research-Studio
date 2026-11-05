"""Validated per-run web-search options."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

FRESHNESS_VALUES = ("pd", "pw", "pm", "py")
_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")
_COUNTRY_RE = re.compile(r"^[A-Za-z]{2}$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_LOCALE_FIELDS = ("country", "search_lang", "ui_language")


@dataclass(frozen=True)
class SearchOptions:
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    freshness: str | None = None
    extra_subqueries: tuple[str, ...] = ()
    country: str | None = None
    search_lang: str | None = None
    ui_language: str | None = None
    # API requests retain only the user's explicit values above. These private,
    # comparison-neutral fields carry the API-frozen defaults into an RQ worker.
    _effective_country: str | None = field(default=None, compare=False, repr=False)
    _effective_search_lang: str | None = field(default=None, compare=False, repr=False)
    _effective_ui_language: str | None = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "include_domains": list(self.include_domains),
            "exclude_domains": list(self.exclude_domains),
            "freshness": self.freshness,
            "extra_subqueries": list(self.extra_subqueries),
        }
        for field_name in _LOCALE_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload

    def has_search_filters(self) -> bool:
        return bool(self.include_domains or self.exclude_domains or self.freshness)

    def has_any(self) -> bool:
        return self.has_search_filters() or bool(self.extra_subqueries) or bool(self.requested_locale())

    def requested_locale(self) -> dict[str, str]:
        """Return only locale fields explicitly supplied for this run."""

        return {
            field_name: value
            for field_name in _LOCALE_FIELDS
            if (value := getattr(self, field_name)) is not None
        }

    def effective_locale(self, settings: Any | None = None) -> dict[str, str]:
        """Return API-frozen locale, falling back to current search settings."""

        country = self._effective_country or self.country or str(getattr(settings, "country", "US") or "US").upper()
        search_lang = (
            self._effective_search_lang
            or self.search_lang
            or str(getattr(settings, "search_lang", "en") or "en").lower()
        )
        ui_language = self._effective_ui_language or self.ui_language
        payload = {"country": country, "search_lang": search_lang}
        if ui_language:
            payload["ui_language"] = ui_language
        return payload

    def with_effective_locale(self, locale: dict[str, str]) -> "SearchOptions":
        """Attach frozen effective values without changing the public request."""

        return replace(
            self,
            _effective_country=locale.get("country"),
            _effective_search_lang=locale.get("search_lang"),
            _effective_ui_language=locale.get("ui_language"),
        )


def normalize_search_options(value: Any) -> SearchOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("search_options must be an object")
    include = _normalize_domains(value.get("include_domains"), field="include_domains")
    exclude = _normalize_domains(value.get("exclude_domains"), field="exclude_domains")
    overlap = set(include).intersection(exclude)
    if overlap:
        raise ValueError("include_domains and exclude_domains must not overlap")
    freshness = value.get("freshness")
    if freshness in ("", None):
        normalized_freshness = None
    else:
        normalized_freshness = str(freshness).strip().lower()
        if normalized_freshness not in FRESHNESS_VALUES:
            raise ValueError("freshness must be one of: pd, pw, pm, py")
    extra = _normalize_extra_subqueries(value.get("extra_subqueries"))
    country = _normalize_country(value.get("country"))
    search_lang = _normalize_language(value.get("search_lang"), field="search_lang", canonical_ui=False)
    ui_language = _normalize_language(value.get("ui_language"), field="ui_language", canonical_ui=True)
    options = SearchOptions(
        include_domains=include,
        exclude_domains=exclude,
        freshness=normalized_freshness,
        extra_subqueries=extra,
        country=country,
        search_lang=search_lang,
        ui_language=ui_language,
    )
    return options if options.has_any() else None


def freeze_search_locale(
    options: SearchOptions | None,
    settings: Any | None = None,
) -> tuple[SearchOptions | None, dict[str, str], dict[str, str]]:
    """Freeze requested/effective locale before a run crosses the job boundary."""

    requested = options.requested_locale() if options is not None else {}
    base = options or SearchOptions()
    effective = base.effective_locale(settings)
    return (options.with_effective_locale(effective) if options is not None else None), requested, effective


def runtime_search_options(
    options: SearchOptions | None,
    effective_locale: dict[str, str] | None,
) -> SearchOptions:
    """Build provider options containing the frozen defaults but no fake request values."""

    effective = dict(effective_locale or {})
    if options is not None and (not effective or options.effective_locale() == effective):
        return options
    return (options or SearchOptions()).with_effective_locale(effective)


def _normalize_country(value: Any) -> str | None:
    if value in (None, ""):
        return None
    country = str(value).strip()
    if not _COUNTRY_RE.fullmatch(country):
        raise ValueError("country must be exactly two ASCII letters")
    return country.upper()


def _normalize_language(value: Any, *, field: str, canonical_ui: bool) -> str | None:
    if value in (None, ""):
        return None
    language = str(value).strip()
    if len(language) > 35 or not _LANGUAGE_RE.fullmatch(language):
        raise ValueError(f"{field} must be a BCP47-like language tag of at most 35 characters")
    parts = language.split("-")
    if not canonical_ui:
        return "-".join(part.lower() for part in parts)
    canonical = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            canonical.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            canonical.append(part.upper())
        else:
            canonical.append(part.lower())
    return "-".join(canonical)


def _normalize_domains(value: Any, *, field: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        domain = str(item or "").strip().lower()
        if not domain:
            continue
        if "://" in domain or "/" in domain or ":" in domain:
            raise ValueError(f"{field} values must be host names without scheme, path, or port")
        if not _HOST_RE.match(domain):
            raise ValueError(f"{field} contains an invalid host name")
        if domain not in seen:
            out.append(domain)
            seen.add(domain)
    if len(out) > 10:
        raise ValueError(f"{field} accepts at most 10 domains")
    return tuple(out)


def _normalize_extra_subqueries(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("extra_subqueries must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        query = " ".join(str(item or "").split())
        if not query:
            continue
        if len(query) < 2 or len(query) > 200:
            raise ValueError("extra_subqueries values must be 2-200 characters")
        key = query.lower()
        if key not in seen:
            out.append(query)
            seen.add(key)
    if len(out) > 8:
        raise ValueError("extra_subqueries accepts at most 8 values")
    return tuple(out)
