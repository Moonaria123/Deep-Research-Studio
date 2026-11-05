import json
import re
from typing import Any

from deepresearch_core.standalone.llm import parse_json_object, redact_text, user_facing_provider_error


JsonDict = dict[str, Any]
MAX_PROVIDER_CLAIMS = 6
MAX_PROVIDER_SOURCES = 3
EVIDENCE_TEXT_LIMIT = 700
CLAIM_TEXT_LIMIT = 220
NOTE_LIMIT = 220
_LOCAL_PATH_RE = re.compile(
    r"(?i)file:///[^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])(?:/home|/users|/var|/tmp|/mnt|/workspace|/root|/etc|/opt|/private)/[^\r\n,;)]*",
)
_VALID_VERDICTS = {"supported", "uncertain", "contradicted", "unsupported"}


def provider_verification_skipped() -> JsonDict:
    """Return a public provider verification object for the default offline path."""

    return {
        "enabled": False,
        "status": "skipped",
        "mode": "provider_semantic_entailment",
        "model": "",
        "note": "默认离线；设置 DEEPRESEARCH_PROVIDER_SEMANTIC_VERIFICATION=1 后才会使用本机 BYOK 模型做语义核验。",
        "items": [],
    }


def verify_provenance_with_provider(
    llm_client: Any,
    provenance: JsonDict,
    *,
    source_documents: list[JsonDict],
    enabled: bool,
) -> JsonDict:
    """Use the configured BYOK LLM to check claim/source semantic alignment when enabled."""

    if not enabled:
        return provider_verification_skipped()

    claims = _public_claims(provenance)
    if not claims:
        return {
            **provider_verification_skipped(),
            "enabled": True,
            "status": "no_input",
            "model": _safe_text(getattr(llm_client, "model", ""), 120),
            "note": "未找到可交给模型核验的公开主张。",
        }

    prompt_payload = {
        "task": "provider_semantic_entailment",
        "allowed_verdicts": ["supported", "uncertain", "contradicted", "unsupported"],
        "claims": [_claim_prompt_item(claim, source_documents) for claim in claims],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You verify whether visible source excerpts semantically support each claim. "
                "Return JSON only. Do not include chain-of-thought, hidden reasoning, secrets, or source full text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return JSON in this shape: "
                '{"claims":[{"claim_id":"C1","verdict":"supported|uncertain|contradicted|unsupported",'
                '"score":0-100,"note":"short user-facing note"}]}.\n\n'
                f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        data = parse_json_object(llm_client.chat(messages, max_tokens=1200, temperature=0.0), {"claims": []})
    except Exception as error:
        return {
            **provider_verification_skipped(),
            "enabled": True,
            "status": "failed",
            "model": _safe_text(getattr(llm_client, "model", ""), 120),
            "note": _safe_text(f"模型语义核验失败：{user_facing_provider_error(error)}", NOTE_LIMIT),
        }

    return {
        "enabled": True,
        "status": "completed",
        "mode": "provider_semantic_entailment",
        "model": _safe_text(getattr(llm_client, "model", ""), 120),
        "note": "模型语义核验已基于可见主张和来源摘要执行；结果仍需人工核对。",
        "items": _safe_result_items(data, {claim["id"] for claim in claims}),
    }


def _public_claims(provenance: JsonDict) -> list[JsonDict]:
    """Return public claims that can be sent to the user's configured provider."""

    claims = provenance.get("claims") if isinstance(provenance, dict) else []
    if not isinstance(claims, list):
        return []
    safe_claims: list[JsonDict] = []
    for item in claims:
        if not isinstance(item, dict) or len(safe_claims) >= MAX_PROVIDER_CLAIMS:
            continue
        claim_id = _safe_text(item.get("id"), 24)
        text = _safe_text(item.get("text"), CLAIM_TEXT_LIMIT)
        if not claim_id or not text:
            continue
        source_indexes = []
        raw_indexes = item.get("source_indexes") if isinstance(item.get("source_indexes"), list) else []
        for value in raw_indexes:
            try:
                source_index = int(value)
            except (TypeError, ValueError):
                continue
            if source_index >= 1 and source_index not in source_indexes:
                source_indexes.append(source_index)
        safe_claims.append({"id": claim_id, "text": text, "source_indexes": source_indexes[:MAX_PROVIDER_SOURCES]})
    return safe_claims


def _claim_prompt_item(claim: JsonDict, source_documents: list[JsonDict]) -> JsonDict:
    """Return one bounded prompt item for provider-side semantic verification."""

    return {
        "claim_id": claim["id"],
        "text": claim["text"],
        "sources": [_source_prompt_item(source_index, source_documents) for source_index in claim.get("source_indexes", [])],
    }


def _source_prompt_item(source_index: int, source_documents: list[JsonDict]) -> JsonDict:
    """Return one bounded source excerpt for the provider prompt."""

    source = source_documents[source_index - 1] if 1 <= source_index <= len(source_documents) else {}
    title = _safe_text(source.get("title") or source.get("url") or f"Source {source_index}", 180)
    url = _safe_text(source.get("url") or "", 300)
    evidence = _safe_text(source.get("description") or source.get("markdown") or source.get("text") or "", EVIDENCE_TEXT_LIMIT)
    return {"index": source_index, "title": title, "url": url, "evidence": evidence}


def _safe_result_items(data: JsonDict, claim_ids: set[str]) -> list[JsonDict]:
    """Return provider verdict items limited to known public claim ids."""

    raw_items = data.get("claims")
    if not isinstance(raw_items, list):
        raw_items = data.get("items") if isinstance(data.get("items"), list) else []
    items: list[JsonDict] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict) or len(items) >= MAX_PROVIDER_CLAIMS:
            continue
        claim_id = _safe_text(item.get("claim_id") or item.get("claimId") or item.get("id"), 24)
        if claim_id not in claim_ids or claim_id in seen:
            continue
        seen.add(claim_id)
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "uncertain"
        items.append(
            {
                "claim_id": claim_id,
                "verdict": verdict,
                "score": _score(item.get("score")),
                "note": _safe_text(item.get("note"), NOTE_LIMIT),
            }
        )
    return items


def _score(value: Any) -> int:
    """Return a bounded provider score."""

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _safe_text(value: Any, limit: int) -> str:
    """Return compact public text without secrets or local paths."""

    text = _LOCAL_PATH_RE.sub("[local path]", redact_text(str(value or "")))
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
