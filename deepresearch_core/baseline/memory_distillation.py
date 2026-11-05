"""Memory continuity distillation for workspace sessions and user profiles."""

from __future__ import annotations

import re
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .adapters import LLMAdapter
from .errors import BaselineError, ErrorCode
from .memory import MemoryLayer
from .memory_json import (
    MemoryJsonError,
    attempt_diagnostics,
    build_repair_prompt,
    diagnostics_from_attempts,
    parse_json_object,
)
from .memory_schema import (
    MEMORY_FACT_JSON_SCHEMA,
    SESSION_SUMMARY_JSON_SCHEMA,
    USER_PROFILE_JSON_SCHEMA,
    MemorySchemaError,
    validate_memory_fact_candidates,
    validate_session_memory_summary,
    validate_user_profile_sections,
)
from .store import Store

JsonDict = dict[str, Any]

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{6,}|ghp_[A-Za-z0-9_]{6,}|AKIA[0-9A-Z]{8,}|"
    r"xox[abp]-[A-Za-z0-9-]{6,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


class MemoryDistillationError(RuntimeError):
    """Raised when memory distillation cannot produce safe structured output."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "MEMORY_DISTILLATION_FAILED",
        failed_stage: str | None = None,
        failed_session_id: str | None = None,
        attempted_sessions: int = 0,
        successful_sessions: int = 0,
        failed_sessions: int = 0,
        user_profile_attempted: bool = False,
        user_profile_updated: bool = False,
        user_profile_failed: bool = False,
        failure_count: int = 0,
        diagnostics: JsonDict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failed_stage = failed_stage
        self.failed_session_id = failed_session_id
        self.attempted_sessions = attempted_sessions
        self.successful_sessions = successful_sessions
        self.failed_sessions = failed_sessions
        self.user_profile_attempted = user_profile_attempted
        self.user_profile_updated = user_profile_updated
        self.user_profile_failed = user_profile_failed
        self.failure_count = failure_count
        self.diagnostics = diagnostics or {}

    def status_metadata(self) -> JsonDict:
        """Return secret-free status fields suitable for Dream run metadata."""

        data: JsonDict = {
            "error_code": self.error_code,
            "failed_stage": self.failed_stage,
            "attempted_sessions": self.attempted_sessions,
            "successful_sessions": self.successful_sessions,
            "failed_sessions": self.failed_sessions,
            "user_profile_attempted": self.user_profile_attempted,
            "user_profile_updated": self.user_profile_updated,
            "user_profile_failed": self.user_profile_failed,
            "failure_count": self.failure_count,
        }
        if self.failed_session_id:
            data["failed_session_id"] = self.failed_session_id
        data.update(_diagnostic_metadata(self.diagnostics))
        return data


class MemoryDistillationCancelled(RuntimeError):
    """Raised at a safe stage boundary after a Dream cancellation request."""


@dataclass(frozen=True)
class MemoryDistillationConfig:
    """Bounds for one memory continuity refresh."""

    recent_session_limit: int = 8
    messages_per_session: int = 12
    existing_memory_limit: int = 40
    report_limit: int = 6
    max_input_chars: int = 12000
    max_summary_chars: int = 1200
    max_section_chars: int = 900


class MemoryDistillationService:
    """Distill run/session activity into durable local memory rows."""

    def __init__(
        self,
        *,
        store: Store,
        memory_llm: Callable[[], LLMAdapter | None],
        memory_thinking_budget: Callable[[], int | None] | None = None,
        config: MemoryDistillationConfig | None = None,
    ) -> None:
        self._store = store
        self._memory_llm = memory_llm
        self._memory_thinking_budget = memory_thinking_budget or (lambda: 0)
        self._config = config or MemoryDistillationConfig()
        self._last_structured_strategy = ""

    def is_configured(self) -> bool:
        """Return whether a memory-processing model is available."""

        return self._memory_llm() is not None

    def refresh(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        session_id: str | None = None,
        include_user: bool = True,
        strategy: str = "incremental",
        should_cancel: Callable[[], bool] | None = None,
    ) -> JsonDict:
        """Refresh workspace session summaries and the optional user profile."""

        llm = self._memory_llm()
        if llm is None:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "memory processing model is not configured",
                details={"role": "memory"},
            )

        strategy = "full" if str(strategy or "").strip().lower() == "full" else "incremental"
        self._last_structured_strategy = ""
        session_ids = [session_id] if session_id else self._recent_session_ids()
        session_ids = [sid for sid in session_ids if sid]
        workspace_updates = 0
        workspace_skipped = 0
        failures: list[JsonDict] = []
        for sid in session_ids:
            _raise_if_cancelled(should_cancel)
            watermark = self._session_watermark(sid)
            if strategy == "incremental" and self._workspace_summary_current(
                session_id=sid,
                workspace_key=workspace_key,
                watermark=watermark,
            ):
                workspace_skipped += 1
                continue
            try:
                summary = self._distill_session(llm, session_id=sid, workspace_key=workspace_key, user_key=user_key)
            except Exception as error:  # noqa: BLE001 - one bad session should not invalidate old memory
                failures.append(_failure_record(error, stage="session_distill", session_id=sid))
                continue
            if not summary:
                continue
            _raise_if_cancelled(should_cancel)
            session = self._store.get_session(sid) or {}
            persisted_watermark = {**watermark, "last_distilled_at": time.time()}
            self._store.upsert_workspace_session_memory(
                session_id=sid,
                workspace_key=workspace_key,
                content=summary,
                metadata={
                    "session_title": str(session.get("title") or "Untitled"),
                    "session_updated_at": session.get("updated_at"),
                    **persisted_watermark,
                },
            )
            workspace_updates += 1

        user_updates = 0
        user_failed = False
        if include_user:
            _raise_if_cancelled(should_cancel)
            try:
                sections = self._distill_user_profile(llm, workspace_key=workspace_key, user_key=user_key)
                if sections:
                    _raise_if_cancelled(should_cancel)
                    content = _render_user_profile(sections)
                    self._store.upsert_user_profile_memory(user_key=user_key, content=content, sections=sections)
                    user_updates = 1
            except MemoryDistillationCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - user profile refresh is isolated from workspace summaries
                user_failed = True
                failures.append(_failure_record(error, stage="user_profile"))

        if workspace_updates == 0 and user_updates == 0 and workspace_skipped == 0:
            first = failures[0] if failures else {}
            raise MemoryDistillationError(
                "memory distillation failed for all attempted stages"
                if failures
                else "memory distillation produced no durable memory output",
                error_code="MEMORY_DISTILLATION_ALL_FAILED" if failures else "NO_DURABLE_MEMORY_OUTPUT",
                failed_stage=first.get("failed_stage") if isinstance(first.get("failed_stage"), str) else None,
                failed_session_id=first.get("failed_session_id") if isinstance(first.get("failed_session_id"), str) else None,
                attempted_sessions=len(session_ids),
                successful_sessions=workspace_updates,
                failed_sessions=sum(1 for item in failures if item.get("failed_stage") == "session_distill"),
                user_profile_attempted=include_user,
                user_profile_updated=bool(user_updates),
                user_profile_failed=user_failed,
                failure_count=len(failures),
                diagnostics=failures[0] if failures else {},
            )
        result: JsonDict = {
            "workspace_updated": workspace_updates,
            "workspace_skipped": workspace_skipped,
            "user_updated": user_updates,
            "memories_created": workspace_updates + user_updates,
            "attempted_sessions": len(session_ids),
            "successful_sessions": workspace_updates,
            "failed_sessions": sum(1 for item in failures if item.get("failed_stage") == "session_distill"),
            "user_profile_attempted": include_user,
            "user_profile_updated": bool(user_updates),
            "user_profile_failed": user_failed,
            "failure_count": len(failures),
            "structured_strategy": self._last_structured_strategy or "unknown",
        }
        if failures:
            first = failures[0]
            result.update(
                {
                    "error_code": "PARTIAL_FAILURE",
                    "failed_stage": first.get("failed_stage"),
                    "failed_session_id": first.get("failed_session_id"),
                    **_diagnostic_metadata(first),
                    "failures": failures[:5],
                }
            )
        incomplete_reasons = _incomplete_reasons(result)
        result["incomplete"] = bool(incomplete_reasons)
        result["incomplete_reasons"] = incomplete_reasons
        return result

    def refresh_post_turn(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        session_id: str,
        run_id: str = "",
    ) -> JsonDict:
        """Capture the last completed turn without refreshing full session/user memory."""

        llm = self._memory_llm()
        if llm is None:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "memory processing model is not configured",
                details={"role": "memory"},
            )
        self._last_structured_strategy = ""
        try:
            summary = self._distill_post_turn(
                llm,
                session_id=session_id,
                run_id=run_id,
                workspace_key=workspace_key,
                user_key=user_key,
            )
        except Exception as error:  # noqa: BLE001 - normalize for audit/status callers
            record = _failure_record(error, stage="post_turn_capture", session_id=session_id)
            raise MemoryDistillationError(
                "post-turn memory capture failed",
                error_code=str(record.get("error_code") or "MEMORY_DISTILLATION_FAILED"),
                failed_stage=str(record.get("failed_stage") or "post_turn_capture"),
                failed_session_id=session_id,
                attempted_sessions=1,
                failed_sessions=1,
                failure_count=1,
                diagnostics=record,
            ) from error
        if not summary:
            raise MemoryDistillationError(
                "post-turn memory capture produced no durable output",
                error_code="NO_DURABLE_MEMORY_OUTPUT",
                failed_stage="post_turn_capture",
                failed_session_id=session_id,
                attempted_sessions=1,
                failed_sessions=1,
                failure_count=1,
            )
        session = self._store.get_session(session_id) or {}
        watermark = self._turn_watermark(session_id, run_id=run_id)
        persisted = {**watermark, "checked_at": time.time()}
        memory_row = self._store.upsert_post_turn_memory(
            session_id=session_id,
            run_id=run_id,
            workspace_key=workspace_key,
            content=summary,
            metadata={
                "session_title": str(session.get("title") or "Untitled"),
                "session_updated_at": session.get("updated_at"),
                "structured_strategy": self._last_structured_strategy or "unknown",
                **persisted,
            },
        )
        return {
            "status": "completed",
            "source": "post_turn_memory_v1",
            "attempted": 1,
            "succeeded": 1,
            "skipped": 0,
            "failed": 0,
            "memories_created": 1,
            "memory_id": str(memory_row.get("memory_id") or ""),
            "session_id": session_id,
            "run_id": run_id,
            "checked_at": persisted["checked_at"],
            "provider_ref": "",
            "structured_strategy": self._last_structured_strategy or "unknown",
            "incomplete": False,
            "incomplete_reasons": [],
        }

    def refresh_memory_facts(
        self,
        *,
        workspace_key: str = "default",
        user_key: str = "default",
        session_id: str | None = None,
        run_id: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> JsonDict:
        """Extract ADD-only memory_fact_v1 candidates and persist new fact rows."""

        llm = self._memory_llm()
        if llm is None:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "memory processing model is not configured",
                details={"role": "memory"},
            )
        self._last_structured_strategy = ""
        session_ids = [session_id] if session_id else self._recent_session_ids()
        session_ids = [sid for sid in session_ids if sid]
        accepted = 0
        skipped = 0
        failed = 0
        failures: list[JsonDict] = []
        for sid in session_ids:
            _raise_if_cancelled(should_cancel)
            try:
                candidates = self._distill_memory_facts(
                    llm,
                    session_id=sid,
                    workspace_key=workspace_key,
                    user_key=user_key,
                )
            except Exception as error:  # noqa: BLE001 - one bad session should not block others
                failed += 1
                failures.append(_failure_record(error, stage="memory_fact", session_id=sid))
                continue
            for candidate in candidates:
                _raise_if_cancelled(should_cancel)
                content = _clean_text(str(candidate.get("content") or ""), max_chars=500)
                if not content or _SECRET_RE.search(content):
                    skipped += 1
                    continue
                digest = _fact_content_hash(
                    session_id=sid,
                    run_id=run_id,
                    content=content,
                    kind=str(candidate.get("kind") or "fact"),
                )
                created = self._store.add_workspace_memory_fact_once(
                    workspace_key=workspace_key,
                    content=content,
                    content_hash=digest,
                    metadata={
                        "session_id": sid,
                        "run_id": run_id,
                        "kind": str(candidate.get("kind") or "fact"),
                        "confidence": str(candidate.get("confidence") or "medium"),
                        "checked_at": time.time(),
                        "structured_strategy": self._last_structured_strategy or "unknown",
                    },
                ).get("created")
                if created:
                    accepted += 1
                else:
                    skipped += 1
        result: JsonDict = {
            "source": "memory_fact_v1",
            "accepted": accepted,
            "skipped": skipped,
            "failed": failed,
            "attempted": accepted + skipped + failed,
            "memories_created": accepted,
            "structured_strategy": self._last_structured_strategy or "unknown",
        }
        if failures:
            first = failures[0]
            result.update(
                {
                    "error_code": "PARTIAL_FAILURE",
                    "failed_stage": first.get("failed_stage"),
                    "failed_session_id": first.get("failed_session_id"),
                    **_diagnostic_metadata(first),
                    "failures": failures[:5],
                }
            )
        return result

    def _distill_memory_facts(
        self,
        llm: LLMAdapter,
        *,
        session_id: str,
        workspace_key: str,
        user_key: str,
    ) -> list[JsonDict]:
        prompt = self._memory_fact_prompt(session_id=session_id, workspace_key=workspace_key, user_key=user_key)
        data = self._complete_json(
            llm,
            prompt=prompt,
            max_tokens=800,
            stage="memory_fact",
            session_id=session_id,
            schema=MEMORY_FACT_JSON_SCHEMA,
            validator=lambda item: validate_memory_fact_candidates(item).to_dict(),
        )
        raw = data.get("memories") if isinstance(data, dict) else []
        return [item for item in raw if isinstance(item, dict)]
    def _distill_post_turn(
        self,
        llm: LLMAdapter,
        *,
        session_id: str,
        run_id: str,
        workspace_key: str,
        user_key: str,
    ) -> str:
        if run_id and not self._turn_messages(session_id, run_id=run_id):
            raise MemoryDistillationError(
                "post-turn source messages are unavailable",
                error_code="POST_TURN_SOURCE_MISSING",
                failed_stage="post_turn_capture",
                failed_session_id=session_id,
            )
        prompt = self._post_turn_prompt(
            session_id=session_id,
            run_id=run_id,
            workspace_key=workspace_key,
            user_key=user_key,
        )
        data = self._complete_json(
            llm,
            prompt=prompt,
            max_tokens=500,
            stage="post_turn_capture",
            session_id=session_id,
            schema=SESSION_SUMMARY_JSON_SCHEMA,
            validator=lambda item: validate_session_memory_summary(item).to_dict(),
        )
        summary = _clean_text(str(data.get("summary") or ""), max_chars=self._config.max_summary_chars)
        if not summary or _SECRET_RE.search(summary):
            return ""
        return summary

    def _distill_session(
        self,
        llm: LLMAdapter,
        *,
        session_id: str,
        workspace_key: str,
        user_key: str,
    ) -> str:
        prompt = self._session_prompt(session_id=session_id, workspace_key=workspace_key, user_key=user_key)
        data = self._complete_json(
            llm,
            prompt=prompt,
            max_tokens=700,
            stage="session_distill",
            session_id=session_id,
            schema=SESSION_SUMMARY_JSON_SCHEMA,
            validator=lambda item: validate_session_memory_summary(item).to_dict(),
        )
        summary = _clean_text(str(data.get("summary") or ""), max_chars=self._config.max_summary_chars)
        if not summary or _SECRET_RE.search(summary):
            return ""
        return summary

    def _distill_user_profile(self, llm: LLMAdapter, *, workspace_key: str, user_key: str) -> list[JsonDict]:
        prompt = self._user_prompt(workspace_key=workspace_key, user_key=user_key)
        data = self._complete_json(
            llm,
            prompt=prompt,
            max_tokens=900,
            stage="user_profile",
            schema=USER_PROFILE_JSON_SCHEMA,
            validator=lambda item: validate_user_profile_sections(item).to_dict(),
        )
        raw_sections = data.get("sections") if isinstance(data, dict) else None
        if not isinstance(raw_sections, list):
            return []
        sections: list[JsonDict] = []
        for raw in raw_sections:
            if not isinstance(raw, dict):
                continue
            title = _clean_text(str(raw.get("title") or raw.get("id") or ""), max_chars=80)
            body = _clean_text(str(raw.get("body") or ""), max_chars=self._config.max_section_chars)
            if not title or not body or _SECRET_RE.search(title) or _SECRET_RE.search(body):
                continue
            section_id = _section_id(str(raw.get("id") or title))
            sections.append({"id": section_id, "title": title, "body": body})
        return sections

    def _complete_json(
        self,
        llm: LLMAdapter,
        *,
        prompt: str,
        max_tokens: int,
        stage: str,
        session_id: str | None = None,
        schema: JsonDict | None = None,
        validator: Callable[[JsonDict], JsonDict] | None = None,
    ) -> JsonDict:
        attempts: list[JsonDict] = []
        response_format = schema or {"type": "json_object"}
        try:
            result, strategy = self._call_json_with_fallbacks(
                llm,
                prompt=prompt,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except BaselineError as error:
            if _response_format_unsupported(error):
                result, strategy = self._call_json_with_fallbacks(
                    llm,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                raise
        self._last_structured_strategy = strategy
        try:
            return parse_json_object(result.text or "", validator=validator)
        except MemoryJsonError as error:
            attempts.append(
                attempt_diagnostics(
                    result.text,
                    result.finish_reason,
                    error.error_code,
                    strategy,
                )
            )
            repair_prompt = build_repair_prompt(stage=stage, error_code=error.error_code, schema=response_format)
            try:
                repair, repair_strategy = self._call_json_with_fallbacks(
                    llm,
                    prompt=repair_prompt,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self._last_structured_strategy = repair_strategy
                return parse_json_object(repair.text or "", validator=validator)
            except Exception as repair_error:  # noqa: BLE001 - normalize below
                if "repair" in locals():
                    attempts.append(
                        attempt_diagnostics(
                            repair.text,
                            repair.finish_reason,
                            getattr(repair_error, "error_code", "JSON_PARSE_FAILED"),
                            locals().get("repair_strategy", strategy),
                        )
                    )
                error = repair_error if isinstance(repair_error, (MemoryDistillationError, MemorySchemaError)) else error
            raise MemoryDistillationError(
                str(error),
                error_code=getattr(error, "error_code", "JSON_PARSE_FAILED"),
                failed_stage=stage,
                failed_session_id=session_id,
                diagnostics=diagnostics_from_attempts(attempts),
            ) from error

    def _call_json_with_fallbacks(
        self,
        llm: LLMAdapter,
        *,
        prompt: str,
        max_tokens: int,
        response_format: JsonDict | None,
    ):
        formats: list[tuple[str, JsonDict | None]] = []
        if response_format:
            strategy = "json_schema_strict" if response_format.get("type") == "json_schema" else "json_object"
            formats.append((strategy, response_format))
        if response_format != {"type": "json_object"}:
            formats.append(("json_object", {"type": "json_object"}))
        formats.append(("parser_fallback", None))

        last_error: BaselineError | None = None
        for strategy, fmt in formats:
            try:
                return self._call_json_llm(llm, prompt=prompt, max_tokens=max_tokens, response_format=fmt), strategy
            except BaselineError as error:
                if not _response_format_unsupported(error):
                    raise
                last_error = error
                continue
        if last_error is not None:
            raise last_error
        return self._call_json_llm(llm, prompt=prompt, max_tokens=max_tokens, response_format=None), "parser_fallback"

    def _call_json_llm(
        self,
        llm: LLMAdapter,
        *,
        prompt: str,
        max_tokens: int,
        response_format: JsonDict | None,
    ):
        return llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            thinking_budget=self._memory_thinking_budget(),
            response_format=response_format,
        )

    def _session_watermark(self, session_id: str) -> JsonDict:
        messages = self._store.list_messages(session_id)
        latest_message = messages[-1] if messages else {}
        latest_run = self._store.get_latest_run_for_session(session_id)
        digest = hashlib.sha256()
        for message in messages:
            digest.update(str(message.get("id") or "").encode("utf-8"))
            digest.update(str(message.get("content") or "").encode("utf-8"))
        return {
            "last_message_id": str(latest_message.get("id") or ""),
            "last_run_id": latest_run[0] if latest_run else "",
            "content_hash": digest.hexdigest()[:16],
        }

    def _turn_watermark(self, session_id: str, *, run_id: str = "") -> JsonDict:
        messages = self._turn_messages(session_id, run_id=run_id)
        latest_message = messages[-1] if messages else {}
        digest = hashlib.sha256()
        for message in messages:
            digest.update(str(message.get("id") or "").encode("utf-8"))
            digest.update(str(message.get("content") or "").encode("utf-8"))
        return {
            "last_message_id": str(latest_message.get("id") or ""),
            "last_run_id": str(run_id or ""),
            "content_hash": digest.hexdigest()[:16],
        }

    def _workspace_summary_current(self, *, session_id: str, workspace_key: str, watermark: JsonDict) -> bool:
        for row in self._store.list_memories(namespace=MemoryLayer.WORKSPACE.value, scope_key=workspace_key, limit=1000):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("source") != "session_distill_v1" or metadata.get("session_id") != session_id:
                continue
            return all(str(metadata.get(key) or "") == str(watermark.get(key) or "") for key in ("last_message_id", "last_run_id", "content_hash"))
        return False

    def _session_prompt(self, *, session_id: str, workspace_key: str, user_key: str) -> str:
        session = self._store.get_session(session_id) or {}
        sections = [
            "Distill a session research summary for DeepResearch memory continuity.",
            'Return strict JSON only: {"summary":"..."}',
            "The summary should be durable, non-sensitive, and useful for future research context.",
            "Do not include API keys, credentials, private URLs, local filesystem paths, or implementation details.",
            "",
            f"Session: {str(session.get('title') or 'Untitled')}",
            "Messages:",
            self._message_lines(session_id),
            "",
            "Existing workspace memories:",
            self._memory_lines(MemoryLayer.WORKSPACE.value, workspace_key),
            "",
            "Existing user profile:",
            self._memory_lines(MemoryLayer.USER.value, user_key),
            "",
            "Recent reports:",
            self._report_lines(),
        ]
        return _truncate("\n".join(sections), self._config.max_input_chars)

    def _post_turn_prompt(self, *, session_id: str, run_id: str, workspace_key: str, user_key: str) -> str:
        session = self._store.get_session(session_id) or {}
        sections = [
            "Capture the last completed DeepResearch turn for memory continuity.",
            'Return strict JSON only: {"summary":"..."}',
            "Write only durable, non-sensitive facts or preferences from the latest turn.",
            "Do not include API keys, credentials, private URLs, local filesystem paths, or implementation details.",
            "",
            f"Session: {str(session.get('title') or 'Untitled')}",
            "Latest turn messages:",
            self._turn_message_lines(session_id, run_id=run_id),
            "",
            "Existing workspace memories:",
            self._memory_lines(MemoryLayer.WORKSPACE.value, workspace_key),
            "",
            "Existing user profile:",
            self._memory_lines(MemoryLayer.USER.value, user_key),
        ]
        return _truncate("\n".join(sections), min(self._config.max_input_chars, 6000))

    def _memory_fact_prompt(self, *, session_id: str, workspace_key: str, user_key: str) -> str:
        session = self._store.get_session(session_id) or {}
        sections = [
            "Extract ADD-only memory_fact_v1 candidates for DeepResearch memory continuity.",
            'Return strict JSON only: {"memories":[{"content":"...","kind":"fact","confidence":"medium"}]}',
            "The model may only propose new facts, preferences, or constraints. Do not update, delete, merge, or rewrite existing memories.",
            "Do not include API keys, credentials, private URLs, local filesystem paths, or implementation details.",
            "",
            f"Session: {str(session.get('title') or 'Untitled')}",
            "Messages:",
            self._message_lines(session_id),
            "",
            "Existing workspace memories:",
            self._memory_lines(MemoryLayer.WORKSPACE.value, workspace_key),
            "",
            "Existing user profile:",
            self._memory_lines(MemoryLayer.USER.value, user_key),
        ]
        return _truncate("\n".join(sections), self._config.max_input_chars)

    def _user_prompt(self, *, workspace_key: str, user_key: str) -> str:
        sections = [
            "Distill a user memory profile for DeepResearch memory continuity.",
            'Return strict JSON only: {"sections":[{"id":"...","title":"...","body":"..."}]}',
            "Use a small set of stable sections such as research preferences, domains, constraints, and style.",
            "Do not include API keys, credentials, private URLs, local filesystem paths, or implementation details.",
            "",
            "Recent sessions:",
            self._recent_sessions(),
            "",
            "Workspace session summaries:",
            self._memory_lines(MemoryLayer.WORKSPACE.value, workspace_key),
            "",
            "Existing user profile:",
            self._memory_lines(MemoryLayer.USER.value, user_key),
        ]
        return _truncate("\n".join(sections), self._config.max_input_chars)

    def _recent_session_ids(self) -> list[str]:
        return [str(row.get("id") or "") for row in self._store.list_sessions(limit=self._config.recent_session_limit)]

    def _recent_sessions(self) -> str:
        out: list[str] = []
        for session in self._store.list_sessions(limit=self._config.recent_session_limit):
            sid = str(session.get("id") or "")
            title = _one_line(str(session.get("title") or "Untitled"), 120)
            out.append(f"- Session: {title}")
            messages = self._store.list_messages(sid)[-self._config.messages_per_session :]
            for message in messages:
                role = str(message.get("role") or "message")[:20]
                content = _one_line(str(message.get("content") or ""), 500)
                if content:
                    out.append(f"  - {role}: {content}")
        return "\n".join(out) if out else "(none)"

    def _message_lines(self, session_id: str) -> str:
        lines: list[str] = []
        messages = self._store.list_messages(session_id)[-self._config.messages_per_session :]
        for message in messages:
            role = str(message.get("role") or "message")[:20]
            content = _one_line(str(message.get("content") or ""), 700)
            if content:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines) if lines else "(none)"

    def _turn_messages(self, session_id: str, *, run_id: str = "") -> list[JsonDict]:
        messages = self._store.list_messages(session_id)
        if not run_id:
            return messages[-4:]
        return [
            message
            for message in messages
            if isinstance(message.get("metadata"), dict)
            and str(message["metadata"].get("run_id") or "") == run_id
        ]

    def _turn_message_lines(self, session_id: str, *, run_id: str = "") -> str:
        lines: list[str] = []
        messages = self._turn_messages(session_id, run_id=run_id)
        for message in messages:
            role = str(message.get("role") or "message")[:20]
            content = _one_line(str(message.get("content") or ""), 700)
            if content:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines) if lines else "(none)"

    def _memory_lines(self, namespace: str, scope_key: str) -> str:
        rows = self._store.list_memories(
            namespace=namespace,
            scope_key=scope_key,
            limit=self._config.existing_memory_limit,
        )
        lines = [_one_line(str(row.get("content") or ""), 500) for row in rows]
        lines = [line for line in lines if line]
        return "\n".join(f"- {line}" for line in lines) if lines else "(none)"

    def _report_lines(self) -> str:
        lines: list[str] = []
        for row in self._store.list_reports(limit=self._config.report_limit):
            title = _one_line(str(row.get("title") or "Untitled report"), 120)
            preview = _one_line(str(row.get("body_preview") or ""), 400)
            lines.append(f"- {title}: {preview}" if preview else f"- {title}")
        return "\n".join(lines) if lines else "(none)"


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise MemoryDistillationCancelled("Memory Dream cancellation requested")


def _failure_record(error: Exception, *, stage: str, session_id: str | None = None) -> JsonDict:
    if isinstance(error, MemoryDistillationError):
        failed_stage = error.failed_stage or stage
        code = error.error_code
        failed_session_id = error.failed_session_id or session_id
        diagnostics = error.diagnostics
    elif isinstance(error, BaselineError):
        failed_stage = stage
        code = getattr(error.code, "value", str(error.code))
        failed_session_id = session_id
        diagnostics = {"validation_error": str(code)}
    else:
        failed_stage = stage
        code = "MEMORY_DISTILLATION_FAILED"
        failed_session_id = session_id
        diagnostics = {"validation_error": "MEMORY_DISTILLATION_FAILED"}
    record: JsonDict = {"failed_stage": failed_stage, "error_code": str(code)}
    if failed_session_id:
        record["failed_session_id"] = failed_session_id
    record.update(_diagnostic_metadata(diagnostics))
    return record


def _response_format_unsupported(error: BaselineError) -> bool:
    text = str(error).lower()
    return "response_format" in text or "json_schema" in text or "json_object" in text or "structured output" in text


def _diagnostic_metadata(raw: JsonDict | None) -> JsonDict:
    if not isinstance(raw, dict):
        return {}
    out: JsonDict = {}
    for key in ("response_shape", "finish_reason", "validation_error", "response_empty", "attempts", "structured_strategy"):
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def _incomplete_reasons(result: JsonDict) -> list[str]:
    reasons: list[str] = []
    if int(result.get("failed_sessions") or 0) > 0:
        reasons.append("session_failures")
    if bool(result.get("user_profile_failed")):
        reasons.append("user_profile_failed")
    validation_error = str(result.get("validation_error") or "")
    if validation_error in {"TRUNCATED_OUTPUT", "EMPTY_CONTENT", "SCHEMA_VALIDATION_FAILED", "JSON_PARSE_FAILED"}:
        reasons.append(validation_error.lower())
    return reasons


def _fact_content_hash(*, session_id: str, run_id: str, content: str, kind: str) -> str:
    digest = hashlib.sha256()
    for value in (session_id, run_id, kind, content):
        digest.update(str(value or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]

def _render_user_profile(sections: list[JsonDict]) -> str:
    parts: list[str] = []
    for section in sections:
        title = _clean_text(str(section.get("title") or ""), max_chars=80)
        body = _clean_text(str(section.get("body") or ""), max_chars=900)
        if title and body:
            parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts).strip()


def _clean_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\t\r\n")
    return cleaned[:max_chars].strip()


def _section_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "section"


def _one_line(text: str, max_chars: int) -> str:
    return _truncate(re.sub(r"\s+", " ", text).strip(), max_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
