"""Absolute-deadline helpers shared by provider transports and run orchestration."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

from .errors import BaselineError, ErrorCode


Clock = Callable[[], float]
InterruptCheck = Callable[[], None]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class ProviderCallBudget:
    """A monotonic deadline and retry budget for one provider call chain.

    ``request_deadline`` and ``phase_deadline`` are monotonic timestamps.
    Epoch deadlines are converted by the worker/orchestrator before constructing this
    object. A non-positive deadline is treated as unbounded for legacy call sites.
    """

    request_deadline: float = 0.0
    phase_deadline: float = 0.0
    writer_reserve_s: float = 0.0
    finalize_reserve_s: float = 0.0
    cleanup_reserve_s: float = 0.0
    attempt: int = 0
    max_attempts: int = 1
    cancel_or_timeout_check: InterruptCheck | None = field(default=None, repr=False, compare=False)
    clock: Clock = field(default=time.monotonic, repr=False, compare=False)

    @property
    def request_deadline_at(self) -> float:
        """Compatibility alias for the earlier ``*_at`` terminology."""

        return self.request_deadline

    @property
    def phase_deadline_at(self) -> float:
        """Compatibility alias for the earlier ``*_at`` terminology."""

        return self.phase_deadline

    @property
    def attempt_index(self) -> int:
        """Compatibility alias for the zero-based provider attempt."""

        return self.attempt

    @property
    def cleanup_grace_s(self) -> float:
        """Compatibility alias; public contracts use ``cleanup_reserve_s``."""

        return self.cleanup_reserve_s

    @property
    def downstream_reserve_s(self) -> float:
        """Return all configured downstream reserves."""

        return max(0.0, self.writer_reserve_s) + max(0.0, self.finalize_reserve_s) + max(
            0.0, self.cleanup_reserve_s
        )

    def effective_deadline_at(self) -> float:
        """Return the earliest configured deadline, or infinity when unbounded."""

        deadlines = [
            value
            for value in (self.request_deadline, self.phase_deadline)
            if value > 0
        ]
        return min(deadlines) if deadlines else math.inf

    def remaining_s(self, *, reserve_s: float = 0.0) -> float:
        """Return usable seconds after subtracting an optional downstream reserve."""

        deadline = self.effective_deadline_at()
        if math.isinf(deadline):
            return math.inf
        return max(0.0, deadline - float(self.clock()) - max(0.0, reserve_s))

    def check(self, stage: str = "provider_call", *, reserve_s: float = 0.0) -> None:
        """Run the cancellation probe and raise ``RUN_TIMEOUT`` when exhausted."""

        # Cancellation/terminal checks deliberately run first so cancellation wins a
        # race with the absolute deadline.
        if self.cancel_or_timeout_check is not None:
            self.cancel_or_timeout_check()
        if self.remaining_s(reserve_s=reserve_s) <= 0:
            raise BaselineError(
                ErrorCode.RUN_TIMEOUT,
                f"Run deadline was exceeded during {stage}.",
            )

    def call_timeout_s(self, cap_s: float, *, reserve_s: float = 0.0) -> float:
        """Return a positive transport timeout bounded by the remaining deadline."""

        self.check("provider_call_start", reserve_s=reserve_s)
        remaining = self.remaining_s(reserve_s=reserve_s)
        cap = float(cap_s)
        if cap <= 0:
            return remaining
        return min(cap, remaining)

    def retry_allowed(self, next_delay_s: float = 0.0, *, reserve_s: float = 0.0) -> bool:
        """Return whether another attempt and its delay fit the call budget."""

        if self.attempt + 1 >= max(1, int(self.max_attempts)):
            return False
        needed = max(0.0, next_delay_s) + max(0.0, reserve_s)
        return self.remaining_s() > needed

    def sleep(
        self,
        delay_s: float,
        *,
        sleeper: Sleeper = time.sleep,
        stage: str = "provider_retry_backoff",
        reserve_s: float = 0.0,
    ) -> None:
        """Sleep only when the full delay fits, checking both boundaries."""

        delay = max(0.0, float(delay_s))
        self.check(stage, reserve_s=delay + max(0.0, reserve_s))
        if delay:
            sleeper(delay)
        self.check(stage, reserve_s=reserve_s)

    def for_attempt(self, attempt_index: int) -> "ProviderCallBudget":
        """Return this budget with a new zero-based attempt index."""

        return replace(self, attempt=max(0, int(attempt_index)))

    def with_timeout_cap(self, cap_s: float) -> "ProviderCallBudget":
        """Return a single-attempt child budget capped from the current instant.

        Optional enrichment may improve retrieval, but it must not consume the
        full research phase deadline. The absolute request boundary is retained
        while the child phase boundary is shortened.
        """

        cap_deadline = float(self.clock()) + max(0.0, float(cap_s))
        return replace(
            self,
            phase_deadline=min(self.effective_deadline_at(), cap_deadline),
            attempt=0,
            max_attempts=1,
        )


_CURRENT_PROVIDER_CALL_BUDGET: ContextVar[ProviderCallBudget | None] = ContextVar(
    "deepresearch_provider_call_budget",
    default=None,
)


@contextmanager
def use_provider_call_budget(budget: ProviderCallBudget | None) -> Iterator[ProviderCallBudget | None]:
    """Install a provider call budget for the current thread/task context."""

    token = _CURRENT_PROVIDER_CALL_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _CURRENT_PROVIDER_CALL_BUDGET.reset(token)


def current_provider_call_budget() -> ProviderCallBudget | None:
    """Return the provider budget active in the current context, if any."""

    return _CURRENT_PROVIDER_CALL_BUDGET.get()
