"""Deep Research state and evidence matrix primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceMatrixRow:
    """One researched Deep subtask and its evidence summary."""

    subtask: str
    queries: list[str]
    notes: str
    gaps: list[str] = field(default_factory=list)
    source_count: int = 0
    status: str = "researched"

    def to_dict(self) -> dict:
        """Return a JSON-serializable row."""

        return {
            "subtask": self.subtask,
            "queries": list(self.queries),
            "notes": self.notes,
            "gaps": list(self.gaps),
            "source_count": self.source_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceMatrixRow":
        """Restore one allowlisted row from a checkpoint payload."""

        if not isinstance(value, dict):
            raise ValueError("evidence matrix row must be an object")
        return cls(
            subtask=str(value.get("subtask") or "").strip(),
            queries=[str(item) for item in (value.get("queries") or [])],
            notes=str(value.get("notes") or ""),
            gaps=[str(item) for item in (value.get("gaps") or [])],
            source_count=max(0, int(value.get("source_count") or 0)),
            status=str(value.get("status") or "researched"),
        )


@dataclass
class ResearchState:
    """Mutable Deep Research progress snapshot persisted in run results."""

    plan: list[str]
    max_searches: int
    evidence_matrix: list[EvidenceMatrixRow] = field(default_factory=list)
    searches_used: int = 0
    pages_fetched: int = 0
    think_tokens_used: int = 0
    gap_rounds: int = 0
    _pending: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: list[str], *, max_searches: int) -> "ResearchState":
        """Create an initial state from a planned subtask list."""

        cleaned = [item.strip() for item in plan if item.strip()]
        return cls(plan=cleaned, max_searches=max(1, max_searches), _pending=list(cleaned))

    def next_subtask(self) -> str | None:
        """Return the next subtask, or None when done."""

        if self.budget_exhausted() or not self._pending:
            return None
        return self._pending[0]

    def add_evidence(
        self,
        subtask: str,
        *,
        queries: list[str],
        notes: str,
        source_count: int,
        gaps: list[str] | None = None,
        status: str = "researched",
        think_tokens: int = 0,
    ) -> None:
        """Record evidence for one subtask and advance counters."""

        normalized = subtask.strip()
        if self._pending and self._pending[0].lower() == normalized.lower():
            self._pending.pop(0)
        elif normalized in self._pending:
            self._pending.remove(normalized)
        self.searches_used += 1
        self.pages_fetched += max(0, int(source_count))
        self.think_tokens_used += max(0, int(think_tokens))
        if status == "gap_closed":
            self.gap_rounds += 1
        self.evidence_matrix.append(
            EvidenceMatrixRow(
                subtask=normalized,
                queries=list(queries),
                notes=notes,
                gaps=list(gaps or []),
                source_count=max(0, int(source_count)),
                status=status,
            )
        )

    def apply_gap_analysis(self, gaps: list[str], *, max_new: int = 3) -> list[str]:
        """Add deduplicated gaps to the pending plan within remaining budget."""

        existing = {item.lower() for item in self.plan}
        existing.update(row.subtask.lower() for row in self.evidence_matrix)
        existing.update(item.lower() for item in self._pending)
        remaining = max(0, self.max_searches - self.searches_used - len(self._pending))
        limit = min(max_new, remaining)
        added: list[str] = []
        for gap in gaps:
            cleaned = " ".join(str(gap or "").split())
            key = cleaned.lower()
            if not cleaned or key in existing:
                continue
            self.plan.append(cleaned)
            self._pending.append(cleaned)
            existing.add(key)
            added.append(cleaned)
            if len(added) >= limit:
                break
        if added and self.evidence_matrix:
            self.evidence_matrix[-1].gaps = added
        return added

    def budget_exhausted(self) -> bool:
        """Return whether no more search rounds can be used."""

        return self.searches_used >= self.max_searches

    def done(self) -> bool:
        """Return whether the state has no more executable work."""

        return self.budget_exhausted() or not self._pending

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot."""

        return {
            "plan": list(self.plan),
            "evidence_matrix": [row.to_dict() for row in self.evidence_matrix],
            "searches_used": self.searches_used,
            "searches_max": self.max_searches,
            "pages_fetched": self.pages_fetched,
            "think_tokens_used": self.think_tokens_used,
            "gap_rounds": self.gap_rounds,
        }

    def to_checkpoint(self) -> dict:
        """Return complete resumable state, including the private pending queue."""

        return {
            **self.to_dict(),
            "pending": list(self._pending),
        }

    @classmethod
    def from_checkpoint(cls, value: object) -> "ResearchState":
        """Restore counters and pending order from a validated checkpoint object."""

        if not isinstance(value, dict):
            raise ValueError("research state checkpoint must be an object")
        plan = [str(item).strip() for item in (value.get("plan") or []) if str(item).strip()]
        pending = [
            str(item).strip()
            for item in (value.get("pending") or [])
            if str(item).strip()
        ]
        max_searches = max(1, int(value.get("searches_max") or 1))
        searches_used = max(0, int(value.get("searches_used") or 0))
        if searches_used > max_searches:
            raise ValueError("research state searches_used exceeds searches_max")
        if any(item not in plan for item in pending):
            raise ValueError("research state pending task is not present in plan")
        return cls(
            plan=plan,
            max_searches=max_searches,
            evidence_matrix=[
                EvidenceMatrixRow.from_dict(item)
                for item in (value.get("evidence_matrix") or [])
            ],
            searches_used=searches_used,
            pages_fetched=max(0, int(value.get("pages_fetched") or 0)),
            think_tokens_used=max(0, int(value.get("think_tokens_used") or 0)),
            gap_rounds=max(0, int(value.get("gap_rounds") or 0)),
            _pending=pending,
        )
