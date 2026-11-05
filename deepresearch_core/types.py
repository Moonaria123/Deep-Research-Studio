from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchRequest:
    """Normalized research request consumed by the local research core."""

    user_query: str
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    previous_search_context: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Return the existing workflow payload shape."""

        return {
            "user_query": self.user_query,
            "chat_history": self.chat_history,
            "previous_search_context": self.previous_search_context,
        }
