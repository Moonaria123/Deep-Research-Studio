"""Budget presets + RunRequest normalization."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.budget import Budget, Mode, RunRequest, preset_for
from deepresearch_core.baseline.errors import BaselineError, ErrorCode


def test_every_mode_has_a_preset():
    for mode in Mode:
        budget = preset_for(mode)
        assert isinstance(budget, Budget)
        assert budget.max_searches >= 1
        assert budget.top_k >= 1


def test_fast_is_cheaper_than_deep():
    fast, deep = preset_for(Mode.FAST), preset_for(Mode.DEEP)
    assert fast.max_searches < deep.max_searches
    assert fast.max_output_tokens < deep.max_output_tokens


def test_overrides_only_apply_known_fields():
    req = RunRequest(query="q", mode=Mode.FAST, overrides={"max_searches": 9, "bogus": 100})
    resolved = req.resolved_budget()
    assert resolved.max_searches == 9
    assert not hasattr(resolved, "bogus")


def test_from_payload_requires_query():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "  ", "mode": "fast"})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_from_payload_rejects_unknown_mode():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "hello", "mode": "turbo"})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert "allowed" in (exc.value.details or {})


def test_from_payload_maps_budget_to_overrides():
    req = RunRequest.from_payload({"query": "hi", "mode": "deep", "budget": {"top_k": 3}})
    assert req.mode is Mode.DEEP
    assert req.resolved_budget().top_k == 3


def test_from_payload_council_size_defaults_to_three():
    req = RunRequest.from_payload({"query": "hi", "mode": "council"})
    assert req.council_size == 3


def test_from_payload_council_size_clamps_to_range():
    assert RunRequest.from_payload({"query": "q", "mode": "council", "council_size": 1}).council_size == 2
    assert RunRequest.from_payload({"query": "q", "mode": "council", "council_size": 9}).council_size == 5
    assert RunRequest.from_payload({"query": "q", "mode": "council", "council_size": 4}).council_size == 4
    assert RunRequest.from_payload({"query": "q", "mode": "council", "council_size": "bad"}).council_size == 3


def test_from_payload_council_strategy_defaults_to_judge_synthesis():
    req = RunRequest.from_payload({"query": "hi", "mode": "council"})
    assert req.council_strategy == "judge_synthesis"


def test_from_payload_council_strategy_accepts_debate_once():
    req = RunRequest.from_payload({"query": "hi", "mode": "council", "council_strategy": "debate_once"})
    assert req.council_strategy == "debate_once"


def test_from_payload_council_strategy_accepts_parallel_vote():
    req = RunRequest.from_payload({"query": "hi", "mode": "council", "council_strategy": "parallel_vote"})
    assert req.council_strategy == "parallel_vote"


def test_from_payload_rejects_unknown_council_strategy():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "hi", "mode": "council", "council_strategy": "vote"})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert "allowed" in (exc.value.details or {})


def test_from_payload_memory_policy_defaults_to_off():
    req = RunRequest.from_payload({"query": "hi", "mode": "fast"})
    assert req.memory_policy == "off"
    assert req.workspace_key == "default"
    assert req.user_key == "default"


def test_from_payload_accepts_memory_policies_and_keys():
    req = RunRequest.from_payload(
        {
            "query": "hi",
            "mode": "fast",
            "memory_policy": "workspace_user",
            "workspace_key": "team-a",
            "user_key": "alice",
        }
    )
    assert req.memory_policy == "workspace_user"
    assert req.workspace_key == "team-a"
    assert req.user_key == "alice"


def test_from_payload_rejects_unknown_memory_policy():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "hi", "mode": "fast", "memory_policy": "everything"})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert "allowed" in (exc.value.details or {})


def test_memory_policy_round_trips_through_payload():
    req = RunRequest.from_payload(
        {"query": "hi", "mode": "fast", "memory_policy": "workspace", "workspace_key": "w1"}
    )
    again = RunRequest.from_payload(req.to_payload())
    assert again.memory_policy == "workspace"
    assert again.workspace_key == "w1"


def test_from_payload_library_scope_defaults_to_off():
    req = RunRequest.from_payload({"query": "hi", "mode": "fast"})
    assert req.library_scope == "off"
    assert req.to_payload()["library_scope"] == "off"


def test_library_scope_round_trips_through_payload():
    req = RunRequest.from_payload({"query": "hi", "mode": "deep", "library_scope": "all"})
    again = RunRequest.from_payload(req.to_payload())
    assert again.library_scope == "all"


def test_from_payload_rejects_unknown_library_scope():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "hi", "mode": "fast", "library_scope": "everything"})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert "allowed" in (exc.value.details or {})


def test_from_payload_defaults_context_budget_to_none():
    req = RunRequest.from_payload({"query": "hi", "mode": "fast"})
    assert req.context_budget is None
    assert req.to_payload()["context_budget"] is None


def test_context_budget_preset_round_trips_through_payload():
    req = RunRequest.from_payload(
        {"query": "hi", "mode": "fast", "context_budget": {"preset": "followup"}}
    )
    assert req.context_budget is not None
    assert req.context_budget["preset"] == "followup"
    again = RunRequest.from_payload(req.to_payload())
    assert again.context_budget is not None
    assert again.context_budget["preset"] == "followup"
    # Normalized to full caps that sum to max_chars.
    caps = req.context_budget
    assert caps["recent_chars"] + caps["semantic_chars"] + caps["memory_chars"] == caps["max_chars"]


def test_from_payload_rejects_invalid_context_budget():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload(
            {"query": "hi", "mode": "fast", "context_budget": {"preset": "nonsense"}}
        )
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_search_options_round_trips_through_payload():
    req = RunRequest.from_payload(
        {
            "query": "hi",
            "mode": "multiple",
            "search_options": {
                "include_domains": ["Example.com"],
                "exclude_domains": ["spam.example"],
                "freshness": "pd",
                "extra_subqueries": ["alpha", "Alpha", "beta"],
            },
        }
    )
    assert req.search_options is not None
    assert req.search_options.include_domains == ("example.com",)
    assert req.search_options.extra_subqueries == ("alpha", "beta")
    again = RunRequest.from_payload(req.to_payload())
    assert again.search_options == req.search_options


def test_from_payload_omits_empty_search_options():
    req = RunRequest.from_payload({"query": "hi", "mode": "fast", "search_options": {}})
    assert req.search_options is None
    assert req.to_payload()["search_options"] is None


def test_from_payload_rejects_invalid_search_options():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload({"query": "hi", "mode": "fast", "search_options": {"freshness": "forever"}})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_from_payload_rejects_context_budget_shares_not_summing_to_100():
    with pytest.raises(BaselineError) as exc:
        RunRequest.from_payload(
            {
                "query": "hi",
                "mode": "fast",
                "context_budget": {
                    "recent_share": 50,
                    "semantic_share": 20,
                    "memory_share": 10,
                },
            }
        )
    assert exc.value.code is ErrorCode.INVALID_REQUEST
