"""Context injection budget config: resolution, share->chars, and recommend (DR-ITER-057)."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.context_budget import (
    DEFAULT_MAX_CHARS,
    PRESET_CHARS,
    ContextBudgetConfig,
    coerce_payload,
    default_config,
    from_env,
    from_preset,
    recommend,
    resolve_context_budget,
)


def test_balanced_preset_matches_t5_defaults():
    cfg = default_config()
    assert cfg.preset == "balanced"
    assert cfg.caps() == (3500, 1500, 1000)
    assert cfg.max_chars == DEFAULT_MAX_CHARS


@pytest.mark.parametrize("preset", list(PRESET_CHARS))
def test_every_preset_caps_sum_to_max_chars(preset):
    cfg = from_preset(preset)
    assert sum(cfg.caps()) == cfg.max_chars == DEFAULT_MAX_CHARS


def test_from_preset_scales_to_custom_max_chars():
    cfg = from_preset("balanced", max_chars=3000)
    assert sum(cfg.caps()) == 3000
    # Remainder folded into recent (highest-priority block).
    assert cfg.recent_chars >= cfg.semantic_chars


def test_coerce_payload_named_preset():
    cfg = coerce_payload({"preset": "memory_rich"})
    assert cfg.preset == "memory_rich"
    assert cfg.caps() == PRESET_CHARS["memory_rich"]


def test_coerce_payload_custom_shares_to_caps_with_remainder_on_recent():
    cfg = coerce_payload(
        {"recent_share": 50, "semantic_share": 30, "memory_share": 20, "max_chars": 1000}
    )
    assert cfg.preset == "custom"
    assert cfg.caps() == (500, 300, 200)
    assert sum(cfg.caps()) == 1000


def test_coerce_payload_share_remainder_goes_to_recent():
    # 33/33/34 of 1000 -> semantic=330, memory=340, recent=remainder 330.
    cfg = coerce_payload({"recent_share": 33, "semantic_share": 33, "memory_share": 34, "max_chars": 1000})
    assert cfg.semantic_chars == 330
    assert cfg.memory_chars == 340
    assert cfg.recent_chars == 1000 - 330 - 340
    assert sum(cfg.caps()) == 1000


def test_coerce_payload_explicit_caps_round_trip():
    original = from_preset("long_thread")
    cfg = coerce_payload(original.to_dict())
    assert cfg.caps() == original.caps()


def test_coerce_payload_rejects_unknown_preset():
    with pytest.raises(ValueError):
        coerce_payload({"preset": "turbo"})


def test_coerce_payload_rejects_shares_not_summing_to_100():
    with pytest.raises(ValueError):
        coerce_payload({"recent_share": 50, "semantic_share": 20, "memory_share": 10})


def test_coerce_payload_accepts_share_tolerance():
    # 99 and 101 are within +/-1 tolerance.
    assert coerce_payload({"recent_share": 60, "semantic_share": 25, "memory_share": 14})
    assert coerce_payload({"recent_share": 60, "semantic_share": 25, "memory_share": 16})


def test_coerce_payload_rejects_zero_recent_share():
    with pytest.raises(ValueError):
        coerce_payload({"recent_share": 0, "semantic_share": 50, "memory_share": 50})


# --- resolution precedence ----------------------------------------------------


def test_resolve_prefers_run_override():
    saved = from_preset("memory_rich")
    cfg = resolve_context_budget({"preset": "followup"}, saved, {})
    assert cfg.preset == "followup"


def test_resolve_falls_back_to_saved():
    saved = from_preset("long_thread")
    cfg = resolve_context_budget(None, saved, {})
    assert cfg.preset == "long_thread"


def test_resolve_falls_back_to_env():
    env = {"DEEPRESEARCH_CONTEXT_RECENT_CHARS": "2000", "DEEPRESEARCH_CONTEXT_SEMANTIC_CHARS": "1000"}
    cfg = resolve_context_budget(None, None, env)
    assert cfg.recent_chars == 2000
    assert cfg.semantic_chars == 1000


def test_resolve_falls_back_to_default():
    cfg = resolve_context_budget(None, None, {})
    assert cfg.preset == "balanced"


def test_from_env_returns_none_when_unset():
    assert from_env({}) is None


# --- heuristic recommendation -------------------------------------------------


def test_recommend_no_embedder_zeros_semantic():
    out = recommend(
        embed_ready=False, workspace_memory_count=0, session_turn_count=2, memory_policy="off"
    )
    rec = out["recommendation"]
    assert rec["semantic_chars"] == 0
    assert out["reasons"]


def test_recommend_memory_rich_when_many_memories():
    out = recommend(
        embed_ready=True, workspace_memory_count=8, session_turn_count=2, memory_policy="workspace"
    )
    assert out["recommendation"]["preset"] == "memory_rich"


def test_recommend_long_thread_for_long_session():
    out = recommend(
        embed_ready=True, workspace_memory_count=0, session_turn_count=20, memory_policy="off"
    )
    assert out["recommendation"]["preset"] == "long_thread"


def test_recommend_defaults_to_balanced():
    out = recommend(
        embed_ready=True, workspace_memory_count=0, session_turn_count=2, memory_policy="off"
    )
    assert out["recommendation"]["preset"] == "balanced"


def test_shares_sum_to_about_100():
    for preset in PRESET_CHARS:
        cfg = from_preset(preset)
        recent, semantic, memory = cfg.shares()
        assert 99 <= recent + semantic + memory <= 101


def test_to_dict_exposes_caps_and_shares():
    cfg = ContextBudgetConfig()
    d = cfg.to_dict()
    assert d["recent_chars"] == 3500
    assert "recent_share" in d
    assert "semantic_share" in d
    assert "memory_share" in d
