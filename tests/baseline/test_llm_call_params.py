"""Resolution + payload mapping for effective per-call LLM parameters (DR-ITER-049)."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.llm_call_params import (
    DEFAULT_TEMPERATURE,
    budget_to_effort,
    clamp_temperature,
    effort_to_budget,
    normalize_effort,
    resolve_llm_call_params,
    thinking_payload_kwargs,
)
from deepresearch_core.providers_registry import ModelProfile


# --- effort <-> budget mapping ------------------------------------------------


@pytest.mark.parametrize(
    "effort,budget",
    [("off", None), ("low", 1024), ("medium", 4096), ("high", 8192)],
)
def test_effort_to_budget(effort, budget):
    assert effort_to_budget(effort) == budget


@pytest.mark.parametrize(
    "budget,effort",
    [(None, "off"), (0, "off"), (512, "low"), (1024, "low"), (4096, "medium"), (9000, "high")],
)
def test_budget_to_effort(budget, effort):
    assert budget_to_effort(budget) == effort


def test_normalize_effort_accepts_known_and_rejects_unknown():
    assert normalize_effort(None) is None
    assert normalize_effort("") is None
    assert normalize_effort(" High ") == "high"
    with pytest.raises(ValueError):
        normalize_effort("turbo")


def test_clamp_temperature():
    assert clamp_temperature(None) is None
    assert clamp_temperature("") is None
    assert clamp_temperature("abc") is None
    assert clamp_temperature(-1) == 0.0
    assert clamp_temperature(5) == 2.0
    assert clamp_temperature(0.7) == 0.7


# --- priority chain -----------------------------------------------------------


def test_resolve_falls_back_to_system_defaults_without_model():
    params = resolve_llm_call_params(None)
    assert params.temperature == DEFAULT_TEMPERATURE
    assert params.thinking_budget is None


def test_resolve_uses_model_defaults_when_run_absent():
    model = ModelProfile(id="m", temperature=0.9, thinking_budget=4096)
    params = resolve_llm_call_params(model)
    assert params.temperature == 0.9
    assert params.thinking_budget == 4096
    assert params.effort == "medium"


def test_run_overrides_take_precedence_over_model():
    model = ModelProfile(id="m", temperature=0.9, thinking_budget=4096)
    params = resolve_llm_call_params(model, run_temperature=0.2, run_thinking_effort="high")
    assert params.temperature == 0.2
    assert params.thinking_budget == 8192


def test_run_off_explicitly_disables_thinking():
    model = ModelProfile(id="m", thinking_budget=8192)
    params = resolve_llm_call_params(model, run_thinking_effort="off")
    assert params.thinking_budget is None


# --- payload emission ---------------------------------------------------------


def test_thinking_payload_for_openai():
    assert thinking_payload_kwargs("openai", 4096) == {"reasoning_effort": "medium"}


def test_thinking_payload_skipped_for_ollama():
    assert thinking_payload_kwargs("ollama", 8192) == {}


def test_thinking_payload_empty_when_no_budget():
    assert thinking_payload_kwargs("openai", None) == {}
    assert thinking_payload_kwargs("openai", 0) == {}
