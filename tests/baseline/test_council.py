"""Models Council: parallel members, judge, confidence formula."""

from __future__ import annotations

import time

import pytest

from deepresearch_core.baseline.adapters import LLMResult
from deepresearch_core.baseline.citations import enrich_council_structure
from deepresearch_core.baseline.council import (
    CouncilConflictDirective,
    CouncilMemberSpec,
    CouncilTrustConstraints,
    build_council_trust_constraints,
    run_council,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.evidence_trust import TemporalCoverage
from deepresearch_core.baseline.retrieval import EvidenceItem

from tests.baseline.fakes import FakeLLM


_TEST_SECRET = "sk" + "-test-secret"


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=1, document_id="d1", url="https://e.com/a", title="Tides",
            text="Ocean tides are caused mainly by the gravitational pull of the moon and the sun.",
            score=0.9,
        ),
        EvidenceItem(
            chunk_id=2, document_id="d2", url="https://e.com/b", title="Spring tides",
            text="Spring tides occur when the sun moon and earth align producing larger tidal ranges.",
            score=0.6,
        ),
    ]


def test_council_returns_consensus_and_weighted_breakdown():
    members = [FakeLLM("m-a"), FakeLLM("m-b"), FakeLLM("m-c")]
    result = run_council(members, "What causes ocean tides?", _evidence(), judge=members[0])

    assert len(result.members) == 3
    assert set(result.breakdown) == {"coverage", "agreement", "citation_quality", "validator"}
    assert 0.0 <= result.confidence <= 1.0
    # Identical deterministic members => high agreement and coverage.
    assert result.breakdown["agreement"] > 0.8
    assert result.consensus.text


def test_confidence_matches_weighted_sum():
    members = [FakeLLM("m-a"), FakeLLM("m-b")]
    result = run_council(members, "What causes ocean tides?", _evidence(), judge=members[0])
    b = result.breakdown
    expected = 0.40 * b["coverage"] + 0.25 * b["agreement"] + 0.20 * b["citation_quality"] + 0.15 * b["validator"]
    assert result.confidence == round(expected, 10) or abs(result.confidence - expected) < 1e-9


class _RecordingLLM(FakeLLM):
    """FakeLLM that records the prompts it was asked, to assert perspective injection."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompts: list[str] = []
        self.message_batches: list[list[dict]] = []

    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.message_batches.append(list(messages))
        self.prompts.append(messages[-1]["content"])
        return super().complete(messages, **kwargs)


class _FixedLLM(_RecordingLLM):
    """Recording LLM that always returns a fixed draft."""

    def __init__(self, name: str, text: str) -> None:
        super().__init__(name)
        self._text = text

    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.message_batches.append(list(messages))
        self.prompts.append(messages[-1]["content"])
        return LLMResult(text=self._text, model=self.model)


class _DelayedFixedLLM(_FixedLLM):
    """Fixed member with deterministic completion order different from input order."""

    def __init__(self, name: str, text: str, delay: float) -> None:
        super().__init__(name, text)
        self._delay = delay

    def complete(self, messages, **kwargs):  # type: ignore[override]
        time.sleep(self._delay)
        return super().complete(messages, **kwargs)


class _FailingMember(FakeLLM):
    """Member that raises an unsafe provider error; no detail may cross the wire."""

    def complete(self, messages, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"provider failed with {_TEST_SECRET} and private upstream detail")


def test_council_specs_inject_perspectives_and_names():
    a, b = _RecordingLLM("model-x"), _RecordingLLM("model-x")
    specs = [
        CouncilMemberSpec(adapter=a, perspective="optimistic advocate", name="Advocate"),
        CouncilMemberSpec(adapter=b, perspective="skeptical risk analyst", name="Skeptic"),
    ]
    result = run_council(specs, "What causes ocean tides?", _evidence(), judge=a)

    names = {m.model for m in result.members}
    perspectives = {m.perspective for m in result.members}
    assert names == {"Advocate", "Skeptic"}
    assert perspectives == {"optimistic advocate", "skeptical risk analyst"}
    assert any("optimistic advocate" in p for p in a.prompts)
    assert any("skeptical risk analyst" in p for p in b.prompts)
    # Member dicts surface the perspective for the UI.
    assert all("perspective" in m.to_dict() for m in result.members)


def test_council_accepts_bare_adapters_without_perspective():
    members = [FakeLLM("m-a"), FakeLLM("m-b")]
    result = run_council(members, "What causes ocean tides?", _evidence(), judge=members[0])
    assert all(m.perspective == "" for m in result.members)


# --- DR-ITER-044: debate_once critique round -------------------------------


def _is_critique_prompt(prompt: str) -> bool:
    return "reviewing peer answers" in prompt and prompt.rstrip().endswith("Critique:")


def test_debate_once_runs_critique_round_and_feeds_judge():
    a, b = _RecordingLLM("model-a"), _RecordingLLM("model-b")
    judge = _RecordingLLM("judge")
    result = run_council(
        [a, b], "What causes ocean tides?", _evidence(), judge=judge, strategy="debate_once"
    )

    # Each member critiqued its peer, so the result carries critiques.
    assert len(result.critiques) == 2
    assert {c["model"] for c in result.critiques} == {"model-a", "model-b"}
    # Members were asked to critique (an extra prompt beyond their draft).
    assert any(_is_critique_prompt(p) for p in a.prompts)
    assert any(_is_critique_prompt(p) for p in b.prompts)
    # The judge prompt embeds the peer critiques block.
    assert any("Peer critiques" in p for p in judge.prompts)
    assert result.consensus.text


def test_trusted_constraints_reach_members_critiques_and_judge_without_free_text():
    a, b = _RecordingLLM("model-a"), _RecordingLLM("model-b")
    judge = _RecordingLLM("judge")
    constraints = CouncilTrustConstraints(
        temporal=TemporalCoverage(
            intent="latest",
            checked_at=123.0,
            status="partially_bounded",
            basis="source_published_or_updated_at",
            evidence_ids=("c1", "https://untrusted.example/private"),
            reason_codes=("missing_effective_at", "raw free-form reason"),
            statement="raw free-form disclosure must not enter the trusted block",
            note_required=True,
            supported_through=100.0,
        ),
        conflict_directives=(
            CouncilConflictDirective(
                evidence_ids=("c1", "c2", "provider-response-text"),
                resolution="contested",
                basis="peer_unresolved",
            ),
        ),
    )

    run_council(
        [a, b],
        "What causes ocean tides?",
        _evidence(),
        judge=judge,
        strategy="debate_once",
        trust_constraints=constraints,
    )

    batches = [*a.message_batches, *b.message_batches, *judge.message_batches]
    assert len(batches) == 5
    assert all([message["role"] for message in batch] == ["system", "user"] for batch in batches)
    for batch in batches:
        trusted = batch[0]["content"].split("<trusted_council_constraints>", 1)[1].split(
            "</trusted_council_constraints>",
            1,
        )[0]
        assert '"intent":"latest"' in trusted
        assert '"resolution":"contested"' in trusted
        assert '"evidence_ids":["c1","c2"]' in trusted
        assert "raw free-form disclosure" not in trusted
        assert "raw free-form reason" not in trusted
        assert "untrusted.example" not in trusted
        assert "provider-response-text" not in trusted
        assert "What causes ocean tides?" not in trusted
        assert "What causes ocean tides?" in batch[1]["content"]


def test_trust_builder_maps_temporal_evidence_and_resolved_winner_to_citation_ids():
    constraints = build_council_trust_constraints(
        TemporalCoverage(
            intent="latest",
            checked_at=123.0,
            status="bounded",
            basis="claim_effective_at",
            evidence_ids=("external:web_document:d1",),
            reason_codes=(),
            statement="not copied",
            note_required=False,
            supported_through=100.0,
        ),
        [],
        [
            {
                "evidence_ids": ["c1", "c2"],
                "resolution": "resolved",
                "basis": "authority_directness",
                "winner_evidence_id": "c1",
            }
        ],
        evidence=_evidence(),
    )

    assert tuple(constraints.temporal.evidence_ids) == ("c1",)
    assert constraints.conflict_directives == (
        CouncilConflictDirective(
            evidence_ids=("c1", "c2"),
            resolution="resolved",
            basis="authority_directness",
            winner_evidence_id="c1",
        ),
    )


def test_judge_synthesis_default_skips_critique():
    a, b = _RecordingLLM("model-a"), _RecordingLLM("model-b")
    judge = _RecordingLLM("judge")
    result = run_council([a, b], "What causes ocean tides?", _evidence(), judge=judge)

    assert result.critiques == []
    assert not any(_is_critique_prompt(p) for p in a.prompts)
    assert not any(_is_critique_prompt(p) for p in b.prompts)
    assert not any("Peer critiques" in p for p in judge.prompts)


class _FlakyCritic(_RecordingLLM):
    """Drafts normally but raises during the critique round."""

    def complete(self, messages, **kwargs):  # type: ignore[override]
        prompt = messages[-1]["content"]
        if _is_critique_prompt(prompt):
            raise RuntimeError("critique unavailable")
        return super().complete(messages, **kwargs)


def test_debate_once_degrades_when_critics_fail():
    a, b = _FlakyCritic("model-a"), _FlakyCritic("model-b")
    judge = _RecordingLLM("judge")
    result = run_council(
        [a, b], "What causes ocean tides?", _evidence(), judge=judge, strategy="debate_once"
    )

    # Failed critiques are dropped; the run still produces a grounded consensus.
    assert result.critiques == []
    assert not any("Peer critiques" in p for p in judge.prompts)
    assert result.consensus.text


def test_debate_once_single_member_has_no_critique():
    a = _RecordingLLM("solo")
    result = run_council([a], "What causes ocean tides?", _evidence(), judge=a, strategy="debate_once")
    assert result.critiques == []
    assert not any(_is_critique_prompt(p) for p in a.prompts)


# --- DR-ITER-055: parallel_vote strategy -----------------------------------


def test_parallel_vote_uses_highest_coverage_member_without_judge():
    winner_text = "Ocean tides are caused mainly by the gravitational pull of the moon and the sun. [c1]"
    loser_text = "Market forecasts and battery pricing shape electric cars. [c1]"
    winner, loser = _FixedLLM("winner", winner_text), _FixedLLM("loser", loser_text)
    judge = _RecordingLLM("judge")

    result = run_council(
        [loser, winner],
        "What causes ocean tides?",
        _evidence(),
        judge=judge,
        strategy="parallel_vote",
    )

    assert judge.prompts == []
    assert result.critiques == []
    assert result.consensus.text == winner_text
    assert result.consensus.coverage == 1.0


def test_parallel_vote_single_member_returns_that_answer():
    text = "Ocean tides are caused mainly by the gravitational pull of the moon and the sun. [c1]"
    member = _FixedLLM("solo", text)
    judge = _RecordingLLM("judge")

    result = run_council([member], "What causes ocean tides?", _evidence(), judge=judge, strategy="parallel_vote")

    assert judge.prompts == []
    assert result.critiques == []
    assert result.consensus.text == text


def test_parallel_vote_consensus_copy_keeps_member_draft_intact_after_enrich():
    winner_text = (
        "Ocean tides are caused mainly by the gravitational pull of the moon and the sun. [c1]\n\n"
        "## Limitations\n"
        "- Spring tide timing needs alignment data. [c2]"
    )
    loser_text = "Market forecasts and battery pricing shape electric cars. [c1]"
    winner, loser = _FixedLLM("winner", winner_text), _FixedLLM("loser", loser_text)

    result = run_council([winner, loser], "What causes ocean tides?", _evidence(), strategy="parallel_vote")
    member_text_before = result.members[0].answer.text

    enrich_council_structure(result.consensus)

    assert "## Limitations" not in result.consensus.text
    assert "Spring tide timing" in result.consensus.limitations
    assert result.members[0].answer.text == member_text_before
    assert "## Limitations" in result.members[0].answer.text


# --- DR-ITER-055 follow-up: on_stage emits truthful per-phase sequences -----


def test_on_stage_sequences_follow_strategy():
    expected = {
        "parallel_vote": ["drafts", "vote"],
        "judge_synthesis": ["drafts", "synthesis"],
        "debate_once": ["drafts", "critique", "synthesis"],
    }
    for strategy, stages in expected.items():
        recorded: list[str] = []
        members = [FakeLLM("m-a"), FakeLLM("m-b")]
        judge = None if strategy == "parallel_vote" else FakeLLM("judge")
        run_council(
            members, "What causes ocean tides?", _evidence(),
            judge=judge, strategy=strategy, on_stage=recorded.append,
        )
        assert recorded == stages, f"{strategy}: {recorded}"


def test_one_member_failure_is_isolated_and_success_order_is_stable():
    text = "Ocean tides are caused mainly by the gravitational pull of the moon and the sun. [c1]"
    slow = _DelayedFixedLLM("first", text, 0.02)
    failed = CouncilMemberSpec(
        adapter=_FailingMember("unsafe"),
        name=f"bad\nmember-{_TEST_SECRET} -----BEGIN PRIVATE KEY-----",
    )
    fast = _DelayedFixedLLM("third", text, 0.0)

    result = run_council([slow, failed, fast], "What causes ocean tides?", _evidence())

    assert [member.model for member in result.members] == ["first", "third"]
    assert result.consensus.text
    assert result.consensus.coverage == 1.0
    assert result.member_failures == [
        {"model": "bad member-[redacted] [redacted]", "code": "MEMBER_EXECUTION_FAILED"}
    ]
    payload = result.to_dict()
    assert payload["member_failures"] == result.member_failures
    assert "private upstream detail" not in str(payload)
    assert _TEST_SECRET not in str(payload)
    assert "PRIVATE KEY" not in str(payload)


@pytest.mark.parametrize(
    ("unsafe_label", "forbidden"),
    [
        ("Authorization: Bearer credential-value", "credential-value"),
        ('{"api_key":"credential-value"}', "credential-value"),
        ("https://alice:credential-value@example.invalid/model", "alice:credential-value"),
        (r"C:\Users\Alice\private\model.json", "Alice"),
        ("/home/alice/.config/model", "alice"),
    ],
)
def test_member_failure_labels_redact_credentials_and_host_paths(
    unsafe_label: str,
    forbidden: str,
):
    text = "Ocean tides are caused mainly by the gravitational pull of the moon and the sun. [c1]"
    failed = CouncilMemberSpec(adapter=_FailingMember("unsafe"), name=unsafe_label)

    result = run_council(
        [_FixedLLM("healthy", text), failed],
        "What causes ocean tides?",
        _evidence(),
    )

    failure = result.member_failures[0]
    assert failure["code"] == "MEMBER_EXECUTION_FAILED"
    assert forbidden not in failure["model"]
    assert len(failure["model"]) <= 80


def test_all_member_failures_raise_sanitized_upstream_error_before_synthesis():
    members = [_FailingMember("failed-a"), _FailingMember("failed-b")]
    stages: list[str] = []
    judge = _RecordingLLM("judge")
    validator = _RecordingLLM("validator")

    with pytest.raises(BaselineError) as exc:
        run_council(
            members,
            "What causes ocean tides?",
            _evidence(),
            judge=judge,
            validator=validator,
            on_stage=stages.append,
        )

    assert exc.value.code is ErrorCode.UPSTREAM_ERROR
    assert exc.value.message == "All council members failed to produce a draft."
    assert exc.value.details == {"failed_members": 2}
    assert stages == ["drafts"]
    assert judge.prompts == []
    assert validator.prompts == []
    assert "private upstream detail" not in str(exc.value.to_payload())
