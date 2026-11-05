"""DR-ITER-072: Deep Research state machine primitives."""

from __future__ import annotations

from deepresearch_core.baseline.research_state import ResearchState


def test_research_state_tracks_subtasks_budget_and_matrix() -> None:
    state = ResearchState.from_plan(["Alpha", "Beta"], max_searches=2)

    assert state.next_subtask() == "Alpha"
    state.add_evidence(
        "Alpha",
        queries=["Alpha"],
        notes="Alpha notes",
        source_count=3,
        status="researched",
    )

    assert state.searches_used == 1
    assert state.next_subtask() == "Beta"
    assert state.evidence_matrix[0].subtask == "Alpha"
    assert state.evidence_matrix[0].source_count == 3


def test_research_state_dedups_gaps_and_stops_at_budget() -> None:
    state = ResearchState.from_plan(["Alpha"], max_searches=2)
    state.add_evidence("Alpha", queries=["Alpha"], notes="Alpha notes", source_count=1)

    added = state.apply_gap_analysis(["Beta", "Beta", "alpha", ""], max_new=3)

    assert added == ["Beta"]
    assert state.next_subtask() == "Beta"
    state.add_evidence("Beta", queries=["Beta"], notes="Beta notes", source_count=1, status="gap_closed")
    assert state.budget_exhausted()
    assert state.done()


def test_research_state_serializes_evidence_matrix() -> None:
    state = ResearchState.from_plan(["Alpha"], max_searches=1)
    state.add_evidence("Alpha", queries=["Alpha"], notes="Alpha notes", gaps=["Beta"], source_count=2)

    payload = state.to_dict()

    assert payload["plan"] == ["Alpha"]
    assert payload["searches_used"] == 1
    assert payload["evidence_matrix"] == [
        {
            "subtask": "Alpha",
            "queries": ["Alpha"],
            "notes": "Alpha notes",
            "gaps": ["Beta"],
            "source_count": 2,
            "status": "researched",
        }
    ]
