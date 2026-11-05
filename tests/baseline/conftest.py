"""Shared fixtures for the baseline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.baseline.fakes import FakeLLM, FakeSearch  # noqa: E402


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_search() -> FakeSearch:
    return FakeSearch(
        {
            "https://example.com/quantum": (
                "Quantum computing uses qubits to represent superpositions of states. "
                "Error correction is essential for scalable quantum computers. "
                "Companies are racing to build fault tolerant quantum hardware."
            ),
            "https://example.com/qubits": (
                "A qubit can be in a superposition of zero and one. "
                "Decoherence limits qubit lifetime. Surface codes help correct errors."
            ),
        }
    )
