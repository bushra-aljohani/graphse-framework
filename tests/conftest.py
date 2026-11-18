import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphse.knowledge import build_graph, load_corpus, load_proposals  # noqa: E402


@pytest.fixture
def corpus():
    return load_corpus(ROOT / "examples/toy_corpus.json")


@pytest.fixture
def kg(corpus):
    return build_graph(corpus, load_proposals(ROOT / "examples/toy_proposals.json"))
