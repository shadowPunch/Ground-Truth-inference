"""Integration tests against real (but small) external libraries. Slower and
network-dependent — run these separately from the fast unit tests if you're
iterating (``pytest -m "not integration"`` to skip)."""
import pytest

from biasneut.eval.preservation import PreservationScorer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scorer():
    return PreservationScorer(
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        bertscore_model="distilbert-base-uncased",
    )


def test_preservation_identical_sentences_score_near_perfect(scorer):
    sentences = ["the committee met on friday", "a calm discussion occurred"]
    result = scorer.score(sentences, sentences)
    assert all(f1 > 0.95 for f1 in result.bertscore_f1)
    assert all(cos > 0.95 for cos in result.sbert_cosine)
    assert result.bleu > 90


def test_preservation_unrelated_sentences_score_lower(scorer):
    sources = ["the committee met on friday to discuss the budget"]
    outputs = ["bananas are a good source of potassium"]
    result = scorer.score(sources, outputs)
    assert result.sbert_cosine[0] < 0.5
