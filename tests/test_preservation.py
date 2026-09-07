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


def test_preservation_handles_empty_generation_without_crashing(scorer):
    # An undertrained seq2seq editor can degenerate to an empty string. On
    # some transformers/bert_score version combinations (hit on Kaggle: a
    # RobertaTokenizer missing build_inputs_with_special_tokens),
    # bert_score's own empty-string special case crashes with an
    # AttributeError instead of scoring it as a preservation failure.
    sources = ["the corrupt regime cracked down on protesters", "a calm meeting occurred"]
    outputs = ["", "a calm meeting occurred"]
    result = scorer.score(sources, outputs)
    assert len(result.bertscore_f1) == 2
    assert result.bertscore_f1[1] > result.bertscore_f1[0]
