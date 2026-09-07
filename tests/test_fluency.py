import pytest

from biasneut.eval.fluency import FluencyScorer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scorer():
    return FluencyScorer(lm_name="gpt2", grammaticality_model="textattack/roberta-base-CoLA")


def test_fluent_sentence_has_lower_perplexity_than_word_salad(scorer):
    fluent = "The committee met on Friday to discuss the annual budget."
    salad = "Friday budget committee annual discuss the on met the."
    assert scorer.perplexity(fluent) < scorer.perplexity(salad)


def test_grammatical_prob_in_range(scorer):
    prob = scorer.grammatical_prob("The committee met on Friday.")
    assert 0.0 <= prob <= 1.0
