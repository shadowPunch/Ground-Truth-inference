from pathlib import Path

import pytest

from biasneut.eval.lexicon import LexiconBiasScorer, load_lexicons

LEXICON_DIR = Path(__file__).parent.parent / "data" / "lexicons"


@pytest.mark.skipif(not LEXICON_DIR.exists(), reason="Pryzant lexicon files not vendored in this checkout")
def test_load_lexicons_reads_real_files():
    lexicons = load_lexicons(LEXICON_DIR)
    assert "hedges" in lexicons
    assert "factives" in lexicons
    assert len(lexicons["hedges"]) > 0


@pytest.mark.skipif(not LEXICON_DIR.exists(), reason="Pryzant lexicon files not vendored in this checkout")
def test_lexicon_bias_scorer_scores_higher_for_loaded_language():
    scorer = LexiconBiasScorer(LEXICON_DIR)
    neutral = scorer.score("the committee met on friday to discuss the budget")
    loaded = scorer.score("critics claim the committee allegedly refused to discuss the budget")
    assert loaded >= neutral


def test_lexicon_bias_scorer_empty_sentence(tmp_path):
    (tmp_path / "hedges_hyland2005.txt").write_text("somewhat\nperhaps\n")
    scorer = LexiconBiasScorer(tmp_path)
    assert scorer.score("") == 0.0


def test_lexicon_bias_scorer_toy_lexicon(tmp_path):
    (tmp_path / "factives_hooper1975.txt").write_text("realize\nknow\n")
    scorer = LexiconBiasScorer(tmp_path)
    score = scorer.score("they realize the plan failed")
    assert score == pytest.approx(1 / 5)
