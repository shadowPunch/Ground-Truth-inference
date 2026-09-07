"""Linguistic lexicon features (Recasens et al. 2013's word lists, as
released with Pryzant et al. 2020) — one of the few directly reusable
assets from the existing codebase (``neutralizing-bias-master/src/lexicons``).
Used as one signal inside the Aggregate Bias Score (§6.1), independent of any
trained classifier.
"""
from __future__ import annotations

from pathlib import Path

from biasneut.data.tokenize_utils import whitespace_tokenize

_LEXICON_FILES = {
    "assertives": "assertives_hooper1975.txt",
    "entailed_arg": "entailed_arg_berant2012.txt",
    "entailed": "entailed_berant2012.txt",
    "entailing_arg": "entailing_arg_berant2012.txt",
    "entailing": "entailing_berant2012.txt",
    "factives": "factives_hooper1975.txt",
    "hedges": "hedges_hyland2005.txt",
    "implicatives": "implicatives_karttunen1971.txt",
    "negative": "negative_liu2005.txt",
    "npov": "npov_lexicon.txt",
    "positive": "positive_liu2005.txt",
    "report_verbs": "report_verbs.txt",
    "strong_subjectives": "strong_subjectives_riloff2003.txt",
    "weak_subjectives": "weak_subjectives_riloff2003.txt",
}

# Lists that indicate loaded/subjective language, as opposed to purely
# descriptive resources (positive/negative, entailment lexicons) which we
# load for completeness but don't count toward the bias score.
_BIAS_INDICATIVE = {"assertives", "factives", "hedges", "implicatives", "strong_subjectives", "weak_subjectives"}


def load_lexicons(lexicon_dir: str | Path) -> dict[str, set[str]]:
    lexicon_dir = Path(lexicon_dir)
    lexicons = {}
    for name, filename in _LEXICON_FILES.items():
        path = lexicon_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            words = {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
        lexicons[name] = words
    return lexicons


class LexiconBiasScorer:
    def __init__(self, lexicon_dir: str | Path):
        self.lexicons = load_lexicons(lexicon_dir)
        self.bias_words: set[str] = set()
        for name in _BIAS_INDICATIVE:
            self.bias_words |= self.lexicons.get(name, set())

    def score(self, sentence: str) -> float:
        """Fraction of tokens in ``sentence`` found in a bias-indicative lexicon."""
        tokens = [t.strip('"“”‘’\'.,!?;:()[]').lower() for t in whitespace_tokenize(sentence)]
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in self.bias_words)
        return hits / len(tokens)
