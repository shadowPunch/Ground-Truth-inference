"""Whitespace tokenization + BIO tagging shared by BABE and BASIL loaders.

Both source datasets give us *words* that are biased, not char offsets into a
canonical tokenization, so we standardize on simple whitespace tokenization
for BIO tags at the data layer; the detector's collation step (§data/collate)
re-aligns these to subword tokens at train time.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")


def whitespace_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _normalize(tok: str) -> str:
    return tok.strip('"“”‘’\'.,!?;:()[]').lower()


def bio_tags_for_words(tokens: list[str], biased_words: list[str]) -> list[str]:
    """Tag each whitespace token O/B-BIAS/I-BIAS by matching against a bag of
    biased surface words (BABE's ``biased_words`` list). Contiguous runs of
    matched tokens become a single B-/I- span; this is an approximation since
    BABE gives an unordered bag of words rather than spans, but it is the best
    signal available and matches how Pryzant-style detectors consume BABE.
    """
    normalized_targets = {_normalize(w) for w in biased_words if w.strip()}
    tags = ["O"] * len(tokens)
    if not normalized_targets:
        return tags

    in_span = False
    for i, tok in enumerate(tokens):
        if _normalize(tok) in normalized_targets:
            tags[i] = "I-BIAS" if in_span else "B-BIAS"
            in_span = True
        else:
            in_span = False
    return tags


def bio_tags_for_char_span(tokens: list[str], text: str, span_text: str) -> list[str]:
    """Tag tokens covered by the first occurrence of ``span_text`` inside
    ``text`` (BASIL's phrase-level ``txt`` annotations)."""
    tags = ["O"] * len(tokens)
    idx = text.find(span_text)
    if idx == -1 or not span_text.strip():
        return tags

    span_end = idx + len(span_text)
    cursor = 0
    in_span = False
    for i, tok in enumerate(tokens):
        start = text.find(tok, cursor)
        if start == -1:
            start = cursor
        end = start + len(tok)
        cursor = end
        overlaps = start < span_end and end > idx
        if overlaps:
            tags[i] = "I-BIAS" if in_span else "B-BIAS"
            in_span = True
        else:
            in_span = False
    return tags


def merge_bio_tags(a: list[str], b: list[str]) -> list[str]:
    """OR two BIO taggings of the same token sequence together (used when a
    sentence has multiple, possibly overlapping, biased spans)."""
    out = []
    in_span = False
    for ta, tb in zip(a, b):
        biased = ta != "O" or tb != "O"
        if biased:
            out.append("I-BIAS" if in_span else "B-BIAS")
        else:
            out.append("O")
        in_span = biased
    return out
