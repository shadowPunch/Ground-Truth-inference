"""Span marking: the interface between Stage-1 localization and Stage-2
editing. The editor always sees its input as a sentence with the flagged
span wrapped in marker tokens — whether that span came from a gold
biased/neutral diff (training, Strategies A/B) or from the Stage-1 detector's
predictions (inference, and Strategy C's mask-and-infill).
"""
from __future__ import annotations

import difflib

from biasneut.data.tokenize_utils import whitespace_tokenize


def diff_word_tags(source: str, target: str) -> list[str]:
    """BIO-tag source words that were deleted or replaced going source->target.

    This is how Strategy A/B training pairs (which have a real target but no
    explicit span annotation) get span supervision for the editor: the
    surface diff of a real biased->neutral rewrite *is* the span that
    mattered, which is exactly Pryzant's own single-word-subset framing.
    """
    src_words = whitespace_tokenize(source)
    tgt_words = whitespace_tokenize(target)
    matcher = difflib.SequenceMatcher(a=src_words, b=tgt_words)

    tags = ["O"] * len(src_words)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            for i in range(i1, i2):
                tags[i] = "I-BIAS" if i > i1 else "B-BIAS"
    return tags


def mark_span_from_tags(
    words: list[str],
    word_tags: list[str],
    span_start: str = "<bias>",
    span_end: str = "</bias>",
) -> str:
    """Wrap contiguous B-/I-BIAS runs in ``words`` with marker tokens."""
    out: list[str] = []
    in_span = False
    for word, tag in zip(words, word_tags):
        is_biased = tag != "O"
        if is_biased and not in_span:
            out.append(span_start)
        if not is_biased and in_span:
            out.append(span_end)
        out.append(word)
        in_span = is_biased
    if in_span:
        out.append(span_end)
    return " ".join(out)


def mark_text(text: str, word_tags: list[str], span_start: str = "<bias>", span_end: str = "</bias>") -> str:
    return mark_span_from_tags(whitespace_tokenize(text), word_tags, span_start, span_end)
