"""Sentence-level and BIO span-level metrics for the detector.

Deliberately dependency-free (no seqeval) — span extraction and P/R/F1 are
~20 lines and this is the only place we need them.
"""
from __future__ import annotations

from biasneut.data.collate import ID2TAG


def sentence_prf1(preds: list[int], golds: list[int]) -> dict[str, float]:
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(1 for p, g in zip(preds, golds) if p == g) / len(preds) if preds else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def extract_spans(tag_ids: list[int]) -> set[tuple[int, int]]:
    """Extract (start, end_exclusive) index pairs for each B-/I- run."""
    spans = set()
    start = None
    for i, tag_id in enumerate(tag_ids + [-100]):
        tag = ID2TAG.get(tag_id, "O") if tag_id != -100 else "O"
        if tag == "B-BIAS":
            if start is not None:
                spans.add((start, i))
            start = i
        elif tag == "I-BIAS":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.add((start, i))
                start = None
    return spans


def span_prf1(pred_tags: list[list[int]], gold_tags: list[list[int]]) -> dict[str, float]:
    """Exact-match span F1, ignoring positions labeled -100 in the gold (they
    contribute no gold spans there, matching standard NER eval conventions)."""
    tp = fp = fn = 0
    for preds, golds in zip(pred_tags, gold_tags):
        # Mask predictions at ignored gold positions so padding/continuation
        # subwords never manufacture spurious spans.
        masked_preds = [p if g != -100 else -100 for p, g in zip(preds, golds)]
        pred_spans = extract_spans(masked_preds)
        gold_spans = extract_spans(golds)
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
