"""BABE loader (§3.2, §4.1) — sentence-level bias label + a bag of biased
surface words, used for Stage-1 sentence-classification and (approximate)
span supervision.
"""
from __future__ import annotations

import ast
import logging

from biasneut.data.schema import DetectionExample
from biasneut.data.tokenize_utils import bio_tags_for_words, whitespace_tokenize

logger = logging.getLogger(__name__)

HF_DATASET_ID = "mediabiasgroup/BABE"


def _parse_biased_words(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        logger.warning("Could not parse biased_words field: %r", raw)
        return []


def load_babe(split: str = "train") -> list[DetectionExample]:
    """Load BABE from the Hugging Face Hub and normalize to DetectionExample.

    BABE ships as a single split; callers should build their own leakage-safe
    train/dev/test split via ``biasneut.data.splits`` rather than relying on
    an HF-provided split name.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split=split)
    examples = []
    for row in ds:
        text = row["text"]
        tokens = whitespace_tokenize(text)
        biased_words = _parse_biased_words(row.get("biased_words"))
        bio_tags = bio_tags_for_words(tokens, biased_words) if biased_words else ["O"] * len(tokens)
        examples.append(
            DetectionExample(
                text=text,
                is_biased=bool(row["label"] == 1),
                bio_tags=bio_tags,
                source="babe",
                story_id=row.get("uuid"),
                outlet=row.get("outlet"),
            )
        )
    return examples
