"""Shared example schemas so BABE/BASIL/WNC/pseudo-parallel data all normalize
to the same shape before hitting the detector or editor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionExample:
    """One sentence for Stage-1 training/eval.

    ``bio_tags`` is None when only a sentence-level label is available (BABE
    without word spans); the token-classification loss is masked out for
    those examples rather than penalizing absent span supervision.
    """

    text: str
    is_biased: bool
    bio_tags: list[str] | None  # "O" / "B-BIAS" / "I-BIAS" per whitespace token, or None
    source: str  # "babe" | "basil"
    story_id: str | None = None  # for leakage-safe splitting (§6.5)
    outlet: str | None = None


@dataclass
class EditExample:
    """One (biased -> neutral) pair for Stage-2 training, from any strategy."""

    source: str
    target: str
    biased_span: str | None
    strategy: str  # "wnc" | "llm_synth" | "unsupervised"
    provenance: str  # free-text note, e.g. generating model name for Strategy B
