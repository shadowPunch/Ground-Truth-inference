"""Strategy C, arm 2 (§3.4): LEWIS-style pseudo-parallel synthesis + generator
training.

Scope note: LEWIS's full architecture is a RoBERTa insert/replace/delete
*tagger* plus a BART *generator*, where the tagger is trained on labeled
edits and used to synthesize pseudo-parallel data at scale. We already have
a trained tagger — Stage 1's span detector — so we reuse it for masking
decisions instead of training a second bespoke tagger, and reuse the
Mask-and-Infill MLM (``biasneut.editor.mask_infill``) as the "style-specific
LM" that synthesizes the neutral side. The synthesized pairs then train the
*same* small seq2seq editor used by Strategies A/B via
``train_seq2seq_editor``, giving a real, working LEWIS-lite arm rather than a
from-scratch reimplementation of its tagger.
"""
from __future__ import annotations

import logging

from biasneut.data.schema import DetectionExample, EditExample
from biasneut.detector.infer import DetectorInference
from biasneut.editor.mask_infill import MaskInfiller

logger = logging.getLogger(__name__)


def synthesize_lewis_pairs(
    biased_examples: list[DetectionExample],
    detector: DetectorInference,
    infiller: MaskInfiller,
) -> list[EditExample]:
    """For each detector-flagged sentence, mask the predicted biased span(s)
    and infill with the (optionally neutral-domain-adapted) MLM, producing a
    synthetic (source, target) pseudo-parallel pair."""
    pairs = []
    predictions = detector.predict([ex.text for ex in biased_examples])
    for ex, pred in zip(biased_examples, predictions):
        if not pred.is_biased or all(t == "O" for t in pred.word_tags):
            continue
        target = infiller.infill(ex.text, pred.word_tags)
        if target.strip() == ex.text.strip():
            continue
        pairs.append(
            EditExample(
                source=ex.text,
                target=target,
                biased_span=pred.biased_span_text,
                strategy="unsupervised",
                provenance="lewis_lite: detector-masked + mlm-infilled",
            )
        )
    logger.info("Synthesized %d LEWIS-lite pseudo-parallel pairs from %d candidates",
                len(pairs), len(biased_examples))
    return pairs
