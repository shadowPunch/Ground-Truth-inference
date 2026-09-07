"""End-to-end detect-then-edit inference pipeline (§5): Stage-1 localization
feeds Stage-2 span-marked editing. This is the modular architecture the
proposal keeps from Pryzant (§2.3, §3.1) — as opposed to a concurrent model
where detection is folded into the encoder rather than a separate module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biasneut.detector.infer import DetectorInference, DetectorPrediction


@dataclass
class PipelineResult:
    source: str
    neutralized: str
    was_flagged_biased: bool
    bias_prob: float
    flagged_span: str | None


class Editor(Protocol):
    def neutralize(self, sentences: list[str], word_tags_per_sentence: list[list[str]], **kwargs) -> list[str]: ...


class BiasNeutralizationPipeline:
    """Wraps any Stage-2 editor that exposes ``neutralize(sentences,
    word_tags_per_sentence)`` — this matches ``SeqEditor`` (Strategies A/B and
    LEWIS-lite) directly; for the pure mask-and-infill arm, wrap
    ``MaskInfiller.infill`` in a small adapter with the same signature.
    """

    def __init__(self, detector: DetectorInference, editor: Editor, skip_unflagged: bool = True):
        self.detector = detector
        self.editor = editor
        self.skip_unflagged = skip_unflagged

    def __call__(self, sentences: list[str], **editor_kwargs) -> list[PipelineResult]:
        predictions: list[DetectorPrediction] = self.detector.predict(sentences)

        to_edit_idx = [
            i for i, p in enumerate(predictions) if p.is_biased or not self.skip_unflagged
        ]
        edited = [""] * len(sentences)
        if to_edit_idx:
            edit_sentences = [sentences[i] for i in to_edit_idx]
            edit_tags = [predictions[i].word_tags for i in to_edit_idx]
            outputs = self.editor.neutralize(edit_sentences, edit_tags, **editor_kwargs)
            for i, out in zip(to_edit_idx, outputs):
                edited[i] = out

        results = []
        for i, (sentence, pred) in enumerate(zip(sentences, predictions)):
            neutralized = edited[i] if i in to_edit_idx else sentence
            results.append(
                PipelineResult(
                    source=sentence,
                    neutralized=neutralized,
                    was_flagged_biased=pred.is_biased,
                    bias_prob=pred.bias_prob,
                    flagged_span=pred.biased_span_text,
                )
            )
        return results


class MaskInfillEditorAdapter:
    """Adapts ``MaskInfiller.infill`` (single sentence at a time, no beam
    search) to the batched ``Editor`` protocol so it can drive the same
    pipeline as the seq2seq editors."""

    def __init__(self, infiller):
        self.infiller = infiller

    def neutralize(self, sentences: list[str], word_tags_per_sentence: list[list[str]], **_) -> list[str]:
        return [self.infiller.infill(s, tags) for s, tags in zip(sentences, word_tags_per_sentence)]
