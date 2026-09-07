"""Transfer strength / Aggregate Bias Score (§6.1, axis 1).

"fraction of outputs flagged neutral by an independent BABE-trained
classifier, plus the mean drop in its bias probability (source -> output).
Independence from the training signal is essential; watch for Goodhart once
optimizing against it."

The ``detector`` passed in here MUST be a separately-trained instance from
any detector used elsewhere in the pipeline (training-time span localization,
Strategy-B filtering, Strategy-C masking) — see §4.3 / §5.1's "doubles as
evaluator" note. We don't enforce that at the type level (nothing stops a
caller from passing the same instance), so callers are responsible for
keeping two on-disk checkpoints and only ever loading the "_eval" one here.
"""
from __future__ import annotations

from dataclasses import dataclass

from biasneut.detector.infer import DetectorInference
from biasneut.eval.lexicon import LexiconBiasScorer


@dataclass
class TransferStrengthResult:
    frac_neutral: float
    mean_prob_drop: float
    source_probs: list[float]
    output_probs: list[float]

    @property
    def per_example_drop(self) -> list[float]:
        return [s - o for s, o in zip(self.source_probs, self.output_probs)]


def aggregate_bias_score(
    sources: list[str],
    outputs: list[str],
    independent_detector: DetectorInference,
    lexicon_scorer: LexiconBiasScorer | None = None,
) -> TransferStrengthResult:
    source_preds = independent_detector.predict(sources)
    output_preds = independent_detector.predict(outputs)

    source_probs = [p.bias_prob for p in source_preds]
    output_probs = [p.bias_prob for p in output_preds]

    if lexicon_scorer is not None:
        # Blend classifier probability with the lexicon signal (mean of the
        # two), matching the multi-signal spirit of §6.1 without collapsing
        # to a single opaque score computed by one model alone.
        output_probs = [
            0.5 * p + 0.5 * lexicon_scorer.score(o) for p, o in zip(output_probs, outputs)
        ]
        source_probs = [
            0.5 * p + 0.5 * lexicon_scorer.score(s) for p, s in zip(source_probs, sources)
        ]

    frac_neutral = sum(1 for p in output_preds if not p.is_biased) / len(output_preds) if output_preds else 0.0
    mean_prob_drop = (
        sum(s - o for s, o in zip(source_probs, output_probs)) / len(source_probs) if source_probs else 0.0
    )

    return TransferStrengthResult(
        frac_neutral=frac_neutral,
        mean_prob_drop=mean_prob_drop,
        source_probs=source_probs,
        output_probs=output_probs,
    )
