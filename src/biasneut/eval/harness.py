"""Full evaluation harness (§6): orchestrates the triad + baselines + Pareto
+ significance testing into one report per experiment run.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from biasneut.common import tracking
from biasneut.detector.infer import DetectorInference
from biasneut.eval.fluency import FluencyResult, FluencyScorer
from biasneut.eval.lexicon import LexiconBiasScorer
from biasneut.eval.pareto import ParetoPoint, plot_pareto
from biasneut.eval.preservation import PreservationResult, PreservationScorer
from biasneut.eval.significance import bootstrap_ci, mcnemar_test
from biasneut.eval.transfer_strength import TransferStrengthResult, aggregate_bias_score

logger = logging.getLogger(__name__)


@dataclass
class SystemEvalResult:
    name: str
    transfer: TransferStrengthResult
    preservation: PreservationResult
    fluency: FluencyResult

    def summary(self) -> dict:
        return {
            "name": self.name,
            "frac_neutral": self.transfer.frac_neutral,
            "mean_prob_drop": self.transfer.mean_prob_drop,
            "mean_bertscore_f1": sum(self.preservation.bertscore_f1) / len(self.preservation.bertscore_f1),
            "mean_sbert_cosine": sum(self.preservation.sbert_cosine) / len(self.preservation.sbert_cosine),
            "sari": self.preservation.sari,
            "bleu": self.preservation.bleu,
            "mean_perplexity": sum(self.fluency.perplexities) / len(self.fluency.perplexities),
            "mean_grammatical_prob": sum(self.fluency.grammatical_probs) / len(self.fluency.grammatical_probs),
        }


class EvaluationHarness:
    def __init__(
        self,
        independent_detector: DetectorInference,
        preservation_scorer: PreservationScorer,
        fluency_scorer: FluencyScorer,
        lexicon_scorer: LexiconBiasScorer | None = None,
    ):
        self.independent_detector = independent_detector
        self.preservation_scorer = preservation_scorer
        self.fluency_scorer = fluency_scorer
        self.lexicon_scorer = lexicon_scorer

    def evaluate_system(
        self, name: str, sources: list[str], outputs: list[str], references: list[str] | None = None,
    ) -> SystemEvalResult:
        transfer = aggregate_bias_score(sources, outputs, self.independent_detector, self.lexicon_scorer)
        preservation = self.preservation_scorer.score(sources, outputs, references)
        fluency = self.fluency_scorer.score(outputs)
        return SystemEvalResult(name=name, transfer=transfer, preservation=preservation, fluency=fluency)

    def evaluate_all(
        self, systems: dict[str, list[str]], sources: list[str], references: list[str] | None = None,
    ) -> dict[str, SystemEvalResult]:
        return {name: self.evaluate_system(name, sources, outputs, references) for name, outputs in systems.items()}


def compare_to_baseline(results: dict[str, SystemEvalResult], system_name: str, baseline_name: str) -> dict:
    """Significance tests for one system against one baseline (§6.5)."""
    system, baseline = results[system_name], results[baseline_name]

    system_correct = [prob < 0.5 for prob in system.transfer.output_probs]
    baseline_correct = [prob < 0.5 for prob in baseline.transfer.output_probs]
    mcnemar = mcnemar_test(system_correct, baseline_correct)

    sbert_mean, sbert_lo, sbert_hi = bootstrap_ci(system.preservation.sbert_cosine)

    return {
        "system": system_name,
        "baseline": baseline_name,
        "mcnemar_transfer_p_value": mcnemar.p_value,
        "mcnemar_statistic": mcnemar.statistic,
        "sbert_cosine_mean": sbert_mean,
        "sbert_cosine_ci": (sbert_lo, sbert_hi),
    }


def build_markdown_report(results: dict[str, SystemEvalResult]) -> str:
    header = "| System | Frac Neutral | Mean Prob Drop | SBERT Cos | BERTScore F1 | SARI | BLEU | Perplexity | P(grammatical) |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for name, r in results.items():
        s = r.summary()
        rows.append(
            f"| {name} | {s['frac_neutral']:.3f} | {s['mean_prob_drop']:.3f} | {s['mean_sbert_cosine']:.3f} | "
            f"{s['mean_bertscore_f1']:.3f} | {s['sari']:.2f} | {s['bleu']:.2f} | {s['mean_perplexity']:.1f} | "
            f"{s['mean_grammatical_prob']:.3f} |"
        )
    return header + sep + "\n".join(rows)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _log_evaluation(tracker: tracking.Tracker, results: dict[str, SystemEvalResult],
                    comparisons: dict[str, dict], output_dir: Path) -> None:
    metric_keys = [k for k in next(iter(results.values())).summary() if k != "name"]
    rows = []
    for name, r in results.items():
        s = r.summary()
        # Non-finite perplexity = empty or 1-token output, i.e. a degenerate generation.
        n_degenerate = sum(not math.isfinite(p) for p in r.fluency.perplexities)
        rows.append([name] + [_finite_or_none(s[k]) for k in metric_keys] + [n_degenerate])
        tracker.set_summary(**{f"{name}/{k}": _finite_or_none(s[k]) for k in metric_keys},
                            **{f"{name}/n_degenerate": n_degenerate})
    tracker.log_table("eval/systems", ["system"] + metric_keys + ["n_degenerate"], rows)
    tracker.log_table(
        "eval/significance",
        ["system", "baseline", "mcnemar_p", "mcnemar_stat", "sbert_mean", "sbert_ci_lo", "sbert_ci_hi"],
        [[c["system"], c["baseline"], c["mcnemar_transfer_p_value"], c["mcnemar_statistic"],
          c["sbert_cosine_mean"], *c["sbert_cosine_ci"]] for c in comparisons.values()],
    )
    tracker.log_image("eval/pareto", output_dir / "pareto.png")
    tracker.set_summary(n_test_sentences=len(next(iter(results.values())).fluency.perplexities))


def run_full_evaluation(
    results: dict[str, SystemEvalResult],
    output_dir: str | Path,
    baseline_name: str = "copy_input",
) -> None:
    with tracking.run(name="eval", job_type="eval",
                      config={"systems": list(results), "baseline": baseline_name}) as tracker:
        _run_full_evaluation(results, Path(output_dir), baseline_name, tracker)


def _run_full_evaluation(
    results: dict[str, SystemEvalResult],
    output_dir: Path,
    baseline_name: str,
    tracker: tracking.Tracker,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_markdown_report(results)
    (output_dir / "report.md").write_text(report)

    comparisons = {
        name: compare_to_baseline(results, name, baseline_name)
        for name in results if name != baseline_name
    }
    (output_dir / "significance.json").write_text(json.dumps(comparisons, indent=2))

    points = [
        ParetoPoint(label=name, bias_reduction=r.transfer.mean_prob_drop,
                    preservation=sum(r.preservation.sbert_cosine) / len(r.preservation.sbert_cosine))
        for name, r in results.items()
    ]
    plot_pareto(points, str(output_dir / "pareto.png"))

    summaries = {name: r.summary() for name, r in results.items()}
    (output_dir / "summaries.json").write_text(json.dumps(summaries, indent=2))
    _log_evaluation(tracker, results, comparisons, output_dir)
    logger.info("Evaluation report written to %s", output_dir)
