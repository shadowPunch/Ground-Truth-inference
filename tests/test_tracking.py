import math

import pytest

from biasneut.common import tracking
from biasneut.data.pseudo_parallel import log_generation_stats
from biasneut.data.schema import EditExample
from biasneut.eval.fluency import FluencyResult
from biasneut.eval.harness import SystemEvalResult, _log_evaluation
from biasneut.eval.preservation import PreservationResult
from biasneut.eval.transfer_strength import TransferStrengthResult
from biasneut.pipeline import PipelineResult, log_results


class RecordingTracker(tracking.Tracker):
    """Stands in for a live run so the logging helpers can be checked without W&B."""

    def __init__(self):
        super().__init__()
        self.logged, self.tables, self.images, self.summary = [], {}, {}, {}

    def log(self, metrics, step=None):
        self.logged.append(metrics)

    def log_table(self, key, columns, rows):
        self.tables[key] = (columns, rows)

    def log_image(self, key, path):
        self.images[key] = path

    def set_summary(self, **values):
        self.summary.update(values)


def test_disabled_tracking_yields_inert_tracker():
    with tracking.run(name="t", job_type="test") as tracker:
        assert not tracker.active
        tracker.log({"x": 1})
        tracker.log_table("t", ["a"], [[1]])
        tracker.set_summary(y=2)


def test_enabled_without_credentials_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("BIASNEUT_WANDB", "1")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc either
    with pytest.raises(RuntimeError, match="no credentials"):
        with tracking.run(name="t", job_type="test"):
            pass


def test_enabled_offline_run_logs_and_marks_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("BIASNEUT_WANDB", "1")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)

    with tracking.run(name="ok", job_type="test", config={"lr": 1e-3}) as tracker:
        assert tracker.active
        tracker.log({"loss": 0.5}, step=1)
        tracker.set_summary(final=1.0)

    with pytest.raises(ValueError):
        with tracking.run(name="boom", job_type="test") as tracker:
            raise ValueError("training crashed")
    assert "failed" in tracker._run.tags


def test_generation_stats_counts_kept_pairs_per_llm():
    def pair(model):
        return EditExample(source="s", target="t", biased_span=None, strategy="llm_synth",
                           provenance=f"model={model}")

    generated = [pair("gemini"), pair("gemini"), pair("anthropic"), pair("anthropic")]
    kept = [generated[0], generated[2], generated[3]]
    tracker = RecordingTracker()

    log_generation_stats(tracker, n_sources=2, pairs=generated, kept=kept)

    assert tracker.summary["pseudo_parallel/n_generated"] == 4
    assert tracker.summary["pseudo_parallel/keep_rate"] == 0.75
    assert tracker.summary["pseudo_parallel/n_kept/model=gemini"] == 1
    assert tracker.summary["pseudo_parallel/n_kept/model=anthropic"] == 2
    assert len(tracker.tables["pseudo_parallel/kept_sample"][1]) == 3


def test_inference_log_counts_flagged_changed_and_empty():
    results = [
        PipelineResult("a b", "a", True, 0.9, "b"),
        PipelineResult("c d", "", True, 0.8, "d"),
        PipelineResult("e f", "e f", False, 0.1, None),
    ]
    tracker = RecordingTracker()

    log_results(tracker, results)

    assert tracker.summary == {"n_inputs": 3, "n_flagged": 2, "n_changed": 2, "n_empty_output": 1}
    assert len(tracker.tables["inference/outputs"][1]) == 3


def test_evaluation_log_counts_degenerate_outputs_and_drops_inf(tmp_path):
    def result(name, perplexities):
        return SystemEvalResult(
            name=name,
            transfer=TransferStrengthResult(frac_neutral=0.5, mean_prob_drop=0.2,
                                            source_probs=[0.9, 0.9], output_probs=[0.4, 0.6]),
            preservation=PreservationResult(bertscore_f1=[0.9, 0.8], sbert_cosine=[0.7, 0.6], sari=10.0, bleu=5.0),
            fluency=FluencyResult(perplexities=perplexities, grammatical_probs=[0.5, 0.5]),
        )

    results = {"copy_input": result("copy_input", [20.0, 30.0]), "strategy_a": result("strategy_a", [25.0, math.inf])}
    comparisons = {"strategy_a": {"system": "strategy_a", "baseline": "copy_input", "mcnemar_transfer_p_value": 0.5,
                                  "mcnemar_statistic": 1.0, "sbert_cosine_mean": 0.65,
                                  "sbert_cosine_ci": (0.6, 0.7)}}
    tracker = RecordingTracker()

    _log_evaluation(tracker, results, comparisons, tmp_path)

    assert tracker.summary["strategy_a/n_degenerate"] == 1
    assert tracker.summary["copy_input/n_degenerate"] == 0
    assert tracker.summary["strategy_a/mean_perplexity"] is None  # inf isn't JSON-safe
    assert tracker.summary["copy_input/mean_perplexity"] == 25.0
    assert tracker.summary["n_test_sentences"] == 2
    assert "eval/pareto" in tracker.images
