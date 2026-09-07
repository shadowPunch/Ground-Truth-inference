from transformers import AutoTokenizer

from biasneut.detector.infer import DetectorInference
from biasneut.detector.model import BiasDetector
from biasneut.eval.transfer_strength import aggregate_bias_score

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


def _tiny_detector() -> DetectorInference:
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    return DetectorInference(model, tokenizer, max_length=32)


def test_aggregate_bias_score_shapes_and_ranges():
    detector = _tiny_detector()
    sources = ["the regime collapsed", "a total disaster occurred"]
    outputs = ["the government changed", "an event occurred"]

    result = aggregate_bias_score(sources, outputs, detector)
    assert 0.0 <= result.frac_neutral <= 1.0
    assert len(result.source_probs) == len(sources)
    assert len(result.output_probs) == len(outputs)
    assert len(result.per_example_drop) == len(sources)


def test_aggregate_bias_score_identical_source_and_output_has_zero_mean_drop():
    detector = _tiny_detector()
    sentences = ["nothing changed here", "still the same sentence"]
    result = aggregate_bias_score(sentences, sentences, detector)
    assert result.mean_prob_drop == 0.0
    assert result.source_probs == result.output_probs
