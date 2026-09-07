import torch
from transformers import AutoTokenizer

from biasneut.detector.infer import DetectorInference
from biasneut.detector.model import BiasDetector
from biasneut.detector.train import save_pretrained

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


def _tiny_inference() -> DetectorInference:
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    return DetectorInference(model, tokenizer, max_length=32)


def test_predict_returns_one_prediction_per_sentence():
    inference = _tiny_inference()
    predictions = inference.predict(["the regime collapsed", "a calm meeting"])
    assert len(predictions) == 2
    for pred, sentence in zip(predictions, ["the regime collapsed", "a calm meeting"]):
        assert pred.text == sentence
        assert 0.0 <= pred.bias_prob <= 1.0
        assert len(pred.word_tags) == len(pred.words)


def test_predict_word_tags_align_to_whitespace_words():
    inference = _tiny_inference()
    sentence = "the regime collapsed quickly"
    pred = inference.predict([sentence])[0]
    assert pred.words == ["the", "regime", "collapsed", "quickly"]
    assert all(t in ("O", "B-BIAS", "I-BIAS") for t in pred.word_tags)


def test_bias_score_single_sentence_convenience_method():
    inference = _tiny_inference()
    score = inference.bias_score("a total disaster")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_detector_save_and_load_roundtrip_with_explicit_backbone(tmp_path):
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    save_pretrained(model, tokenizer, tmp_path)

    reloaded = DetectorInference.from_pretrained(tmp_path, backbone=TINY_BACKBONE)
    predictions = reloaded.predict(["the regime collapsed"])
    assert len(predictions) == 1


def test_detector_save_and_load_roundtrip_from_saved_config(tmp_path):
    """The default reload path: no --backbone given, architecture rebuilt
    entirely from the config.json written by save_pretrained (regression
    test for a real bug caught during a live CLI smoke run, where
    from_pretrained tried to AutoConfig.from_pretrained() the checkpoint
    directory itself, which had no config.json)."""
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    save_pretrained(model, tokenizer, tmp_path)
    assert (tmp_path / "config.json").exists()

    reloaded = DetectorInference.from_pretrained(tmp_path)
    predictions = reloaded.predict(["the regime collapsed", "a calm meeting"])
    assert len(predictions) == 2

    # Fine-tuned weights actually made the round trip, not just a fresh init.
    original_logits = model.token_head.weight
    reloaded_logits = reloaded.model.token_head.weight
    assert torch.equal(original_logits, reloaded_logits)
