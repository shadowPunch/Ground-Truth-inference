"""Detector wiring smoke tests against a tiny real HF checkpoint (mirrors
Pryzant's own integration_test.sh philosophy: prove the plumbing works, not
that a real model has been trained)."""
import torch
from transformers import AutoTokenizer

from biasneut.data.collate import DetectorCollator
from biasneut.data.schema import DetectionExample
from biasneut.detector.model import BiasDetector

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


def test_detector_forward_shapes_and_loss():
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)

    batch = [
        DetectionExample(text="the regime collapsed", is_biased=True,
                          bio_tags=["O", "B-BIAS", "O"], source="test"),
        DetectionExample(text="the meeting ended", is_biased=False,
                          bio_tags=["O", "O", "O"], source="test"),
    ]
    collated = DetectorCollator(tokenizer=tokenizer, max_length=32)(batch)
    out = model(**collated)

    batch_size, seq_len = collated["input_ids"].shape
    assert out.sentence_logits.shape == (batch_size, 2)
    assert out.token_logits.shape[0] == batch_size
    assert out.token_logits.shape[1] == seq_len
    assert out.loss is not None
    assert out.loss.item() > 0


def test_detector_backward_pass_updates_weights():
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    batch = [DetectionExample(text="a total disaster", is_biased=True,
                               bio_tags=["O", "B-BIAS", "O"], source="test")]
    collated = DetectorCollator(tokenizer=tokenizer, max_length=16)(batch)

    before = model.token_head.weight.clone()
    out = model(**collated)
    out.loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    assert not torch.equal(before, model.token_head.weight)


def test_detector_loss_none_without_labels():
    model = BiasDetector(backbone=TINY_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    encoding = tokenizer(["short sentence"], return_tensors="pt")
    out = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
    assert out.loss is None
