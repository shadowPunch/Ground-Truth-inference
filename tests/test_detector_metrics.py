from biasneut.data.collate import TAG2ID
from biasneut.detector.metrics import extract_spans, sentence_prf1, span_prf1

O, B, I = TAG2ID["O"], TAG2ID["B-BIAS"], TAG2ID["I-BIAS"]


def test_sentence_prf1_perfect():
    metrics = sentence_prf1([1, 0, 1, 0], [1, 0, 1, 0])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_sentence_prf1_partial():
    # 1 true positive, 1 false positive, 1 false negative
    preds = [1, 1, 0]
    golds = [1, 0, 1]
    metrics = sentence_prf1(preds, golds)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_extract_spans_single_run():
    tags = [O, B, I, O]
    assert extract_spans(tags) == {(1, 3)}


def test_extract_spans_two_runs():
    tags = [B, O, B, I, O]
    assert extract_spans(tags) == {(0, 1), (2, 4)}


def test_extract_spans_no_bias():
    assert extract_spans([O, O, O]) == set()


def test_span_prf1_exact_match():
    preds = [[O, B, I, O]]
    golds = [[O, B, I, O]]
    metrics = span_prf1(preds, golds)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_span_prf1_ignores_masked_positions():
    preds = [[O, B, O, O]]
    golds = [[O, B, -100, -100]]  # padding/continuation subwords
    metrics = span_prf1(preds, golds)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_span_prf1_false_positive():
    preds = [[O, B, O]]
    golds = [[O, O, O]]
    metrics = span_prf1(preds, golds)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
