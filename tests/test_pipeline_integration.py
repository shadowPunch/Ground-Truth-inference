"""End-to-end detect-then-edit smoke test (tiny real checkpoints, no
training) — proves the Stage-1 -> Stage-2 wiring holds together, in the same
spirit as Pryzant's own ``src/integration_test.sh``. This does not assert
neutralization quality (the tiny backbones have random weights); it asserts
shapes, types, and that every sentence gets exactly one result.
"""
from transformers import AutoTokenizer

from biasneut.baselines.trivial import copy_input_baseline, delete_flagged_word_baseline
from biasneut.detector.infer import DetectorInference
from biasneut.detector.model import BiasDetector
from biasneut.editor.mask_infill import MaskInfiller
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.pipeline import BiasNeutralizationPipeline, MaskInfillEditorAdapter

DETECTOR_BACKBONE = "hf-internal-testing/tiny-random-bert"
EDITOR_BACKBONE = "hf-internal-testing/tiny-random-t5"
MLM_BACKBONE = "hf-internal-testing/tiny-random-roberta"


def _tiny_detector() -> DetectorInference:
    model = BiasDetector(backbone=DETECTOR_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_BACKBONE)
    return DetectorInference(model, tokenizer, max_length=32)


def test_pipeline_with_seq2seq_editor_end_to_end():
    detector = _tiny_detector()
    editor = SeqEditor.from_pretrained_backbone(EDITOR_BACKBONE)
    pipeline = BiasNeutralizationPipeline(detector, editor, skip_unflagged=False)

    sentences = ["the regime collapsed under pressure", "the committee met on friday"]
    results = pipeline(sentences, num_beams=1, max_new_tokens=8)

    assert len(results) == len(sentences)
    for r, sentence in zip(results, sentences):
        assert r.source == sentence
        # Tiny random-weight T5 can legitimately emit EOS immediately (empty
        # string); the smoke test only needs to prove the wiring holds
        # together end to end, not that generation quality is meaningful.
        assert isinstance(r.neutralized, str)
        assert isinstance(r.was_flagged_biased, bool)


def test_pipeline_skips_unflagged_sentences_by_default():
    detector = _tiny_detector()
    editor = SeqEditor.from_pretrained_backbone(EDITOR_BACKBONE)
    pipeline = BiasNeutralizationPipeline(detector, editor, skip_unflagged=True)

    results = pipeline(["some sentence here"])
    assert len(results) == 1
    # Whether or not it was flagged, the result must be well-formed either way.
    if not results[0].was_flagged_biased:
        assert results[0].neutralized == "some sentence here"


def test_pipeline_with_mask_infill_editor_adapter():
    detector = _tiny_detector()
    infiller = MaskInfiller(backbone=MLM_BACKBONE)
    adapter = MaskInfillEditorAdapter(infiller)
    pipeline = BiasNeutralizationPipeline(detector, adapter, skip_unflagged=False)

    results = pipeline(["the regime fell quickly"])
    assert len(results) == 1
    assert isinstance(results[0].neutralized, str)


def test_baselines_are_drop_in_compatible_with_detector_output():
    detector = _tiny_detector()
    sentences = ["the regime collapsed", "a calm meeting occurred"]
    predictions = detector.predict(sentences)
    word_tags = [p.word_tags for p in predictions]

    copy_out = copy_input_baseline(sentences)
    delete_out = delete_flagged_word_baseline(sentences, word_tags)

    assert copy_out == sentences
    assert len(delete_out) == len(sentences)
    for original, deleted in zip(sentences, delete_out):
        assert len(deleted.split()) <= len(original.split())
