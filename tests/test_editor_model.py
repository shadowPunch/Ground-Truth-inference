"""Stage-2 editor wiring smoke tests against a tiny real T5 checkpoint."""
import pytest
import torch

from biasneut.common.config import EditorConfig
from biasneut.data.schema import EditExample
from biasneut.editor.collate import EditorCollator
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.editor.train_common import train_seq2seq_editor

TINY_BACKBONE = "hf-internal-testing/tiny-random-t5"


def test_seq_editor_adds_span_markers_and_resizes_embeddings():
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    assert "<bias>" in editor.tokenizer.all_special_tokens
    assert "</bias>" in editor.tokenizer.all_special_tokens
    assert editor.model.get_input_embeddings().weight.shape[0] == len(editor.tokenizer)


def test_editor_collator_produces_marked_source_and_masked_labels():
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    collator = EditorCollator(tokenizer=editor.tokenizer, task_prefix=editor.task_prefix)

    batch = [EditExample(source="the regime fell", target="the government fell",
                          biased_span="regime", strategy="wnc", provenance="test")]
    collated = collator(batch)

    assert "input_ids" in collated and "labels" in collated
    assert collated["labels"].shape[0] == 1
    # Padding positions in labels must be -100 (ignored by the loss), not the pad token id.
    assert (collated["labels"] == editor.tokenizer.pad_token_id).sum().item() == 0 or \
        editor.tokenizer.pad_token_id == -100


def test_editor_forward_pass_produces_loss():
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    collator = EditorCollator(tokenizer=editor.tokenizer, task_prefix=editor.task_prefix)
    batch = [
        EditExample(source="the regime fell", target="the government fell", biased_span="regime",
                    strategy="wnc", provenance="test"),
        EditExample(source="all calm here", target="all calm here", biased_span=None,
                    strategy="wnc", provenance="test"),
    ]
    collated = collator(batch)
    out = editor.model(**collated)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_seq_editor_neutralize_returns_one_output_per_sentence():
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    sentences = ["the regime collapsed", "a total disaster happened"]
    word_tags = [["O", "B-BIAS", "O"], ["O", "B-BIAS", "I-BIAS", "O"]]

    outputs = editor.neutralize(sentences, word_tags, num_beams=1, max_new_tokens=8)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_seq_editor_neutralize_default_blocks_repeated_trigrams():
    # The copy-bias boost pushes every non-flagged token up uniformly at every
    # decoding step, which without a repetition guard lets beam search loop on
    # whichever boosted token scores highest (observed on real data: output
    # collapsing to "protest protest protest..."). no_repeat_ngram_size=3 is
    # the default now specifically to prevent that.
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    sentences = ["the regime collapsed under pressure quickly"]
    word_tags = [["O", "B-BIAS", "O", "O", "O", "O"]]

    outputs = editor.neutralize(sentences, word_tags, num_beams=4, max_new_tokens=40, copy_bias_strength=5.0)
    token_ids = editor.tokenizer(outputs[0], add_special_tokens=False)["input_ids"]
    trigrams = [tuple(token_ids[i:i + 3]) for i in range(len(token_ids) - 2)]
    assert len(trigrams) == len(set(trigrams)), f"repeated trigram in output: {outputs[0]!r}"


def test_seq_editor_save_and_load_roundtrip(tmp_path):
    editor = SeqEditor.from_pretrained_backbone(TINY_BACKBONE)
    editor.save(tmp_path)

    reloaded = SeqEditor.load(tmp_path)
    assert reloaded.task_prefix == editor.task_prefix
    assert "<bias>" in reloaded.tokenizer.all_special_tokens
    assert reloaded.model.get_input_embeddings().weight.shape == editor.model.get_input_embeddings().weight.shape


def test_train_seq2seq_editor_rejects_empty_training_set():
    # Strategy C/LEWIS can legitimately synthesize 0 pairs (synthesize_lewis_pairs
    # skips any candidate with an all-'O' detector prediction) — this should fail
    # loudly here, not with PyTorch's opaque "num_samples=0" DataLoader error.
    with pytest.raises(ValueError, match="0 training examples"):
        train_seq2seq_editor([], [], EditorConfig())
