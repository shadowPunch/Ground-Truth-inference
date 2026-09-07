import torch

from biasneut.editor.constrained_decoding import CopyBiasLogitsProcessor, non_flagged_token_ids
from transformers import AutoTokenizer

TINY_BACKBONE = "hf-internal-testing/tiny-random-t5"


def test_copy_bias_logits_processor_boosts_only_allowed_ids():
    vocab_size = 10
    scores = torch.zeros(2, vocab_size)
    allowed = [{2, 5}, set()]  # batch item 0 has allowed ids, batch item 1 has none
    processor = CopyBiasLogitsProcessor(allowed, bias_strength=3.0)

    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = processor(input_ids, scores.clone())

    expected_row0 = torch.zeros(vocab_size)
    expected_row0[2] = 3.0
    expected_row0[5] = 3.0
    assert torch.equal(out[0], expected_row0)
    assert torch.equal(out[1], torch.zeros(vocab_size))


def test_copy_bias_logits_processor_is_additive_not_overwrite():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    processor = CopyBiasLogitsProcessor([{1}], bias_strength=5.0)
    out = processor(torch.zeros(1, 1, dtype=torch.long), scores)
    assert out[0, 1].item() == 7.0
    assert out[0, 0].item() == 1.0  # untouched


def test_non_flagged_token_ids_excludes_flagged_words():
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    sentence = "the regime collapsed quickly"
    word_tags = ["O", "B-BIAS", "O", "O"]

    allowed = non_flagged_token_ids(tokenizer, sentence, word_tags)
    flagged_only_ids = set(tokenizer("regime", add_special_tokens=False)["input_ids"])

    # None of the tokens unique to the flagged word "regime" should be allowed
    # (some subword ids could theoretically overlap with other words; we only
    # assert the allowed set was computed from the non-flagged text).
    non_flagged_text_ids = set(tokenizer("the collapsed quickly", add_special_tokens=False)["input_ids"])
    assert allowed == non_flagged_text_ids
    assert isinstance(flagged_only_ids, set)


def test_non_flagged_token_ids_empty_when_all_flagged():
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    allowed = non_flagged_token_ids(tokenizer, "bad awful terrible", ["B-BIAS", "I-BIAS", "I-BIAS"])
    assert allowed == set()
