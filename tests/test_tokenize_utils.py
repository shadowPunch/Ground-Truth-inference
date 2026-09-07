from biasneut.data.tokenize_utils import (
    bio_tags_for_char_span,
    bio_tags_for_words,
    merge_bio_tags,
    whitespace_tokenize,
)


def test_whitespace_tokenize_basic():
    assert whitespace_tokenize("the quick brown fox") == ["the", "quick", "brown", "fox"]


def test_bio_tags_for_words_single_word():
    tokens = ["the", "regime", "collapsed"]
    tags = bio_tags_for_words(tokens, ["regime"])
    assert tags == ["O", "B-BIAS", "O"]


def test_bio_tags_for_words_contiguous_run():
    tokens = ["a", "total", "disaster", "occurred"]
    tags = bio_tags_for_words(tokens, ["total", "disaster"])
    assert tags == ["O", "B-BIAS", "I-BIAS", "O"]


def test_bio_tags_for_words_no_match():
    tokens = ["all", "calm", "here"]
    assert bio_tags_for_words(tokens, []) == ["O", "O", "O"]
    assert bio_tags_for_words(tokens, ["nonexistent"]) == ["O", "O", "O"]


def test_bio_tags_for_char_span():
    text = "he called it a total disaster today"
    tokens = whitespace_tokenize(text)
    tags = bio_tags_for_char_span(tokens, text, "total disaster")
    assert tags == ["O", "O", "O", "O", "B-BIAS", "I-BIAS", "O"]


def test_bio_tags_for_char_span_missing():
    text = "nothing loaded here"
    tokens = whitespace_tokenize(text)
    tags = bio_tags_for_char_span(tokens, text, "not present")
    assert tags == ["O"] * len(tokens)


def test_merge_bio_tags_or():
    a = ["O", "B-BIAS", "O", "O"]
    b = ["O", "O", "B-BIAS", "O"]
    merged = merge_bio_tags(a, b)
    # The two spans are contiguous once merged, so it's one B-/I- run.
    assert merged == ["O", "B-BIAS", "I-BIAS", "O"]


def test_merge_bio_tags_disjoint_spans():
    a = ["O", "B-BIAS", "O", "O", "O"]
    b = ["O", "O", "O", "B-BIAS", "O"]
    merged = merge_bio_tags(a, b)
    assert merged == ["O", "B-BIAS", "O", "B-BIAS", "O"]
