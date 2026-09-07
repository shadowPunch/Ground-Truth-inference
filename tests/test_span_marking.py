from biasneut.editor.span_marking import diff_word_tags, mark_span_from_tags, mark_text


def test_diff_word_tags_single_word_replace():
    source = "the regime collapsed quickly"
    target = "the government collapsed quickly"
    tags = diff_word_tags(source, target)
    assert tags == ["O", "B-BIAS", "O", "O"]


def test_diff_word_tags_deletion():
    source = "a total disaster occurred here"
    target = "a disaster occurred here"
    tags = diff_word_tags(source, target)
    assert tags == ["O", "B-BIAS", "O", "O", "O"]


def test_diff_word_tags_identical():
    source = target = "nothing changed at all"
    assert diff_word_tags(source, target) == ["O"] * 4


def test_mark_span_from_tags_wraps_contiguous_run():
    words = ["the", "regime", "fell"]
    tags = ["O", "B-BIAS", "O"]
    marked = mark_span_from_tags(words, tags)
    assert marked == "the <bias> regime </bias> fell"


def test_mark_span_from_tags_multi_word_run():
    words = ["a", "total", "disaster", "here"]
    tags = ["O", "B-BIAS", "I-BIAS", "O"]
    marked = mark_span_from_tags(words, tags)
    assert marked == "a <bias> total disaster </bias> here"


def test_mark_span_from_tags_span_at_end():
    words = ["it", "was", "awful"]
    tags = ["O", "O", "B-BIAS"]
    marked = mark_span_from_tags(words, tags)
    assert marked == "it was <bias> awful </bias>"


def test_mark_text_custom_markers():
    marked = mark_text("the regime fell", ["O", "B-BIAS", "O"], span_start="[[", span_end="]]")
    assert marked == "the [[ regime ]] fell"
