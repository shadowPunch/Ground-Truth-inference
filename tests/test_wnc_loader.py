from pathlib import Path

import pytest

from biasneut.data.wnc import load_wnc

WNC_DEV_PATH = Path(__file__).parent.parent / "data" / "bias_data" / "WNC" / "biased.word.dev"


@pytest.mark.skipif(not WNC_DEV_PATH.exists(), reason="WNC not downloaded (see README: Data)")
def test_load_wnc_real_file_smoke():
    examples = load_wnc(WNC_DEV_PATH, max_examples=20)
    assert len(examples) == 20
    for ex in examples:
        assert ex.strategy == "wnc"
        assert ex.source and ex.target
        assert ex.source != ex.target or ex.biased_span is None


def test_load_wnc_parses_columns(tmp_path):
    tsv_path = tmp_path / "toy.tsv"
    tsv_path.write_text(
        "1\tsrc_tok\ttgt_tok\tthe regime fell\tthe government fell\tPOS\tDEP\n"
        "2\tsrc_tok2\ttgt_tok2\tall calm here\tall calm here\tPOS\tDEP\n",
        encoding="utf-8",
    )
    examples = load_wnc(tsv_path)
    assert len(examples) == 2
    assert examples[0].source == "the regime fell"
    assert examples[0].target == "the government fell"
    assert examples[0].biased_span == "regime"
    assert examples[1].biased_span is None  # identical src/tgt, nothing removed


def test_load_wnc_max_examples_cap(tmp_path):
    tsv_path = tmp_path / "toy.tsv"
    lines = "\n".join(f"{i}\tx\ty\tsentence {i}\tsentence {i}\tPOS\tDEP" for i in range(10))
    tsv_path.write_text(lines, encoding="utf-8")
    examples = load_wnc(tsv_path, max_examples=3)
    assert len(examples) == 3
