"""BASIL loader — real-network integration tests (pytest -m integration)."""
import pytest

from biasneut.data.basil import _download_tarball_fallback, load_basil

pytestmark = pytest.mark.integration


def test_tarball_fallback_produces_expected_layout(tmp_path):
    # Exercises the codeload.github.com fallback directly, independent of
    # whether `git clone` itself succeeds in this environment (it doesn't in
    # a network-restricted sandbox, which is exactly why this fallback exists).
    repo_dir = tmp_path / "BASIL"
    _download_tarball_fallback(repo_dir)
    assert (repo_dir / "articles").exists()
    assert (repo_dir / "annotations").exists()


def test_load_basil_real_data_matches_published_corpus_size(tmp_path):
    examples = load_basil(tmp_path)
    assert len(examples) == 7984  # matches the paper's reported corpus size exactly
    assert any(ex.is_biased for ex in examples)
    assert any(not ex.is_biased for ex in examples)
    assert len({ex.story_id for ex in examples}) == 100  # 100 unique stories x 3 outlets
