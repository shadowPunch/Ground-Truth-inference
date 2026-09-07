"""BASIL loader (§3.2, §4.1) — sentence + phrase-level lexical/informational
bias spans, source-clustered by story (same event, 3 outlets: hpo/fox/nyt).

We only load the *lexical* bias spans as detection targets: §2.1 draws the
lexical/informational distinction as the load-bearing scope decision for this
whole project, and Pryzant's copy-heavy editor is explicitly the wrong tool
for informational bias (§2.2). Informational spans are still surfaced on
``DetectionExample`` via the ``basil_info_biased`` note in provenance so
downstream analysis can report on the discarded half without training on it.

BASIL ships as raw per-article JSON in a GitHub repo (no HF Hub mirror with
spans), so this loader shallow-clones ``launchnlp/BASIL`` into a local cache
rather than hitting the GitHub API file-by-file.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from biasneut.data.schema import DetectionExample
from biasneut.data.tokenize_utils import bio_tags_for_char_span, merge_bio_tags, whitespace_tokenize

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/launchnlp/BASIL"
REPO_SLUG = "launchnlp/BASIL"


def _download_tarball_fallback(repo_dir: Path) -> None:
    """Some sandboxed/CI environments allow plain HTTPS to GitHub's asset
    hosts while blocking `git clone` to github.com itself (observed in a
    Claude Code sandbox: `codeload.github.com` reachable, `github.com`
    times out) — fetch the repo as a tarball via stdlib `urllib`/`tarfile`
    instead of shelling out to `git`."""
    tmp_tar = repo_dir.parent / "BASIL.tar.gz"
    last_error: Exception | None = None
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{REPO_SLUG}/tar.gz/refs/heads/{branch}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp_tar, "wb") as f:
                f.write(response.read())
            break
        except urllib.error.HTTPError as e:
            last_error = e
            continue
    else:
        raise RuntimeError(f"Could not fetch {REPO_SLUG} via git clone or tarball fallback") from last_error

    repo_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_tar) as tar:
        # GitHub's tarball wraps everything in a single "<repo>-<branch>/"
        # root directory; strip it so repo_dir/articles, repo_dir/annotations
        # land directly under repo_dir (equivalent to `tar --strip-components=1`).
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            member.name = str(Path(*parts[1:]))
            members.append(member)
        tar.extractall(repo_dir, members=members, filter="data")
    tmp_tar.unlink(missing_ok=True)


def ensure_basil_repo(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    repo_dir = cache_dir / "BASIL"
    if (repo_dir / "articles").exists() and (repo_dir / "annotations").exists():
        return repo_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", REPO_URL, repo_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        logger.warning("git clone failed (%s); falling back to tarball download", detail)
        _download_tarball_fallback(repo_dir)
    return repo_dir


def _flatten_sentences(body_paragraphs: list[list[str]]) -> list[str]:
    return [sent for para in body_paragraphs for sent in para]


def _load_article_pair(article_path: Path, annotation_path: Path) -> list[DetectionExample]:
    article = json.loads(article_path.read_text())
    annotation = json.loads(annotation_path.read_text())

    sentences = _flatten_sentences(article["body-paragraphs"])
    story_id = article.get("triplet-uuid", article.get("uuid"))
    # BASIL's raw "source" field is inconsistently cased across articles
    # (observed: "nyt"/"NYT", "fox"/"FOX", "hpo"/"HPO" for the same outlet).
    outlet = article.get("source", "").lower() or None

    lex_tags_by_sentence = [["O"] * len(whitespace_tokenize(s)) for s in sentences]
    info_flagged = [False] * len(sentences)

    for phrase in annotation.get("phrase-level-annotations", []):
        span_text = phrase.get("txt", "")
        if not span_text.strip():
            continue
        # BASIL doesn't give an explicit sentence index for each phrase
        # annotation, so we locate it by substring search across the
        # article's sentences (first match; spans are short and specific
        # enough in practice that collisions are rare).
        sent_idx = next((i for i, s in enumerate(sentences) if span_text in s), None)
        if sent_idx is None:
            continue
        if phrase.get("bias") == "lex":
            tokens = whitespace_tokenize(sentences[sent_idx])
            span_tags = bio_tags_for_char_span(tokens, sentences[sent_idx], span_text)
            lex_tags_by_sentence[sent_idx] = merge_bio_tags(lex_tags_by_sentence[sent_idx], span_tags)
        elif phrase.get("bias") == "inf":
            info_flagged[sent_idx] = True

    examples = []
    for i, sent in enumerate(sentences):
        bio_tags = lex_tags_by_sentence[i]
        is_lex_biased = any(t != "O" for t in bio_tags)
        examples.append(
            DetectionExample(
                text=sent,
                is_biased=is_lex_biased,
                bio_tags=bio_tags,
                source="basil",
                story_id=story_id,
                outlet=outlet,
            )
        )
    return examples


def load_basil(cache_dir: str | Path = "data_cache") -> list[DetectionExample]:
    repo_dir = ensure_basil_repo(cache_dir)
    articles_dir = repo_dir / "articles"
    annotations_dir = repo_dir / "annotations"

    examples: list[DetectionExample] = []
    n_pairs = 0
    for article_path in sorted(articles_dir.rglob("*.json")):
        rel = article_path.relative_to(articles_dir)
        annotation_path = annotations_dir / rel.parent / (article_path.stem + "_ann.json")
        if not annotation_path.exists():
            logger.warning("No annotation file for %s, skipping", article_path)
            continue
        examples.extend(_load_article_pair(article_path, annotation_path))
        n_pairs += 1

    logger.info("Loaded BASIL: %d article/annotation pairs -> %d sentences", n_pairs, len(examples))
    return examples
