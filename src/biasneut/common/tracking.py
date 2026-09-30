"""Weights & Biases experiment tracking.

Every training, evaluation and inference run is logged to W&B. Tracking is on
by default; set ``BIASNEUT_WANDB=0`` to switch it off (tests, offline work).
When it is on, a missing ``wandb`` install or missing credentials is a hard
error rather than a silent fallback, so no real run goes untracked.

Env vars: ``BIASNEUT_WANDB_PROJECT`` (default "biasneut") and
``BIASNEUT_WANDB_GROUP`` (groups the runs of one pipeline execution, e.g.
"kaggle-v13").
"""
from __future__ import annotations

import netrc
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ENABLED_ENV = "BIASNEUT_WANDB"
PROJECT_ENV = "BIASNEUT_WANDB_PROJECT"
GROUP_ENV = "BIASNEUT_WANDB_GROUP"
DEFAULT_PROJECT = "biasneut"
_WANDB_HOST = "api.wandb.ai"


def is_enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1") != "0"


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return netrc.netrc().authenticators(_WANDB_HOST) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def _require_wandb():
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            f"W&B tracking is on but `wandb` isn't installed. `pip install wandb`, "
            f"or set {ENABLED_ENV}=0 to run untracked on purpose."
        ) from e
    # Checked up front: in a notebook kernel wandb.init would otherwise block on a login prompt.
    if not _has_credentials():
        raise RuntimeError(
            f"W&B tracking is on but no credentials were found (WANDB_API_KEY or ~/.netrc). "
            f"Set WANDB_API_KEY, or set {ENABLED_ENV}=0 to run untracked on purpose."
        )
    return wandb


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


class Tracker:
    """Handle for one run. Every method is a no-op when tracking is disabled."""

    def __init__(self, wb_run=None):
        self._run = wb_run

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self._run is not None:
            import wandb
            self._run.log({key: wandb.Table(columns=columns, data=rows)})

    def log_image(self, key: str, path: str | Path) -> None:
        if self._run is not None:
            import wandb
            self._run.log({key: wandb.Image(str(path))})

    def set_summary(self, **values: Any) -> None:
        if self._run is not None:
            self._run.summary.update(values)

    def add_tags(self, *tags: str) -> None:
        if self._run is not None:
            self._run.tags = tuple(self._run.tags or ()) + tags


@contextmanager
def run(
    name: str,
    job_type: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> Iterator[Tracker]:
    """Open a W&B run for the duration of the block; a raised exception tags it "failed"."""
    if not is_enabled():
        yield Tracker()
        return

    wandb = _require_wandb()
    wb_run = wandb.init(
        project=os.environ.get(PROJECT_ENV, DEFAULT_PROJECT),
        group=os.environ.get(GROUP_ENV) or None,
        name=name,
        job_type=job_type,
        config={**(config or {}), "git_commit": _git_commit()},
        tags=tags,
        notes=notes,
        reinit="finish_previous",
    )
    tracker = Tracker(wb_run)
    try:
        yield tracker
    except BaseException:
        tracker.add_tags("failed")
        wb_run.finish(exit_code=1)
        raise
    wb_run.finish()
