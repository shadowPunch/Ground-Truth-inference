"""Checkpoint/resume discipline for multi-session, quota-capped training (§7.4).

Kaggle sessions die mid-run (weekly quota, ~9-12h cap), so every long-running
loop in this codebase must be able to stop at an arbitrary step and pick back
up. ``CheckpointManager`` writes model/optimizer/scheduler/scaler state plus a
data-cursor (e.g. "shard index, row offset") atomically, and prunes all but
the ``keep_last`` most recent checkpoints to stay under Kaggle's ~20GB working
storage.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingState:
    step: int
    epoch: int
    data_cursor: dict[str, Any]
    best_metric: float | None = None


class CheckpointManager:
    def __init__(self, output_dir: str | Path, keep_last: int = 2):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _ckpt_dir(self, step: int) -> Path:
        return self.output_dir / f"checkpoint-{step:08d}"

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        state: TrainingState,
    ) -> Path:
        ckpt_dir = self._ckpt_dir(step)
        tmp_dir = self.output_dir / f".tmp-{step:08d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        torch.save(model.state_dict(), tmp_dir / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        with open(tmp_dir / "state.json", "w") as f:
            json.dump(
                {"step": state.step, "epoch": state.epoch, "data_cursor": state.data_cursor,
                 "best_metric": state.best_metric},
                f,
            )

        # Atomic-ish: rename fully-written tmp dir into place, then prune.
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)
        self._prune()
        return ckpt_dir

    def _prune(self) -> None:
        ckpts = sorted(self.output_dir.glob("checkpoint-*"), key=lambda p: p.name)
        for stale in ckpts[: max(0, len(ckpts) - self.keep_last)]:
            shutil.rmtree(stale, ignore_errors=True)

    def latest(self) -> Path | None:
        ckpts = sorted(self.output_dir.glob("checkpoint-*"), key=lambda p: p.name)
        return ckpts[-1] if ckpts else None

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
    ) -> TrainingState | None:
        """Resume from the latest checkpoint, or return None if none exists."""
        ckpt_dir = self.latest()
        if ckpt_dir is None:
            return None

        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=map_location))
        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=map_location))
        if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location=map_location))
        with open(ckpt_dir / "state.json") as f:
            payload = json.load(f)
        return TrainingState(**payload)
