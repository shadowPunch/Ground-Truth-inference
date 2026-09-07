"""Leakage-safe splitting (§6.5): "same-story sentences must not straddle the
split boundary." We split whole groups (story id, falling back to outlet, or
the example's own index when the source has no grouping key) into
train/dev/test using a greedy load-balancer so that group sizes need not be
uniform for the split ratios to come out close to the target.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, TypeVar

T = TypeVar("T")

GroupKeyFn = Callable[[T], str | None]


def default_group_key(example) -> str | None:
    """Group by story_id if present, else by outlet, else ungrouped."""
    story_id = getattr(example, "story_id", None)
    if story_id:
        return f"story::{story_id}"
    outlet = getattr(example, "outlet", None)
    if outlet:
        return f"outlet::{outlet}"
    return None


def group_disjoint_split(
    examples: list[T],
    group_key_fn: GroupKeyFn = default_group_key,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Split ``examples`` into (train, dev, test) such that no group (e.g. all
    sentences from one story, across outlets) straddles the boundary.
    Examples with no group key are treated as singleton groups.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        key = group_key_fn(ex) or f"__singleton__::{i}"
        groups[key].append(i)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    n_total = len(examples)
    targets = [r * n_total for r in ratios]
    counts = [0, 0, 0]
    split_indices: tuple[list[int], list[int], list[int]] = ([], [], [])

    for key in group_keys:
        idxs = groups[key]
        # Assign this whole group to whichever split is furthest below its
        # target share (a simple greedy load balancer for approximate ratios).
        deficits = [targets[j] - counts[j] for j in range(3)]
        target_split = max(range(3), key=lambda j: deficits[j])
        split_indices[target_split].extend(idxs)
        counts[target_split] += len(idxs)

    train = [examples[i] for i in split_indices[0]]
    dev = [examples[i] for i in split_indices[1]]
    test = [examples[i] for i in split_indices[2]]
    return train, dev, test


def assert_no_leakage(*splits: list, group_key_fn: GroupKeyFn = default_group_key) -> None:
    """Raise if any group key appears in more than one split (safety net to
    call right before training, not just at split-creation time)."""
    seen: dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for ex in split:
            key = group_key_fn(ex)
            if key is None:
                continue
            if key in seen and seen[key] != split_idx:
                raise ValueError(f"Leakage detected: group {key!r} appears in multiple splits")
            seen[key] = split_idx
