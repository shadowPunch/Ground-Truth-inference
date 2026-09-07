"""Pareto frontier reporting (§6.2): "plot bias-reduction vs. preservation as
a Pareto frontier across decoding settings, rather than collapsing to one
number. The whole point of the task is the trade-off."
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParetoPoint:
    label: str
    bias_reduction: float  # higher is better
    preservation: float  # higher is better


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated points (higher-is-better on both axes)."""
    frontier = []
    for p in points:
        dominated = any(
            (q.bias_reduction >= p.bias_reduction and q.preservation >= p.preservation and
             (q.bias_reduction > p.bias_reduction or q.preservation > p.preservation))
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p.bias_reduction)


def plot_pareto(points: list[ParetoPoint], output_path: str, title: str = "Bias reduction vs. preservation") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = pareto_frontier(points)
    frontier_labels = {p.label for p in frontier}

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [p.bias_reduction for p in points]
    ys = [p.preservation for p in points]
    colors = ["#d62728" if p.label in frontier_labels else "#7f7f7f" for p in points]
    ax.scatter(xs, ys, c=colors, s=60, zorder=3)
    for p in points:
        ax.annotate(p.label, (p.bias_reduction, p.preservation), textcoords="offset points", xytext=(5, 5),
                    fontsize=8)

    if len(frontier) > 1:
        fx = [p.bias_reduction for p in frontier]
        fy = [p.preservation for p in frontier]
        ax.plot(fx, fy, "--", color="#d62728", alpha=0.6, zorder=2, label="Pareto frontier")
        ax.legend()

    ax.set_xlabel("Bias reduction (transfer strength)")
    ax.set_ylabel("Content preservation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
