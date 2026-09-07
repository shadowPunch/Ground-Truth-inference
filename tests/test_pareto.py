from biasneut.eval.pareto import ParetoPoint, pareto_frontier


def test_pareto_frontier_excludes_dominated_points():
    points = [
        ParetoPoint("A", bias_reduction=0.5, preservation=0.9),
        ParetoPoint("B", bias_reduction=0.3, preservation=0.5),  # dominated by A on both axes
        ParetoPoint("C", bias_reduction=0.8, preservation=0.6),  # trade-off point, non-dominated
    ]
    frontier = pareto_frontier(points)
    labels = {p.label for p in frontier}
    assert labels == {"A", "C"}


def test_pareto_frontier_sorted_by_bias_reduction():
    points = [
        ParetoPoint("high_reduction", bias_reduction=0.9, preservation=0.4),
        ParetoPoint("high_preservation", bias_reduction=0.2, preservation=0.95),
    ]
    frontier = pareto_frontier(points)
    assert [p.label for p in frontier] == ["high_preservation", "high_reduction"]


def test_pareto_frontier_single_point():
    points = [ParetoPoint("only", bias_reduction=0.5, preservation=0.5)]
    assert pareto_frontier(points) == points


def test_pareto_frontier_copy_input_is_extreme_but_non_dominated():
    # copy-input: zero transfer, max preservation -- always on the frontier
    points = [
        ParetoPoint("copy_input", bias_reduction=0.0, preservation=1.0),
        ParetoPoint("aggressive_editor", bias_reduction=0.9, preservation=0.3),
    ]
    frontier = pareto_frontier(points)
    assert {p.label for p in frontier} == {"copy_input", "aggressive_editor"}
