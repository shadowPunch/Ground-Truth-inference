from biasneut.eval.significance import aggregate_across_seeds, bootstrap_ci, mcnemar_test


def test_mcnemar_identical_systems_no_signal():
    a = [True, False, True, False, True]
    b = [True, False, True, False, True]
    result = mcnemar_test(a, b)
    assert result.p_value == 1.0
    assert result.n_a_only_correct == 0
    assert result.n_b_only_correct == 0


def test_mcnemar_clear_difference_small_n_uses_exact_test():
    # system_a right where b is wrong, many times, b never right where a is wrong
    a = [True] * 20
    b = [False] * 20
    result = mcnemar_test(a, b)
    assert result.n_a_only_correct == 20
    assert result.n_b_only_correct == 0
    assert result.p_value < 0.001


def test_bootstrap_ci_contains_true_mean_for_constant_values():
    values = [0.5] * 50
    mean, lo, hi = bootstrap_ci(values, n_samples=200, seed=0)
    assert mean == 0.5
    assert lo <= 0.5 <= hi


def test_bootstrap_ci_narrower_with_more_data():
    import random

    rng = random.Random(0)
    small = [rng.gauss(0, 1) for _ in range(20)]
    large = [rng.gauss(0, 1) for _ in range(2000)]

    _, lo_s, hi_s = bootstrap_ci(small, n_samples=500, seed=1)
    _, lo_l, hi_l = bootstrap_ci(large, n_samples=500, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_aggregate_across_seeds():
    result = aggregate_across_seeds([0.8, 0.82, 0.79])
    assert abs(result.mean - 0.8033333) < 1e-5
    assert result.std > 0
    assert result.per_seed == [0.8, 0.82, 0.79]
