from biasneut.data.schema import DetectionExample
from biasneut.data.splits import assert_no_leakage, group_disjoint_split


def _make_examples(n_stories: int, per_story: int) -> list[DetectionExample]:
    examples = []
    for s in range(n_stories):
        for _ in range(per_story):
            examples.append(DetectionExample(text=f"story {s} sentence", is_biased=False, bio_tags=None,
                                              source="test", story_id=f"story-{s}"))
    return examples


def test_group_disjoint_split_no_story_straddles_boundary():
    examples = _make_examples(n_stories=50, per_story=3)
    train, dev, test = group_disjoint_split(examples, ratios=(0.7, 0.15, 0.15), seed=1)
    assert len(train) + len(dev) + len(test) == len(examples)
    assert_no_leakage(train, dev, test)  # should not raise


def test_group_disjoint_split_approximate_ratios():
    examples = _make_examples(n_stories=100, per_story=2)
    train, dev, test = group_disjoint_split(examples, ratios=(0.8, 0.1, 0.1), seed=7)
    total = len(examples)
    assert abs(len(train) / total - 0.8) < 0.05
    assert abs(len(dev) / total - 0.1) < 0.05
    assert abs(len(test) / total - 0.1) < 0.05


def test_group_disjoint_split_singleton_examples_dont_require_grouping():
    examples = [DetectionExample(text=f"s{i}", is_biased=False, bio_tags=None, source="test", story_id=None)
                for i in range(30)]
    train, dev, test = group_disjoint_split(examples, seed=3)
    assert len(train) + len(dev) + len(test) == 30


def test_assert_no_leakage_raises_on_violation():
    ex_a = DetectionExample(text="a", is_biased=False, bio_tags=None, source="t", story_id="dup")
    ex_b = DetectionExample(text="b", is_biased=False, bio_tags=None, source="t", story_id="dup")
    try:
        assert_no_leakage([ex_a], [ex_b])
        raised = False
    except ValueError:
        raised = True
    assert raised
