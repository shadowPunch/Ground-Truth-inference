from biasneut.data.pseudo_parallel import EchoClient, filter_pseudo_parallel, generate_pseudo_parallel
from biasneut.data.schema import DetectionExample, EditExample


def test_echo_client_extracts_sentence_from_prompt():
    client = EchoClient()
    prompt = "You are helping...\n\nSentence: the regime collapsed"
    assert client.generate(prompt) == "the regime collapsed"


def test_generate_pseudo_parallel_skips_non_biased():
    examples = [
        DetectionExample(text="the regime fell", is_biased=True, bio_tags=None, source="babe"),
        DetectionExample(text="the government changed", is_biased=False, bio_tags=None, source="babe"),
    ]
    pairs = generate_pseudo_parallel(examples, [EchoClient()])
    assert len(pairs) == 1
    assert pairs[0].source == "the regime fell"
    assert pairs[0].strategy == "llm_synth"


def test_generate_pseudo_parallel_multiple_clients_multiple_candidates():
    examples = [DetectionExample(text="the regime fell", is_biased=True, bio_tags=None, source="babe")]
    pairs = generate_pseudo_parallel(examples, [EchoClient(name="model_a"), EchoClient(name="model_b")])
    assert len(pairs) == 2
    assert {p.provenance for p in pairs} == {"model=model_a", "model=model_b"}


def test_filter_pseudo_parallel_similarity_floor():
    pairs = [
        EditExample(source="a", target="a", biased_span=None, strategy="llm_synth", provenance="m"),
        EditExample(source="b", target="b", biased_span=None, strategy="llm_synth", provenance="m"),
    ]
    # First pair passes the similarity floor, second doesn't.
    similarities = {"a": 0.9, "b": 0.1}
    filtered = filter_pseudo_parallel(
        pairs,
        bias_score_fn=lambda s: 0.5,
        similarity_fn=lambda s, t: similarities[s],
        similarity_floor=0.6,
        require_bias_drop=False,
    )
    assert len(filtered) == 1
    assert filtered[0].source == "a"


def test_filter_pseudo_parallel_requires_bias_drop():
    pairs = [EditExample(source="biased", target="still_biased", biased_span=None,
                          strategy="llm_synth", provenance="m")]
    scores = {"biased": 0.9, "still_biased": 0.9}  # no drop
    filtered = filter_pseudo_parallel(
        pairs, bias_score_fn=lambda s: scores[s], similarity_fn=lambda s, t: 1.0, require_bias_drop=True,
    )
    assert filtered == []
