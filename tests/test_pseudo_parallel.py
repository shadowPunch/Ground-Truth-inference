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


class _FlakyClient:
    """Fails on the calls listed in ``fail_on`` (0-based call index), echoes otherwise."""

    def __init__(self, name, fail_on):
        self.name, self.fail_on, self.calls = name, set(fail_on), 0

    def generate(self, prompt):
        call, self.calls = self.calls, self.calls + 1
        if call in self.fail_on:
            raise RuntimeError("quota exhausted")
        return EchoClient().generate(prompt)


def _biased(n):
    return [DetectionExample(text=f"sentence {i}", is_biased=True, bio_tags=None, source="babe") for i in range(n)]


def test_generation_drops_client_after_consecutive_failures():
    dead = _FlakyClient("dead", fail_on=range(1, 100))
    healthy = _FlakyClient("healthy", fail_on=[])

    pairs = generate_pseudo_parallel(_biased(10), [dead, healthy], max_consecutive_failures=3)

    assert dead.calls == 4  # 1 success, then 3 failures, then never called again
    assert healthy.calls == 10
    assert sum(p.provenance == "model=dead" for p in pairs) == 1


def test_generation_failure_streak_resets_after_success():
    client = _FlakyClient("flaky", fail_on=[0, 1, 3, 4, 6, 7])
    pairs = generate_pseudo_parallel(_biased(9), [client], max_consecutive_failures=3)
    assert client.calls == 9
    assert len(pairs) == 3


class _FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class _FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Resp", (), {"text": outcome})()


def _gemini_with(outcomes, monkeypatch):
    import pytest
    pytest.importorskip("google.genai")
    from biasneut.data import pseudo_parallel

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr(pseudo_parallel.time, "sleep", sleeps.append)
    client = pseudo_parallel.GeminiClient(min_interval_s=1.0, max_retries=2)
    client._client = type("Fake", (), {"models": _FakeModels(outcomes)})()
    return client, sleeps


def test_gemini_retries_rate_limits_with_backoff(monkeypatch):
    client, sleeps = _gemini_with([_FakeAPIError(429), _FakeAPIError(503), " the regime fell "], monkeypatch)
    assert client.generate("p") == "the regime fell"
    assert client._client.models.calls == 3
    assert 1.0 in sleeps and 2.0 in sleeps  # backoff doubles per attempt


def test_gemini_gives_up_after_max_retries(monkeypatch):
    import pytest
    client, _ = _gemini_with([_FakeAPIError(429)] * 3, monkeypatch)
    with pytest.raises(_FakeAPIError):
        client.generate("p")


def test_gemini_does_not_retry_non_retryable_errors(monkeypatch):
    import pytest
    client, _ = _gemini_with([_FakeAPIError(400), "unused"], monkeypatch)
    with pytest.raises(_FakeAPIError):
        client.generate("p")
    assert client._client.models.calls == 1


def test_gemini_empty_response_is_an_error(monkeypatch):
    import pytest
    client, _ = _gemini_with([""], monkeypatch)
    with pytest.raises(ValueError, match="no text"):
        client.generate("p")
