import pytest


@pytest.fixture(autouse=True)
def _wandb_off_by_default(monkeypatch):
    # Unit tests must never create real W&B runs; tracking tests opt back in explicitly.
    monkeypatch.setenv("BIASNEUT_WANDB", "0")
