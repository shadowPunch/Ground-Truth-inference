import torch

from biasneut.common.checkpoint import CheckpointManager, TrainingState


def _tiny_model():
    return torch.nn.Linear(4, 2)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(tmp_path, keep_last=2)

    mgr.save(10, model, optimizer, None, TrainingState(step=10, epoch=0, data_cursor={"epoch": 0, "step_in_epoch": 10}))

    new_model = _tiny_model()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    state = mgr.load(new_model, new_optimizer)

    assert state.step == 10
    assert state.data_cursor == {"epoch": 0, "step_in_epoch": 10}
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_load_returns_none_when_empty(tmp_path):
    mgr = CheckpointManager(tmp_path / "empty", keep_last=2)
    model = _tiny_model()
    assert mgr.load(model) is None


def test_checkpoint_prunes_old_checkpoints(tmp_path):
    model = _tiny_model()
    mgr = CheckpointManager(tmp_path, keep_last=2)
    for step in [10, 20, 30]:
        mgr.save(step, model, None, None, TrainingState(step=step, epoch=0, data_cursor={}))

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert len(remaining) == 2
    assert remaining == ["checkpoint-00000020", "checkpoint-00000030"]


def test_checkpoint_latest_picks_highest_step(tmp_path):
    model = _tiny_model()
    mgr = CheckpointManager(tmp_path, keep_last=5)
    for step in [5, 50, 15]:
        mgr.save(step, model, None, None, TrainingState(step=step, epoch=0, data_cursor={}))
    assert mgr.latest().name == "checkpoint-00000050"
