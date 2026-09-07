import dataclasses
import typing
from pathlib import Path

import pytest

from biasneut.common.config import DetectorConfig, EditorConfig, EvalConfig, load_config, parse_cli_overrides, save_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_load_config_defaults_when_no_path():
    cfg = load_config(DetectorConfig, None)
    assert cfg.backbone == "launch/POLITICS"
    assert cfg.num_epochs == 5


def test_load_config_from_yaml(tmp_path):
    path = tmp_path / "detector.yaml"
    path.write_text("backbone: distilroberta-base\nnum_epochs: 2\n")
    cfg = load_config(DetectorConfig, path)
    assert cfg.backbone == "distilroberta-base"
    assert cfg.num_epochs == 2


def test_load_config_overrides_take_precedence(tmp_path):
    path = tmp_path / "detector.yaml"
    path.write_text("num_epochs: 2\n")
    cfg = load_config(DetectorConfig, path, overrides={"num_epochs": 99})
    assert cfg.num_epochs == 99


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "detector.yaml"
    path.write_text("not_a_real_field: 1\n")
    with pytest.raises(ValueError):
        load_config(DetectorConfig, path)


def test_parse_cli_overrides_type_coercion():
    overrides = parse_cli_overrides(["learning_rate=3e-4", "num_epochs=5", "backbone=roberta-base", "flag=true"])
    assert overrides["learning_rate"] == 3e-4
    assert overrides["num_epochs"] == 5
    assert overrides["backbone"] == "roberta-base"
    assert overrides["flag"] is True


def test_load_config_dotted_override_for_nested_field():
    cfg = load_config(DetectorConfig, None, overrides={"compute.gradient_checkpointing": False})
    assert cfg.compute.gradient_checkpointing is False
    assert cfg.compute.mixed_precision == "bf16"  # untouched sibling field keeps its default


def test_parse_cli_overrides_supports_dotted_keys():
    overrides = parse_cli_overrides(["compute.mixed_precision=fp16", "num_epochs=3"])
    cfg = load_config(DetectorConfig, None, overrides=overrides)
    assert cfg.compute.mixed_precision == "fp16"
    assert cfg.num_epochs == 3


def test_save_config_roundtrip(tmp_path):
    cfg = DetectorConfig(num_epochs=7)
    path = tmp_path / "out.yaml"
    save_config(cfg, path)
    reloaded = load_config(DetectorConfig, path)
    assert reloaded.num_epochs == 7
    assert dataclasses.asdict(reloaded)["compute"]["mixed_precision"] == "bf16"


def test_load_config_coerces_unquoted_scientific_notation_floats(tmp_path):
    # PyYAML's YAML-1.1 resolver only recognizes scientific notation as a
    # float with an explicit decimal point (`2.0e-5`); `2e-5` (no decimal
    # point) loads as a plain str. configs/detector.yaml used to be written
    # exactly this way, which crashed torch.optim.AdamW at training time
    # instead of failing loudly at config-load time.
    path = tmp_path / "detector.yaml"
    path.write_text("learning_rate: 2e-5\n")
    cfg = load_config(DetectorConfig, path)
    assert cfg.learning_rate == 2e-5
    assert isinstance(cfg.learning_rate, float)


@pytest.mark.parametrize("config_cls,filename", [
    (DetectorConfig, "detector.yaml"),
    (EditorConfig, "editor_strategy_a.yaml"),
    (EditorConfig, "editor_strategy_b.yaml"),
    (EditorConfig, "editor_strategy_c.yaml"),
    (EvalConfig, "eval.yaml"),
])
def test_real_config_files_deserialize_declared_field_types(config_cls, filename):
    loaded = load_config(config_cls, CONFIGS_DIR / filename)
    type_hints = typing.get_type_hints(config_cls)
    for field_name, value in dataclasses.asdict(loaded).items():
        if field_name == "compute":
            continue
        expected = type_hints.get(field_name)
        if expected in (float, int) and not isinstance(value, bool):
            assert isinstance(value, expected), (
                f"{filename}: {field_name} should be {expected.__name__}, got {type(value).__name__} ({value!r})"
            )
