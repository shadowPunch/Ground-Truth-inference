"""Dataclass configs + YAML (de)serialization.

Kept deliberately simple (no Hydra/OmegaConf dependency) — every config is a
plain dataclass with defaults matching the proposal's §7 compute plan, loaded
from a YAML file and overridable via CLI ``--set key=value`` pairs in scripts.
"""
from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _coerce(value: str) -> Any:
    """Best-effort str -> {bool,int,float,str} coercion for CLI overrides."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _expand_dotted_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Turn CLI-style dotted overrides (``compute.gradient_checkpointing=false``)
    into nested dicts (``{"compute": {"gradient_checkpointing": False}}``) so
    they merge cleanly with any nested config loaded from YAML."""
    expanded: dict[str, Any] = {}
    for key, value in data.items():
        if "." in key:
            head, rest = key.split(".", 1)
            expanded.setdefault(head, {})
            if not isinstance(expanded[head], dict):
                raise ValueError(f"Conflicting override for {head!r}: both a scalar and a nested key given")
            expanded[head][rest] = value
        else:
            expanded[key] = value
    # Recurse so multi-level dotted keys (a.b.c) also expand correctly.
    for key, value in list(expanded.items()):
        if isinstance(value, dict):
            expanded[key] = _expand_dotted_keys(value)
    return expanded


def _coerce_scalar_to_field_type(value: Any, field_type: Any) -> Any:
    """PyYAML's default (YAML-1.1) resolver only recognizes scientific notation
    as a float when it has an explicit decimal point (``2.0e-5``, not
    ``2e-5``) — the latter silently loads as a ``str``. Rather than rely on
    every config file getting that spelling right, coerce int/float fields
    back to their declared type when YAML handed us a string instead."""
    if field_type in (float, int) and isinstance(value, str):
        try:
            return field_type(value)
        except ValueError:
            return value
    return value


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively build a dataclass from a dict, reconstructing any nested
    dataclass fields (e.g. ``EditorConfig.compute: ComputeConfig``) instead of
    leaving them as plain dicts — required for the save_config/load_config
    round trip to actually produce a usable config object."""
    data = _expand_dotted_keys(data)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    type_hints = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        field_type = type_hints.get(key)
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = _coerce_scalar_to_field_type(value, field_type)
    return cls(**kwargs)


def load_config(cls: type[T], path: str | Path | None, overrides: dict[str, Any] | None = None) -> T:
    """Build a dataclass ``cls`` from an optional YAML file plus overrides."""
    data: dict[str, Any] = {}
    if path is not None and Path(path).exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data.update(overrides)
    return _from_dict(cls, data)


def parse_cli_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``["lr=3e-4", "epochs=5"]`` style CLI overrides into a dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = _coerce(value)
    return out


def save_config(cfg: Any, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)


@dataclasses.dataclass
class ComputeConfig:
    """§7.2 memory-reduction stack, as knobs rather than hardcoded behavior."""

    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no" — bf16 needs Turing+/tensor cores
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"  # "adamw_8bit" | "adamw_torch"
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 128
    seed: int = 42


@dataclasses.dataclass
class DetectorConfig:
    """Stage 1 — sentence + span detector (§5.1)."""

    backbone: str = "launch/POLITICS"
    fallback_backbone: str = "distilroberta-base"
    max_seq_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 5
    warmup_ratio: float = 0.06
    span_loss_weight: float = 1.0
    sentence_loss_weight: float = 1.0
    output_dir: str = "runs/detector"
    checkpoint_every_steps: int = 200
    compute: ComputeConfig = dataclasses.field(default_factory=ComputeConfig)


@dataclasses.dataclass
class EditorConfig:
    """Stage 2 — seq2seq editor, shared across strategies A/B/C (§5.2)."""

    strategy: str = "a"  # "a" | "b" | "c"
    backbone: str = "t5-small"
    max_source_length: int = 128
    max_target_length: int = 128
    learning_rate: float = 3e-4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 4
    warmup_ratio: float = 0.06
    span_marker_start: str = "<bias>"
    span_marker_end: str = "</bias>"
    use_constrained_decoding: bool = True
    copy_bias_strength: float = 2.5
    num_beams: int = 4
    output_dir: str = "runs/editor"
    checkpoint_every_steps: int = 200
    compute: ComputeConfig = dataclasses.field(default_factory=ComputeConfig)


@dataclasses.dataclass
class QLoRAEditorConfig:
    """Stretch arm (§5.2, §7.3) — QLoRA'd 1-3B instruction-tuned decoder editor."""

    backbone: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_seq_length: int = 256
    learning_rate: float = 2e-4
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    output_dir: str = "runs/editor_qlora"


@dataclasses.dataclass
class EvalConfig:
    """§6 evaluation protocol."""

    independent_classifier_dir: str = "runs/detector_eval_independent"
    fluency_lm: str = "gpt2"
    grammaticality_model: str = "textattack/roberta-base-CoLA"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bertscore_model: str = "roberta-large"
    num_bootstrap_samples: int = 1000
    seeds: tuple[int, ...] = (13, 42, 1337)
    output_dir: str = "runs/eval"
