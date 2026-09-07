"""Memory-reduction stack helpers (§7.2): precision, checkpointing, 8-bit optim.

Each function degrades gracefully when the ideal option isn't available (e.g.
bf16 on a pre-Turing GPU, or bitsandbytes not installed) so the same config
works on a P100, a T4, or a laptop GPU without edits.
"""
from __future__ import annotations

import logging

import torch

from biasneut.common.config import ComputeConfig

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(mixed_precision: str) -> torch.dtype:
    """bf16 requires tensor cores (Turing+); P100 (Pascal) silently has none,
    so callers on a P100 should request "fp16" per §7.2 point 3."""
    if mixed_precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning("bf16 requested but not supported on this device; falling back to fp32")
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        logger.warning("%s has no gradient_checkpointing_enable()", type(model).__name__)


def build_optimizer(model: torch.nn.Module, cfg: ComputeConfig, learning_rate: float,
                     weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """AdamW, 8-bit via bitsandbytes when requested and available (§7.2 point 4)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
        except ImportError:
            logger.warning("bitsandbytes not installed; falling back to torch.optim.AdamW")
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def autocast_context(dtype: torch.dtype):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    enabled = dtype in (torch.bfloat16, torch.float16)
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
