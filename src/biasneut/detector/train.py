"""Stage-1 detector training loop (§5.1, §7.4 checkpoint/resume discipline)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from biasneut.common.checkpoint import CheckpointManager, TrainingState
from biasneut.common.compute import (
    autocast_context,
    build_optimizer,
    enable_gradient_checkpointing,
    get_device,
    resolve_dtype,
)
from biasneut.common.config import DetectorConfig
from biasneut.data.collate import DetectorCollator
from biasneut.data.schema import DetectionExample
from biasneut.detector.metrics import sentence_prf1, span_prf1
from biasneut.detector.model import BiasDetector

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: BiasDetector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_sentence_preds, all_sentence_golds = [], []
    all_token_preds, all_token_golds = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1

        all_sentence_preds.extend(out.sentence_logits.argmax(-1).cpu().tolist())
        all_sentence_golds.extend(batch["sentence_labels"].cpu().tolist())
        all_token_preds.extend(out.token_logits.argmax(-1).cpu().tolist())
        all_token_golds.extend(batch["token_labels"].cpu().tolist())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    metrics.update({f"sentence_{k}": v for k, v in sentence_prf1(all_sentence_preds, all_sentence_golds).items()})
    metrics.update({f"span_{k}": v for k, v in span_prf1(all_token_preds, all_token_golds).items()})
    model.train()
    return metrics


def train_detector(
    train_examples: list[DetectionExample],
    dev_examples: list[DetectionExample],
    cfg: DetectorConfig,
    resume: bool = True,
) -> BiasDetector:
    device = get_device()
    dtype = resolve_dtype(cfg.compute.mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = BiasDetector(
        backbone=cfg.backbone,
        sentence_loss_weight=cfg.sentence_loss_weight,
        span_loss_weight=cfg.span_loss_weight,
    ).to(device)
    if cfg.compute.gradient_checkpointing:
        enable_gradient_checkpointing(model.encoder)

    collator = DetectorCollator(tokenizer=tokenizer, max_length=cfg.max_seq_length)
    train_loader = DataLoader(train_examples, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_examples, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = build_optimizer(model, cfg.compute, cfg.learning_rate, cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs // max(cfg.compute.gradient_accumulation_steps, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg.warmup_ratio * total_steps), num_training_steps=total_steps
    )

    ckpt_mgr = CheckpointManager(cfg.output_dir)
    start_epoch, skip_steps, global_step = 0, 0, 0
    if resume:
        state = ckpt_mgr.load(model, optimizer, scheduler, map_location=device)
        if state is not None:
            start_epoch = state.data_cursor.get("epoch", 0)
            skip_steps = state.data_cursor.get("step_in_epoch", 0)
            global_step = state.step
            logger.info("Resumed from step %d (epoch %d, step_in_epoch %d)", global_step, start_epoch, skip_steps)

    model.train()
    for epoch in range(start_epoch, cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < skip_steps:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_context(dtype):
                out = model(**batch)
                loss = out.loss / cfg.compute.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.compute.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.checkpoint_every_steps == 0:
                    ckpt_mgr.save(
                        global_step, model, optimizer, scheduler,
                        TrainingState(step=global_step, epoch=epoch,
                                       data_cursor={"epoch": epoch, "step_in_epoch": step + 1}),
                    )

        skip_steps = 0
        metrics = evaluate(model, dev_loader, device)
        logger.info("epoch %d dev metrics: %s", epoch, metrics)
        ckpt_mgr.save(
            global_step, model, optimizer, scheduler,
            TrainingState(step=global_step, epoch=epoch + 1, data_cursor={"epoch": epoch + 1, "step_in_epoch": 0},
                          best_metric=metrics["span_f1"]),
        )

    return model


def save_pretrained(model: BiasDetector, tokenizer, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    # Persist the encoder's own config so DetectorInference.from_pretrained
    # can rebuild the exact architecture without redownloading the original
    # pretrained backbone (see biasneut.detector.model.BiasDetector's
    # encoder_config path).
    model.encoder.config.save_pretrained(output_dir)
