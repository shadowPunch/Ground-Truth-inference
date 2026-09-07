"""Generic seq2seq fine-tuning loop shared by Strategy A (WNC pretrain +
adapt) and Strategy B (LLM-synthesized pseudo-parallel) — they differ only in
which `EditExample`s are passed in and whether a checkpoint is used to warm
start (§5.2, §7.4)."""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from biasneut.common.checkpoint import CheckpointManager, TrainingState
from biasneut.common.compute import (
    autocast_context,
    build_optimizer,
    enable_gradient_checkpointing,
    get_device,
    resolve_dtype,
)
from biasneut.common.config import EditorConfig
from biasneut.data.schema import EditExample
from biasneut.editor.collate import EditorCollator
from biasneut.editor.seq2seq_model import SeqEditor

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_loss(editor: SeqEditor, loader: DataLoader, device: torch.device) -> float:
    editor.model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = editor.model(**batch)
        total += out.loss.item()
        n += 1
    editor.model.train()
    return total / max(n, 1)


def train_seq2seq_editor(
    train_examples: list[EditExample],
    dev_examples: list[EditExample],
    cfg: EditorConfig,
    editor: SeqEditor | None = None,
    resume: bool = True,
) -> SeqEditor:
    if not train_examples:
        raise ValueError(
            "train_seq2seq_editor got 0 training examples. For Strategy C/LEWIS this "
            "usually means the detector flagged no biased spans in the candidate set "
            "(synthesize_lewis_pairs skips any sentence with an all-'O' prediction) — "
            "check the detector's span recall before assuming this is a data bug."
        )

    device = get_device()
    dtype = resolve_dtype(cfg.compute.mixed_precision)

    if editor is None:
        editor = SeqEditor.from_pretrained_backbone(cfg.backbone)
    editor.to(device)
    if cfg.compute.gradient_checkpointing:
        enable_gradient_checkpointing(editor.model)

    collator = EditorCollator(
        tokenizer=editor.tokenizer,
        max_source_length=cfg.max_source_length,
        max_target_length=cfg.max_target_length,
        task_prefix=editor.task_prefix,
    )
    train_loader = DataLoader(train_examples, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_examples, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = build_optimizer(editor.model, cfg.compute, cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs // max(cfg.compute.gradient_accumulation_steps, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg.warmup_ratio * total_steps), num_training_steps=max(total_steps, 1)
    )

    ckpt_mgr = CheckpointManager(cfg.output_dir)
    start_epoch, skip_steps, global_step = 0, 0, 0
    if resume:
        state = ckpt_mgr.load(editor.model, optimizer, scheduler, map_location=device)
        if state is not None:
            start_epoch = state.data_cursor.get("epoch", 0)
            skip_steps = state.data_cursor.get("step_in_epoch", 0)
            global_step = state.step
            logger.info("Resumed editor from step %d (epoch %d)", global_step, start_epoch)

    editor.model.train()
    for epoch in range(start_epoch, cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < skip_steps:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_context(dtype):
                out = editor.model(**batch)
                loss = out.loss / cfg.compute.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.compute.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(editor.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.checkpoint_every_steps == 0:
                    ckpt_mgr.save(
                        global_step, editor.model, optimizer, scheduler,
                        TrainingState(step=global_step, epoch=epoch,
                                       data_cursor={"epoch": epoch, "step_in_epoch": step + 1}),
                    )

        skip_steps = 0
        dev_loss = evaluate_loss(editor, dev_loader, device)
        logger.info("epoch %d dev loss: %.4f", epoch, dev_loss)
        ckpt_mgr.save(
            global_step, editor.model, optimizer, scheduler,
            TrainingState(step=global_step, epoch=epoch + 1, data_cursor={"epoch": epoch + 1, "step_in_epoch": 0},
                          best_metric=dev_loss),
        )

    return editor
