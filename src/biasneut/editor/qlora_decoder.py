"""Stretch editor arm (§5.2, §7.2, §7.3): QLoRA'd 1-3B instruction-tuned
decoder for higher fluency than the small seq2seq editors, still feasible on
a single free T4.

QLoRA = 4-bit NF4 quantization of the frozen base + double quantization +
LoRA adapters trained in bf16 + a paged optimizer (bitsandbytes). None of the
core pipeline needs this (§7.3) — it's optional and gated behind the
``qlora`` extra (``pip install -e ".[qlora]"``) so the rest of the codebase
has no hard bitsandbytes dependency.

Checkpointing here intentionally bypasses ``CheckpointManager``: saving a
full state_dict of a 4-bit-quantized base model is both wasteful and
fragile, whereas a LoRA adapter is a few tens of MB and PEFT already knows
how to (de)serialize it — so we save/resume just the adapter + step counter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biasneut.common.config import QLoRAEditorConfig
from biasneut.data.schema import EditExample

logger = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n"
    "Rewrite the sentence to remove loaded political language while preserving its meaning.\n\n"
    "### Sentence:\n{source}\n\n### Neutralized:\n"
)


def _load_quantized_causal_lm(backbone: str):
    import bitsandbytes  # noqa: F401  (import error here is the clear failure point if extra not installed)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForCausalLM.from_pretrained(backbone, quantization_config=bnb_config, device_map="auto")


def build_qlora_model(cfg: QLoRAEditorConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_quantized_causal_lm(cfg.backbone)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


class _InstructionCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[EditExample]) -> dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        for ex in batch:
            prompt = INSTRUCTION_TEMPLATE.format(source=ex.source)
            full_text = prompt + ex.target + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False, truncation=True,
                                       max_length=self.max_length)["input_ids"]

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(lbls)] = torch.tensor(lbls)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _adapter_checkpoint_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "adapter"


def train_qlora_editor(
    train_examples: list[EditExample],
    cfg: QLoRAEditorConfig,
    resume: bool = True,
) -> tuple:
    import bitsandbytes as bnb

    model, tokenizer = build_qlora_model(cfg)
    collator = _InstructionCollator(tokenizer, cfg.max_seq_length)
    loader = DataLoader(train_examples, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collator)

    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    start_epoch, global_step = 0, 0

    if resume and _adapter_checkpoint_dir(output_dir).exists() and progress_path.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, _adapter_checkpoint_dir(output_dir), is_trainable=True)
        progress = json.loads(progress_path.read_text())
        start_epoch, global_step = progress["epoch"], progress["step"]
        logger.info("Resumed QLoRA editor from epoch %d, step %d", start_epoch, global_step)

    model.train()
    for epoch in range(start_epoch, cfg.num_epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        model.save_pretrained(_adapter_checkpoint_dir(output_dir))
        progress_path.write_text(json.dumps({"epoch": epoch + 1, "step": global_step}))
        logger.info("QLoRA editor epoch %d complete (step %d)", epoch, global_step)

    return model, tokenizer
