from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import load_json_list, load_prompt


SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}
PLAIN_PROMPT = load_prompt("training_prompt.txt")
REASONING_PROMPT = load_prompt("training_reasoning_prompt.txt")


def inspect_dataset(path: Path, with_reasoning: bool) -> list[dict[str, Any]]:
    rows = load_json_list(path)
    if not rows:
        raise ValueError(f"Training dataset is empty: {path}")
    invalid: list[int] = []
    for index, row in enumerate(rows):
        judgement = row.get("judgement")
        valid = bool(str(row.get("gen_trace") or "").strip()) and isinstance(judgement, dict)
        if with_reasoning:
            valid = valid and isinstance(row.get("thoughts"), (str, list, dict))
        if not valid:
            invalid.append(index)
    if invalid:
        preview = ", ".join(map(str, invalid[:10]))
        raise ValueError(f"Dataset has {len(invalid)} invalid records; first indices: {preview}")
    return rows


def normalize_label(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in {"yes", "true", "1"}:
        return "Yes"
    if isinstance(value, (bool, int)) and not isinstance(value, str):
        return "Yes" if bool(value) else "No"
    return "No"


def normalize_reasoning(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            if isinstance(item, dict)
            else str(item)
            for item in value
        ).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def train(args: argparse.Namespace) -> None:
    if args.max_length is None:
        args.max_length = 8192 if args.with_reasoning else 16384
    rows = inspect_dataset(args.data, args.with_reasoning)
    print(f"Loaded {len(rows)} training samples from {args.data}")
    if args.dry_run:
        return

    if args.cuda_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_template = REASONING_PROMPT if args.with_reasoning else PLAIN_PROMPT

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        user_message = {
            "role": "user",
            "content": prompt_template.format(actions=example.get("gen_trace", "")),
        }
        prompt_text = tokenizer.apply_chat_template(
            [SYSTEM_MESSAGE, user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.with_reasoning,
        )
        judgement = example.get("judgement") or {}
        target = {
            "is_bug": normalize_label(judgement.get("is_bug", "No")),
            "reason": str(judgement.get("reason") or "").strip(),
        }
        if args.with_reasoning:
            output_text = (
                f"<think>\n{normalize_reasoning(example.get('thoughts'))}\n</think>\n\n"
                + json.dumps(target, ensure_ascii=False)
                + tokenizer.eos_token
            )
        else:
            output_text = json.dumps(target, ensure_ascii=False) + tokenizer.eos_token

        full = tokenizer(
            prompt_text + output_text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = min(len(prompt_ids), len(full["input_ids"]))
        full["labels"] = [-100] * prompt_length + full["input_ids"][prompt_length:]
        return full

    dataset = Dataset.from_list(rows).map(
        format_example,
        remove_columns=list(rows[0].keys()),
        desc="Tokenizing training data",
    )
    dataset = dataset.filter(
        lambda row: bool(row["labels"]) and any(token != -100 for token in row["labels"]),
        desc="Dropping fully truncated records",
    )
    if len(dataset) != len(rows):
        print(f"Retained {len(dataset)}/{len(rows)} examples after tokenization")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    training_arguments = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved the LoRA adapter and tokenizer to {args.output}")
