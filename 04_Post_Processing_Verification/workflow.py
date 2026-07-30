from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import load_json_list, load_prompt, parse_json_object, save_json


DETECT_PROMPT = load_prompt("detection_prompt.txt")
NCF_REASON_PROMPT = load_prompt("ncf_reason_prompt.txt")
TRACE_SUPPORT_PROMPT = load_prompt("trace_support_prompt.txt")


def load_model(base_model: str, adapter: Path | None, max_length: int, load_in_4bit: bool):
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are a precise JSON-only assistant."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=False,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_detection(text: str) -> tuple[bool | None, str]:
    parsed = parse_json_object(text)
    if parsed is None:
        trailing = re.search(r"\{.*\}\s*$", text, re.S)
        if trailing:
            try:
                candidate = json.loads(trailing.group(0))
                parsed = candidate if isinstance(candidate, dict) else None
            except json.JSONDecodeError:
                parsed = None
    if parsed:
        value = parsed.get("is_bug", parsed.get("is_ncf_bug"))
        if isinstance(value, bool):
            return value, str(parsed.get("reason", parsed.get("explanation", "")) or "")
        if isinstance(value, str) and value.strip().lower() in {"yes", "no", "true", "false", "1", "0"}:
            return value.strip().lower() in {"yes", "true", "1"}, str(
                parsed.get("reason", parsed.get("explanation", "")) or ""
            )
    match = re.search(r'"?(?:is_bug|is_ncf_bug)"?\s*:\s*"?(yes|no|true|false|1|0)', text, re.I)
    if not match:
        return None, ""
    return match.group(1).lower() in {"yes", "true", "1"}, ""


def load_test_samples(buggy_path: Path, bug_free_path: Path, excluded_ids: set[str]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for prefix, path in (("tr", buggy_path), ("fl", bug_free_path)):
        for item in load_json_list(path):
            identifier = str(item.get("id"))
            if identifier in excluded_ids:
                continue
            samples.append(
                {
                    "id": f"{prefix}_{identifier}",
                    "trace": str(item.get("trace") or ""),
                }
            )
    return samples


def run_detection(args: argparse.Namespace) -> Path:
    model, tokenizer = load_model(
        args.base_model, args.adapter, args.max_length, args.load_in_4bit
    )
    samples = load_test_samples(args.buggy, args.bug_free, set(args.exclude_id))
    results: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="FineDroid inference"):
        raw = generate(
            model,
            tokenizer,
            DETECT_PROMPT.format(actions=sample["trace"]),
            args.max_length - args.max_new_tokens,
            args.max_new_tokens,
        )
        is_bug, reason = parse_detection(raw)
        results.append(
            {
                "id": sample["id"],
                "is_bug": is_bug,
                "reason": reason,
                "generated": raw,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "predictions.json"
    save_json(path, results)
    print(f"Saved {len(results)} predictions to {path}")
    return path


def bool_field(text: str, field: str) -> bool:
    parsed = parse_json_object(text)
    if isinstance(parsed, dict) and isinstance(parsed.get(field), bool):
        return parsed[field]
    match = re.search(rf'"{re.escape(field)}"\s*:\s*(true|false)', text, re.I)
    return bool(match and match.group(1).lower() == "true")


def verify_predictions(args: argparse.Namespace, predictions_path: Path | None = None) -> Path:
    predictions_path = predictions_path or args.predictions
    if predictions_path is None:
        raise ValueError("--predictions is required when running verification independently")
    predictions = load_json_list(predictions_path)
    samples = load_test_samples(args.buggy, args.bug_free, set(args.exclude_id))
    traces = {sample["id"]: sample["trace"] for sample in samples}
    model, tokenizer = load_model(
        args.verification_model,
        args.verification_adapter,
        args.max_length,
        args.load_in_4bit,
    )
    records: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []
    for prediction in tqdm(predictions, desc="Reason verification"):
        final = dict(prediction)
        if prediction.get("is_bug") is not True:
            final_predictions.append(final)
            continue
        reason = str(prediction.get("reason") or "")
        trace = traces.get(str(prediction.get("id")), "")
        ncf_raw = generate(
            model,
            tokenizer,
            NCF_REASON_PROMPT.format(user_input_text=reason),
            2048,
            args.max_new_tokens,
        )
        is_ncf = bool_field(ncf_raw, "is_ncf_bug")
        support_raw = ""
        is_supported = False
        if is_ncf:
            support_raw = generate(
                model,
                tokenizer,
                TRACE_SUPPORT_PROMPT.format(
                    full_trace_text=trace,
                    user_input_text=reason,
                ),
                args.max_length - args.max_new_tokens,
                args.max_new_tokens,
            )
            is_supported = bool_field(support_raw, "is_correct")
        keep = is_ncf and is_supported
        if not keep:
            final["is_bug"] = False
            final["verification_rejected"] = True
        records.append(
            {
                "id": prediction.get("id"),
                "ncf_output": ncf_raw,
                "is_ncf_bug": is_ncf,
                "support_output": support_raw,
                "is_supported": is_supported,
                "kept": keep,
            }
        )
        final_predictions.append(final)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verification_path = args.output_dir / "reason_verification.json"
    final_path = args.output_dir / "predictions_verified.json"
    save_json(verification_path, records)
    save_json(final_path, final_predictions)
    print(f"Saved verification records to {verification_path}")
    print(f"Saved final predictions to {final_path}")
    return final_path

