from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from common import (
    load_json_list,
    load_prompt,
    normalize_yes_no,
    parse_json_object,
    save_json,
)


BUGGY_TRACE_INSTRUCTIONS = load_prompt("buggy_trace_instructions.txt")
BUGGY_TRACE_EXAMPLE = load_prompt("buggy_trace_example.txt")
BUG_FREE_TRACE_INSTRUCTIONS = load_prompt("bug_free_trace_instructions.txt")
BUG_FREE_TRACE_EXAMPLE = load_prompt("bug_free_trace_example.txt")
BUGGY_VERIFY_STEP1 = load_prompt("verify_buggy_step1.txt")
BUGGY_VERIFY_STEP2 = load_prompt("verify_buggy_step2.txt")
BUG_FREE_VERIFY_SYSTEM = load_prompt("verify_bug_free_system.txt")
BUG_FREE_VERIFY_EXAMPLE = load_prompt("verify_bug_free_example.txt")
REASONING_PROMPT = load_prompt("reasoning_instruction.txt")


def openai_client(api_key: str):
    from openai import OpenAI

    if not api_key.strip():
        raise RuntimeError("Set OPENAI_API_KEY near the top of this module's run.py")
    return OpenAI(api_key=api_key)


def call_openai_text(
    client: Any,
    model: str,
    instructions: str,
    prompt: str,
    json_mode: bool = False,
    retries: int = 3,
) -> tuple[str, list[str]]:
    for attempt in range(1, retries + 1):
        try:
            request: dict[str, Any] = {
                "model": model,
                "instructions": instructions,
                "input": prompt,
            }
            if json_mode:
                request["text"] = {"format": {"type": "json_object"}}
            if model.startswith("gpt-5"):
                request["reasoning"] = {"summary": "auto"}
            response = client.responses.create(**request)
            summaries: list[str] = []
            for output in getattr(response, "output", []) or []:
                if getattr(output, "type", None) != "reasoning":
                    continue
                for summary in getattr(output, "summary", []) or []:
                    text = getattr(summary, "text", None)
                    if text:
                        summaries.append(text)
            return response.output_text or "", summaries
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def normalize_trace(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"(?<!\n)'action':", "\n'action':", text)
    return text + ("\n" if text else "")


def process_with_checkpoint(
    input_path: Path,
    output_path: Path,
    resume: bool,
    limit: int | None,
    processor: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    if input_path.is_dir():
        paths = sorted(input_path.glob("filtered_BR_*.json"))
        if not paths:
            raise FileNotFoundError(f"No filtered_BR_*.json files found in {input_path}")
        source = [item for path in paths for item in load_json_list(path)]
    else:
        source = load_json_list(input_path)
    completed = load_json_list(output_path) if resume and output_path.exists() else []
    start = len(completed)
    if start > len(source):
        raise ValueError("Resume output contains more records than the source input")
    stop = len(source) if limit is None else min(len(source), start + limit)
    results = completed
    for index in tqdm(range(start, stop), desc=output_path.stem):
        results.append(processor(dict(source[index])))
        save_json(output_path, results)
    print(f"Saved {len(results)} records to {output_path}")


def generate_buggy(args: argparse.Namespace) -> None:
    client = openai_client(args.api_key)

    def processor(item: dict[str, Any]) -> dict[str, Any]:
        title = str(item.get("title") or "")
        body = str(item.get("body") or "")
        prompt = (
            "Below is a full example for your reference, including bug report and "
            "Comeplete UI Info Interaction Trace.\n"
            f"{BUGGY_TRACE_EXAMPLE.strip()}\n"
            "Now generate Comeplete UI Info Interaction Trace for the new case below.\n\n"
            "Bug Report:\n-----------\n"
            f"Title: {title}\nBody:\n{body}\n\n"
        )
        output, summaries = call_openai_text(
            client, args.model, BUGGY_TRACE_INSTRUCTIONS, prompt
        )
        item["gen_trace"] = normalize_trace(output)
        item["reasoning_summary"] = summaries
        return item

    process_with_checkpoint(args.input, args.output, args.resume, args.limit, processor)


def verify_buggy(args: argparse.Namespace) -> None:
    client = openai_client(args.api_key)

    def processor(item: dict[str, Any]) -> dict[str, Any]:
        trace = str(item.get("gen_trace") or "")
        if not trace.strip():
            item["judgement"] = {
                "is_ncf_bug": "no",
                "same_as_BR": "no",
                "reason": "Empty generated trace.",
            }
            return item
        raw1, reasoning1 = call_openai_text(
            client,
            args.model,
            "Return one JSON object and no other text.",
            BUGGY_VERIFY_STEP1.format(trace=trace),
            json_mode=True,
        )
        step1 = parse_json_object(raw1) or {"is_bug": "no", "bugs": []}
        bugs = step1.get("bugs") if isinstance(step1.get("bugs"), list) else []
        is_bug = normalize_yes_no(step1.get("is_bug", "no")) == "yes" and bool(bugs)
        step2: dict[str, Any] | None = None
        raw2 = ""
        reasoning2: list[str] = []
        if is_bug:
            raw2, reasoning2 = call_openai_text(
                client,
                args.model,
                "Return one JSON object and no other text.",
                BUGGY_VERIFY_STEP2.format(
                    title=item.get("title", ""),
                    body=item.get("body", ""),
                    bugs=json.dumps(bugs, ensure_ascii=False, indent=2),
                ),
                json_mode=True,
            )
            step2 = parse_json_object(raw2) or {}
        matched = step2.get("matched_bug") if isinstance(step2, dict) else None
        description = matched.get("description") if isinstance(matched, dict) else None
        item["judgement"] = {
            "is_ncf_bug": "yes" if is_bug else "no",
            "same_as_BR": (step2 or {}).get("same_as_BR", "no"),
            "reason": description or (step2 or {}).get("reason") or "No NCF bug detected.",
        }
        item["verification"] = {
            "step1_output": raw1,
            "step1_reasoning_summary": reasoning1,
            "step2_output": raw2,
            "step2_reasoning_summary": reasoning2,
        }
        return item

    process_with_checkpoint(args.input, args.output, args.resume, args.limit, processor)


def extract_bug_reason(item: dict[str, Any]) -> str:
    judgement = item.get("judgement")
    if isinstance(judgement, dict):
        return str(judgement.get("reason") or "")
    if isinstance(judgement, str):
        parsed = parse_json_object(judgement)
        if parsed:
            return str(parsed.get("reason") or "")
        return judgement
    return ""


def generate_bug_free(args: argparse.Namespace) -> None:
    client = openai_client(args.api_key)

    def processor(item: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Below is a full example for your reference, including bug report, bug reason "
            "and UI Info Interaction Traces.\n"
            f"{BUG_FREE_TRACE_EXAMPLE.strip()}\n"
            "Now fix the buggy UI Info Interaction Trace for the new case below and produce "
            "a bug-free UI Info Interaction Trace only.\n\n"
            "Bug Report:\n-----------\n"
            f"Title: {item.get('title', '')}\nBody:\n{item.get('body', '')}\n\n"
            "Bug Reason:\n----------------------------\n"
            f"{extract_bug_reason(item).strip()}\n\n"
            "Buggy UI Info Interaction Trace (to be fixed):\n"
            "---------------------------------------------\n"
            f"{str(item.get('gen_trace', '')).strip()}\n\n"
            "Requirements:\n"
            "- Keep the intent of the original case.\n"
            "- Fix issues indicated by the bug reason and any inconsistencies in the buggy trace.\n"
            "- Produce a clean, self-consistent, bug-free UI Info Interaction Trace.\n"
            "- Do not include explanations; output the final trace only.\n"
        )
        output, summaries = call_openai_text(
            client, args.model, BUG_FREE_TRACE_INSTRUCTIONS, prompt
        )
        item["bug-free-trace"] = normalize_trace(output)
        item["bug_free_reasoning_summary"] = summaries
        return item

    process_with_checkpoint(args.input, args.output, args.resume, args.limit, processor)


def verify_bug_free(args: argparse.Namespace) -> None:
    client = openai_client(args.api_key)

    def processor(item: dict[str, Any]) -> dict[str, Any]:
        trace = str(item.get("bug-free-trace") or "")
        if not trace.strip():
            parsed = {
                "is_ncf_bug": True,
                "categories": [],
                "evidence": [],
                "explanation": "The bug-free trace is empty.",
            }
            raw = ""
            summaries: list[str] = []
        else:
            prompt = (
                BUG_FREE_VERIFY_EXAMPLE.rstrip()
                + trace
                + "\nRespond with a single JSON object exactly matching the schema. "
                "No code fences or extra text. Evidence <= 3 items; explanation <= 60 words.\n"
            )
            raw, summaries = call_openai_text(
                client,
                args.model,
                BUG_FREE_VERIFY_SYSTEM,
                prompt,
                json_mode=True,
            )
            parsed = parse_json_object(raw) or {
                "is_ncf_bug": True,
                "categories": [],
                "evidence": [],
                "explanation": "The verifier output could not be parsed.",
            }
        item["bugfree_judgement"] = parsed
        item["bugfree_verification"] = {
            "output": raw,
            "reasoning_summary": summaries,
        }
        return item

    process_with_checkpoint(args.input, args.output, args.resume, args.limit, processor)


def gemini_reasoning(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    if not args.api_key.strip():
        raise RuntimeError("Set GEMINI_API_KEY near the top of this module's run.py")
    client = genai.Client(api_key=args.api_key)
    fields = args.trace_fields

    def processor(item: dict[str, Any]) -> dict[str, Any]:
        reasoning = item.get("reasoning_by_trace")
        if not isinstance(reasoning, dict):
            reasoning = {}
        for field in fields:
            trace = str(item.get(field) or "").strip()
            if not trace:
                reasoning[field] = {"thoughts": [], "answer": "", "parsed": {}}
                continue
            response = client.models.generate_content(
                model=args.model,
                contents=REASONING_PROMPT + trace,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )
            thoughts: list[str] = []
            answers: list[str] = []
            for part in getattr(response.candidates[0].content, "parts", []) or []:
                text = getattr(part, "text", None)
                if not text:
                    continue
                (thoughts if bool(getattr(part, "thought", False)) else answers).append(text.strip())
            answer = "\n".join(answers)
            reasoning[field] = {
                "thoughts": thoughts,
                "answer": answer,
                "parsed": parse_json_object(answer) or {},
            }
            time.sleep(args.rate_pause)
        item["reasoning_by_trace"] = reasoning
        return item

    process_with_checkpoint(args.input, args.output, args.resume, args.limit, processor)


def accepted_buggy(item: dict[str, Any]) -> bool:
    judgement = item.get("judgement")
    if not isinstance(judgement, dict):
        return False
    try:
        is_bug = normalize_yes_no(judgement.get("is_ncf_bug", judgement.get("is_bug", "no"))) == "yes"
    except ValueError:
        return False
    return is_bug and str(judgement.get("same_as_BR", "yes")).lower() in {"yes", "partial"}


def accepted_bug_free(item: dict[str, Any]) -> bool:
    judgement = item.get("bugfree_judgement")
    return isinstance(judgement, dict) and judgement.get("is_ncf_bug") is False


def combine(args: argparse.Namespace) -> None:
    items = load_json_list(args.input)
    output: list[dict[str, Any]] = []
    rejected = 0
    for item in items:
        buggy_trace = str(item.get("gen_trace") or "").strip()
        clean_trace = str(item.get("bug-free-trace") or "").strip()
        if not buggy_trace or not clean_trace or not accepted_buggy(item) or not accepted_bug_free(item):
            rejected += 1
            continue
        reasoning = item.get("reasoning_by_trace")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        buggy_reasoning = reasoning.get("gen_trace") if isinstance(reasoning.get("gen_trace"), dict) else {}
        clean_reasoning = reasoning.get("bug-free-trace") if isinstance(reasoning.get("bug-free-trace"), dict) else {}
        output.extend(
            [
                {
                    "gen_trace": buggy_trace,
                    "judgement": {
                        "is_bug": "yes",
                        "reason": extract_bug_reason(item),
                    },
                    "thoughts": buggy_reasoning.get("thoughts", []),
                },
                {
                    "gen_trace": clean_trace,
                    "judgement": {
                        "is_bug": "no",
                        "reason": str((clean_reasoning.get("parsed") or {}).get("reason") or "No NCF bug detected."),
                    },
                    "thoughts": clean_reasoning.get("thoughts", []),
                },
            ]
        )
    if args.shuffle:
        random.Random(args.seed).shuffle(output)
    save_json(args.output, output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Saved {len(output)} examples ({len(output) // 2} accepted pairs); rejected {rejected} source records")
    print(f"SHA-256: {digest}")
