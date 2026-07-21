#!/usr/bin/env python3
"""Recompute the three comparison tables from the released case-level labels."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ("TPC (%)", "TPW (%)", "FP (%)", "FA (%)")


TABLES = {
    "open_source_llms": [
        ("Llama3.1-8B-Instruct", ["open_source_llms/llama-3.1-8b-instruct/labels.csv"]),
        ("Mistral-7B-Instruct-v0.3", ["open_source_llms/mistral-7b-instruct-v0.3/labels.csv"]),
        ("Phi-4", ["open_source_llms/phi-4-14b/labels.csv"]),
        ("Gemma3-12B-IT", ["open_source_llms/gemma-3-12b-it/labels.csv"]),
        ("Qwen2.5-14B", ["open_source_llms/qwen2.5-14b/labels.csv"]),
        ("Qwen3-14B (No Thinking)", ["open_source_llms/qwen3-14b-no-thinking/labels.csv"]),
        ("Qwen3-14B (Thinking)", ["open_source_llms/qwen3-14b-thinking/labels.csv"]),
        ("FineDroid", ["finedroid/labels.csv"]),
    ],
    "commercial_llms": [
        ("GPT-4o", [f"commercial_llms/gpt-4o/run_{run}/labels.csv" for run in range(1, 4)]),
        ("GPT-5", [f"commercial_llms/gpt-5/run_{run}/labels.csv" for run in range(1, 4)]),
        ("Gemini 2.5 Flash", [f"commercial_llms/gemini-2.5-flash/run_{run}/labels.csv" for run in range(1, 4)]),
        ("Gemini 2.5 Pro", [f"commercial_llms/gemini-2.5-pro/run_{run}/labels.csv" for run in range(1, 4)]),
        ("DeepSeek-Chat", [f"commercial_llms/deepseek-chat/run_{run}/labels.csv" for run in range(1, 4)]),
        ("DeepSeek-Reasoner", [f"commercial_llms/deepseek-reasoner/run_{run}/labels.csv" for run in range(1, 4)]),
        ("FineDroid", ["finedroid/labels.csv"]),
    ],
    "existing_llm_based_ncf_bug_detectors": [
        ("OLLM", [f"existing_llm_based_ncf_bug_detectors/ollm-gpt-4o/run_{run}/labels.csv" for run in range(1, 4)]),
        ("VisionDroid", ["existing_llm_based_ncf_bug_detectors/visiondroid-gpt-4o/labels.csv"]),
        ("FineDroid", ["finedroid/labels.csv"]),
    ],
}


def label_counts(relative_path: str) -> Counter[str]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"{relative_path}: expected 100 cases, found {len(rows)}")
    counts = Counter(row["label"].strip() for row in rows)
    if counts["TPC"] + counts["TPW"] + counts["FN"] != 50:
        raise ValueError(f"{relative_path}: buggy-case labels do not total 50: {counts}")
    if counts["TN"] + counts["FP"] + counts["PARSE_ERROR"] != 50:
        raise ValueError(f"{relative_path}: bug-free-case labels do not total 50: {counts}")
    return counts


def metrics(label_files: list[str]) -> dict[str, float]:
    runs = [label_counts(path) for path in label_files]
    mean = {
        label: sum(run[label] for run in runs) / len(runs)
        for label in ("TPC", "TPW", "FP")
    }
    return {
        "TPC (%)": mean["TPC"] / 50 * 100,
        "TPW (%)": mean["TPW"] / 50 * 100,
        "FP (%)": mean["FP"] / 50 * 100,
        "FA (%)": mean["TPW"] + mean["FP"],
    }


def expected_rows(table: str) -> dict[str, dict[str, str]]:
    path = ROOT / table / "results.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def check() -> dict[str, list[tuple[str, dict[str, float]]]]:
    computed: dict[str, list[tuple[str, dict[str, float]]]] = {}
    errors: list[str] = []
    for table, models in TABLES.items():
        expected = expected_rows(table)
        computed[table] = []
        for model, files in models:
            actual = metrics(files)
            computed[table].append((model, actual))
            for metric in METRICS:
                published = float(expected[model][metric])
                if abs(actual[metric] - published) > 0.051:
                    errors.append(
                        f"{table}/{model}/{metric}: computed {actual[metric]:.3f}, "
                        f"published {published:.3f}"
                    )
    if errors:
        raise ValueError("\n".join(errors))
    return computed


def print_markdown(computed: dict[str, list[tuple[str, dict[str, float]]]]) -> None:
    for table, rows in computed.items():
        print(f"## {table.replace('_', ' ').title()}\n")
        print("| Model | TPC (%) | TPW (%) | FP (%) | FA (%) |")
        print("|---|---:|---:|---:|---:|")
        for model, values in rows:
            cells = " | ".join(f"{values[metric]:.1f}" for metric in METRICS)
            print(f"| {model} | {cells} |")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only validate the published CSV files")
    args = parser.parse_args()
    try:
        computed = check()
    except (KeyError, OSError, ValueError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    if args.check:
        print("All released labels reproduce the published tables.")
    else:
        print_markdown(computed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
