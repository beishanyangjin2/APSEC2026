from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter

from openai import OpenAI
from tqdm import tqdm

from common import ARTIFACT_DIR, load_json_list, load_prompt, save_json


# Keep the original notebook-style configuration: edit these values in place.
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-5"
INPUT_DIR = ARTIFACT_DIR / "keyword_filtered_issues"
OUTPUT_DIR = ARTIFACT_DIR / "llm_filtered_issues"
MAX_RETRIES = 3

ALLOWED_LABELS = {"yes", "no", "cannot"}
SYSTEM_MESSAGE = (
    "You are an expert software quality assurance engineer. Your task is to analyze "
    "a bug report for an Android application and classify it. You will be given a JSON "
    "object containing the bug report's title and body."
)
CLASSIFICATION_PROMPT = load_prompt("report_filter.txt")


def classify_with_retry(client: OpenAI, payload: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=1,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": payload},
                ],
            )
            verdict = (response.choices[0].message.content or "").strip().lower()
            if verdict in ALLOWED_LABELS:
                return verdict
        except Exception as error:
            print(f"OpenAI call failed on attempt {attempt}: {error}")
        if attempt < MAX_RETRIES:
            time.sleep(1.2 * (2 ** (attempt - 1)) + random.uniform(0, 0.4))
    return "error"


def write_comparison_csv(path, verdicts: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", MODEL_NAME])
        writer.writerows((index, verdict) for index, verdict in enumerate(verdicts, start=1))


def main() -> None:
    if not OPENAI_API_KEY.strip():
        raise RuntimeError(
            "Set OPENAI_API_KEY near the top of filter_reports_with_llm.py. "
            "See this module's README for the key instructions."
        )
    paths = sorted(INPUT_DIR.glob("selected_issue_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No selected_issue_*.json files found in {INPUT_DIR}. Run the rule filter "
            "or extract keyword_filtered_issues.zip as described in the README."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        yes_items: list[dict] = []
        verdicts: list[str] = []
        for item in tqdm(load_json_list(path), desc=f"Processing {path.name}"):
            payload = (
                CLASSIFICATION_PROMPT
                + "\n"
                + json.dumps(item, ensure_ascii=False)
                + "Now, analyze the following JSON and provide your one-word classification:"
            )
            verdict = classify_with_retry(client, payload)
            verdicts.append(verdict)
            if verdict == "yes":
                yes_items.append(item)

        match = re.search(r"(\d+)", path.stem)
        suffix = match.group(1) if match else path.stem
        safe_model_name = MODEL_NAME.replace("/", "-")
        save_json(OUTPUT_DIR / f"filtered_BR_{suffix}_{safe_model_name}.json", yes_items)
        save_json(OUTPUT_DIR / f"verdicts_{suffix}_{safe_model_name}.json", verdicts)
        write_comparison_csv(OUTPUT_DIR / f"results_compare_{suffix}.csv", verdicts)
        print(f"{path.name}: {dict(Counter(verdicts))}")


if __name__ == "__main__":
    main()
