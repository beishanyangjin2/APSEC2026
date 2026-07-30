from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

from common import ARTIFACT_DIR, REPOSITORY_ROOT
from workflow import (
    combine,
    gemini_reasoning,
    generate_bug_free,
    generate_buggy,
    verify_bug_free,
    verify_buggy,
)


# Fill in both keys before running the complete module.
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_MODEL = "gpt-5"
GEMINI_MODEL = "gemini-2.5-pro"

REPORT_DIR = (
    REPOSITORY_ROOT
    / "artifacts"
    / "01_ncf_bug_report_collection"
    / "llm_filtered_issues"
)
RELEASED_REPORT_ARCHIVE = (
    REPOSITORY_ROOT
    / "01_NCF_Bug_Report_Collection"
    / "data"
    / "llm_filtered_issues.zip"
)
FINAL_DATASET = ARTIFACT_DIR / "trace_all_with_reasoning.json"
RESUME = True
MAX_RECORDS: int | None = None


def prepare_report_input() -> Path:
    if list(REPORT_DIR.glob("filtered_BR_*.json")):
        return REPORT_DIR
    if not RELEASED_REPORT_ARCHIVE.is_file():
        raise FileNotFoundError(
            "No Step 1 output or released report archive was found. Run "
            "01_NCF_Bug_Report_Collection/run.py or restore its tracked data archive."
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RELEASED_REPORT_ARCHIVE) as archive:
        archive.extractall(REPORT_DIR)
    print(f"Using the released Step 1 output from {RELEASED_REPORT_ARCHIVE}")
    return REPORT_DIR


def openai_settings(input_path: Path, output_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        input=input_path,
        output=output_path,
        resume=RESUME,
        limit=MAX_RECORDS,
    )


def main() -> None:
    report_input = prepare_report_input()
    buggy = ARTIFACT_DIR / "buggy_traces.json"
    buggy_verified = ARTIFACT_DIR / "buggy_traces_verified.json"
    pairs = ARTIFACT_DIR / "trace_pairs.json"
    pairs_verified = ARTIFACT_DIR / "trace_pairs_verified.json"
    pairs_with_reasoning = ARTIFACT_DIR / "trace_pairs_with_reasoning.json"

    generate_buggy(openai_settings(report_input, buggy))
    verify_buggy(openai_settings(buggy, buggy_verified))
    generate_bug_free(openai_settings(buggy_verified, pairs))
    verify_bug_free(openai_settings(pairs, pairs_verified))
    gemini_reasoning(
        SimpleNamespace(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            input=pairs_verified,
            output=pairs_with_reasoning,
            trace_fields=["gen_trace", "bug-free-trace"],
            rate_pause=2.0,
            resume=RESUME,
            limit=MAX_RECORDS,
        )
    )
    combine(
        SimpleNamespace(
            input=pairs_with_reasoning,
            output=FINAL_DATASET,
            shuffle=False,
            seed=42,
        )
    )


if __name__ == "__main__":
    main()
