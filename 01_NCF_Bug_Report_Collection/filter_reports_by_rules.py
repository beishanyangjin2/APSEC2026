from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from common import ARTIFACT_DIR, load_json_list, save_json


# Edit these paths only if you do not use the repository's artifacts layout.
INPUT_DIR = ARTIFACT_DIR / "issues"
OUTPUT_DIR = ARTIFACT_DIR / "keyword_filtered_issues"
BLOCKED_REPOSITORIES_CSV: Path | None = None
OUTPUT_BATCH_SIZE = 1000

RE_ACTUAL = re.compile(r"\bactual(?:\s*(?:result|behavior|behaviour|output))?\b", re.I)
RE_EXPECT = re.compile(r"\bexpect(?:ed|ation|ations|s)?(?:\s*(?:result|behavior|behaviour|output))?\b", re.I)
RE_REPRODUCE = re.compile(r"\b(?:reproduction|to\s+reproduce)\b", re.I)
RE_CRASH = re.compile(
    r"\b(?:crash(?:es|ed|ing)?|freeze(?:s|d|ing)?|ANR|(?:app\s+)?not\s+responding|unresponsive|terminated)\b"
    r"|[A-Za-z]+Exception\b|\b[A-Za-z]+Error\b",
    re.I,
)
RE_NEGATED_CRASH_LINE = re.compile(
    r"^[^\n]*\b(?:no|does\s+not|did\s+not|never)\s+crash(?:es|ed|ing)?\b[^\n]*$",
    re.I | re.M,
)
RE_CRASH_QUESTION_LINE = re.compile(
    r"^[^\n]*\bcrash(?:es|ed|ing)?\b[^\n]*\?[^\n]*$", re.I | re.M
)
RE_CRASH_LOG = re.compile(
    r"(?<![A-Za-z0-9])crash(?:\s*|[-_])?logs?(?![A-Za-z0-9])", re.I
)
RE_CODE_FENCE = re.compile(r"```.+?```", re.S)
RE_NEXT_HEADER = re.compile(r"^\s*#{2,}\s", re.I | re.M)
RE_PLACEHOLDER = re.compile(r"^\s*(?:n/?a|no\s*response|none|N\.?A\.?)\s*$", re.I)
RE_IMAGE = re.compile(r"!\[.*?\]\(.*?\)|<img .*?>", re.I)
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_TEMPLATE_CRASH_PROMPT = re.compile(
    r"^\s{0,3}(?:\#{1,6}\s*)?(?:\*\*|__)?\s*(?:describe|summary)\s+"
    r"(?:the\s+)?(?:bug|issue|problem)(?:\s*/\s*crash)?(?:\s*\(.*?\))?"
    r"(?:\*\*|__)?\s*:?\s*$",
    re.I | re.M,
)
RE_KEYBOARD = re.compile(r"\bkeyboard\b", re.I)
LABEL_EXCLUSIONS = ("crash", "forceout")
SOFT_EXCLUSIONS = {
    "out_of_scope_action": [
        "rotate", "rotation", "landscape", "portrait", "permission", "proxy",
        "firewall", "bluetooth", "NFC",
    ],
    "visual_or_media": [
        "color", "font", "size", "UI", "UX", "layout", "overlap", "look",
        "appearance", "visual", "graphic", "audio", "video", "gesture", "sound",
        "music", "picture", "thumbnail", "distracting", "blur", "theme", "style",
        "interface", "cursor", "zoom", "render",
    ],
    "external_environment": [
        "server", "desktop", "PC", "windows", "linux", "mac", "WebDAV", "nginx",
        "apache", "S3", "iOS", "iPad", "another device", "multiple accounts",
        "backend", "API",
    ],
    "long_wait_or_sync": [
        "wait", "large file", "slow", "sync", "synchronization", "downloading",
        "performance", "lag", "delay",
    ],
    "notification": ["notification", "status bar", "notification center"],
}
SOFT_PATTERNS = {
    category: re.compile(
        r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b", re.I
    )
    for category, words in SOFT_EXCLUSIONS.items()
}


def normalize_github_repository(url: str) -> str | None:
    parsed = urlsplit(url.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return urlunsplit(("https", "github.com", f"/{parts[0]}/{repository}", "", ""))


def minimal_issue(issue: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(issue.get("title") or ""),
        "body": str(issue.get("body") or ""),
        "html_url": str(issue.get("html_url") or ""),
    }


def has_nonempty_crash_log(text: str) -> bool:
    match = RE_CRASH_LOG.search(text)
    if not match:
        return False
    tail = text[match.end() :]
    next_header = RE_NEXT_HEADER.search(tail)
    segment = tail[: next_header.start()] if next_header else tail
    if not segment.strip():
        return False
    if RE_CODE_FENCE.search(segment):
        return True
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped or RE_PLACEHOLDER.fullmatch(stripped):
            continue
        if len(stripped) >= 50 or re.search(
            r"\b(Exception|Error|SIG[A-Z]+|stack|trace|at\s+\S+\()", stripped, re.I
        ):
            return True
    return False


def has_definitive_crash_line(text: str) -> bool:
    for line in text.splitlines():
        if not re.search(r"\bcrash(?:es|ed|ing)?\b", line, re.I):
            continue
        if "?" in line or RE_NEGATED_CRASH_LINE.fullmatch(line):
            continue
        return True
    return False


def section_presence_and_nonempty(
    text: str, header_variants: tuple[str, ...]
) -> tuple[bool, bool]:
    for header in header_variants:
        match = re.search(
            rf"^\s*(?:[-*]\s*)?(?:#+\s*)?{header}\s*:?\s*$",
            text,
            re.I | re.M,
        )
        if not match:
            continue
        for line in text[match.end() :].splitlines():
            stripped = line.strip()
            if stripped:
                return True, not bool(RE_PLACEHOLDER.fullmatch(stripped))
        return True, False
    return False, False


def remove_android_ios_lines(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not (
            re.search(r"\bandroid\b", line, re.I)
            and re.search(r"\bios\b", line, re.I)
        )
    )


def load_blocked_repositories(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    blocked: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            for cell in row:
                repository = normalize_github_repository(cell)
                if repository:
                    blocked.add(repository.lower())
    return blocked


def issue_is_blocked(url: str, blocked_repositories: set[str]) -> bool:
    normalized = url.lower().rstrip("/")
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in blocked_repositories
    )


def write_chunks(
    output_dir: Path,
    prefix: str,
    items: list[dict[str, Any]],
    chunk_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, offset in enumerate(range(0, len(items), chunk_size), start=1):
        save_json(
            output_dir / f"{prefix}{index:04d}.json",
            items[offset : offset + chunk_size],
        )


def keyword_filter(
    input_dir: Path,
    output_dir: Path,
    blocked_repositories_csv: Path | None = None,
    chunk_size: int = 1000,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    blocked = load_blocked_repositories(blocked_repositories_csv)
    selected: list[dict[str, Any]] = []
    excluded: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def exclude(reason: str, issue: dict[str, Any]) -> None:
        excluded[reason].append(minimal_issue(issue))

    paths = sorted(
        input_dir.glob("issue_*.json"),
        key=lambda path: int(re.search(r"(\d+)", path.stem).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No issue_*.json batches found in {input_dir}")

    for issue in (item for path in paths for item in load_json_list(path)):
        title = str(issue.get("title") or "")
        body = str(issue.get("body") or "")
        url = str(issue.get("html_url") or "")
        if issue_is_blocked(url, blocked):
            exclude("blocked_repository", issue)
            continue
        if not (
            RE_ACTUAL.search(body)
            and RE_EXPECT.search(body)
            and RE_REPRODUCE.search(body)
        ):
            exclude("missing_report_sections", issue)
            continue

        cleaned = RE_HTML_TAG.sub(" ", RE_HTML_COMMENT.sub("", RE_IMAGE.sub("", body)))
        cleaned = RE_TEMPLATE_CRASH_PROMPT.sub("", cleaned)
        expected_present, expected_nonempty = section_presence_and_nonempty(
            cleaned,
            ("Expected behaviour", "Expected behavior", "Expected result", "Expected results"),
        )
        actual_present, actual_nonempty = section_presence_and_nonempty(
            cleaned,
            ("Actual behaviour", "Actual behavior", "Actual result", "Actual results"),
        )
        if (expected_present and not expected_nonempty) or (
            actual_present and not actual_nonempty
        ):
            exclude("empty_report_section", issue)
            continue

        labels = issue.get("labels") or []
        label_names = [
            str(label.get("name") or "").lower()
            for label in labels
            if isinstance(label, dict)
        ]
        if any(word in name for name in label_names for word in LABEL_EXCLUSIONS):
            exclude("crash_label", issue)
            continue

        crash_text = title + "\n" + cleaned
        if RE_CRASH.search(crash_text):
            allow_crash_reference = False
            if RE_CRASH_LOG.search(cleaned):
                without_phrase = RE_CRASH_LOG.sub("", cleaned)
                other_signal = RE_CRASH.search(title + "\n" + without_phrase)
                if not has_nonempty_crash_log(cleaned) and not other_signal:
                    allow_crash_reference = True
            if not allow_crash_reference:
                negated_or_question = RE_NEGATED_CRASH_LINE.search(
                    cleaned
                ) or RE_CRASH_QUESTION_LINE.search(cleaned)
                if not negated_or_question or has_definitive_crash_line(cleaned):
                    exclude("crash_report", issue)
                    continue

        if RE_KEYBOARD.search(title + "\n" + cleaned + "\n" + url):
            exclude("keyboard_dependency", issue)
            continue

        soft_reason = None
        for category, pattern in SOFT_PATTERNS.items():
            searchable = title + "\n" + cleaned
            if category == "external_environment":
                searchable = remove_android_ios_lines(title) + "\n" + remove_android_ios_lines(cleaned)
            if pattern.search(searchable):
                soft_reason = category
                break
        if soft_reason == "visual_or_media" and re.search(
            r"\bfile\s+size\b", title + "\n" + cleaned, re.I
        ):
            soft_reason = None
        if soft_reason:
            exclude(soft_reason, issue)
            continue
        selected.append(minimal_issue(issue))

    write_chunks(output_dir, "selected_issue_", selected, chunk_size)
    for reason, items in excluded.items():
        write_chunks(output_dir / "excluded" / reason, f"{reason}_", items, chunk_size)
    total = len(selected) + sum(map(len, excluded.values()))
    counts = dict(sorted((key, len(value)) for key, value in excluded.items()))
    print(f"Selected {len(selected)} issues from {total}")
    print(f"Exclusions: {counts}")


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Issue directory not found: {INPUT_DIR}. Run crawl_issues.py or extract "
            "all_issues_new.zip as described in this module's README."
        )
    keyword_filter(
        INPUT_DIR,
        OUTPUT_DIR,
        BLOCKED_REPOSITORIES_CSV,
        OUTPUT_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
