from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from common import ARTIFACT_DIR, MODULE_DIR, load_json_list, save_json


# Replace the placeholder with your GitHub token, as in the original notebook.
# The module README explains how to create one.
GITHUB_TOKEN = "xxxx"

# Use the tracked CSV immediately, or replace this path with the output produced
# by discover_repositories.py.
REPOSITORIES_CSV = MODULE_DIR / "data" / "fdroid_github_only.csv"
OUTPUT_DIR = ARTIFACT_DIR / "issues"
BATCH_SIZE = 1000
PER_PAGE = 100

CHECKPOINT_PATH = OUTPUT_DIR / "crawl_checkpoint.json"
CURRENT_BATCH_PATH = OUTPUT_DIR / "current_batch.json"


def parse_owner_repository(url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return parts[0], repository


def respectful_get(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    max_retries: int = 5,
) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        response = session.get(url, headers=headers, params=params, timeout=60)
        remaining = response.headers.get("X-RateLimit-Remaining")
        if response.status_code == 403 and remaining == "0":
            reset = int(response.headers.get("X-RateLimit-Reset", "0"))
            wait = max(1, reset - int(time.time()) + 1)
            print(f"[GitHub] Rate limit reached; waiting {wait} seconds")
            time.sleep(wait)
            continue
        if response.status_code in {403, 429} or response.status_code >= 500:
            if attempt == max_retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else backoff
            print(f"[GitHub] HTTP {response.status_code}; retrying in {wait:.1f} seconds")
            time.sleep(wait)
            backoff = min(backoff * 2, 60)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"GitHub request failed after {max_retries} attempts: {url}")


def issue_identity(issue: dict[str, Any]) -> str:
    github_id = issue.get("id")
    if github_id is not None:
        return f"id:{github_id}"
    return f"url:{issue.get('html_url', '')}"


def numbered_batches() -> list[Path]:
    def batch_number(path: Path) -> int:
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    return sorted(OUTPUT_DIR.glob("issue_*.json"), key=batch_number)


class RollingWriter:
    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        batches = numbered_batches()
        self.batch_index = 1
        if batches:
            match = re.search(r"(\d+)", batches[-1].stem)
            self.batch_index = int(match.group(1)) + 1 if match else len(batches) + 1
        stored_ids: set[str] = set()
        for path in batches:
            stored_ids.update(issue_identity(item) for item in load_json_list(path))
        self.stored_ids = stored_ids
        current = load_json_list(CURRENT_BATCH_PATH) if CURRENT_BATCH_PATH.exists() else []
        self.buffer = [item for item in current if issue_identity(item) not in stored_ids]
        if len(self.buffer) != len(current):
            save_json(CURRENT_BATCH_PATH, self.buffer)

    def append(self, issue: dict[str, Any]) -> None:
        self.buffer.append(issue)
        save_json(CURRENT_BATCH_PATH, self.buffer)
        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = OUTPUT_DIR / f"issue_{self.batch_index:04d}.json"
        save_json(path, self.buffer)
        print(f"[GitHub] Saved {len(self.buffer)} issues to {path}")
        self.buffer = []
        save_json(CURRENT_BATCH_PATH, self.buffer)
        self.batch_index += 1


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {"repository_index": 0, "page": 1, "complete": False}
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(repository_index: int, page: int, complete: bool = False) -> None:
    save_json(
        CHECKPOINT_PATH,
        {"repository_index": repository_index, "page": page, "complete": complete},
    )


def main() -> None:
    if not GITHUB_TOKEN.strip() or GITHUB_TOKEN == "xxxx":
        raise RuntimeError(
            "Replace GITHUB_TOKEN = 'xxxx' in crawl_issues.py before running. "
            "See this module's README for the GitHub token instructions."
        )
    if not REPOSITORIES_CSV.is_file():
        raise FileNotFoundError(
            f"Repository list not found: {REPOSITORIES_CSV}. "
            "Run discover_repositories.py or use the tracked CSV described in the README."
        )

    with REPOSITORIES_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint = load_checkpoint()
    if checkpoint.get("complete"):
        print(f"Crawl already completed according to {CHECKPOINT_PATH}")
        return

    start_repository = int(checkpoint.get("repository_index", 0))
    start_page = int(checkpoint.get("page", 1))
    writer = RollingWriter()
    seen = writer.stored_ids | {issue_identity(item) for item in writer.buffer}
    print(f"Loaded {len(seen)} existing issue IDs; duplicates will be skipped")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "User-Agent": "FineDroid issue crawler",
    }
    session = requests.Session()

    for repository_index in range(start_repository, len(rows)):
        row = rows[repository_index]
        parsed = parse_owner_repository(row.get("source_code_url", ""))
        if parsed is None:
            print(f"[GitHub] Skipping invalid URL: {row.get('source_code_url')}")
            save_checkpoint(repository_index + 1, 1)
            continue
        owner, repository = parsed
        page = start_page if repository_index == start_repository else 1
        while True:
            response = respectful_get(
                session,
                f"https://api.github.com/repos/{owner}/{repository}/issues",
                headers,
                {"state": "all", "per_page": PER_PAGE, "page": page},
            )
            payload = response.json()
            if not isinstance(payload, list):
                print(f"[GitHub] Unexpected response for {owner}/{repository} page {page}")
                break
            if not payload:
                break
            added = 0
            for issue in payload:
                if "pull_request" in issue:
                    continue
                identity = issue_identity(issue)
                if identity in seen:
                    continue
                issue["_repo_full_name"] = f"{owner}/{repository}"
                if row.get("package_id"):
                    issue["_package_id"] = row["package_id"]
                writer.append(issue)
                seen.add(identity)
                added += 1
            page += 1
            save_checkpoint(repository_index, page)
            print(
                f"[GitHub] Repository {repository_index + 1}/{len(rows)} "
                f"{owner}/{repository}, page {page - 1}: added {added} issues"
            )
        save_checkpoint(repository_index + 1, 1)

    writer.flush()
    save_checkpoint(len(rows), 1, complete=True)
    print(f"Crawl complete: {len(seen)} unique issues are stored in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
