from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parent
ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts" / "01_ncf_bug_report_collection"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_prompt(name: str) -> str:
    return (MODULE_DIR / "prompts" / name).read_text(encoding="utf-8").rstrip()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value
