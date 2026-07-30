from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parent
PROMPT_DIR = MODULE_DIR / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").rstrip()


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


def parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = (text or "").strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.I)
    if not fence:
        return None
    try:
        value = json.loads(fence.group(1))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
