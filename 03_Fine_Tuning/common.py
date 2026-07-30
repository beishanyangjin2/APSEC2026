from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parent
MODEL_DIR = REPOSITORY_ROOT / "artifacts" / "models" / "finedroid-qwen3-14b"


def load_prompt(name: str) -> str:
    return (MODULE_DIR / "prompts" / name).read_text(encoding="utf-8").rstrip()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]
