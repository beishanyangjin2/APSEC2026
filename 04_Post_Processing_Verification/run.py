from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace

from common import REPOSITORY_ROOT
from workflow import run_detection, verify_predictions


# Both a newly trained adapter and the downloaded released adapter should be
# placed at this exact location. See this module's README for both routes.
ADAPTER_DIR = REPOSITORY_ROOT / "artifacts" / "models" / "finedroid-qwen3-14b"
BASE_MODEL = "unsloth/Qwen3-14B"
VERIFICATION_MODEL = "unsloth/Qwen3-14B"
OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "runs" / "finedroid"

TEST_SET_DIR = REPOSITORY_ROOT / "Evaluation" / "test_set"
BUGGY_TRACES = TEST_SET_DIR / "buggy_traces_withGT.json"
BUG_FREE_TRACES = TEST_SET_DIR / "bug_free_traces_withGT.json"
EXCLUDED_IDS = ["33"]
CUDA_DEVICES = "0"
MAX_LENGTH = 16384
MAX_NEW_TOKENS = 2048
LOAD_IN_4BIT = False


def require_adapter(path: Path) -> None:
    config = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if config.is_file() and any(candidate.is_file() for candidate in weights):
        return
    raise FileNotFoundError(
        "FineDroid's LoRA adapter was not found at:\n"
        f"  {path}\n\n"
        "Either:\n"
        "1. download fine-tuned_qwen3-14b.zip and extract/rename its adapter directory "
        "to that exact path, or\n"
        "2. run 03_Fine_Tuning/run.py, whose default output is that path.\n\n"
        "The directory must directly contain adapter_config.json and adapter_model.safetensors "
        "(or adapter_model.bin)."
    )


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    require_adapter(ADAPTER_DIR)
    for path in (BUGGY_TRACES, BUG_FREE_TRACES):
        if not path.is_file():
            raise FileNotFoundError(f"Test trace file not found: {path}")
    if CUDA_DEVICES:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES

    settings = SimpleNamespace(
        buggy=BUGGY_TRACES,
        bug_free=BUG_FREE_TRACES,
        output_dir=OUTPUT_DIR,
        base_model=BASE_MODEL,
        adapter=ADAPTER_DIR,
        verification_model=VERIFICATION_MODEL,
        verification_adapter=None,
        predictions=None,
        exclude_id=EXCLUDED_IDS,
        max_length=MAX_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        load_in_4bit=LOAD_IN_4BIT,
    )
    predictions = run_detection(settings)
    release_gpu_memory()
    verify_predictions(settings, predictions)


if __name__ == "__main__":
    main()
