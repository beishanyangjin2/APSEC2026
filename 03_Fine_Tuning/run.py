from __future__ import annotations

from types import SimpleNamespace

from common import MODEL_DIR, MODULE_DIR, REPOSITORY_ROOT
from training import train


GENERATED_DATASET = (
    REPOSITORY_ROOT
    / "artifacts"
    / "02_trace_pair_dataset_construction"
    / "trace_all_with_reasoning.json"
)
RELEASED_DATASET = (
    MODULE_DIR / "data" / "trace_all_thinking_summary_by_gemini_sorted.json"
)

# The generated Step 2 output is used when present; otherwise the tracked
# released dataset lets this module run independently after cloning.
DATA_PATH = GENERATED_DATASET if GENERATED_DATASET.is_file() else RELEASED_DATASET
OUTPUT_MODEL_DIR = MODEL_DIR
BASE_MODEL = "unsloth/Qwen3-14B"
CUDA_DEVICES = "0"
LOAD_IN_4BIT = False


def main() -> None:
    print(f"Training with {DATA_PATH}")
    train(
        SimpleNamespace(
            data=DATA_PATH,
            output=OUTPUT_MODEL_DIR,
            base_model=BASE_MODEL,
            with_reasoning=True,
            max_length=8192,
            cuda_devices=CUDA_DEVICES,
            load_in_4bit=LOAD_IN_4BIT,
            epochs=3.0,
            learning_rate=1e-4,
            batch_size=1,
            gradient_accumulation=1,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=42,
            resume_from_checkpoint=None,
            dry_run=False,
        )
    )


if __name__ == "__main__":
    main()
