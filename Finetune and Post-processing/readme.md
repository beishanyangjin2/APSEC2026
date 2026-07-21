# Fine-Tuning and Post-Processing

This module fine-tunes Qwen3-14B with LoRA and runs FineDroid's inference and description-verification pipeline.

## Contents

- `data_train/`: accepted training traces and distilled reasoning annotations.
- `Script/fintune_qwen3.ipynb`: Qwen3-14B LoRA fine-tuning.
- `Script/fintune_gemini_thinking_summary_qwen3.ipynb`: fine-tuning with distilled Gemini reasoning summaries.
- `Script/all_round_temp0.ipynb`: model loading, inference, initial judging, and post-processing verification.

The canonical compact test set is stored in [`Evaluation/test_set`](../Evaluation/test_set/README.md), separate from the fine-tuning data.

The notebook filenames retain the original `fintune` spelling to avoid breaking existing references.

## Fine-tuned adapter

Download `fine-tuned_qwen3-14b.zip` from the [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing). Extract it and set `MODEL_DIR` in `Script/all_round_temp0.ipynb` to the adapter/checkpoint directory. To evaluate the base model, load Qwen3-14B without applying the LoRA adapter.

The same Drive folder contains `testingset.zip`, which provides the complete APKs, execution screenshots, XML hierarchies, and trace files corresponding to the compact records in `Evaluation/test_set/`.

The training notebooks use PyTorch, Hugging Face `datasets`/`transformers`, PEFT, and Unsloth. GPU memory requirements depend on the chosen quantization and training configuration. Do not store provider credentials directly in the notebooks.

Final case-level outputs and comparisons are available in [Evaluation](../Evaluation/readme.md).
