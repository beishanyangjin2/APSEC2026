# Step 4: Post-Processing Verification

This module is the complete FineDroid execution project. It loads the fine-tuned LoRA adapter, predicts NCF bugs from Android UI interaction traces, and verifies each positive reason against the NCF definition and the trace evidence.

Its input is the adapter at `artifacts/models/finedroid-qwen3-14b/` and the compact traces under `Evaluation/test_set/`. Its output is written to `artifacts/runs/finedroid/`: initial predictions, reason-verification records, and final post-processed predictions.

Install the GPU dependencies and run the complete module:

```bash
python3 -m pip install -r requirements-gpu.txt
python3 -m pip install unsloth
python3 04_Post_Processing_Verification/run.py
```

Before loading Qwen3-14B, `run.py` checks that the adapter directory directly contains `adapter_config.json` and `adapter_model.safetensors` or `adapter_model.bin`. If it is missing, either run `03_Fine_Tuning/run.py` or download `fine-tuned_qwen3-14b.zip` from the [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing). The archive is the LoRA output produced by Step 3; extract and place it at `artifacts/models/finedroid-qwen3-14b/`. Inference and verification prompts are loaded from `prompts/`.

The released paper outputs are already available under `Evaluation/finedroid/`, so this module can be skipped when the goal is only to audit the published evaluation.
