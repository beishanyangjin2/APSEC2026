# Step 3: Fine-Tuning

This module is the complete FineDroid fine-tuning project. It fine-tunes Qwen3-14B with LoRA using the reasoning-supervised trace dataset and saves the adapter in the location expected by Step 4.

Its input is the dataset produced by Step 2. When that generated file is absent, `run.py` automatically uses the released training dataset tracked at `data/trace_all_thinking_summary_by_gemini_sorted.json`. That file contains 2,328 trace-level training samples: 1,164 buggy and 1,164 bug-free. The other tracked file, `data/trace_all_sorted.json`, contains 3,546 samples without reasoning supervision and is retained as the corresponding released dataset.

Its output is the LoRA adapter under `artifacts/models/finedroid-qwen3-14b/`.

Install the GPU dependencies and run the complete module:

```bash
python3 -m pip install -r requirements-gpu.txt
python3 -m pip install unsloth
python3 03_Fine_Tuning/run.py
```

The original reasoning-supervised settings are retained in `run.py`: Qwen3-14B, an 8,192-token maximum length, BF16, LoRA rank 16, batch size 1, three epochs, and learning rate `1e-4`. The training prompt is loaded from `prompts/`.

To skip this complete module, download `fine-tuned_qwen3-14b.zip` from the [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing). This archive is the LoRA adapter produced by this training module, not a separate input dataset. Extract and rename/move its adapter directory to `artifacts/models/finedroid-qwen3-14b/`. `adapter_config.json` and `adapter_model.safetensors` must be directly inside that directory.
