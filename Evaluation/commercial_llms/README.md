# API-Based Commercial LLMs

Each model was evaluated three times with the baseline prompt. Every `run_N` directory contains raw predictions and case-level labels; `summary_counts.csv` records the per-run and mean label counts. The table reports the mean percentages.

DeepSeek-V3.2-Exp is evaluated through its chat and reasoner API endpoints and grouped here by deployment mode, despite being open-weight, because the full model does not fit on the evaluation A100 80GB GPU.
