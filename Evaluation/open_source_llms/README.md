# Locally Deployed Open-Source LLMs

Each baseline has one raw prediction file and one final case-level label file for the 100-trace test set. Qwen3 is evaluated separately with thinking disabled and enabled. `label_review.csv` records the stricter review retained for the four model suites where that review file was available.

The Qwen3 rows in the paper use `labels.csv`, i.e. the raw round-1 labels, rather than the later FineDroid reason-verification stage. The FineDroid comparison row links to `../finedroid/labels.csv`.
