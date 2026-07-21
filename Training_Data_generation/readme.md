# Training-Data Generation

This module converts filtered NCF bug reports into paired buggy and bug-free GUI testing traces, verifies the generated traces, and prepares reasoning annotations for fine-tuning.

## Recommended order

1. `buggy_trace_gen.ipynb`: generate a trace intended to reproduce each selected bug report.
2. `bug-free_trace_gen.ipynb`: rewrite the buggy trace into a bug-free counterpart.
3. `filter_gen_buggy_trace.ipynb`: verify that generated buggy traces match their source reports.
4. `filter_bugfree_gen_trace.ipynb`: verify that bug-free traces do not exhibit an NCF bug.
5. `reasoning_gemini2.5pro_thinking.ipynb`: optionally generate Gemini 2.5 Pro reasoning annotations.
6. `gen_trace_combine.ipynb`: combine, normalize, and shuffle the accepted examples.

## Inputs and outputs

The input is the bug-report collection produced by [NCF bug-report filtering](../NCF_bug_report_filtering/readme.md). Configure the input/output paths and provider credentials in the first cell of each notebook; keep credentials in environment variables rather than notebook cells.

The compact training datasets used by the artifact are stored in [`Finetune and Post-processing/data_train`](../Finetune%20and%20Post-processing/data_train). They contain traces, labels, descriptions, and distilled reasoning used for the Qwen3-14B LoRA training stage.
