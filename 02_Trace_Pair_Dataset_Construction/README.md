# Step 2: Buggy and Bug-Free Trace-Pair Dataset Construction

This module is the complete trace-pair construction project. It takes the LLM-filtered reports produced by Step 1, generates buggy and bug-free Android UI interaction traces, verifies the generated traces, adds Gemini reasoning supervision, and combines the accepted pairs into a balanced training dataset.

Its input is `artifacts/01_ncf_bug_report_collection/llm_filtered_issues/`. If Step 1 was skipped, the runner automatically extracts the tracked released archive from `01_NCF_Bug_Report_Collection/data/llm_filtered_issues.zip`. Its final output is `artifacts/02_trace_pair_dataset_construction/trace_all_with_reasoning.json`.

Fill in the OpenAI and Gemini keys near the top of `run.py`, then run the complete module:

```bash
python3 -m pip install -r requirements.txt
python3 02_Trace_Pair_Dataset_Construction/run.py
```

OpenAI keys are created through the [OpenAI API quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request). Gemini keys are created through the [Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key). The original `gpt-5` Responses API and `gemini-2.5-pro` thinking calls are retained, and prompts are read from `prompts/`. Outputs are saved incrementally so the module can resume after interruption.

To skip this complete module, proceed directly to Step 3. Its runner uses the released final training dataset tracked under `03_Fine_Tuning/data/` when no newly generated Step 2 dataset exists. This module does not require a Google Drive download because its released input and the Step 3 training datasets are tracked in the Git repository.
