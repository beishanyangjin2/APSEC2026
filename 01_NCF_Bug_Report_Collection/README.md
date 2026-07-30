# Step 1: NCF Bug Report Collection

This module is the complete NCF bug report collection project. It discovers Android projects from F-Droid, crawls their GitHub issues, applies rule-based exclusions, and performs the final LLM filtering.

Its input is the public F-Droid/GitHub data plus GitHub and OpenAI credentials. Its output is a set of LLM-filtered NCF bug reports under `artifacts/01_ncf_bug_report_collection/llm_filtered_issues/`, which is the input to Step 2.

Install the dependencies, fill in `GITHUB_TOKEN` and `OPENAI_API_KEY` near the top of `run.py`, and run the complete module:

```bash
python3 -m pip install -r requirements.txt
python3 01_NCF_Bug_Report_Collection/run.py
```

Create the GitHub token by following [GitHub's personal access token instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). A fine-grained token with read access to the required public repositories is sufficient. GitHub's REST API limit is normally 60 requests per hour without authentication and 5,000 requests per hour for authenticated personal requests, so the full crawl should use a token. Replace the literal `xxxx` in `run.py`; do not commit the real value. The crawler records its repository/page checkpoint and skips previously stored GitHub issue IDs when the module is resumed.

Create the OpenAI key using the [OpenAI API quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request) and paste it into the same `run.py` configuration block. The model remains `gpt-5` by default, and the original API-call style is retained. Prompts are read from the module's `prompts/` directory.

To skip this complete module, use the released output already tracked as `data/llm_filtered_issues.zip`. The Step 2 runner detects and extracts that archive automatically when no newly generated Step 1 output exists.

The [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing) also contains `all_issues_new.zip`. This is the complete raw issue corpus produced by this crawler before rule-based and LLM filtering. It is provided for inspecting or re-filtering the released crawl and is not required when running the module from the beginning.
