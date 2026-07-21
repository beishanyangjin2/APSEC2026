# NCF Bug-Report Filtering

This module reduces the crawled GitHub issues to reports that are likely to describe Android non-crash functional bugs.

## Pipeline

1. `filtering_issues_keyword.ipynb` applies keyword, label, and repository exclusion rules.
2. `filtering_issues_llm.ipynb` performs semantic filtering with a configurable LLM provider.

The intermediate outputs currently included in this folder are:

- `keyword_filtered_issues.zip`: reports retained by keyword-based filtering;
- `llm_filtered_issues.zip`: reports retained by LLM-based semantic filtering.

## Input data

Download `all_issues_new.zip` from the [FineDroid Project Data folder](https://drive.google.com/drive/folders/1Gcd3DOvtz2brau-ETGziX9KAn1119fgM?usp=sharing), extract it, and set `INPUT_DIR` in `filtering_issues_keyword.ipynb` to the extracted issue directory.

Configure the model name, provider, and API credential before running the semantic filter. Do not commit API credentials; use an environment variable or the `xxx` placeholder. The selected reports are used by the [training-data generation module](../Training_Data_generation/readme.md).
