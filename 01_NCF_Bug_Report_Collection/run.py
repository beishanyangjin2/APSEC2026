from __future__ import annotations

import crawl_issues
import discover_repositories
import filter_reports_by_rules
import filter_reports_with_llm


# Fill in both keys before running the complete module.
GITHUB_TOKEN = "xxxx"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5"


def main() -> None:
    discover_repositories.main()

    crawl_issues.GITHUB_TOKEN = GITHUB_TOKEN
    crawl_issues.REPOSITORIES_CSV = discover_repositories.OUTPUT_CSV
    crawl_issues.main()

    filter_reports_by_rules.INPUT_DIR = crawl_issues.OUTPUT_DIR
    filter_reports_by_rules.main()

    filter_reports_with_llm.OPENAI_API_KEY = OPENAI_API_KEY
    filter_reports_with_llm.MODEL_NAME = OPENAI_MODEL
    filter_reports_with_llm.INPUT_DIR = filter_reports_by_rules.OUTPUT_DIR
    filter_reports_with_llm.main()


if __name__ == "__main__":
    main()
