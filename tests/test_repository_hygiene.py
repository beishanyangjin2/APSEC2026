from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u4e00-\u9fff]")
NON_ASCII_NAME = re.compile(r"[^\x00-\x7f]")
MARKDOWN_LINK = re.compile(r"\[[^]]*]\(([^)]+)\)")


class RepositoryHygieneTests(unittest.TestCase):
    def test_source_documentation_and_prompts_are_english(self) -> None:
        candidates = [
            *ROOT.rglob("*.py"),
            *ROOT.rglob("*.md"),
            *(
                path
                for module in (
                    ROOT / "01_NCF_Bug_Report_Collection",
                    ROOT / "02_Trace_Pair_Dataset_Construction",
                    ROOT / "03_Fine_Tuning",
                    ROOT / "04_Post_Processing_Verification",
                )
                for path in (module / "prompts").glob("*.txt")
            ),
        ]
        candidates = [path for path in candidates if ".git" not in path.parts]
        matches = [
            str(path.relative_to(ROOT))
            for path in candidates
            if CJK.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(matches, [])

    def test_repository_file_names_are_ascii(self) -> None:
        matches = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if ".git" not in path.parts
            and NON_ASCII_NAME.search(str(path.relative_to(ROOT)))
        ]
        self.assertEqual(matches, [])

    def test_obsolete_notebooks_are_removed(self) -> None:
        notebooks = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*.ipynb")
            if ".git" not in path.parts
        ]
        self.assertEqual(notebooks, [])

    def test_relative_markdown_links_exist(self) -> None:
        missing: list[str] = []
        for path in ROOT.rglob("*.md"):
            if ".git" in path.parts:
                continue
            for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
                target = target.strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = path.parent / unquote(target)
                if not resolved.exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
