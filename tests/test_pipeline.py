from __future__ import annotations

import collections
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleasedDataTests(unittest.TestCase):
    def test_training_dataset_counts_and_balance(self) -> None:
        expected = {
            "trace_all_sorted.json": (3546, {"yes": 1773, "no": 1773}),
            "trace_all_thinking_summary_by_gemini_sorted.json": (
                2328,
                {"yes": 1164, "no": 1164},
            ),
        }
        data_dir = ROOT / "03_Fine_Tuning" / "data"
        for name, (expected_count, expected_labels) in expected.items():
            with (data_dir / name).open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            labels = collections.Counter(
                str(row.get("judgement", {}).get("is_bug", "")).lower()
                for row in rows
            )
            self.assertEqual(len(rows), expected_count, name)
            self.assertEqual(dict(labels), expected_labels, name)

    def test_final_runner_has_an_adapter_preflight(self) -> None:
        runner = (
            ROOT / "04_Post_Processing_Verification" / "run.py"
        ).read_text(encoding="utf-8")
        self.assertIn("adapter_config.json", runner)
        self.assertIn("adapter_model.safetensors", runner)
        self.assertIn("require_adapter(ADAPTER_DIR)", runner)

    def test_each_method_module_has_one_runner(self) -> None:
        self.assertFalse((ROOT / "run.py").exists())
        for module in (
            "01_NCF_Bug_Report_Collection",
            "02_Trace_Pair_Dataset_Construction",
            "03_Fine_Tuning",
            "04_Post_Processing_Verification",
        ):
            self.assertTrue((ROOT / module / "run.py").is_file(), module)


if __name__ == "__main__":
    unittest.main()
