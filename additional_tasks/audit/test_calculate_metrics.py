"""Metrics use document-local identities and matching denominator units."""
import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from additional_tasks.audit import calculate_metrics as metrics_script
from additional_tasks.audit.calculate_metrics import calculate_metrics


def candidate(kind, index, excluded=False):
    return {"type": kind, "pseudonym": f"{kind}_{index}", "excluded": excluded}


class MetricsTests(unittest.TestCase):
    def test_repeats_exclusions_and_document_local_identity(self):
        rows = [
            {"text": "ОСОБА_1 ОСОБА_1 АДРЕСА_1", "candidates": [
                candidate("ОСОБА", 1), candidate("ОСОБА", 1),
                candidate("ОСОБА", 2, excluded=True), candidate("АДРЕСА", 1)]},
            {"text": "ОСОБА_1", "candidates": [candidate("ОСОБА", 1)]},
            {"text": "НОМЕР_1", "candidates": [candidate("ОСОБА", 1, excluded=True)]},
            {"text": "Без масок та знахідок", "candidates": []},
        ]
        metrics = calculate_metrics(rows)
        self.assertEqual(metrics["documents"], 4)
        self.assertEqual(metrics["documents_with_leaks_pct"], 50)
        people = metrics["categories"]["ОСОБА"]
        self.assertEqual(people["leaked_unique"], 2)
        self.assertEqual(people["anonymized_unique"], 2)
        self.assertEqual(people["leaked_unique_share_pct"], 50)
        self.assertEqual(people["documents_with_leaks_pct"], 50)
        self.assertEqual(metrics["categories"]["АДРЕСА"]["documents_with_leaks_pct"], 25)
        self.assertEqual(metrics["categories"]["НОМЕР"]["leaked_unique_share_pct"], 0)
        self.assertIsNone(metrics["categories"]["ІНФОРМАЦІЯ"]["leaked_unique_share_pct"])

    def test_nonexcluded_repeat_survives_and_leaks_without_masks(self):
        metrics = calculate_metrics([{"text": "Без масок", "candidates": [
            candidate("ОСОБА", 1, excluded=True), candidate("ОСОБА", 1)]}])
        self.assertEqual(metrics["categories"]["ОСОБА"]["leaked_unique_share_pct"], 100)
        self.assertEqual(metrics["categories"]["ОСОБА"]["leaked_unique"], 1)

    def test_masks_are_complete_tokens(self):
        metrics = calculate_metrics([{"text": "ОСОБА_1, ОСОБА_2. НЕОСОБА_3 ОСОБА_4abc", "candidates": []}])
        self.assertEqual(metrics["categories"]["ОСОБА"]["anonymized_unique"], 2)

    def test_raw_results_and_missing_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline pseudonyms"):
            calculate_metrics([{"text": "Текст", "candidates": [{"type": "ОСОБА", "excluded": False}]}])
        with self.assertRaisesRegex(ValueError, "published text"):
            calculate_metrics([{"candidates": []}])
        with self.assertRaisesRegex(ValueError, "No documents"):
            calculate_metrics([])

    def test_cli_markdown_default_and_custom_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            directory = root / "results"
            directory.mkdir()
            row = {"text": "ОСОБА_1 приватне значення", "candidates": [candidate("ОСОБА", 2)]}
            (directory / "candidates.jsonl").write_text(json.dumps(row), encoding="utf8")
            for md_args, expected in [
                    (["--md"], scripts / "METRICS.md"),
                    (["--md", str(root / "custom" / "report.md")], root / "custom" / "report.md")]:
                with self.subTest(md_args=md_args), patch.object(metrics_script, "HERE", scripts), \
                        patch("sys.argv", ["calculate_metrics.py", str(directory), *md_args]), \
                        redirect_stdout(io.StringIO()):
                    metrics_script.main()
                report = expected.read_text(encoding="utf8")
                self.assertIn("**100.00%**", report)
                self.assertIn("| ОСОБА | 1 | 1 | 2 | 50.00% |", report)
                self.assertIn("| ІНФОРМАЦІЯ | 0 | 0 | 0 | — |", report)
                self.assertIn("Ручні оцінки не враховані", report)
                self.assertNotIn("приватне значення", report)
                self.assertTrue((directory / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
