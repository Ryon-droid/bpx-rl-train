"""Qualification must not hide a weak direction or accept incomplete reports."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("stair_suite", Path(__file__).resolve().parents[1] /
                                            "scripts/summarize_stair_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class StairSuiteTests(unittest.TestCase):
    def write_suite(self, folder):
        for condition in ("nominal", "robust"):
            for direction in ("forward", "backward"):
                for seed in (2026, 2027, 2028):
                    report = {"checkpoint": "/teacher.pt", "height_m": 0.15, "num_envs": 64,
                              "seed": seed, "condition": condition,
                              "speed_m_s": 0.3 if direction == "forward" else -0.3,
                              "attempts": [{"env": i, "terrain": "up" if i < 32 else "down",
                                            "success": True, "physical_failure": False,
                                            "lateral_offset_m": 0.0} for i in range(64)]}
                    (folder / f"test_{condition}_{direction}_{seed}.json").write_text(json.dumps(report))

    def test_weak_backward_descent_cannot_be_averaged_away(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_suite(folder)
            self.assertTrue(suite.summarize(folder, "test")["teacher_passed"])
            path = folder / "test_nominal_backward_2026.json"
            report = json.loads(path.read_text())
            for attempt in report["attempts"][32:37]:
                attempt.update(success=False, physical_failure=True)
            path.write_text(json.dumps(report))
            result = suite.summarize(folder, "test")
            self.assertFalse(result["teacher_passed"])
            failed = [row for row in result["rows"] if not row["passed"]]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["successes"], 91)
            self.assertEqual(failed[0]["physical_failures"], 5)

    def test_missing_duplicate_or_wrong_height_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_suite(folder)
            path = folder / "test_robust_forward_2027.json"
            original = json.loads(path.read_text())
            for change in ("missing_attempt", "duplicate", "height", "physical_success"):
                report = copy.deepcopy(original)
                if change == "missing_attempt":
                    report["attempts"].pop()
                elif change == "duplicate":
                    report["attempts"][-1]["env"] = 0
                elif change == "height":
                    report["height_m"] = 0.16
                else:
                    report["attempts"][0]["physical_failure"] = True
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    suite.summarize(folder, "test")
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                suite.summarize(folder, "test")


if __name__ == "__main__":
    unittest.main()
