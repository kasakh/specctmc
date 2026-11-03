from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = dict(os.environ)
ENVIRONMENT["PYTHONPATH"] = str(ROOT / "src")


class CommandLineDryRunTests(unittest.TestCase):
    def run_main(self, family: str, method: str, *extra: str) -> None:
        command = [
            sys.executable,
            "-m",
            "specctmc.run",
            "--model-family",
            family,
            "--method",
            method,
            "--model-path",
            "/path/supplied/by/user",
            "--prompt",
            "test prompt",
            "--dry-run",
            *extra,
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=ENVIRONMENT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"dry_run": true', completed.stdout)

    def test_all_main_methods_parse_without_model_access(self):
        self.run_main("llada", "direct")
        self.run_main("llada", "spec")
        self.run_main("llada", "tau")
        self.run_main("llada", "trapezoidal")
        self.run_main(
            "llada", "cached-direct", "--refresh-interval", "16", "--target-batch-size", "4"
        )
        self.run_main(
            "llada", "cached-spec", "--refresh-interval", "16", "--target-batch-size", "4"
        )
        self.run_main("dream", "direct")
        self.run_main("dream", "spec")

    def test_s2d2_methods_parse_without_model_access(self):
        for method in ("direct", "s2d2-spec"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/run_s2d2.py",
                    "--method",
                    method,
                    "--model-factory",
                    "user_loader:load",
                    "--model-path",
                    "/path/supplied/by/user",
                    "--prompt",
                    "test prompt",
                    "--mask-id",
                    "1",
                    "--eos-id",
                    "2",
                    "--dry-run",
                ],
                cwd=ROOT,
                env=ENVIRONMENT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"dry_run": true', completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
