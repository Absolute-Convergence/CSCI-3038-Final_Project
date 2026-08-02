"""End-to-end module CLI acceptance test with an opaque fixture worker."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_application import write_configuration


_REPOSITORY = Path(__file__).resolve().parents[2]


class CliAcceptanceTests(unittest.TestCase):
    def test_module_cli_runs_worker_and_reports_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            configuration_path = directory / "configuration.json"
            output_directory = directory / "runs"
            write_configuration(configuration_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "black_box_optimizer",
                    str(configuration_path),
                    "--output-dir",
                    str(output_directory),
                ],
                cwd=_REPOSITORY,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Status: completed", completed.stdout)
            self.assertIn(
                "Termination reason: maximum_trials",
                completed.stdout,
            )
            run_directories = tuple(output_directory.iterdir())
            self.assertEqual(len(run_directories), 1)
            self.assertTrue(
                (run_directories[0] / "pareto_front.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
