"""Tests for the global source-file hygiene checker."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.check_monoliths import (
    ConfigurationError,
    check_repository,
    load_configuration,
    main,
    write_report,
)


class SourceHygieneCheckerTests(unittest.TestCase):
    """Verify hard limits, exceptions, exclusions, and advisories."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def write_configuration(
        self,
        *,
        maximum: int = 3,
        preferred: int = 10,
        exclusions: list[str] | None = None,
        exceptions: list[dict[str, str]] | None = None,
    ) -> Path:
        path = self.root / "source_hygiene.json"
        path.write_text(
            json.dumps(
                {
                    "max_physical_lines": maximum,
                    "preferred_line_length": preferred,
                    "source_extensions": [".py"],
                    "exclude_paths": exclusions or ["excluded/**"],
                    "approved_exceptions": exceptions or [],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_under_limit_passes(self) -> None:
        (self.root / "small.py").write_text("one\ntwo\n", encoding="utf-8")
        configuration = load_configuration(self.write_configuration())

        report = check_repository(self.root, configuration)

        self.assertTrue(report.passed)
        self.assertEqual(report.files[0].physical_lines, 2)
        self.assertEqual(report.unapproved_oversized_files, ())

    def test_file_at_exact_limit_passes(self) -> None:
        (self.root / "limit.py").write_text(
            "one\ntwo\nthree\n", encoding="utf-8"
        )
        configuration = load_configuration(self.write_configuration(maximum=3))

        report = check_repository(self.root, configuration)

        self.assertTrue(report.passed)
        self.assertEqual(report.files[0].physical_lines, 3)

    def test_unapproved_oversized_file_fails(self) -> None:
        (self.root / "large.py").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )
        configuration = load_configuration(self.write_configuration())

        report = check_repository(self.root, configuration)

        self.assertFalse(report.passed)
        self.assertEqual(report.unapproved_oversized_files[0].path, "large.py")
        output = io.StringIO()
        write_report(report, output)
        self.assertIn("large.py: 4 lines", output.getvalue())

    def test_command_returns_failure_for_oversized_file(self) -> None:
        (self.root / "large.py").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )
        self.write_configuration()
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--root", str(self.root)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Source hygiene check: FAIL", output.getvalue())

    def test_complete_human_approval_allows_oversized_file(self) -> None:
        (self.root / "cohesive.py").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )
        exception = {
            "path": "cohesive.py",
            "approved_by": "Mel Bailey",
            "approved_on": "2026-07-21",
            "coherent_responsibility": (
                "This file implements one cohesive parser state machine."
            ),
            "why_splitting_is_worse": (
                "Splitting would scatter transitions and obscure invariants."
            ),
        }
        configuration = load_configuration(
            self.write_configuration(exceptions=[exception])
        )

        report = check_repository(self.root, configuration)

        self.assertTrue(report.passed)
        self.assertEqual(report.approved_oversized_files[0].path, "cohesive.py")

    def test_approval_requires_both_written_explanations(self) -> None:
        exception = {
            "path": "cohesive.py",
            "approved_by": "Mel Bailey",
            "approved_on": "2026-07-21",
            "coherent_responsibility": "Too short",
            "why_splitting_is_worse": (
                "Splitting would scatter transitions and obscure invariants."
            ),
        }

        with self.assertRaises(ConfigurationError):
            load_configuration(
                self.write_configuration(exceptions=[exception])
            )

    def test_approval_rejects_automated_approver(self) -> None:
        exception = {
            "path": "cohesive.py",
            "approved_by": "Codex",
            "approved_on": "2026-07-21",
            "coherent_responsibility": (
                "This file implements one cohesive parser state machine."
            ),
            "why_splitting_is_worse": (
                "Splitting would scatter transitions and obscure invariants."
            ),
        }

        with self.assertRaises(ConfigurationError):
            load_configuration(
                self.write_configuration(exceptions=[exception])
            )

    def test_explicitly_excluded_source_is_not_scanned(self) -> None:
        excluded = self.root / "excluded"
        excluded.mkdir()
        (excluded / "generated.py").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )
        configuration = load_configuration(self.write_configuration())

        report = check_repository(self.root, configuration)

        self.assertTrue(report.passed)
        self.assertEqual(report.files, ())

    def test_long_lines_are_advisory_only(self) -> None:
        (self.root / "wide.py").write_text(
            "this line is deliberately wide\n", encoding="utf-8"
        )
        configuration = load_configuration(
            self.write_configuration(maximum=3, preferred=10)
        )

        report = check_repository(self.root, configuration)

        self.assertTrue(report.passed)
        self.assertEqual(report.files[0].long_line_numbers, (1,))


if __name__ == "__main__":
    unittest.main()
