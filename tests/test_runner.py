"""Focused tests for the synchronous Popen worker boundary."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from black_box_optimizer.models import CandidateConfiguration, WorkerSpec
from black_box_optimizer.runner import _bounded_message, _output_text, execute


def make_worker_spec(timeout: float = 30.0) -> WorkerSpec:
    return WorkerSpec(
        command=("python", "worker.py"),
        metrics_argument="--metrics-out",
        timeout_seconds=timeout,
    )


def make_candidate() -> CandidateConfiguration:
    return CandidateConfiguration(
        parameters={"first_value": 3, "second": "careful"}
    )


class RunnerTests(unittest.TestCase):
    def test_command_order_and_shell_false(self) -> None:
        process = Mock(returncode=0)
        process.communicate.return_value = ("output", "")

        with patch("black_box_optimizer.runner.subprocess.Popen") as popen:
            popen.return_value = process
            result = execute(
                make_worker_spec(),
                make_candidate(),
                Path("trial") / "metrics.csv",
            )

        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                "python",
                "worker.py",
                "--first-value",
                "3",
                "--second",
                "careful",
                "--metrics-out",
                str(Path("trial") / "metrics.csv"),
            ],
        )
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["stdout"], "output")

    def test_nonzero_exit_uses_last_nonblank_stderr_line(self) -> None:
        process = Mock(returncode=7)
        process.communicate.return_value = (
            "full stdout",
            "traceback line\nactual cause\n\n",
        )

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(result["execution_status"], "process_failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(
            result["error_message"],
            "Worker exited with code 7: actual cause",
        )
        self.assertEqual(
            result["stderr"], "traceback line\nactual cause\n\n"
        )

    def test_large_error_message_is_bounded_to_one_thousand_chars(self) -> None:
        process = Mock(returncode=1)
        process.communicate.return_value = ("", "x" * 2_000)

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(len(result["error_message"]), 1_000)
        self.assertTrue(result["error_message"].startswith("Worker exited"))
        self.assertEqual(len(result["stderr"]), 2_000)

    def test_empty_stderr_uses_stable_exit_code_fallback(self) -> None:
        process = Mock(returncode=3)
        process.communicate.return_value = ("", "")

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(result["error_message"], "Worker exited with code 3")

    def test_timeout_terminates_and_collects_output(self) -> None:
        process = Mock(returncode=-15)
        process.communicate.side_effect = (
            subprocess.TimeoutExpired(["worker"], 0.5),
            ("partial output", "still running"),
        )

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(timeout=0.5),
                make_candidate(),
                "metrics.csv",
            )

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertEqual(
            process.communicate.call_args_list[1].kwargs["timeout"],
            2.0,
        )
        self.assertEqual(result["execution_status"], "timed_out")
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["stdout"], "partial output")

    def test_interrupt_terminates_worker_and_returns_cancelled(self) -> None:
        process = Mock(returncode=-15)
        process.communicate.side_effect = (
            KeyboardInterrupt(),
            ("partial output", "cancelled cleanly"),
        )

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertEqual(result["execution_status"], "cancelled")
        self.assertFalse(result["timed_out"])
        self.assertEqual(
            result["error_message"],
            "Worker cancelled by user: cancelled cleanly",
        )

    def test_interrupt_kills_worker_after_two_second_grace(self) -> None:
        process = Mock(returncode=-9)
        process.communicate.side_effect = (
            KeyboardInterrupt(),
            subprocess.TimeoutExpired(["worker"], 2.0),
            ("", "forced shutdown"),
        )

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(result["execution_status"], "cancelled")

    def test_launch_error_is_bounded_and_has_empty_streams(self) -> None:
        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            side_effect=FileNotFoundError("missing worker"),
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(result["execution_status"], "launch_failed")
        self.assertIn("missing worker", result["error_message"])
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")

    def test_non_ascii_output_is_preserved(self) -> None:
        process = Mock(returncode=1)
        process.communicate.return_value = ("résultat", "原因")

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(result["stdout"], "résultat")
        self.assertEqual(result["stderr"], "原因")
        self.assertIn("原因", result["error_message"])

    def test_output_is_not_speculatively_redacted(self) -> None:
        process = Mock(returncode=1)
        process.communicate.return_value = (
            "",
            "worker detail TOKEN=example-value",
        )

        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            return_value=process,
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertIn("TOKEN=example-value", result["stderr"])
        self.assertIn("TOKEN=example-value", result["error_message"])

    def test_interrupt_before_launch_completes_is_cancelled(self) -> None:
        # Ctrl+C landing while Popen() itself is still starting the child
        # process, before there's even a process to terminate/kill.
        with patch(
            "black_box_optimizer.runner.subprocess.Popen",
            side_effect=KeyboardInterrupt(),
        ):
            result = execute(
                make_worker_spec(), make_candidate(), "metrics.csv"
            )

        self.assertEqual(result["execution_status"], "cancelled")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["timed_out"])
        self.assertIn("before launch completed", result["error_message"])


class BoundedMessageTests(unittest.TestCase):
    """_bounded_message() directly -- the truncation math has an edge case
    (a prefix that alone already meets or exceeds the limit) that no real
    caller in this file can currently produce, since every prefix runner.py
    builds is short and fixed-shape."""

    def test_detail_is_appended_within_the_limit(self) -> None:
        message = _bounded_message("Worker failed", "some detail")
        self.assertEqual(message, "Worker failed: some detail")

    def test_prefix_alone_at_or_over_the_limit_drops_the_detail(self) -> None:
        long_prefix = "x" * 1_000
        message = _bounded_message(long_prefix, "detail that cannot fit")
        self.assertEqual(message, long_prefix)
        self.assertEqual(len(message), 1_000)


class OutputTextTests(unittest.TestCase):
    """_output_text() directly -- runner.py always opens Popen with
    text=True, so communicate() never actually returns bytes in real
    usage; this pins down the conversion helper's own contract instead."""

    def test_none_becomes_empty_string(self) -> None:
        self.assertEqual(_output_text(None), "")

    def test_bytes_are_decoded_as_utf8(self) -> None:
        self.assertEqual(_output_text("café".encode("utf-8")), "café")

    def test_str_passes_through_unchanged(self) -> None:
        self.assertEqual(_output_text("already text"), "already text")


if __name__ == "__main__":
    unittest.main()
